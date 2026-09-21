# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""End-to-end KV-cache elasticity under load (SGLang offline engine).

The twin of ``test_elastic_serving.py`` for the engine an Intel test environment
can actually run: that file drives vLLM, which has no XPU build here, so no real
engine has ever exercised the grow/shrink paths on Level Zero -- only the
manager-level APIs and the vmm_ops tests have.

Phases:
  1. idle baseline    -> small mapped footprint (lazy)
  2. heavy batch      -> footprint GROWS (mem_map on demand)
  3. drain (idle)     -> footprint falls as freed blocks are unmapped
  4. limit cut        -> engine keeps serving within a halved limit
  5. recover + check  -> engine healthy after the cycle, output unchanged

Phase 4 asserts rather than merely reporting, unlike the vLLM twin: the limit is
cut to a quarter of the observed peak, a smaller batch is served under it, and
the mapped footprint must stay inside the new limit. The quarter matters -- that
batch maps ~3.7 GB when the limit leaves it room and ~2.1 GB under a 2.1 GB cut,
so a limit that went unenforced would fail the phase rather than fit inside it by
luck. What it does not cover is giving back pages that live requests still hold:
with a drained pool there is nothing held to evict, so that multi-tenant path
still needs a test of its own.

The limit is addressed by the name ``kvctl list`` reports
(``kvcached_<engine>_<pid>``, written by the C++ MemInfoTracker), not by
DEFAULT_IPC_NAME; addressing the wrong segment silently measures nothing, which
is what ``test_kvcache_manager.py``'s skipped resize test runs into.

Radix caching MUST be off, or finished requests keep their KV resident and no
shrink is observable -- the same constraint the vLLM twin states for prefix
caching.

    ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=true KVCACHED_BACKEND=xpu \
        ZE_AFFINITY_MASK=0 pytest tests/test_elastic_serving_sglang.py -s
"""
import glob
import hashlib
import os
import site
import threading
import time
from typing import Optional

import pytest

# kvcached reaches SGLang through a wrapt when_imported("sglang") hook that arms
# itself only when KVCACHED_AUTOPATCH is set, so the hook has to be registered
# before SGLang is imported. Doing it here rather than relying on
# kvcached_autopatch.pth is what makes the test work from a source checkout: a
# PEP 660 editable install never runs setup.py's develop command, so it never
# lays that .pth down and nothing patches the pools.
os.environ.setdefault("KVCACHED_AUTOPATCH", "true")
# Two separate gates, and both are needed: KVCACHED_AUTOPATCH installs the
# elastic pools, while ENABLE_KVCACHED is what SchedulerMemoryLeakPatch checks
# before suppressing SGLang's pool-leak invariant. With only the first set, the
# pools are elastic but the check still runs against them, finds the slots
# kvcached has not mapped, and sigquits the scheduler during warmup.
os.environ.setdefault("ENABLE_KVCACHED", "true")
import kvcached.integration.sglang.autopatch  # noqa: E402,F401
from kvcached.cli.utils import (  # noqa: E402
    get_kv_cache_limit,
    update_kv_cache_limit,
)
from kvcached.utils import get_device_module  # noqa: E402

sgl = pytest.importorskip("sglang")

PTH = "kvcached_autopatch.pth"


def _autopatch_pth_installed() -> bool:
    """Will a freshly spawned interpreter patch SGLang too?

    SGLang runs its scheduler -- and therefore the KV pool -- in a subprocess
    that imports sglang from scratch, so patching this process is not enough:
    the .pth in site-packages is what arms the hook everywhere. Without it the
    engine would quietly serve from its own allocator, which is an unusable
    environment rather than a kvcached failure, hence a skip and not an assert.
    """
    roots = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    roots += [user_site] if isinstance(user_site, str) else list(user_site)
    return any(os.path.exists(os.path.join(root, PTH)) for root in roots)


if not _autopatch_pth_installed():
    pytest.skip(
        f"{PTH} is not in site-packages, so SGLang's scheduler subprocess would "
        "import sglang unpatched. `pip install kvcached` lays it down; a PEP 660 "
        f"editable install does not, so copy it: cp {PTH} $(python -c 'import "
        "site;print(site.getsitepackages()[0])')",
        allow_module_level=True,
    )

try:
    DEVICE_MODULE = get_device_module()
    _HAS_DEVICE = DEVICE_MODULE.is_available()
except Exception:
    _HAS_DEVICE = False

if not _HAS_DEVICE:
    pytest.skip("requires an accelerator", allow_module_level=True)

MODEL = os.getenv("KVCACHED_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MB = 1024 * 1024


def list_segments():
    return {os.path.basename(p) for p in glob.glob("/dev/shm/kvcached_*")}


def read_seg(name):
    mi = get_kv_cache_limit(name)
    return None if mi is None else (mi.total_size, mi.used_size, mi.prealloc_size)


def fmt(v):
    return f"{v / MB:8.1f} MB" if v is not None else "   n/a"


class Sampler:
    """Background poll of one run's IPC segment, scoped to a `with` block.

    Per-run state stays off the module for the same reason as in the vLLM twin:
    a leftover stop flag or sample list from an earlier collection would produce
    a verdict with no data behind it.
    """

    def __init__(self):
        self.samples: list[tuple[float, int, int, int]] = []  # t, total, used, prealloc
        self.seg_name: Optional[str] = None  # this run's segment, set after init
        self._stop = threading.Event()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _poll(self):
        while not self._stop.is_set():
            if self.seg_name is not None:
                v = read_seg(self.seg_name)
                if v is not None:
                    self.samples.append((self.elapsed(), *v))
            time.sleep(0.2)

    def elapsed(self):
        return time.time() - self._t0

    def used_now(self):
        v = read_seg(self.seg_name) if self.seg_name else None
        return v[1] if v else None

    def peak_used(self, t_lo, t_hi):
        xs = [u for (t, _t, u, _p) in self.samples if t_lo <= t <= t_hi]
        return max(xs) if xs else None


def test_kv_footprint_grows_and_shrinks_under_load():
    pre = list_segments()
    with Sampler() as sampler:
        print("=== building offline SGLang engine (kvcached) ===", flush=True)
        # The device is left to SGLang's own detection so this runs wherever the
        # engine runs; only the KV pool underneath it is kvcached's.
        engine = sgl.Engine(
            model_path=MODEL,
            disable_radix_cache=True,  # required: else freed KV stays resident
            disable_cuda_graph=True,
            mem_fraction_static=0.40,
            context_length=8192,
            tp_size=1,
            random_seed=1234,
            log_level="warning",
        )
        try:
            _check_elasticity(engine, sampler, pre)
        finally:
            # The engine holds mem_fraction_static of the card; leaving it
            # resident would make every later accelerator test in the session
            # start on a partly occupied device.
            engine.shutdown()
            get_device_module().empty_cache()


def _generate(engine, prompts, params):
    out = engine.generate(prompts, params)
    return [o["text"] for o in ([out] if isinstance(out, dict) else out)]


def _check_elasticity(engine, sampler, pre):
    for _ in range(50):
        new = list_segments() - pre
        if new:
            sampler.seg_name = sorted(new)[0]
            break
        time.sleep(0.2)
    print(f"[ipc] segment: {sampler.seg_name}", flush=True)
    assert sampler.seg_name is not None, (
        "no kvcached IPC segment detected: the SGLang pools were not patched, so "
        "this run would measure stock SGLang. Check KVCACHED_AUTOPATCH."
    )

    det = {"temperature": 0.0, "max_new_tokens": 24}
    base_txt = _generate(engine, ["The capital of France is"], det)[0]
    base_md5 = hashlib.md5(base_txt.encode()).hexdigest()[:10]
    print(f"[correctness] baseline md5={base_md5} :: {base_txt!r}", flush=True)

    time.sleep(3.0)
    base_used = sampler.used_now()
    print(f"\n[PHASE 1] idle baseline      used={fmt(base_used)}", flush=True)

    print("[PHASE 2] heavy batch (grow) ...", flush=True)
    prompts = [f"Write a long, detailed essay number {i} about distributed systems, "
               f"GPU memory management, and virtual memory paging." for i in range(128)]
    load = {"temperature": 0.7, "max_new_tokens": 1024}
    t_lo = sampler.elapsed()
    _generate(engine, prompts, load)
    t_hi = sampler.elapsed()
    grow_peak = sampler.peak_used(t_lo, t_hi)
    print(f"[PHASE 2] peak used during load = {fmt(grow_peak)}", flush=True)

    for _ in range(18):
        time.sleep(1.0)
    drained = sampler.used_now()
    print(f"[PHASE 3] after drain        used={fmt(drained)}", flush=True)

    total_before = read_seg(sampler.seg_name)[0]
    cut = max(int((grow_peak or 0) // 4), 256 * MB)
    print(f"\n[PHASE 4] limit {fmt(total_before)} -> {fmt(cut)}, then a "
          f"quarter-size batch", flush=True)
    cut_peak = None
    try:
        update_kv_cache_limit(sampler.seg_name, cut)
        # The resize watcher polls the segment every 100ms but lands the new
        # target on the next allocation, so the batch below is what applies it.
        time.sleep(1.0)
        t_lo = sampler.elapsed()
        _generate(engine, prompts[:32], load)
        t_hi = sampler.elapsed()
        cut_peak = sampler.peak_used(t_lo, t_hi)
    finally:
        # A cut left in place would starve whatever runs next in this session.
        update_kv_cache_limit(sampler.seg_name, total_before)
    print(f"[PHASE 4] peak used under the cut = {fmt(cut_peak)}", flush=True)

    time.sleep(2.0)
    txt2 = _generate(engine, ["The capital of France is"], det)[0]
    md5_2 = hashlib.md5(txt2.encode()).hexdigest()[:10]
    print(f"\n[PHASE 5] post-cycle md5={md5_2} :: {txt2!r}", flush=True)

    grew = (grow_peak or 0) > (base_used or 0) * 1.5
    shrank = drained is not None and grow_peak is not None and drained < grow_peak
    honored = cut_peak is not None and cut_peak <= cut
    correct = md5_2 == base_md5
    print("\n==================== VERDICT ====================", flush=True)
    print(f"  baseline used : {fmt(base_used)}")
    print(f"  peak used     : {fmt(grow_peak)}")
    print(f"  drained used  : {fmt(drained)}")
    print(f"  cut limit     : {fmt(cut)}  peak under it: {fmt(cut_peak)}")
    print(f"  GREW under load ........ {'PASS' if grew else 'FAIL'}")
    print(f"  SHRANK on free ......... {'PASS' if shrank else 'FAIL'}")
    print(f"  LIMIT CUT HONORED ...... {'PASS' if honored else 'FAIL'}")
    print(f"  CORRECT after cycle .... {'PASS' if correct else 'FAIL'} "
          f"(base={base_md5} post={md5_2})")
    print("=================================================", flush=True)
    assert grew and shrank and honored and correct, "elasticity check failed"


if __name__ == "__main__":
    test_kv_footprint_grows_and_shrinks_under_load()
