# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""End-to-end KV-cache elasticity under load (vLLM offline engine).

Complements ``test_kvcache_manager.py`` (which exercises the manager-level
``resize``/``trim`` APIs directly) by driving the *real* engine and watching the
physically mapped KV footprint grow and shrink through the /dev/shm IPC that
``kvtop``/``kvctl`` read.

Phases:
  1. idle baseline       -> small mapped footprint (lazy)
  2. heavy batch         -> footprint GROWS (mem_map on demand)
  3. drain (idle)        -> footprint falls as freed blocks are unmapped
  4. forced limit cut    -> kvctl-style limit cut (informational; see note)
  5. recover + check     -> engine healthy after shrink, output unchanged

Validated on AMD MI300X (ROCm/HIP) to confirm the hipMemMap (grow) and
hipMemUnmap (shrink) paths, and on NVIDIA. vLLM selects the device, so nothing
here names one, but this file has never run on XPU: vLLM has no XPU build in the
Intel test environment. ``test_elastic_serving_sglang.py`` drives the grow and
shrink paths there through SGLang instead.

Run inside the engine venv with kvcached enabled:
    ENABLE_KVCACHED=true VLLM_USE_V1=1 pytest tests/test_elastic_serving.py -s

Note: prefix caching MUST be off (enable_prefix_caching=False) or finished
requests keep their KV resident and no shrink is observable. The forced
limit-cut phase is informational only -- with the natural drain already
reclaiming freed pages, it does not independently exercise eviction of *held*
(prefix-cached) blocks; that multi-tenant giveback path needs a dedicated test.
"""
import gc
import glob
import hashlib
import os
import threading
import time
from typing import Optional

import pytest

from kvcached.cli.utils import get_kv_cache_limit, update_kv_cache_limit
from kvcached.utils import get_device_module

pytest.importorskip("vllm")

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

    Per-run state has to stay off the module: as a test this can be collected
    more than once in a process, and a leftover stop flag or sample list from an
    earlier run would silently produce a verdict with no data behind it.
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
    from vllm import LLM

    pre = list_segments()
    with Sampler() as sampler:
        print("=== building offline vLLM engine (kvcached) ===", flush=True)
        llm = LLM(
            model=MODEL,
            enforce_eager=True,
            gpu_memory_utilization=0.40,
            max_model_len=8192,
            enable_prefix_caching=False,  # required: else freed KV stays resident
            disable_log_stats=True,
        )
        try:
            _check_elasticity(llm, sampler, pre)
        finally:
            # The engine holds gpu_memory_utilization of the card. Without this
            # it stays resident for the rest of the session and every later
            # accelerator test starts on a partly occupied device.
            del llm
            gc.collect()
            get_device_module().empty_cache()


def _check_elasticity(llm, sampler, pre):
    from vllm import SamplingParams

    for _ in range(50):
        new = list_segments() - pre
        if new:
            sampler.seg_name = sorted(new)[0]
            break
        time.sleep(0.2)
    print(f"[ipc] segment: {sampler.seg_name}", flush=True)
    assert sampler.seg_name is not None, "no kvcached IPC segment detected"

    det = SamplingParams(temperature=0.0, max_tokens=24)
    base_txt = llm.generate(["The capital of France is"], det)[0].outputs[0].text
    base_md5 = hashlib.md5(base_txt.encode()).hexdigest()[:10]
    print(f"[correctness] baseline md5={base_md5} :: {base_txt!r}", flush=True)

    time.sleep(3.0)
    base_used = sampler.used_now()
    print(f"\n[PHASE 1] idle baseline      used={fmt(base_used)}", flush=True)

    print("[PHASE 2] heavy batch (grow) ...", flush=True)
    prompts = [f"Write a long, detailed essay number {i} about distributed systems, "
               f"GPU memory management, and virtual memory paging." for i in range(128)]
    load = SamplingParams(temperature=0.7, max_tokens=1024, seed=1234)
    t_lo = sampler.elapsed()
    llm.generate(prompts, load)
    t_hi = sampler.elapsed()
    grow_peak = sampler.peak_used(t_lo, t_hi)
    print(f"[PHASE 2] peak used during load = {fmt(grow_peak)}", flush=True)

    for _ in range(18):
        time.sleep(1.0)
    drained = sampler.used_now()
    print(f"[PHASE 3] after drain        used={fmt(drained)}", flush=True)

    total_before = read_seg(sampler.seg_name)[0]
    small_limit = max(int(max(grow_peak or 0, 256 * MB) // 2), 256 * MB)
    print(f"\n[PHASE 4] limit {fmt(total_before)} -> {fmt(small_limit)} "
          f"(informational)", flush=True)
    try:
        update_kv_cache_limit(sampler.seg_name, small_limit)
        time.sleep(10.0)
        cur2 = read_seg(sampler.seg_name)
        print(f"[PHASE 4] after cut  total={fmt(cur2[0])} used={fmt(cur2[1])} "
              f"prealloc={fmt(cur2[2])}", flush=True)
    finally:
        # A cut left in place would starve the engine for whatever runs next.
        update_kv_cache_limit(sampler.seg_name, total_before)
    time.sleep(2.0)
    txt2 = llm.generate(["The capital of France is"], det)[0].outputs[0].text
    md5_2 = hashlib.md5(txt2.encode()).hexdigest()[:10]
    print(f"\n[PHASE 5] post-shrink md5={md5_2} :: {txt2!r}", flush=True)

    grew = (grow_peak or 0) > (base_used or 0) * 1.5
    shrank = drained is not None and grow_peak is not None and drained < grow_peak
    correct = md5_2 == base_md5
    print("\n==================== VERDICT ====================", flush=True)
    print(f"  baseline used : {fmt(base_used)}")
    print(f"  peak used     : {fmt(grow_peak)}")
    print(f"  drained used  : {fmt(drained)}")
    print(f"  GREW under load ........ {'PASS' if grew else 'FAIL'}")
    print(f"  SHRANK on free ......... {'PASS' if shrank else 'FAIL'}")
    print(f"  CORRECT after cycle .... {'PASS' if correct else 'FAIL'} "
          f"(base={base_md5} post={md5_2})")
    print("=================================================", flush=True)
    assert grew and shrank and correct, "elasticity check failed"


if __name__ == "__main__":
    test_kv_footprint_grows_and_shrinks_under_load()
