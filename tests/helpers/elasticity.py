# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Shared harness for the end-to-end KV-cache elasticity checks.

Everything an elasticity run observes -- the /dev/shm segment that `kvtop` and
`kvctl` read, the mapped-footprint samples, the limit cut -- is engine
agnostic. Only starting the engine and issuing a request differ, so the engine
scripts supply an `Engine` and this module runs the phases:

  1. idle baseline       -> small mapped footprint (lazy)
  2. heavy batch         -> footprint GROWS (mem_map on demand)
  3. drain (idle)        -> footprint falls as freed blocks are unmapped
  4. forced limit cut    -> kvctl-style limit cut (informational; see note)
  5. recover + check     -> engine healthy after shrink, output unchanged

Note: prefix reuse MUST be off, or finished requests keep their KV resident and
no shrink is observable. The forced limit-cut phase is informational only --
with the natural drain already reclaiming freed pages, it does not
independently exercise eviction of *held* (prefix-cached) blocks; that
multi-tenant giveback path needs a dedicated test.
"""
from __future__ import annotations

import glob
import hashlib
import os
import threading
import time
from typing import List, Optional, Protocol, Sequence, Tuple

from kvcached.cli.utils import get_kv_cache_limit, update_kv_cache_limit

MB = 1024 * 1024

# The same hybrid model the engine smoke tests use, for the same reason: a
# GQA-only model exercises none of the KV layouts where kvcached's bugs have
# actually been. Its per-block recurrent state does not fit the default 2MB
# page, so raise the page size before kvcached initializes -- both engines
# inherit this environment when they fork their worker.
DEFAULT_MODEL = os.getenv("KVCACHED_TEST_MODEL", "Qwen/Qwen3.5-4B")
os.environ.setdefault("KVCACHED_PAGE_SIZE_MB", "4")

PROBE_PROMPT = "The capital of France is"


class Engine(Protocol):
    """The two things an elasticity run needs from a serving engine."""

    name: str

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int,
        temperature: float,
        seed: Optional[int] = None,
    ) -> List[str]:
        ...


def list_segments() -> set:
    return {os.path.basename(p) for p in glob.glob("/dev/shm/kvcached_*")}


def read_seg(name: str):
    mi = get_kv_cache_limit(name)
    return None if mi is None else (mi.total_size, mi.used_size,
                                    mi.prealloc_size)


def fmt(v) -> str:
    return f"{v / MB:8.1f} MB" if v is not None else "   n/a"


class _Sampler:
    """Poll the IPC segment on a background thread for the whole run."""

    def __init__(self) -> None:
        self.samples: List[Tuple[float, int, int, int]] = []
        self.seg_name: Optional[str] = None
        self._stop = threading.Event()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def now(self) -> float:
        return time.time() - self._t0

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.seg_name is not None:
                v = read_seg(self.seg_name)
                if v is not None:
                    self.samples.append((self.now(), *v))
            time.sleep(0.2)

    def used_now(self) -> Optional[int]:
        v = read_seg(self.seg_name) if self.seg_name else None
        return v[1] if v else None

    def peak_used(self, t_lo: float, t_hi: float) -> Optional[int]:
        xs = [u for (t, _total, u, _p) in self.samples if t_lo <= t <= t_hi]
        return max(xs) if xs else None


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:10]


def run(engine_factory) -> None:
    """Run the five phases against the engine `engine_factory()` returns.

    The factory is called only after the segment watcher is running, so the
    segment this run creates can be told apart from any other process's.
    """
    before = list_segments()
    sampler = _Sampler()
    sampler.start()

    engine = engine_factory()
    print(f"=== {engine.name} engine built (kvcached) ===", flush=True)

    for _ in range(50):
        new = list_segments() - before
        if new:
            sampler.seg_name = sorted(new)[0]
            break
        time.sleep(0.2)
    print(f"[ipc] segment: {sampler.seg_name}", flush=True)
    assert sampler.seg_name is not None, "no kvcached IPC segment detected"

    base_txt = engine.generate([PROBE_PROMPT], max_tokens=24,
                               temperature=0.0)[0]
    base_md5 = _md5(base_txt)
    print(f"[correctness] baseline md5={base_md5} :: {base_txt!r}", flush=True)

    time.sleep(3.0)
    base_used = sampler.used_now()
    print(f"\n[PHASE 1] idle baseline      used={fmt(base_used)}", flush=True)

    print("[PHASE 2] heavy batch (grow) ...", flush=True)
    prompts = [
        f"Write a long, detailed essay number {i} about distributed systems, "
        f"GPU memory management, and virtual memory paging." for i in range(128)
    ]
    t_lo = sampler.now()
    engine.generate(prompts, max_tokens=1024, temperature=0.7, seed=1234)
    t_hi = sampler.now()
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
    update_kv_cache_limit(sampler.seg_name, small_limit)
    time.sleep(10.0)
    after_cut = read_seg(sampler.seg_name)
    print(f"[PHASE 4] after cut  total={fmt(after_cut[0])} "
          f"used={fmt(after_cut[1])} prealloc={fmt(after_cut[2])}", flush=True)

    update_kv_cache_limit(sampler.seg_name, total_before)
    time.sleep(2.0)
    post_txt = engine.generate([PROBE_PROMPT], max_tokens=24,
                               temperature=0.0)[0]
    post_md5 = _md5(post_txt)
    print(f"\n[PHASE 5] post-shrink md5={post_md5} :: {post_txt!r}", flush=True)

    sampler.stop()

    grew = (grow_peak or 0) > (base_used or 0) * 1.5
    shrank = (drained is not None and grow_peak is not None
              and drained < grow_peak)
    correct = post_md5 == base_md5
    print("\n==================== VERDICT ====================", flush=True)
    print(f"  engine        : {engine.name}")
    print(f"  baseline used : {fmt(base_used)}")
    print(f"  peak used     : {fmt(grow_peak)}")
    print(f"  drained used  : {fmt(drained)}")
    print(f"  GREW under load ........ {'PASS' if grew else 'FAIL'}")
    print(f"  SHRANK on free ......... {'PASS' if shrank else 'FAIL'}")
    print(f"  CORRECT after cycle .... {'PASS' if correct else 'FAIL'} "
          f"(base={base_md5} post={post_md5})")
    print("=================================================", flush=True)
    assert grew and shrank and correct, "elasticity check failed"
