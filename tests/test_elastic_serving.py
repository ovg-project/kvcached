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
hipMemUnmap (shrink) paths; runs on NVIDIA too (device "cuda:0").

Run inside the engine venv with kvcached enabled:
    ENABLE_KVCACHED=true VLLM_USE_V1=1 python tests/test_elastic_serving.py

Note: prefix caching MUST be off (enable_prefix_caching=False) or finished
requests keep their KV resident and no shrink is observable. The forced
limit-cut phase is informational only -- with the natural drain already
reclaiming freed pages, it does not independently exercise eviction of *held*
(prefix-cached) blocks; that multi-tenant giveback path needs a dedicated test.
"""
import glob
import hashlib
import os
import threading
import time
from typing import Optional

from kvcached.cli.utils import get_kv_cache_limit, update_kv_cache_limit

MODEL = os.getenv("KVCACHED_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MB = 1024 * 1024


def list_segments():
    return {os.path.basename(p) for p in glob.glob("/dev/shm/kvcached_*")}


def read_seg(name):
    mi = get_kv_cache_limit(name)
    return None if mi is None else (mi.total_size, mi.used_size, mi.prealloc_size)


def fmt(v):
    return f"{v / MB:8.1f} MB" if v is not None else "   n/a"


samples: list[tuple[float, int, int, int]] = []  # (t, total, used, prealloc)
seg_name: list[Optional[str]] = [None]            # this run's segment, set after init
stop = threading.Event()


def sampler(t0):
    while not stop.is_set():
        nm = seg_name[0]
        if nm is not None:
            v = read_seg(nm)
            if v is not None:
                samples.append((time.time() - t0, *v))
        time.sleep(0.2)


def used_now():
    nm = seg_name[0]
    v = read_seg(nm) if nm else None
    return v[1] if v else None


def peak_used(t_lo, t_hi):
    xs = [u for (t, _t, u, _p) in samples if t_lo <= t <= t_hi]
    return max(xs) if xs else None


def main():
    pre = list_segments()
    t0 = time.time()
    threading.Thread(target=sampler, args=(t0,), daemon=True).start()

    print("=== building offline vLLM engine (kvcached) ===", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        gpu_memory_utilization=0.40,
        max_model_len=8192,
        enable_prefix_caching=False,  # required: else freed KV stays resident
        disable_log_stats=True,
    )

    for _ in range(50):
        new = list_segments() - pre
        if new:
            seg_name[0] = sorted(new)[0]
            break
        time.sleep(0.2)
    print(f"[ipc] segment: {seg_name[0]}", flush=True)
    assert seg_name[0] is not None, "no kvcached IPC segment detected"

    det = SamplingParams(temperature=0.0, max_tokens=24)
    base_txt = llm.generate(["The capital of France is"], det)[0].outputs[0].text
    base_md5 = hashlib.md5(base_txt.encode()).hexdigest()[:10]
    print(f"[correctness] baseline md5={base_md5} :: {base_txt!r}", flush=True)

    time.sleep(3.0)
    base_used = used_now()
    print(f"\n[PHASE 1] idle baseline      used={fmt(base_used)}", flush=True)

    print("[PHASE 2] heavy batch (grow) ...", flush=True)
    prompts = [f"Write a long, detailed essay number {i} about distributed systems, "
               f"GPU memory management, and virtual memory paging." for i in range(128)]
    load = SamplingParams(temperature=0.7, max_tokens=1024, seed=1234)
    t_lo = time.time() - t0
    llm.generate(prompts, load)
    t_hi = time.time() - t0
    grow_peak = peak_used(t_lo, t_hi)
    print(f"[PHASE 2] peak used during load = {fmt(grow_peak)}", flush=True)

    drain_series = []
    for _ in range(18):
        time.sleep(1.0)
        drain_series.append(used_now())
    drained = used_now()
    print(f"[PHASE 3] after drain        used={fmt(drained)}", flush=True)

    cur = read_seg(seg_name[0])
    total_before = cur[0]
    small_limit = max(int(max(grow_peak or 0, 256 * MB) // 2), 256 * MB)
    print(f"\n[PHASE 4] limit {fmt(total_before)} -> {fmt(small_limit)} "
          f"(informational)", flush=True)
    update_kv_cache_limit(seg_name[0], small_limit)
    time.sleep(10.0)
    cur2 = read_seg(seg_name[0])
    print(f"[PHASE 4] after cut  total={fmt(cur2[0])} used={fmt(cur2[1])} "
          f"prealloc={fmt(cur2[2])}", flush=True)

    update_kv_cache_limit(seg_name[0], total_before)
    time.sleep(2.0)
    txt2 = llm.generate(["The capital of France is"], det)[0].outputs[0].text
    md5_2 = hashlib.md5(txt2.encode()).hexdigest()[:10]
    print(f"\n[PHASE 5] post-shrink md5={md5_2} :: {txt2!r}", flush=True)

    stop.set()
    time.sleep(0.5)

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
    main()
