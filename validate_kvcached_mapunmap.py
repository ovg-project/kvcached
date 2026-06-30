#!/usr/bin/env python
"""Validate kvcached on-demand KV-page MAP / UNMAP on vLLM 0.20 (NVIDIA/CUDA).

What it proves
--------------
kvcached reserves GPU *virtual* memory up front and maps *physical* 2 MB pages
on demand as the KV cache grows, then unmaps them when the cache is freed.
We watch physical GPU memory (``torch.cuda.mem_get_info``) with a background
sampler across three phases:

  1. idle  : right after model load — only kvcached's prealloc pages are mapped
  2. load  : a big batch with long outputs — KV grows → pages MAP → free drops
  3. settle: requests finished, blocks freed → pages UNMAP → free recovers

Run both arms to see the contrast:
  baseline (no kvcached): vLLM pre-allocates the whole KV pool at load → free is
  low from the start and *flat* (no map/unmap dynamics).
  kvcached: free starts high, dips under load (MAP), recovers after (UNMAP).

Usage
-----
  # kvcached arm
  ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_GPU_UTILIZATION=0.6 \
    python validate_kvcached_mapunmap.py Qwen/Qwen3-8B
  # baseline arm
  GPU_UTIL=0.6 python validate_kvcached_mapunmap.py Qwen/Qwen3-8B
"""
import os
import sys
import time
import threading

import torch
from vllm import LLM, SamplingParams


def free_gb() -> float:
    return torch.cuda.mem_get_info()[0] / 1e9


class MemSampler(threading.Thread):
    """Background thread sampling free GPU memory (GB)."""

    def __init__(self, interval: float = 0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[float] = []
        self._run = True

    def run(self):
        while self._run:
            try:
                self.samples.append(free_gb())
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._run = False
        self.join()

    @property
    def min(self):
        return min(self.samples) if self.samples else float("nan")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
    kvc_on = os.getenv("ENABLE_KVCACHED", "").lower() in ("1", "true")
    util = float(os.getenv("GPU_UTIL", os.getenv("KVCACHED_GPU_UTILIZATION", "0.6")))
    page_mb = float(os.getenv("KVCACHED_PAGE_SIZE_MB", "2"))
    N = int(os.getenv("VAL_BATCH", "64"))
    OUT = int(os.getenv("VAL_OUT", "1024"))

    arm = "kvcached" if kvc_on else "BASELINE (no kvcached)"
    print(f"\n########## kvcached MAP/UNMAP validation ##########")
    print(f"model={model} | arm={arm} | gpu_util={util} | batch={N} x out={OUT}")

    total_gb = torch.cuda.mem_get_info()[1] / 1e9
    free_before_load = free_gb()
    print(f"[boot] GPU total={total_gb:.1f} GB | free before load={free_before_load:.2f} GB")

    # prefix caching OFF: otherwise finished requests' KV blocks are retained for
    # reuse and never freed back to kvcached, so pages never unmap.
    llm = LLM(model=model, enforce_eager=True, max_model_len=4096,
              gpu_memory_utilization=util, enable_prefix_caching=False)
    torch.cuda.synchronize()
    time.sleep(1.5)

    # ---- Phase 1: idle ----
    idle = free_gb()
    print(f"[idle ] free after model load        = {idle:6.2f} GB")

    # ---- Phase 2: heavy load (drives on-demand MAP) ----
    prompt = ("Write an extremely long, richly detailed adventure story that "
              "spans many chapters, characters, and distant lands. Keep going.")
    sp = SamplingParams(max_tokens=OUT, temperature=0.0, ignore_eos=True)
    sampler = MemSampler(0.05)
    sampler.start()
    t0 = time.time()
    llm.generate([prompt] * N, sp)
    dur = time.time() - t0
    sampler.stop()
    min_free = sampler.min
    mapped = idle - min_free
    print(f"[load ] min free during generation   = {min_free:6.2f} GB   "
          f"(generated {N}x{OUT} toks in {dur:.1f}s)")
    print(f"        => MAP delta = idle - min_free = {mapped:6.2f} GB "
          f"(~{mapped*1024/page_mb:.0f} x {page_mb:.0f}MB pages)")

    # ---- Phase 3: settle (drives UNMAP) ----
    # unmap can be gradual; poll for up to ~12 s and take the best (max) recovery.
    after = free_gb()
    for _ in range(24):
        time.sleep(0.5)
        after = max(after, free_gb())
    recovered = after - min_free
    print(f"[settle] free after requests freed    = {after:6.2f} GB")
    print(f"        => UNMAP recovery = after - min_free = {recovered:6.2f} GB")

    # ---- Verdict ----
    print("\n=== VERDICT ===")
    if kvc_on:
        map_ok = mapped > 2.0                       # mapped >2GB of pages on demand
        unmap_ok = recovered > 0.5 * mapped and recovered > 1.0  # gave most of it back
        print(f"  on-demand MAP   : {'PASS' if map_ok else 'FAIL'} "
              f"(free dropped {mapped:.2f} GB under load)")
        print(f"  on-demand UNMAP : {'PASS' if unmap_ok else 'FAIL'} "
              f"(free recovered {recovered:.2f} GB after free; "
              f"back to {after:.2f} vs idle {idle:.2f})")
        print(f"  RESULT: {'✅ MAP + UNMAP both confirmed' if (map_ok and unmap_ok) else '❌ see above'}")
    else:
        print(f"  baseline static pool: idle free={idle:.2f} GB, min under load="
              f"{min_free:.2f} GB, MAP delta={mapped:.2f} GB (expected ~0 / flat).")
        print(f"  RESULT: baseline reference captured (no on-demand dynamics expected).")


if __name__ == "__main__":
    main()
