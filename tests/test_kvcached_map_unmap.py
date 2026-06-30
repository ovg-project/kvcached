# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Validate kvcached on-demand KV-page MAP / UNMAP through the vLLM integration.

kvcached reserves GPU *virtual* memory up front and maps *physical* pages on
demand as the KV cache grows, then unmaps them when the cache is freed.  We
watch physical GPU memory (``torch.cuda.mem_get_info``) with a background
sampler across three phases:

  1. idle   : right after model load -- only kvcached's prealloc pages mapped
  2. load   : a large batch with long outputs -- KV grows -> pages MAP -> free drops
  3. settle : requests finished, blocks freed -> pages UNMAP -> free recovers

The MAP signal is itself proof that kvcached is engaged: vanilla vLLM
pre-allocates the whole KV pool at load, so its free memory is flat (no
on-demand dynamics).  Only kvcached makes free memory dip under load and
recover afterwards.

Run as a test (small default model, ~minutes, needs a CUDA GPU)::

    pytest tests/test_kvcached_map_unmap.py -s

Run as a CLI for a specific model / for the baseline contrast::

    python tests/test_kvcached_map_unmap.py --model Qwen/Qwen3-8B --batch 64 --out 1024
    python tests/test_kvcached_map_unmap.py --model Qwen/Qwen3-8B --baseline

Notes
-----
* prefix caching is disabled: otherwise finished requests' KV blocks are
  retained for reuse and never freed back to kvcached, so pages never unmap.
* Enabling kvcached requires its env vars to be set *before* ``vllm`` is
  imported (the autopatch hook fires at import time), so this module imports
  vLLM lazily inside :func:`measure_map_unmap`.
"""
import argparse
import os
import sys
import threading
import time

import torch

try:
    import pytest
    _HAS_PYTEST = True
except ImportError:  # CLI usage does not require pytest
    _HAS_PYTEST = False

DEFAULT_MODEL = os.getenv("KVCACHED_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
_GB = 1024 ** 3


def _free_bytes() -> int:
    return torch.cuda.mem_get_info()[0]


class _MemSampler(threading.Thread):
    """Background thread sampling free GPU memory (bytes)."""

    def __init__(self, interval: float = 0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[int] = []
        self._run = True

    def run(self):
        while self._run:
            try:
                self.samples.append(_free_bytes())
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._run = False
        self.join()

    @property
    def min(self) -> int:
        return min(self.samples) if self.samples else _free_bytes()


def measure_map_unmap(
    model: str = DEFAULT_MODEL,
    *,
    kvcached: bool = True,
    batch: int = 128,
    out_len: int = 1024,
    max_model_len: int = 4096,
    gpu_util: float = 0.5,
    settle_s: float = 12.0,
    verbose: bool = True,
) -> dict:
    """Load ``model`` under vLLM and measure idle/load/settle physical memory.

    Returns a dict with byte/GB metrics: ``idle``, ``min_free`` (peak mapped),
    ``settled``, ``mapped_gb`` (idle - min_free) and ``recovered_gb``
    (settled - min_free).
    """
    # Must be set before vLLM is imported (autopatch fires at import time).
    if kvcached:
        os.environ["ENABLE_KVCACHED"] = "true"
        os.environ["KVCACHED_AUTOPATCH"] = "1"
    else:
        os.environ["ENABLE_KVCACHED"] = "false"

    # Use spawn for the engine-core worker: we touch the CUDA driver in this
    # parent process (mem_get_info sampling), and vLLM's default 'fork' would
    # then fail with "Cannot re-initialize CUDA in forked subprocess".
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams  # lazy import (see module docstring)

    def log(msg):
        if verbose:
            print(msg, flush=True)

    arm = "kvcached" if kvcached else "BASELINE (no kvcached)"
    log(f"\n### MAP/UNMAP: model={model} arm={arm} batch={batch} out={out_len} util={gpu_util}")

    llm = LLM(
        model=model,
        enforce_eager=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        enable_prefix_caching=False,  # required so freed blocks return to kvcached
    )
    torch.cuda.synchronize()
    time.sleep(1.5)

    # Phase 1: idle
    idle = _free_bytes()
    log(f"[idle ] free after load        = {idle / _GB:6.2f} GB")

    # Phase 2: heavy load -> on-demand MAP
    prompt = ("Write an extremely long, richly detailed adventure story that "
              "spans many chapters, characters, and distant lands. Keep going.")
    sp = SamplingParams(max_tokens=out_len, temperature=0.0, ignore_eos=True)
    sampler = _MemSampler(0.05)
    sampler.start()
    t0 = time.time()
    llm.generate([prompt] * batch, sp)
    dur = time.time() - t0
    sampler.stop()
    min_free = sampler.min
    mapped = idle - min_free
    log(f"[load ] min free during gen    = {min_free / _GB:6.2f} GB "
        f"({batch}x{out_len} toks in {dur:.1f}s)")
    log(f"        MAP delta              = {mapped / _GB:6.2f} GB")

    # Phase 3: settle -> UNMAP (poll, take best recovery; unmap can be gradual)
    settled = _free_bytes()
    deadline = time.time() + settle_s
    while time.time() < deadline:
        time.sleep(0.5)
        settled = max(settled, _free_bytes())
    recovered = settled - min_free
    log(f"[settle] free after free       = {settled / _GB:6.2f} GB")
    log(f"        UNMAP recovery         = {recovered / _GB:6.2f} GB")

    return {
        "model": model,
        "kvcached": kvcached,
        "idle": idle,
        "min_free": min_free,
        "settled": settled,
        "mapped_gb": mapped / _GB,
        "recovered_gb": recovered / _GB,
    }


if _HAS_PYTEST:

  @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
  def test_on_demand_map_unmap():
    """kvcached maps KV pages on demand under load and unmaps them after free."""
    pytest.importorskip("kvcached")

    m = measure_map_unmap(DEFAULT_MODEL, kvcached=True)

    # MAP: free must drop meaningfully under load (vanilla vLLM would be flat).
    # This doubles as proof kvcached actually engaged.
    assert m["mapped_gb"] > 0.5, (
        f"expected on-demand MAP to drop >0.5 GB, got {m['mapped_gb']:.2f} GB "
        f"(kvcached may not be engaged)")
    # UNMAP: most of what was mapped must be given back after requests free.
    assert m["recovered_gb"] > 0.5 * m["mapped_gb"], (
        f"expected UNMAP to recover >50% of {m['mapped_gb']:.2f} GB, "
        f"got {m['recovered_gb']:.2f} GB")


def _cli(argv=None) -> int:
    p = argparse.ArgumentParser(description="kvcached on-demand MAP/UNMAP validation")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--out", type=int, default=1024, dest="out_len")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-util", type=float, default=0.5)
    p.add_argument("--baseline", action="store_true",
                   help="run without kvcached (static pool; for contrast)")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("no CUDA GPU available", file=sys.stderr)
        return 2

    m = measure_map_unmap(
        args.model, kvcached=not args.baseline, batch=args.batch,
        out_len=args.out_len, max_model_len=args.max_model_len, gpu_util=args.gpu_util,
    )

    print("\n=== VERDICT ===")
    if args.baseline:
        print(f"  baseline static pool: MAP delta={m['mapped_gb']:.2f} GB (expected ~0 / flat)")
        return 0
    map_ok = m["mapped_gb"] > 0.5
    unmap_ok = m["recovered_gb"] > 0.5 * m["mapped_gb"]
    print(f"  on-demand MAP   : {'PASS' if map_ok else 'FAIL'} "
          f"(free dropped {m['mapped_gb']:.2f} GB under load)")
    print(f"  on-demand UNMAP : {'PASS' if unmap_ok else 'FAIL'} "
          f"(free recovered {m['recovered_gb']:.2f} GB after free)")
    ok = map_ok and unmap_ok
    print(f"  RESULT: {'PASS - MAP + UNMAP confirmed' if ok else 'FAIL - see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
