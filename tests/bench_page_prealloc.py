#!/usr/bin/env python3
"""Benchmark the effectiveness of PageAllocator's page preallocation across different batch sizes.

This script measures allocation and free operation performance with preallocation disabled and enabled
for various batch sizes. It simulates LLM token generation and deletion patterns.

Examples:
    # Test batch sizes from 8 to 128 with step 8
    python test/bench_page_prealloc.py --batch-start 8 --batch-end 128 --batch-step 8 --iters 1000

    # Test specific batch sizes
    python test/bench_page_prealloc.py --batch-sizes "16,32,64,128" --iters 1000

    # Quick test with smaller range
    python test/bench_page_prealloc.py --batch-start 16 --batch-end 64 --batch-step 16 --iters 1000

The script prints detailed results for each batch size and provides a summary table showing:
- Allocation time with preallocation OFF/ON
- Free time with preallocation OFF/ON
- Speedup percentages for allocation and free operations
"""

import argparse
import importlib
import time
from typing import Tuple

import torch

from kvcached.integration.sglang.interfaces import (alloc_kv_cache,
                                                    init_kvcached,
                                                    shutdown_kvcached)
from kvcached.kv_cache_manager import KVCacheManager

# Relative import of the package works when executed from repo root or installed.
MODULE_PATH = "kvcached.kv_cache_manager"


def _llm_sim_kvcache_benchmark(kv_cache_manager, iterations: int,
                               batch_size: int,
                               delete_after: int) -> tuple[float, float]:
    """Simulate LLM token generation and deletion using KVCacheManager.

    Returns:
        tuple: (total_alloc_time, total_free_time) in seconds
    """
    live_batches = [
    ]  # Each entry is a list of indices allocated in one iteration
    total_alloc_time = 0.0
    total_free_time = 0.0

    for i in range(iterations):
        # Measure allocation time
        alloc_start = time.perf_counter()
        indices = kv_cache_manager.alloc(batch_size)
        alloc_end = time.perf_counter()
        total_alloc_time += alloc_end - alloc_start

        if indices is not None:
            live_batches.append(indices)
        else:
            print(f"Warning: Allocation failed at iteration {i}.")

        if i + 1 >= delete_after and live_batches:
            # Measure free time
            free_start = time.perf_counter()
            to_free = live_batches.pop(0)
            kv_cache_manager.free(to_free)
            free_end = time.perf_counter()
            total_free_time += free_end - free_start

        time.sleep(0.005)  # sleep to simulate LLM token generation

    return total_alloc_time, total_free_time


def _run_single_benchmark_kvcache(prealloc_enabled: bool, total_tokens: int,
                                  iterations: int, batch_size: int,
                                  delete_after: int, head_num: int,
                                  head_dim: int, dtype: str, device: str,
                                  num_layers: int) -> tuple[float, float]:
    """Run one benchmark variant using KVCacheManager and return the alloc and free times (seconds)."""

    # Reload the module to reset global state between variants.
    kv_cache_manager = importlib.import_module(MODULE_PATH)
    importlib.reload(kv_cache_manager)

    # Set preallocation flag before any PageAllocator/KVCacheManager is created
    setattr(kv_cache_manager, "PAGE_PREALLOC_ENABLED", prealloc_enabled)

    init_kvcached()

    # Allocate dummy KV buffers (simulate real usage)
    dtype_obj = getattr(torch, dtype)
    k_buffer, v_buffer = alloc_kv_cache(total_tokens, head_num, head_dim,
                                        dtype_obj, device, num_layers)

    print(f"cell_size =  {head_num * head_dim * dtype_obj.itemsize} bytes")

    kv_cache_manager = KVCacheManager(
        num_blocks=total_tokens,
        block_size=1,
        cell_size=head_num * head_dim * dtype_obj.itemsize,
        num_layers=num_layers,
    )

    # Warm-up
    _llm_sim_kvcache_benchmark(kv_cache_manager, iterations // 10, batch_size,
                               delete_after)

    # Run the actual benchmark
    total_alloc_time, total_free_time = _llm_sim_kvcache_benchmark(
        kv_cache_manager, iterations, batch_size, delete_after)

    shutdown_kvcached()
    return total_alloc_time, total_free_time


def _benchmark_single_batch_size(
        batch_size: int,
        args) -> Tuple[float, float, float, float, float, float]:
    """Benchmark a single batch size and return performance metrics."""
    print(f"\n{'='*60}")
    print(f"Benchmarking batch_size = {batch_size}")
    print(f"{'='*60}")

    alloc_time_off, free_time_off = _run_single_benchmark_kvcache(
        False, args.tokens, args.iters, batch_size, args.delete_after,
        args.head_num, args.head_dim, args.dtype, args.device, args.num_layers)
    total_time_off = alloc_time_off + free_time_off
    allocs_per_sec_off = args.iters / alloc_time_off if alloc_time_off > 0 else 0
    frees_per_sec_off = (args.iters - args.delete_after
                         ) / free_time_off if free_time_off > 0 else 0

    print(
        f"Preallocation OFF: alloc={alloc_time_off:.4f}s, free={free_time_off:.4f}s, total={total_time_off:.4f}s"
    )
    print(
        f"  Alloc rate: {allocs_per_sec_off:,.0f} ops/s, Free rate: {frees_per_sec_off:,.0f} ops/s"
    )

    alloc_time_on, free_time_on = _run_single_benchmark_kvcache(
        True, args.tokens, args.iters, batch_size, args.delete_after,
        args.head_num, args.head_dim, args.dtype, args.device, args.num_layers)
    total_time_on = alloc_time_on + free_time_on
    allocs_per_sec_on = args.iters / alloc_time_on if alloc_time_on > 0 else 0
    frees_per_sec_on = (args.iters - args.delete_after
                        ) / free_time_on if free_time_on > 0 else 0

    print(
        f"Preallocation ON : alloc={alloc_time_on:.4f}s, free={free_time_on:.4f}s, total={total_time_on:.4f}s"
    )
    print(
        f"  Alloc rate: {allocs_per_sec_on:,.0f} ops/s, Free rate: {frees_per_sec_on:,.0f} ops/s"
    )

    if total_time_off > 0:
        alloc_speedup = (
            alloc_time_off - alloc_time_on
        ) / alloc_time_off * 100.0 if alloc_time_off > 0 else 0.0
        free_speedup = (free_time_off - free_time_on
                        ) / free_time_off * 100.0 if free_time_off > 0 else 0.0
        print("\nSpeed-up with preallocation:")
        print(f"  Alloc: {alloc_speedup:.2f}%")
        print(f"  Free:  {free_speedup:.2f}%")

        return alloc_time_off, alloc_time_on, free_time_off, free_time_on, alloc_speedup, free_speedup

    return alloc_time_off, alloc_time_on, free_time_off, free_time_on, 0.0, 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=
        "Benchmark KVCacheManager with and without preallocation across different batch sizes."
    )
    parser.add_argument("--tokens",
                        type=int,
                        default=1024 * 32 * 10,
                        help="Total number of tokens managed during the test.")
    parser.add_argument("--iters",
                        type=int,
                        default=100_000,
                        help="Number of allocation operations per variant.")
    parser.add_argument("--batch-start",
                        type=int,
                        default=8,
                        help="Starting batch size for the range.")
    parser.add_argument("--batch-end",
                        type=int,
                        default=128,
                        help="Ending batch size for the range (inclusive).")
    parser.add_argument("--batch-step",
                        type=int,
                        default=8,
                        help="Step size between batch sizes.")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help=
        "Comma-separated list of specific batch sizes to test (overrides range)."
    )
    parser.add_argument(
        "--delete-after",
        type=int,
        default=50,
        help="Start deleting tokens after this many iterations.")
    parser.add_argument("--head-num",
                        type=int,
                        default=32,
                        help="Number of attention heads.")
    parser.add_argument("--head-dim",
                        type=int,
                        default=128,
                        help="Dimension per head.")
    parser.add_argument("--dtype",
                        type=str,
                        default="float16",
                        help="Data type (e.g., float16, float32).")
    parser.add_argument("--device",
                        type=str,
                        default="cuda:0",
                        help="CUDA device string.")
    parser.add_argument("--num-layers",
                        type=int,
                        default=32,
                        help="Number of transformer layers.")
    args = parser.parse_args()

    # Determine batch sizes to test
    if args.batch_sizes:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]
    else:
        batch_sizes = list(
            range(args.batch_start, args.batch_end + 1, args.batch_step))

    print("Benchmark configuration:")
    print(f"  total_tokens={args.tokens}, iterations={args.iters}")
    print(f"  batch_sizes={batch_sizes}")
    print(
        f"  delete_after={args.delete_after}, head_num={args.head_num}, head_dim={args.head_dim}"
    )
    print(
        f"  dtype={args.dtype}, device={args.device}, num_layers={args.num_layers}\n"
    )

    # Store results for summary
    results = []

    # Run benchmarks for each batch size
    for batch_size in batch_sizes:
        alloc_time_off, alloc_time_on, free_time_off, free_time_on, alloc_speedup, free_speedup = _benchmark_single_batch_size(
            batch_size, args)
        results.append({
            'batch_size': batch_size,
            'alloc_time_off': alloc_time_off,
            'alloc_time_on': alloc_time_on,
            'free_time_off': free_time_off,
            'free_time_on': free_time_on,
            'alloc_speedup': alloc_speedup,
            'free_speedup': free_speedup
        })

    # Print summary table
    print(f"\n{'='*120}")
    print("SUMMARY TABLE")
    print(f"{'='*120}")
    print(
        f"{'Batch Size':<12} {'Total Alloc OFF (s)':<18} {'Total Alloc ON (s)':<18} {'Total Free OFF (s)':<18} {'Total Free ON (s)':<18} {'Alloc Save (%)':<15} {'Free Save (%)':<15}"
    )
    print(f"{'-'*120}")

    for result in results:
        print(
            f"{result['batch_size']:<12} {result['alloc_time_off']:<18.4f} {result['alloc_time_on']:<18.4f} "
            f"{result['free_time_off']:<18.4f} {result['free_time_on']:<18.4f} {result['alloc_speedup']:<15.2f} {result['free_speedup']:<15.2f}"
        )

    print(f"{'='*120}")


if __name__ == "__main__":
    main()
