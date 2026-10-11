# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""CPU benchmark for the no-reclaimable-page radix eviction fallback.

Each four-block physical page has one active block outside the radix tree and
three cached one-token leaves. No physical page can be reclaimed, so the
page-aware path must fall back to native eviction order. The benchmark evicts
half the leaves and compares direct native eviction with the page-aware path.

Run from the repository root:
    python benchmarks/bench_frag/bench_no_reclaimable_evict.py
"""

import argparse
import gc
import os
import statistics
import time
from array import array
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

# This benchmark intentionally exercises only CPU-side radix-cache bookkeeping.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

from kvcached.integration.sglang.patches import _evict_radix_cache_page_aware

BLOCKS_PER_PHYSICAL_PAGE = 4
CACHED_OFFSETS = (1, 2, 3)


class _PageAllocator:

    def group_indices_by_page(self, indices, block_mem_size):
        pages: dict[int, list[int]] = {}
        for block_id in indices:
            pages.setdefault(
                int(block_id) // BLOCKS_PER_PHYSICAL_PAGE, []
            ).append(int(block_id))
        return pages


class _Manager:

    def __init__(self, num_pages: int):
        self.block_mem_size = 1
        self.page_size = BLOCKS_PER_PHYSICAL_PAGE
        self.page_allocator = _PageAllocator()
        self.allocated = set(range(num_pages * BLOCKS_PER_PHYSICAL_PAGE))
        self._occupancy = Counter(
            block_id // BLOCKS_PER_PHYSICAL_PAGE
            for block_id in self.allocated
        )

    def get_page_occupancy(self, page_ids):
        return {page_id: self._occupancy.get(page_id, 0) for page_id in page_ids}

    def free(self, block_ids):
        for block_id in {int(block_id) for block_id in block_ids}:
            if block_id not in self.allocated:
                continue
            self.allocated.remove(block_id)
            self._occupancy[block_id // BLOCKS_PER_PHYSICAL_PAGE] -= 1


class _Allocator:

    def __init__(self, manager: _Manager):
        self.kvcached_allocator = manager
        self.page_size = 1
        self.device = "cpu"

    def free(self, token_indices):
        self.kvcached_allocator.free(token_indices.tolist())

    def free_segment(self, token_indices, start_pos):
        self.free(token_indices)


@dataclass(frozen=True)
class _Sample:
    elapsed_ms: float
    native_calls: int


def _make_cache(num_leaves: int) -> tuple[Any, _Manager]:
    num_pages, remainder = divmod(num_leaves, len(CACHED_OFFSETS))
    if remainder:
        raise ValueError("leaf counts must be divisible by 3")

    manager = _Manager(num_pages)
    cache = RadixCache.create_simulated(
        mock_allocator=_Allocator(manager),
        page_size=1,
    )
    leaf_index = 0
    for page_id in range(num_pages):
        for offset in CACHED_OFFSETS:
            block_id = page_id * BLOCKS_PER_PHYSICAL_PAGE + offset
            cache.insert(
                InsertParams(
                    key=RadixKey(token_ids=array("q", [1_000_000 + leaf_index])),
                    value=torch.tensor([block_id], dtype=torch.int64),
                    priority=leaf_index,
                )
            )
            leaf_index += 1
    return cache, manager


def _measure(num_leaves: int, evict: Callable[[Any, int], Any]) -> _Sample:
    cache, manager = _make_cache(num_leaves)
    budget = num_leaves // 2
    original_evict = cache.evict
    native_calls = 0

    def counted_evict(params):
        nonlocal native_calls
        native_calls += 1
        return original_evict(params)

    cache.evict = counted_evict
    gc.disable()
    try:
        start_ns = time.perf_counter_ns()
        result = evict(cache, budget)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
    finally:
        gc.enable()

    expected_allocated = num_leaves // 3 + num_leaves - budget
    if result.num_tokens_evicted != budget:
        raise RuntimeError(
            f"evicted {result.num_tokens_evicted} tokens, expected {budget}"
        )
    if native_calls != 1:
        raise RuntimeError(f"native evict called {native_calls} times, expected 1")
    if len(manager.allocated) != expected_allocated:
        raise RuntimeError(
            f"{len(manager.allocated)} blocks remain, expected {expected_allocated}"
        )
    return _Sample(elapsed_ms=elapsed_ms, native_calls=native_calls)


def _native_evict(cache, budget):
    return cache.evict(EvictParams(num_tokens=budget))


def _page_aware_evict(cache, budget):
    return _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=budget,
        evict_params_cls=EvictParams,
    )


def _summarize(samples: list[_Sample]) -> tuple[float, float, float]:
    elapsed = [sample.elapsed_ms for sample in samples]
    return statistics.median(elapsed), min(elapsed), max(elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leaves",
        type=int,
        nargs="+",
        default=[1200, 2400],
        help="leaf counts to measure; each must be divisible by 6",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()

    if args.repeats <= 0 or args.warmups < 0:
        parser.error("--repeats must be positive and --warmups must be non-negative")
    if any(num_leaves <= 0 or num_leaves % 6 for num_leaves in args.leaves):
        parser.error("every --leaves value must be positive and divisible by 6")

    paths = (("native", _native_evict), ("page-aware", _page_aware_evict))
    for num_leaves in args.leaves:
        for _ in range(args.warmups):
            for _name, evict in paths:
                _measure(num_leaves, evict)

    samples_by_case: dict[tuple[int, str], list[_Sample]] = {
        (num_leaves, name): []
        for num_leaves in args.leaves
        for name, _evict in paths
    }
    for repeat in range(args.repeats):
        ordered_paths = paths if repeat % 2 == 0 else tuple(reversed(paths))
        for num_leaves in args.leaves:
            for name, evict in ordered_paths:
                samples_by_case[(num_leaves, name)].append(
                    _measure(num_leaves, evict)
                )

    medians: dict[tuple[int, str], float] = {}
    print("CPU-only; one pinned block and three cached leaves per physical page")
    print(
        f"{'leaves':>8} {'path':>12} {'calls':>7} "
        f"{'median ms':>11} {'min ms':>10} {'max ms':>10}"
    )
    for num_leaves in args.leaves:
        for name, _evict in paths:
            samples = samples_by_case[(num_leaves, name)]
            median_ms, min_ms, max_ms = _summarize(samples)
            medians[(num_leaves, name)] = median_ms
            print(
                f"{num_leaves:>8} {name:>12} {samples[0].native_calls:>7} "
                f"{median_ms:>11.3f} {min_ms:>10.3f} {max_ms:>10.3f}"
            )

    if len(args.leaves) > 1:
        print("\nMedian scaling between adjacent leaf counts:")
        for smaller, larger in zip(args.leaves, args.leaves[1:]):
            for name, _evict in paths:
                scale = medians[(larger, name)] / medians[(smaller, name)]
                print(f"  {name:>10}: {smaller} -> {larger}: {scale:.2f}x")


if __name__ == "__main__":
    main()
