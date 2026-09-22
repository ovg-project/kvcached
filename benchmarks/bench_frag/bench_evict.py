# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Measure memory released by prefix-cache eviction (issue #359).

Pages unmap only once every block on them is free, so an eviction policy that
picks victims by age alone can hit its block cap while releasing no memory: the
survivors stay scattered and each one pins a whole page.

This models that. Cache a large run of blocks, then touch every stride-th block
so it looks recently used. Evicting down to the cap must then choose between the
cold blocks (scattered around the hot ones) and the pages they sit on. The
"freed" column is how much physical memory the eviction actually returned.

Run on each branch and compare:
    python bench_evict.py --backend vllm
    python bench_evict.py --backend sglang
"""

import argparse
import time
import types
from array import array

import torch

from kvcached.kv_cache_manager import KVCacheManager
from kvcached.vmm_ops import kv_tensors_created

TP_RANK, TP_SIZE = 0, 1
NUM_LAYERS = 8
BLOCK_SIZE = 16
CELL_SIZE = 1024
NUM_BLOCKS = 8192
DTYPE = torch.float16
DEVICE = f"cuda:{TP_RANK}"
VLLM_KV_SHAPE = (2, NUM_BLOCKS, BLOCK_SIZE, 8, 64)
SGLANG_KV_SHAPE = (NUM_BLOCKS * BLOCK_SIZE, 8, 64)

CACHED = 4096  # blocks cached before eviction
KEEP = 512  # cap to evict down to; must be well below CACHED


class _Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_cnt = 0
        self.is_null = False


class _Request:
    def __init__(self, block_hashes):
        self.block_hashes = block_hashes


class _SGLangAllocator:
    def __init__(self, manager):
        self.kvcached_allocator = manager
        self.page_size = BLOCK_SIZE
        self.device = "cpu"

    def free(self, token_indices):
        block_ids = torch.unique(token_indices // self.page_size).tolist()
        self.kvcached_allocator.free(block_ids)


def setup(backend):
    torch.cuda.set_device(TP_RANK)
    if backend == "vllm":
        from kvcached.integration.vllm.interfaces import (
            alloc_kv_cache,
            init_kvcached,
            shutdown_kvcached,
        )

        init_kvcached(
            tp_rank=TP_RANK,
            world_size=TP_SIZE,
            is_worker=True,
            async_sched=False,
        )
        alloc_kv_cache(
            kvcache_shape=VLLM_KV_SHAPE,
            block_size=BLOCK_SIZE,
            dtype=DTYPE,
            device=DEVICE,
            num_layers=NUM_LAYERS,
        )
    else:
        from kvcached.integration.sglang.interfaces import (
            alloc_kv_cache,
            init_kvcached,
            shutdown_kvcached,
        )

        init_kvcached(
            tp_rank=TP_RANK,
            world_size=TP_SIZE,
            device=DEVICE,
            async_sched=False,
        )
        alloc_kv_cache(
            kvcache_shape=SGLANG_KV_SHAPE,
            page_size=BLOCK_SIZE,
            dtype=DTYPE,
            device=DEVICE,
            num_layers=NUM_LAYERS,
        )

    t0 = time.time()
    while not kv_tensors_created():
        if time.time() - t0 > 10.0:
            raise RuntimeError("KV tensors not created within 10s")
        time.sleep(0.05)
    return shutdown_kvcached


def build_vllm_pool():
    from kvcached.integration.vllm.patches import ElasticBlockPoolPatch

    mod = types.ModuleType("bench_block_pool")
    mod.BlockPool = object
    mod.KVCacheBlock = _Block
    ElasticBlockPoolPatch().inject_elastic_block_pool(mod)
    return mod.ElasticBlockPool(
        num_gpu_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        cell_size=CELL_SIZE,
        num_layers=NUM_LAYERS,
        enable_caching=True,
        # Cap eviction explicitly: the default (1000) would silently bound the
        # pool below CACHED and evict before the measurement starts.
        max_cached_blocks=-1,
    )


def fill_vllm_cache(pool, n, stride):
    blocks = pool.get_new_blocks(n)
    req = _Request([f"h{block.block_id}" for block in blocks])
    pool.cache_full_blocks(req, blocks, 0, n, BLOCK_SIZE, 0)
    pool.free_blocks(blocks)

    hot = blocks[::stride]
    for block in hot:
        pool.touch([block])
    pool.free_blocks(hot)


def build_sglang_cache():
    from sglang.srt.mem_cache.radix_cache import RadixCache

    manager = KVCacheManager(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        cell_size=CELL_SIZE,
        num_layers=NUM_LAYERS,
        world_size=TP_SIZE,
    )
    cache = RadixCache.create_simulated(
        mock_allocator=_SGLangAllocator(manager),
        page_size=BLOCK_SIZE,
    )
    return cache, manager


def fill_sglang_cache(cache, manager, n, stride):
    from sglang.srt.mem_cache.base_prefix_cache import (
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey

    block_ids = manager.alloc(n)
    assert block_ids is not None and len(block_ids) == n
    keys = []
    for block_id in block_ids:
        first_token = block_id * BLOCK_SIZE + 1
        key = RadixKey(
            token_ids=array("q", range(first_token, first_token + BLOCK_SIZE)),
            extra_key=None,
        )
        token_indices = torch.arange(
            block_id * BLOCK_SIZE,
            (block_id + 1) * BLOCK_SIZE,
            dtype=torch.int64,
        )
        cache.insert(InsertParams(key=key, value=token_indices))
        keys.append(key)

    for key in keys[::stride]:
        cache.match_prefix(MatchPrefixParams(key=key))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("vllm", "sglang"), default="vllm")
    args = parser.parse_args()

    shutdown_kvcached = setup(args.backend)
    try:
        if args.backend == "vllm":
            pool = build_vllm_pool()
            manager = pool.kv_cache_manager
        else:
            pool, manager = build_sglang_cache()

        print(
            f"backend={args.backend} cached={CACHED} keep={KEEP} "
            f"page={manager.page_size // (1024 * 1024)}MB "
            f"block={manager.block_mem_size}B"
        )
        print(f"blocks per page = {manager.page_size // manager.block_mem_size}\n")
        print(
            f"{'stride':>7} {'evictable':>10} {'evicted':>8} "
            f"{'before GB':>10} {'after GB':>9} {'freed GB':>9}"
        )
        for stride in (1, 2, 4, 8):
            if args.backend == "vllm":
                fill_vllm_cache(pool, CACHED, stride)
                evictable = len(pool._evictable_blocks)
                excess = max(0, evictable - KEEP)
                before = manager.get_mapped_memory_size(unit="gb")
                evicted = pool._evict_blocks_from_pool(excess)
                after = manager.get_mapped_memory_size(unit="gb")
                pool.reset_prefix_cache()
            else:
                from sglang.srt.mem_cache.base_prefix_cache import EvictParams

                from kvcached.integration.sglang.patches import (
                    _evict_radix_cache_page_aware,
                )

                fill_sglang_cache(pool, manager, CACHED, stride)
                evictable = pool.evictable_size() // BLOCK_SIZE
                excess = max(0, evictable - KEEP)
                before = manager.get_mapped_memory_size(unit="gb")
                result = _evict_radix_cache_page_aware(
                    radix_cache=pool,
                    num_tokens=excess * BLOCK_SIZE,
                    evict_params_cls=EvictParams,
                )
                evicted = result.num_tokens_evicted // BLOCK_SIZE
                after = manager.get_mapped_memory_size(unit="gb")
                pool.evict(EvictParams(num_tokens=pool.evictable_size()))
                pool.reset()

            print(
                f"{stride:>7} {evictable:>10} {evicted:>8} "
                f"{before:>10.2f} {after:>9.2f} {before - after:>9.2f}"
            )
    finally:
        shutdown_kvcached()


if __name__ == "__main__":
    main()
