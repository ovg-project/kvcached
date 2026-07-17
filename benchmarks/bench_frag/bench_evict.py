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
    python bench_evict.py
"""
import time
import types

import torch

from kvcached.integration.vllm.interfaces import alloc_kv_cache, init_kvcached, shutdown_kvcached
from kvcached.integration.vllm.patches import ElasticBlockPoolPatch
from kvcached.vmm_ops import kv_tensors_created

TP_RANK, TP_SIZE = 0, 1
NUM_LAYERS = 8
BLOCK_SIZE = 16
CELL_SIZE = 1024
NUM_BLOCKS = 8192
DTYPE = torch.float16
DEVICE = f"cuda:{TP_RANK}"
KV_SHAPE = (2, NUM_BLOCKS, BLOCK_SIZE, 8, 64)

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


def build_pool():
    torch.cuda.set_device(TP_RANK)
    init_kvcached(tp_rank=TP_RANK, world_size=TP_SIZE, is_worker=True,
                  async_sched=False)
    alloc_kv_cache(kvcache_shape=KV_SHAPE, block_size=BLOCK_SIZE, dtype=DTYPE,
                   device=DEVICE, num_layers=NUM_LAYERS)
    t0 = time.time()
    while not kv_tensors_created():
        if time.time() - t0 > 10.0:
            raise RuntimeError("KV tensors not created within 10s")
        time.sleep(0.05)

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


def fill_cache(pool, n, stride):
    """Cache n blocks, then re-cache every stride-th one so it is newest.

    Re-caching under a fresh hash moves those blocks to the end of the LRU
    order, so an age-only policy spares them. They are spread every stride
    blocks, so sparing them pins one page per survivor.
    """
    blocks = pool.get_new_blocks(n)
    req = _Request([f"h{b.block_id}" for b in blocks])
    pool.cache_full_blocks(req, blocks, 0, n, BLOCK_SIZE, 0)
    pool.free_blocks(blocks)

    hot = blocks[::stride]
    for block in hot:
        pool.touch([block])
    pool.free_blocks(hot)
    return blocks


if __name__ == "__main__":
    pool = build_pool()
    mgr = pool.kv_cache_manager
    print(f"cached={CACHED} keep={KEEP} "
          f"page={mgr.page_size // (1024 * 1024)}MB block={mgr.block_mem_size}B")
    print(f"blocks per page = {mgr.page_size // mgr.block_mem_size}\n")
    print(f"{'stride':>7} {'evictable':>10} {'evicted':>8} {'before GB':>10} "
          f"{'after GB':>9} {'freed GB':>9}")
    for stride in (1, 2, 4, 8):
        fill_cache(pool, CACHED, stride)
        evictable = len(pool._evictable_blocks)
        excess = max(0, evictable - KEEP)
        before = mgr.get_mapped_memory_size(unit='gb')
        evicted = pool._evict_blocks_from_pool(excess)
        after = mgr.get_mapped_memory_size(unit='gb')
        print(f"{stride:>7} {evictable:>10} {evicted:>8} {before:>10.2f} "
              f"{after:>9.2f} {before - after:>9.2f}")
        pool.reset_prefix_cache()
    shutdown_kvcached()
