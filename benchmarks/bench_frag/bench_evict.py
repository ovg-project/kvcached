# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Measure memory released by prefix-cache eviction (issue #359).

Fills ElasticBlockPool's prefix cache, releases the blocks so they become
evictable, then evicts down to a cap and reports how much physical memory came
back. Pages only unmap once every block on them is free, so an eviction policy
that ignores pages can hit the cap while releasing nothing.

Run on each branch and compare the "freed" column:
    python bench_evict.py
"""
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

# Cache this many blocks, then evict down to KEEP. Mirrors an instance going
# idle with MAX_CACHED_TOKENS worth of prefix cache left over.
CACHED = 4096
KEEP = 512


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
    import time
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
    )


def fill_cache(pool, n, stride):
    """Cache n blocks, releasing every stride-th one last.

    Interleaving which blocks are cached vs freed spreads the survivors across
    pages, which is what a real prefix cache looks like after mixed requests.
    """
    blocks = pool.get_new_blocks(n)
    req = _Request([f"h{b.block_id}" for b in blocks])
    pool.cache_full_blocks(req, blocks, 0, n, BLOCK_SIZE, 0)
    pool.free_blocks(blocks)
    # Re-activate blocks we do not want cached, so they leave the pool.
    drop = [b for i, b in enumerate(blocks) if i % stride != 0]
    if drop:
        pool.evict_blocks({b.block_id for b in drop})
    return blocks


if __name__ == "__main__":
    pool = build_pool()
    mgr = pool.kv_cache_manager
    print(f"cached={CACHED} keep={KEEP} "
          f"page={mgr.page_size // (1024 * 1024)}MB block={mgr.block_mem_size}B\n")
    print(f"{'stride':>7} {'evictable':>10} {'before GB':>10} {'after GB':>9} "
          f"{'freed GB':>9}")
    for stride in (1, 2, 4):
        fill_cache(pool, CACHED, stride)
        evictable = len(pool._evictable_blocks)
        before = mgr.get_mapped_memory_size(unit='gb')
        excess = max(0, evictable - KEEP)
        pool._evict_blocks_from_pool(excess)
        after = mgr.get_mapped_memory_size(unit='gb')
        print(f"{stride:>7} {evictable:>10} {before:>10.2f} {after:>9.2f} "
              f"{before - after:>9.2f}")
        pool.reset_prefix_cache()
    shutdown_kvcached()
