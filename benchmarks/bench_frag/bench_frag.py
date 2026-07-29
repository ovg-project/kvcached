# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Measure physical memory pinned by a scattered cache (issue #359).

An idle instance keeps up to KVCACHED_MAX_CACHED_TOKENS worth of prefix cache,
but memory only comes back once every block on a 2MB page is freed. When the
retained blocks are scattered, the pinned set is bounded by pages touched rather
than blocks kept, and the instance holds far more than the cap implies.

This reproduces that without a model: allocate many blocks, then free all but a
scattered residue and compare what is pinned against what is held.

Run on each branch and compare:
    python bench_frag.py
"""
import time

import torch

from kvcached.integration.vllm.interfaces import alloc_kv_cache, init_kvcached, shutdown_kvcached
from kvcached.kv_cache_manager import KVCacheManager
from kvcached.vmm_ops import kv_tensors_created

TP_RANK, TP_SIZE = 0, 1
NUM_LAYERS = 16
BLOCK_SIZE = 16
NUM_BLOCKS = 65536
CELL_SIZE = 1024
DTYPE = torch.float16
DEVICE = f"cuda:{TP_RANK}"
KV_SHAPE = (2, NUM_BLOCKS, BLOCK_SIZE, 8, 64)

# Blocks to allocate before thinning, and how many to keep. KEEP is the residue
# an idle instance would retain as prefix cache.
ALLOC = 16384
KEEP = 1024


def setup():
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
    return KVCacheManager(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE,
                          cell_size=CELL_SIZE, num_layers=NUM_LAYERS,
                          world_size=TP_SIZE)


def measure(manager, keep_stride):
    """Keep every keep_stride-th block, free the rest, report what stays pinned.

    keep_stride=1 keeps a contiguous prefix (the best case: retained blocks pack
    into as few pages as possible). Larger strides spread the same number of
    retained blocks over more pages, which is what a real prefix cache looks like
    after mixed-length requests come and go.
    """
    blocks = manager.alloc(ALLOC)
    assert blocks is not None and len(blocks) == ALLOC

    kept = blocks[::keep_stride][:KEEP]
    kept_set = set(kept)
    manager.free([b for b in blocks if b not in kept_set])

    pinned_gb = manager.get_mapped_memory_size(unit='gb')
    held_bytes = len(kept) * manager.block_mem_size * NUM_LAYERS * 2
    held_gb = held_bytes / (1024**3)

    manager.free(kept)
    return len(kept), held_gb, pinned_gb


if __name__ == "__main__":
    manager = setup()
    print(f"kept={KEEP} blocks of {ALLOC} allocated, "
          f"page={manager.page_size // (1024 * 1024)}MB, "
          f"block={manager.block_mem_size}B\n")
    print(f"{'stride':>7} {'kept':>6} {'held GB':>9} {'pinned GB':>10} {'waste':>7}")
    for stride in (1, 2, 4, 8, 16):
        kept, held_gb, pinned_gb = measure(manager, stride)
        ratio = pinned_gb / held_gb if held_gb else 0.0
        print(f"{stride:>7} {kept:>6} {held_gb:>9.2f} {pinned_gb:>10.2f} "
              f"{ratio:>6.1f}x")
    shutdown_kvcached()
