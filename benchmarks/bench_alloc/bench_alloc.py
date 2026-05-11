# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Microbench for KVCacheManager.alloc() hot path.

Run on each branch and compare:
    python bench_alloc.py
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
DTYPE = torch.float16
DEVICE = f"cuda:{TP_RANK}"
KV_SHAPE = (2, NUM_BLOCKS, BLOCK_SIZE, 8, 64)


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
                          cell_size=1024, num_layers=NUM_LAYERS,
                          world_size=TP_SIZE)


def bench_alloc_free(manager, k, iters):
    # warm up
    for _ in range(100):
        h = manager.alloc(k)
        manager.free(h)

    t0 = time.perf_counter()
    for _ in range(iters):
        h = manager.alloc(k)
        manager.free(h)
    elapsed = time.perf_counter() - t0
    per_op_us = elapsed / iters * 1e6
    return per_op_us


if __name__ == "__main__":
    manager = setup()
    print(f"{'k':>6} {'iters':>8} {'us/alloc+free':>16}")
    for k, iters in [(1, 50000), (4, 50000), (16, 50000), (64, 20000),
                     (256, 10000)]:
        per_op = bench_alloc_free(manager, k, iters)
        print(f"{k:>6} {iters:>8} {per_op:>16.2f}")
    shutdown_kvcached()
