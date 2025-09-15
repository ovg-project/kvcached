# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Pytest script for KVCacheManager APIs:
- alloc
- free
- resize
- trim
- reserve and free reserved blocks
Please run with pytest in the vLLM venv:
pytest test_kvcache_manager.py
"""

import os
import time

import pytest
import torch

from kvcached.cli.utils import get_kv_cache_limit, update_kv_cache_limit
from kvcached.integration.vllm.interfaces import (
    alloc_kv_cache,
    init_kvcached,
    shutdown_kvcached,
)
from kvcached.kv_cache_manager import KVCacheManager
from kvcached.utils import DEFAULT_IPC_NAME
from kvcached.vmm_ops import kv_tensors_created

IPC_NAME = DEFAULT_IPC_NAME

TP_RANK = 0
TP_SIZE = 1
NUM_LAYERS = 16
BLOCK_SIZE = 16
NUM_BLOCKS = 65536
DTYPE = torch.float16
DEVICE = f"cuda:{TP_RANK}"
KV_SHAPE = (2, NUM_BLOCKS, BLOCK_SIZE, 8, 64)


@pytest.fixture(scope="function")
def setup_kvcache():
    # initialize kvcached
    os.environ["KVCACHED_CONTIGUOUS_LAYOUT"] = "true"
    torch.cuda.set_device(TP_RANK)
    init_kvcached(tp_rank=TP_RANK, tp_size=TP_SIZE, is_worker=True, async_sched=False)

    # allocate kv cache tensors in virtual memory
    alloc_kv_cache(
        kvcache_shape=KV_SHAPE,
        block_size=BLOCK_SIZE,
        dtype=DTYPE,
        device=DEVICE,
        num_layers=NUM_LAYERS,
    )

    while True:
        created = kv_tensors_created()
        if created:
            break
        time.sleep(0.1)

    # instantiate a kv cache manager
    manager = KVCacheManager(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        cell_size=1024,
        num_layers=NUM_LAYERS,
        tp_size=TP_SIZE,
    )

    # wait a bit for pre-allocation to finish
    time.sleep(1)

    yield manager

    shutdown_kvcached()


def test_basic_alloc_free(setup_kvcache):
    # instantiate a kv cache manager with known size
    manager = setup_kvcache

    # initial available blocks
    initial_available = manager.available_size()
    print(f"Initial available size: {initial_available}")

    # allocate some blocks
    n_blocks = 256
    handle = manager.alloc(n_blocks)
    after_alloc = manager.available_size()
    print(f"Available size after allocating {n_blocks} tokens: {after_alloc}")
    assert after_alloc + n_blocks == initial_available

    # free the allocated blocks
    manager.free(handle)
    after_free = manager.available_size()
    print(f"Available size after freeing: {after_free}")
    assert after_free == initial_available


def test_over_allocation_fails(setup_kvcache):
    # instantiate a kv cache manager with known size
    manager = setup_kvcache

    # try to allocate 1 block more than the available size
    too_many_tokens = manager.available_size() + 1
    handle = manager.alloc(too_many_tokens)
    assert handle is None


def test_resize_smaller_and_larger(setup_kvcache):
    # instantiate a kv cache manager with known size
    # Terminology:
    # - kv_cache_limit:
    # stored in shm total_size field, corresponds to GPU memory, typically in tens of GBs
    # - mem_size:
    # used by resize method, corresponds K (or V) tensor size in 1 layer, typically in few GBs
    manager = setup_kvcache
    initial_total_pages = manager.page_allocator.num_total_pages
    print(f"Initial page_allocator.num_total_pages: {initial_total_pages}")
    initial_attribute_mem_size = manager.mem_size
    print(f"Initial self.mem_size: {initial_attribute_mem_size}")
    initial_shm_mem_size = get_kv_cache_limit(IPC_NAME).total_size // NUM_LAYERS // 2
    print(f"Initial shm mem_size: {initial_shm_mem_size}")
    print(f"Initial shm total_size: {get_kv_cache_limit(IPC_NAME).total_size}")
    print(f"Initial shm used_size: {get_kv_cache_limit(IPC_NAME).used_size}")
    assert initial_attribute_mem_size == initial_shm_mem_size

    # RESIZE SMALLER: deduct half of initial total pages
    shrink_kv_cache_limit = get_kv_cache_limit(IPC_NAME).total_size - (initial_total_pages // 2) * manager.page_size * NUM_LAYERS * 2
    print(f"Shrinking to kv cache limit: {shrink_kv_cache_limit}")
    # update the shm total_size field
    update_kv_cache_limit(IPC_NAME, shrink_kv_cache_limit)
    print(f"After shrinking, shm total_size: {get_kv_cache_limit(IPC_NAME).total_size}")
    # infer the new mem_size based on shm total_size --- workflow in kvcached
    shrink_shm_mem_size = manager.page_allocator.mem_info_tracker.check_and_get_resize_target(
            manager.mem_size, manager.num_layers)
    print(f"Shrinking to mem_size from check_and_get_resize_target: {shrink_shm_mem_size}")
    # actual resize method
    manager.resize(shrink_shm_mem_size)
    shrink_total_pages = manager.page_allocator.num_total_pages
    print(f"After shrinking, : page_allocator.num_total_pages: {shrink_total_pages}")
    assert initial_total_pages == shrink_total_pages + initial_total_pages // 2

    # RESIZE LARGER: add back the deducted half of initial total pages
    expand_kv_cache_limit =  shrink_kv_cache_limit + (initial_total_pages // 2) * manager.page_size * NUM_LAYERS * 2
    # update the shm total_size field
    update_kv_cache_limit(IPC_NAME, expand_kv_cache_limit)
    print(f"After expanding, shm total_size: {get_kv_cache_limit(IPC_NAME).total_size}")
    # infer the new mem_size based on shm total_size --- workflow in kvcached
    expand_shm_mem_size = manager.page_allocator.mem_info_tracker.check_and_get_resize_target(
            shrink_shm_mem_size, manager.num_layers)
    print(f"Expanding to mem_size from check_and_get_resize_target: {expand_shm_mem_size}")
    # actual resize method
    manager.resize(expand_shm_mem_size)
    expand_total_pages = manager.page_allocator.num_total_pages
    print(f"After expanding, : page_allocator.num_total_pages: {expand_total_pages}")
    assert expand_total_pages == initial_total_pages


def test_trim(setup_kvcache):
    # instantiate a kv cache manager with known size
    manager = setup_kvcache

    # initial reserved pages
    initial_reserved = len(manager.page_allocator.reserved_page_list)
    print(f"Initial reserved pages: {initial_reserved}")
    if manager.page_allocator.enable_page_prealloc:
        assert initial_reserved == manager.page_allocator.min_reserved_pages

    # trim reserved pages
    manager.trim()
    time.sleep(1)
    after_trim_reserved = len(manager.page_allocator.reserved_page_list)
    print(f"Reserved pages after trim: {after_trim_reserved}")
    assert after_trim_reserved == 0


def test_reserve_and_free_blocks(setup_kvcache):
    # instantiate a kv cache manager with known size
    manager = setup_kvcache

    # initial reserved blocks
    initial_reserved_blocks = len(manager.reserved_blocks)
    print(f"Initial reserved blocks: {initial_reserved_blocks}")

    # reserve some blocks
    n_blocks = 512
    manager.try_to_reserve(n_blocks)
    time.sleep(1)
    after_reserve_blocks = len(manager.reserved_blocks)
    print(f"Reserved blocks after reserving {n_blocks} tokens: {after_reserve_blocks}")
    assert after_reserve_blocks == initial_reserved_blocks + n_blocks

    # free the reserved blocks
    manager.free_reserved()
    after_free_blocks = len(manager.reserved_blocks)
    print(f"Reserved blocks after freeing: {after_free_blocks}")
    assert after_free_blocks == 0