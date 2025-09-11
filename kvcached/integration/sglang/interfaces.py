# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import math
from typing import List, Optional, Tuple

import torch

from kvcached.kv_cache_manager import KVCacheManager
from kvcached.tp_ipc_util import start_worker_listener_thread
from kvcached.utils import CONTIGUOUS_LAYOUT, PAGE_SIZE, get_kvcached_logger
from kvcached.vmm_ops import (
    create_kv_tensors,
    init_kvcached as _init_kvcached_impl,
    shutdown_kvcached as _shutdown_kvcached_impl,
)

logger = get_kvcached_logger()

_kvcached_initialized: bool = False
_kvcached_device = None
_async_sched = False
_contiguous_layout = CONTIGUOUS_LAYOUT


def init_kvcached(
    tp_rank: int = 0,
    tp_size: int = 1,
    device: Optional[str] = None,
    async_sched: bool = False,
) -> None:
    global _kvcached_initialized, _kvcached_device, _async_sched
    if _kvcached_initialized:
        return

    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"

    _init_kvcached_impl(device, PAGE_SIZE, _contiguous_layout)
    _kvcached_initialized = True
    _kvcached_device = device
    _async_sched = async_sched

    if tp_size > 1:
        # start the listener thread for tensor parallel kv cache management
        start_worker_listener_thread(torch.cuda.current_device())


def shutdown_kvcached() -> None:
    global _kvcached_initialized, _kvcached_device, _async_sched
    if not _kvcached_initialized:
        return

    _shutdown_kvcached_impl()
    _kvcached_initialized = False
    _kvcached_device = None
    _async_sched = False


def alloc_kv_cache(
    kvcache_shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    num_layers: int,
    page_size: int = 1,
    attention_type: str = "MHA",  # TODO: support MLA
    kv_layout: str = "NHD",  # NHD: (num_tokens, head_num, head_dim)
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if not _kvcached_initialized:
        raise RuntimeError(
            "kvcached is not initialized. Please call init_kvcached() first.")

    if attention_type != "MHA":
        raise ValueError(f"Attention type {attention_type} is not supported.")

    if kv_layout != "NHD":
        raise ValueError(f"KV layout {kv_layout} is not supported.")

    if len(kvcache_shape) <= 2:
        raise ValueError(f"Unsupported kv cache shape: {kvcache_shape}")

    assert torch.cuda.is_available(), "CUDA is not available."
    if page_size != 1:
        logger.warning("kvcached is only tested with page_size=1 for SGLang.")

    # SGLang named it "page" to be consistent with PagedAttention. But we call
    # it "block" to distinguish a KV cache block and a physical memory page.
    block_size = page_size
    num_tokens = kvcache_shape[0]
    block_mem_size = math.prod(kvcache_shape[1:]) * dtype.itemsize
    blocks_per_page = PAGE_SIZE // block_mem_size

    gpu_mem_size = torch.cuda.get_device_properties(device).total_memory

    # Calculate virtual memory size based on layout
    # For contiguous layout, C++ will handle num_layers multiplication
    # So we still calculate per-layer size and let C++ multiply
    num_pages = gpu_mem_size // num_layers // 2 // PAGE_SIZE
    virtual_mem_size = num_pages * PAGE_SIZE * 2

    raw_kv_tensors = create_kv_tensors(virtual_mem_size, dtype.itemsize,
                                       device, num_layers)

    assert block_size * blocks_per_page * num_pages >= num_tokens, \
        "Not enough memory to allocate KV cache."
    num_tokens = block_size * blocks_per_page * num_pages
    actual_kvcache_shape: List[int] = list(kvcache_shape)
    actual_kvcache_shape[0] = num_tokens

    k_tensors, v_tensors = [], []

    if not _contiguous_layout:
        for t in raw_kv_tensors:
            t = t.view(2, *actual_kvcache_shape).view(dtype=dtype)
            k_tensors.append(t.narrow(0, 0, 1).view(actual_kvcache_shape))
            v_tensors.append(t.narrow(0, 1, 1).view(actual_kvcache_shape))
    else:
        contiguous_tensor = raw_kv_tensors[0].view(
            num_tokens, num_layers, 2,
            *actual_kvcache_shape[1:]).view(dtype=dtype)
        for i in range(num_layers):
            k_tensors.append(contiguous_tensor[:, i, 0, :, :])
            v_tensors.append(contiguous_tensor[:, i, 1, :, :])

    return k_tensors, v_tensors


def get_kv_cache_manager(num_blocks: int,
                         block_size: int,
                         cell_size: int,
                         num_layers: int,
                         reserve_null_block: bool = True) -> KVCacheManager:
    if not _kvcached_initialized:
        raise RuntimeError(
            "kvcached is not initialized. Please call init_kvcached() first.")

    return KVCacheManager(
        num_blocks,
        block_size,
        cell_size,
        num_layers,
        async_sched=_async_sched,
        reserve_null_block=reserve_null_block,
    )
