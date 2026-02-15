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
_kvcached_device: Optional[str] = None
_async_sched: bool = False
_tp_size: int = 1
_contiguous_layout: bool = CONTIGUOUS_LAYOUT


def init_kvcached(
    tp_rank: int = 0,
    tp_size: int = 1,
    is_worker: bool = False,
    device: Optional[str] = None,
    async_sched: bool = False,
) -> None:
    """Initialize kvcached for SGLang integration.

    Args:
        tp_rank: Tensor parallel rank of this process.
        tp_size: Total number of tensor parallel processes.
        is_worker: Whether this process is a worker (not the main scheduler).
            Only workers should start the IPC listener thread.
        device: CUDA device string (e.g., "cuda:0"). If None, uses current device.
        async_sched: Whether to enable asynchronous scheduling.
    """
    global _kvcached_initialized, _kvcached_device, _tp_size, _async_sched
    if _kvcached_initialized:
        return

    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"

    _init_kvcached_impl(device, PAGE_SIZE, _contiguous_layout)
    _kvcached_initialized = True
    _kvcached_device = device
    _tp_size = tp_size
    _async_sched = async_sched

    if tp_size > 1 and is_worker:
        # Start the listener thread for tensor parallel KV cache management.
        # Use tp_rank (not device ID) to ensure correct socket path matching.
        start_worker_listener_thread(tp_rank)


def shutdown_kvcached() -> None:
    """Shutdown kvcached and release resources."""
    global _kvcached_initialized, _kvcached_device, _tp_size, _async_sched
    if not _kvcached_initialized:
        return

    _shutdown_kvcached_impl()
    _kvcached_initialized = False
    _kvcached_device = None
    _tp_size = 1
    _async_sched = False


def alloc_kv_cache(
    kvcache_shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    num_layers: int,
    page_size: int = 1,
    attention_type: str = "MHA",  # GQA is also supported. TODO: support MLA
    kv_layout: str = "NHD",  # NHD: (num_tokens, head_num, head_dim)
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Allocate KV cache tensors with elastic memory management.

    This function creates K and V tensors for each layer using kvcached's
    virtual memory abstraction. Memory is allocated on-demand rather than
    pre-allocated.

    Args:
        kvcache_shape: Shape of the KV cache as (num_tokens, head_num, head_dim).
        dtype: Data type of the tensors (e.g., torch.float16, torch.bfloat16).
        device: CUDA device string (e.g., "cuda", "cuda:0").
        num_layers: Number of transformer layers.
        page_size: Number of tokens per page (default 1 for SGLang).
        attention_type: Attention mechanism type ("MHA" or "GQA").
        kv_layout: Layout of KV cache tensors (only "NHD" supported).

    Returns:
        Tuple of (k_tensors, v_tensors) where each is a list of tensors,
        one per layer.

    Raises:
        RuntimeError: If kvcached is not initialized.
        ValueError: If attention_type or kv_layout is not supported.
    """
    if not _kvcached_initialized:
        raise RuntimeError("kvcached is not initialized. Please call init_kvcached() first.")

    if attention_type not in ["MHA", "GQA"]:
        raise ValueError(f"Attention type {attention_type} is not supported.")
    num_k_or_v = 2
    requested_num_tokens = kvcache_shape[0]

    if kv_layout != "NHD":
        raise ValueError(f"KV layout {kv_layout} is not supported.")

    if len(kvcache_shape) <= 2:
        raise ValueError(f"Unsupported kv cache shape: {kvcache_shape}")

    assert torch.cuda.is_available(), "CUDA is not available."
    # if page_size != 1:
    #     logger.warning("kvcached is only tested with page_size=1 for SGLang.")

    # SGLang named it "page" to be consistent with PagedAttention. But we call
    # it "block" to distinguish a KV cache block and a physical memory page.
    block_size = page_size
    block_mem_size = block_size * math.prod(kvcache_shape[1:]) * dtype.itemsize

    gpu_mem_bytes = torch.cuda.get_device_properties(device).total_memory
    gpu_mem_bytes_per_layer_k_or_v = gpu_mem_bytes // num_layers // num_k_or_v
    # round down to page size
    gpu_mem_bytes_per_layer_k_or_v = (gpu_mem_bytes_per_layer_k_or_v // PAGE_SIZE) * PAGE_SIZE

    raw_kv_tensors = create_kv_tensors(
        gpu_mem_bytes_per_layer_k_or_v * num_k_or_v, dtype.itemsize, device, num_layers,
        num_kv_buffers=num_k_or_v,
    )

    num_blocks_per_layer = gpu_mem_bytes_per_layer_k_or_v // block_mem_size
    num_tokens = num_blocks_per_layer * block_size
    if requested_num_tokens > num_tokens:
        logger.warning(
            f"Requested {requested_num_tokens} tokens, but only {num_tokens} tokens are available."
        )

    actual_kvcache_shape: List[int] = list(kvcache_shape)
    actual_kvcache_shape[0] = block_size * num_blocks_per_layer

    k_tensors, v_tensors = [], []

    if not _contiguous_layout:
        num_eles = num_k_or_v * math.prod(actual_kvcache_shape)
        for t in raw_kv_tensors:
            t = t.view(dtype=dtype)[:num_eles].view(num_k_or_v, *actual_kvcache_shape)
            k_tensors.append(t.narrow(0, 0, 1).view(actual_kvcache_shape))
            v_tensors.append(t.narrow(0, 1, 1).view(actual_kvcache_shape))
    else:
        contiguous_shape = (num_tokens, num_layers, num_k_or_v, *actual_kvcache_shape[1:])
        num_eles = math.prod(contiguous_shape)
        contiguous_tensor = raw_kv_tensors[0].view(dtype=dtype)[:num_eles].view(contiguous_shape)
        for i in range(num_layers):
            k_tensors.append(contiguous_tensor[:, i, 0, :, :])
            v_tensors.append(contiguous_tensor[:, i, 1, :, :])

    return k_tensors, v_tensors


def alloc_mla_kv_cache(
    kvcache_shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    num_layers: int,
    page_size: int = 1,
) -> List[torch.Tensor]:
    """Allocate MLA-style KV cache with a single combined kv_buffer per layer.

    MLA uses a single buffer of shape (num_tokens, 1, kv_cache_dim) per layer
    instead of separate K and V buffers.
    """
    if not _kvcached_initialized:
        raise RuntimeError("kvcached is not initialized. Please call init_kvcached() first.")

    num_k_or_v = 1
    requested_num_tokens = kvcache_shape[0]

    if len(kvcache_shape) <= 2:
        raise ValueError(f"Unsupported kv cache shape: {kvcache_shape}")

    assert torch.cuda.is_available(), "CUDA is not available."
    if page_size != 1:
        logger.warning("kvcached is only tested with page_size=1 for SGLang.")

    block_size = page_size
    block_mem_size = block_size * math.prod(kvcache_shape[1:]) * dtype.itemsize

    gpu_mem_bytes = torch.cuda.get_device_properties(device).total_memory
    gpu_mem_bytes_per_layer = gpu_mem_bytes // num_layers // num_k_or_v
    # round down to page size
    gpu_mem_bytes_per_layer = (gpu_mem_bytes_per_layer // PAGE_SIZE) * PAGE_SIZE

    raw_kv_tensors = create_kv_tensors(
        gpu_mem_bytes_per_layer * num_k_or_v, dtype.itemsize, device, num_layers,
        num_kv_buffers=num_k_or_v,
    )

    num_blocks_per_layer = gpu_mem_bytes_per_layer // block_mem_size
    num_tokens = num_blocks_per_layer * block_size
    if requested_num_tokens > num_tokens:
        logger.warning(
            f"Requested {requested_num_tokens} tokens, but only {num_tokens} tokens are available."
        )

    actual_kvcache_shape: List[int] = list(kvcache_shape)
    actual_kvcache_shape[0] = block_size * num_blocks_per_layer

    kv_tensors: List[torch.Tensor] = []

    if not _contiguous_layout:
        num_eles = math.prod(actual_kvcache_shape)
        for t in raw_kv_tensors:
            kv_tensors.append(t.view(dtype=dtype)[:num_eles].view(actual_kvcache_shape))
    else:
        contiguous_shape = (num_tokens, num_layers, *actual_kvcache_shape[1:])
        num_eles = math.prod(contiguous_shape)
        contiguous_tensor = raw_kv_tensors[0].view(dtype=dtype)[:num_eles].view(contiguous_shape)
        for i in range(num_layers):
            kv_tensors.append(contiguous_tensor[:, i, :, :])

    return kv_tensors


def get_kv_cache_manager(
    num_blocks: int,
    block_size: int,
    cell_size: int,
    num_layers: int,
    reserve_null_block: bool = True,
    num_kv_buffers: int = 2,
) -> KVCacheManager:
    """Create a KVCacheManager for SGLang.

    Args:
        num_blocks: Number of KV cache blocks to manage.
        block_size: Size of each block in tokens.
        cell_size: Size of each cell in bytes (head_num * head_dim * dtype_size).
        num_layers: Number of transformer layers.
        reserve_null_block: Whether to reserve the first block as null block
            for padding tokens. Required by SGLang.

    Returns:
        KVCacheManager instance configured for the specified parameters.

    Raises:
        RuntimeError: If kvcached is not initialized.
    """
    if not _kvcached_initialized:
        raise RuntimeError("kvcached is not initialized. Please call init_kvcached() first.")

    return KVCacheManager(
        num_blocks,
        block_size,
        cell_size,
        num_layers,
        _tp_size,
        async_sched=_async_sched,
        reserve_null_block=reserve_null_block,
        num_kv_buffers=num_kv_buffers,
    )
