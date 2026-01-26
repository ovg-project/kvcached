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
    """Initialize kvcached for vLLM integration.

    Args:
        tp_rank: Tensor parallel rank of this process.
        tp_size: Total number of tensor parallel processes.
        is_worker: Whether this process is a worker (not the main engine).
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
        # Use tp_rank to ensure correct socket path matching.
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
    kvcache_shape: Tuple[int, ...],  # (2, num_blocks, block_size, head_num, head_dim)
    block_size: int,
    dtype: torch.dtype,
    device: str,
    num_layers: int,
    attention_type: str = "MHA",  # GQA is also supported. TODO: support MLA
    kv_layout: str = "NHD",  # NHD: (num_tokens, head_num, head_dim)
) -> List[torch.Tensor]:
    """Allocate KV cache tensors with elastic memory management for vLLM.

    This function creates combined K/V tensors for each layer using kvcached's
    virtual memory abstraction. Supports both FlashAttn and FlashInfer layouts.

    Args:
        kvcache_shape: Shape of the KV cache. Supports two layouts:
            - FlashAttn: (2, num_blocks, block_size, head_num, head_dim)
            - FlashInfer: (num_blocks, 2, block_size, head_num, head_dim)
        block_size: Number of tokens per block.
        dtype: Data type of the tensors (e.g., torch.float16, torch.bfloat16).
        device: CUDA device type string (e.g., "cuda").
        num_layers: Number of transformer layers.
        attention_type: Attention mechanism type ("MHA" or "GQA").
        kv_layout: Layout of KV cache tensors (only "NHD" supported).

    Returns:
        List of KV tensors, one per layer.

    Raises:
        RuntimeError: If kvcached is not initialized.
        ValueError: If attention_type, kv_layout, or kvcache_shape is not supported.
    """
    if not _kvcached_initialized:
        raise RuntimeError("kvcached is not initialized. Please call init_kvcached() first.")

    if attention_type not in ["MHA", "GQA"]:
        raise ValueError(f"Attention type {attention_type} is not supported.")
    num_k_or_v = 2

    if kv_layout != "NHD":
        raise ValueError(f"KV layout {kv_layout} is not supported.")

    if (len(kvcache_shape) <= 3 or (kvcache_shape[0] != num_k_or_v and kvcache_shape[1] != num_k_or_v
    ) or kvcache_shape[2] != block_size):
        raise ValueError(f"Unsupported kv cache shape: {kvcache_shape}")

    # FlashAttn (num_k_or_v, num_blocks, block_size, head_num, head_dim)
    if kvcache_shape[0] == num_k_or_v:
        blocks_dim_idx = 1
        permute_order = [1, 0] + list(range(2, len(kvcache_shape)))
    # FlashInfer (num_blocks, num_k_or_v, block_size, head_num, head_dim)
    elif kvcache_shape[1] == num_k_or_v:
        blocks_dim_idx = 0
        permute_order = [0, 1] + list(range(2, len(kvcache_shape)))
    else:
        raise ValueError(f"Unsupported kv cache shape: {kvcache_shape}")

    requested_num_blocks = kvcache_shape[blocks_dim_idx]

    assert torch.cuda.is_available(), "CUDA is not available."

    gpu_mem_bytes = torch.cuda.get_device_properties(device).total_memory
    gpu_mem_bytes_per_layer_k_or_v = gpu_mem_bytes // num_layers // num_k_or_v
    # round down to page size
    gpu_mem_bytes_per_layer_k_or_v = (gpu_mem_bytes_per_layer_k_or_v // PAGE_SIZE) * PAGE_SIZE

    block_mem_bytes = math.prod(kvcache_shape[2:]) * dtype.itemsize
    num_blocks_per_layer = gpu_mem_bytes_per_layer_k_or_v // block_mem_bytes
    if requested_num_blocks > num_blocks_per_layer:
        logger.warning(
            f"Requested {requested_num_blocks} blocks, but only {num_blocks_per_layer} blocks are available."
        )

    raw_kv_tensors = create_kv_tensors(
        gpu_mem_bytes_per_layer_k_or_v * num_k_or_v, dtype.itemsize, device, num_layers
    )

    actual_kvcache_shape: List[int] = list(kvcache_shape)
    actual_kvcache_shape[blocks_dim_idx] = num_blocks_per_layer
    if not _contiguous_layout:
        num_eles = math.prod(actual_kvcache_shape)
        kv_tensors = [
            t.view(dtype=dtype)[:num_eles].view(tuple(actual_kvcache_shape)) for t in raw_kv_tensors
        ]
    else:
        contiguous_shape = (num_blocks_per_layer, num_layers, num_k_or_v, *actual_kvcache_shape[2:])
        num_eles = math.prod(contiguous_shape)
        contiguous_tensor = raw_kv_tensors[0].view(dtype=dtype)[:num_eles].view(contiguous_shape)
        kv_tensors = [
            contiguous_tensor[:, i, :, :, :].permute(*permute_order)
            for i in range(num_layers)
        ]
    return kv_tensors


def get_kv_cache_manager(
    num_blocks: int, block_size: int, cell_size: int, num_layers: int
) -> KVCacheManager:
    """Create a KVCacheManager for vLLM.

    Args:
        num_blocks: Number of KV cache blocks to manage.
        block_size: Size of each block in tokens.
        cell_size: Size of each cell in bytes (page_size_bytes / block_size / 2).
        num_layers: Number of transformer layers.

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
    )
