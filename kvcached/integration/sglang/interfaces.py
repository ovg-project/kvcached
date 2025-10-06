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
    attention_type: str = "MHA",  # GQA is also supported. TODO: support MLA
    kv_layout: str = "NHD",  # NHD: (num_tokens, head_num, head_dim)
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
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
    if page_size != 1:
        logger.warning("kvcached is only tested with page_size=1 for SGLang.")

    # SGLang named it "page" to be consistent with PagedAttention. But we call
    # it "block" to distinguish a KV cache block and a physical memory page.
    block_size = page_size
    block_mem_size = math.prod(kvcache_shape[1:]) * dtype.itemsize

    gpu_mem_bytes = torch.cuda.get_device_properties(device).total_memory
    gpu_mem_bytes_per_layer_k_or_v = gpu_mem_bytes // num_layers // num_k_or_v
    # round down to page size
    gpu_mem_bytes_per_layer_k_or_v = (gpu_mem_bytes_per_layer_k_or_v // PAGE_SIZE) * PAGE_SIZE

    raw_kv_tensors = create_kv_tensors(
        gpu_mem_bytes_per_layer_k_or_v * num_k_or_v, dtype.itemsize, device, num_layers
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


def get_kv_cache_manager(
    num_blocks: int,
    block_size: int,
    cell_size: int,
    num_layers: int,
    reserve_null_block: bool = True,
) -> KVCacheManager:
    if not _kvcached_initialized:
        raise RuntimeError("kvcached is not initialized. Please call init_kvcached() first.")

    return KVCacheManager(
        num_blocks,
        block_size,
        cell_size,
        num_layers,
        async_sched=_async_sched,
        reserve_null_block=reserve_null_block,
    )
