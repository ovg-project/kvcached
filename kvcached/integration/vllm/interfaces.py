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
_world_size: int = 1
_pp_rank: int = 0
_contiguous_layout: bool = CONTIGUOUS_LAYOUT
_is_worker: bool = False
# Physical blocks per layer kvcached allocates for the GPU's KV region.
# Read by the NixlConnector patch in autopatch.py to satisfy the connector's
# shape[blocks_dim] == self.num_blocks assertion when kvcached over-allocates.
_num_blocks_per_layer: int = 0


def should_use_worker_ipc() -> bool:
    return _kvcached_initialized and not _is_worker

def init_kvcached(
    tp_rank: int = 0,
    world_size: int = 1,
    pp_rank: int = 0,
    is_worker: bool = False,
    device: Optional[str] = None,
    async_sched: bool = False,
) -> None:
    global _kvcached_initialized, _kvcached_device, _world_size, _async_sched, _pp_rank, _is_worker
    if _kvcached_initialized:
        # EngineCore call init_kvcached(is_worker=False) first. When TP=1 GPUModelRunner
        # then calls init_kvcached(is_worker=True) in the same process; without this branch
        # the early return would leave _is_worker False, so KVCacheManager would try Unix IPC
        # (broadcast_kv_tensors_created) and fail with ENOENT on the socket path.
        if is_worker and not _is_worker:
            _is_worker = True
            start_worker_listener_thread(tp_rank, pp_rank)
        if async_sched and not _async_sched:
            _async_sched = True
            logger.info("kvcached async scheduler enabled")
        _pp_rank = pp_rank
        _world_size = world_size
        return

    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"

    _init_kvcached_impl(device, PAGE_SIZE, _contiguous_layout)
    _kvcached_initialized = True
    _kvcached_device = device
    _world_size = world_size
    _pp_rank = pp_rank
    _async_sched = async_sched
    _is_worker = is_worker

    if _async_sched:
        logger.info("kvcached async scheduler enabled")

    if is_worker:
        # start the listener thread for kv cache management regardless of TP size
        # because the vLLM EngineCore might need to reach this worker if PP > 1
        start_worker_listener_thread(tp_rank, pp_rank)


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
    block_size: int,
    dtype: torch.dtype,
    device: str,
    num_layers: int,
    attention_type: str = "MHA",  # MHA, GQA, MLA, or HYBRID_LINEAR.
    kv_layout: str = "NHD",  # NHD: (num_tokens, head_num, head_dim)
    group_id: int = 0,
    kernel_block_size: Optional[int] = None,
) -> List[torch.Tensor]:
    """Allocate KV cache tensors for all supported attention types.

    For MHA/GQA, kvcache_shape is expected to be:
      - FlashAttn:  (2, num_blocks, block_size, head_num, head_dim)
      - FlashInfer: (num_blocks, 2, block_size, head_num, head_dim)
    For MLA, kvcache_shape is expected to be:
      - (num_blocks, block_size, head_size)

    ``attention_type="HYBRID_LINEAR"`` selects the layout for hybrid
    models that mix full attention with linear attention (mamba/SSM).
    It collapses K and V into a single FTensor per pool so VM page
    mappings use page_size_bytes granularity (K+V combined), matching
    the ``as_strided_`` access pattern that vLLM's
    ``_update_hybrid_attention_mamba_layout`` applies. Callers should
    set ``num_layers`` to the group_size in this mode.

    Returns:
        List[torch.Tensor] for MHA/GQA/MLA.
        For HYBRID_LINEAR, returns (kv_tensors, raw_info) where
        raw_info is a dict with:
          buffers            - flat int8 tensors, one per pool
          num_blocks         - number of blocks per pool
          page_size_bytes    - uniform page size (bytes) shared by all groups
          block_stride_bytes - byte stride between consecutive blocks of the
                               same pool
          num_pools          - number of pools (== num_layers)
    """
    if not _kvcached_initialized:
        raise RuntimeError("kvcached is not initialized. Please call init_kvcached() first.")

    if attention_type not in ["MHA", "GQA", "MLA", "HYBRID_LINEAR"]:
        raise ValueError(f"Attention type {attention_type} is not supported.")

    if kv_layout != "NHD":
        raise ValueError(f"KV layout {kv_layout} is not supported.")

    is_mla = attention_type == "MLA"
    is_hybrid_linear = attention_type == "HYBRID_LINEAR"
    unified_pool = is_hybrid_linear

    if is_hybrid_linear and _contiguous_layout:
        raise ValueError(
            "kvcached detected a hybrid linear-attention model "
            "(e.g. Jamba/Bamba/NemotronH/Zamba2/Plamo2), which requires the "
            "non-contiguous KV layout. Re-launch with "
            "KVCACHED_CONTIGUOUS_LAYOUT=false. Also do NOT pass "
            "--disable-hybrid-kv-cache-manager to vLLM for these models — "
            "it only applies to attention-only hybrids like sliding-window "
            "models (e.g. GPT-OSS).")

    num_k_or_v = 1 if is_mla else 2

    # Kernel-block granularity. vLLM may split a virtual block (``block_size``
    # tokens) into ``ratio`` kernel-sized blocks. The attention zero kernel
    # assumes the per-layer tensor is strided at kernel-block granularity, so
    # the as_strided view we hand back must match.
    if kernel_block_size is None:
        kernel_block_size = block_size
    if block_size % kernel_block_size != 0:
        raise ValueError(
            f"block_size ({block_size}) must be a multiple of "
            f"kernel_block_size ({kernel_block_size})")
    ratio = block_size // kernel_block_size

    # --- Validate shape and determine layout indices ---
    if is_mla:
        # MLA shape: (num_blocks, block_size, head_size)
        if len(kvcache_shape) <= 2:
            raise ValueError(f"Unsupported MLA kv cache shape: {kvcache_shape}")
        if kvcache_shape[1] != block_size:
            raise ValueError(
                f"block_size mismatch: kvcache_shape[1]={kvcache_shape[1]} != block_size={block_size}"
            )
        blocks_dim_idx = 0
        permute_order = list(range(len(kvcache_shape)))
        block_mem_bytes = math.prod(kvcache_shape[1:]) * dtype.itemsize
    else:
        # MHA/GQA shape with K/V dimension
        if (len(kvcache_shape) <= 3
                or (kvcache_shape[0] != num_k_or_v and kvcache_shape[1] != num_k_or_v)
                or kvcache_shape[2] != block_size):
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

        block_mem_bytes = math.prod(kvcache_shape[2:]) * dtype.itemsize

    requested_num_blocks = kvcache_shape[blocks_dim_idx]

    assert torch.cuda.is_available(), "CUDA is not available."

    # --- Compute per-layer memory budget and number of blocks ---
    gpu_mem_bytes = torch.cuda.get_device_properties(device).total_memory
    gpu_mem_bytes_per_layer_k_or_v = gpu_mem_bytes // num_layers // num_k_or_v
    # round down to page size
    gpu_mem_bytes_per_layer_k_or_v = (gpu_mem_bytes_per_layer_k_or_v // PAGE_SIZE) * PAGE_SIZE

    num_blocks_per_layer = gpu_mem_bytes_per_layer_k_or_v // block_mem_bytes
    global _num_blocks_per_layer
    _num_blocks_per_layer = num_blocks_per_layer
    if requested_num_blocks > num_blocks_per_layer:
        logger.warning(
            f"Requested {requested_num_blocks} blocks, but only {num_blocks_per_layer} blocks are available."
        )

    ftensor_bytes_per_layer = gpu_mem_bytes_per_layer_k_or_v * num_k_or_v

    raw_kv_tensors = create_kv_tensors(
        ftensor_bytes_per_layer, dtype.itemsize, device, num_layers,
        num_kv_buffers=num_k_or_v, group_id=group_id,
        unified_pool=unified_pool,
    )

    actual_kvcache_shape: List[int] = list(kvcache_shape)
    actual_kvcache_shape[blocks_dim_idx] = num_blocks_per_layer

    page_size_bytes = math.prod(
        actual_kvcache_shape[:blocks_dim_idx] + actual_kvcache_shape[blocks_dim_idx + 1:]
    ) * dtype.itemsize

    # Build a second shape expressed at kernel-block granularity. vLLM's zero
    # kernel and attention kernels index the KV tensor using ``kernel_bs``-
    # token blocks; each virtual block is ``ratio`` contiguous kernel blocks.
    # When ratio == 1, kernel_kvcache_shape == actual_kvcache_shape.
    kernel_kvcache_shape: List[int] = list(actual_kvcache_shape)
    if ratio > 1:
        kernel_kvcache_shape[blocks_dim_idx] = num_blocks_per_layer * ratio
        # Token dim index: for MHA it's 2, for MLA it's 1 (right after block dim).
        token_dim_idx = 2 if not is_mla else 1
        kernel_kvcache_shape[token_dim_idx] = kernel_block_size

    # --- Reshape raw tensors into per-layer KV cache views ---
    if not _contiguous_layout:
        kv_tensors: List[torch.Tensor] = []
        if is_mla:
            num_eles = math.prod(kernel_kvcache_shape)
            kv_tensors = [
                t.view(dtype=dtype)[:num_eles].view(kernel_kvcache_shape)
                for t in raw_kv_tensors
            ]
        else:
            # Build attention view with as_strided. Two modes:
            #   split-half (default): K occupies [0, v_offset), V occupies
            #     [v_offset, 2*v_offset). K/V dim stride = v_offset_eles.
            #   unified_pool (HYBRID_LINEAR): K and V interleaved per kernel
            #     block. This mirrors native vLLM's
            #     _update_hybrid_attention_mamba_layout and lets an attached
            #     linear-attention / mamba layer read the same flat buffer
            #     (mamba still indexes by virtual block; each virtual block
            #     spans ``ratio`` kernel blocks).
            shape = list(kernel_kvcache_shape)
            strides = [0] * len(shape)
            strides[-1] = 1
            for i in range(len(shape) - 2, 1, -1):
                strides[i] = strides[i + 1] * shape[i + 1]
            # hidden_size_eles uses kernel_block_size (shape[2]), not block_size.
            hidden_size_eles = strides[2] * shape[2]  # = kernel_bs * h * d
            if unified_pool:
                # Block-interleaved at kernel granularity: inter-(kernel-)block
                # stride = 2*hidden_size; K/V dim stride = hidden_size.
                if blocks_dim_idx == 1:          # FlashAttn (2, N*ratio, ...)
                    strides[1] = 2 * hidden_size_eles
                    strides[0] = hidden_size_eles
                else:                             # FlashInfer (N*ratio, 2, ...)
                    strides[0] = 2 * hidden_size_eles
                    strides[1] = hidden_size_eles
            else:
                v_offset_eles = gpu_mem_bytes_per_layer_k_or_v // dtype.itemsize
                if blocks_dim_idx == 1:          # FlashAttn (2, N*ratio, ...)
                    strides[1] = hidden_size_eles
                    strides[0] = v_offset_eles
                else:                             # FlashInfer (N*ratio, 2, ...)
                    strides[0] = hidden_size_eles
                    strides[1] = v_offset_eles
            for t in raw_kv_tensors:
                kv_tensors.append(
                    torch.as_strided(t.view(dtype=dtype), shape, strides))
    else:
        layer_elem_shape = actual_kvcache_shape[:blocks_dim_idx] + actual_kvcache_shape[blocks_dim_idx + 1:]
        contiguous_shape = [num_blocks_per_layer, num_layers] + layer_elem_shape
        num_eles = math.prod(contiguous_shape)
        contiguous_tensor = raw_kv_tensors[0].view(dtype=dtype)[:num_eles].view(contiguous_shape)
        kv_tensors = [
            contiguous_tensor[:, i].permute(*permute_order) for i in range(num_layers)
        ]

    if not unified_pool:
        return kv_tensors

    # --- Build raw int8 buffers for hybrid model (mamba) support ---
    pool_bytes = num_blocks_per_layer * page_size_bytes
    raw_int8 = [t.view(torch.int8)[:pool_bytes] for t in raw_kv_tensors]

    raw_info = {
        "buffers": raw_int8,
        "num_blocks": num_blocks_per_layer,
        "page_size_bytes": page_size_bytes,
        "block_stride_bytes": page_size_bytes,
        "num_pools": num_layers,
    }
    return kv_tensors, raw_info  # type: ignore[return-value]


def get_kv_cache_manager(
    num_blocks: int,
    block_size: int,
    cell_size: int,
    num_layers: int,
    num_kv_buffers: int = 2,
    group_id: int = 0,
) -> KVCacheManager:
    if not _kvcached_initialized:
        raise RuntimeError("kvcached is not initialized. Please call init_kvcached() first.")

    return KVCacheManager(
        num_blocks,
        block_size,
        cell_size,
        num_layers,
        _world_size,
        pp_rank=_pp_rank,
        async_sched=_async_sched,
        num_kv_buffers=num_kv_buffers,
        group_id=group_id,
    )
