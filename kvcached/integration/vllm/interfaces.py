# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from kvcached.kv_cache_manager import KVCacheManager
from kvcached.observability import (
    build_runtime_snapshot,
    get_registered_kv_cache_pool_operation_snapshot_dicts,
    get_registered_kv_cache_pool_operation_snapshots,
    get_registered_kv_cache_pool_snapshot_dicts,
    get_registered_kv_cache_pool_snapshots,
)
from kvcached.pool_registry import (
    clear_registered_kv_cache_pools,
    register_kv_cache_pool,
)
from kvcached.tp_ipc_util import resolve_gpu_device_index, start_worker_listener_thread
from kvcached.utils import CONTIGUOUS_LAYOUT, PAGE_SIZE, get_kvcached_logger, normalize_gpu_device
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


def should_use_worker_ipc() -> bool:
    return _kvcached_initialized and not _is_worker


def get_world_size() -> int:
    """Return the TP world size recorded by the latest initialization."""
    if not _kvcached_initialized:
        raise RuntimeError(
            "kvcached is not initialized. Please call init_kvcached() first."
        )
    return _world_size


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
            listener_device = _kvcached_device or device
            start_worker_listener_thread(
                tp_rank,
                pp_rank,
                device_index=resolve_gpu_device_index(listener_device),
            )
        if async_sched and not _async_sched:
            _async_sched = True
            logger.info("kvcached async scheduler enabled")
        _pp_rank = pp_rank
        _world_size = world_size
        return

    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"
    device = normalize_gpu_device(device)

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
        start_worker_listener_thread(
            tp_rank,
            pp_rank,
            device_index=resolve_gpu_device_index(device),
        )


def shutdown_kvcached() -> None:
    global _kvcached_initialized, _kvcached_device, _async_sched
    if not _kvcached_initialized:
        clear_registered_kv_cache_pools(integration="vllm")
        return

    _shutdown_kvcached_impl()
    clear_registered_kv_cache_pools(integration="vllm")
    _kvcached_initialized = False
    _kvcached_device = None
    _async_sched = False


def build_kv_views(
    raw_kv_tensors: List[torch.Tensor],
    kvcache_shape: Tuple[int, ...],
    block_size: int,
    dtype: torch.dtype,
    attention_type: str,
    num_blocks_per_layer: int,
    gpu_mem_bytes_per_layer_k_or_v: int,
    num_layers: int,
    kernel_block_size: Optional[int] = None,
) -> Tuple[List[torch.Tensor], int]:
    """Reinterpret already-allocated raw KV pools as per-layer KV views.

    Mirrors the view-building math inside ``alloc_kv_cache`` but takes
    already-allocated ``raw_kv_tensors`` plus the (uniform) physical quantities
    ``num_blocks_per_layer`` / ``gpu_mem_bytes_per_layer_k_or_v``. This lets a
    heterogeneous hybrid model (e.g. Gemma: sliding-window + full-attention
    groups with different ``(block_size, num_kv_heads, head_size)`` but identical
    ``block_mem_size``) build a DIFFERENT view per group over the SAME physical
    pools. Returns ``(kv_tensors, page_size_bytes)``.

    Both layouts are supported for heterogeneous callers: the shared
    ``block_mem_size`` means block N occupies the same bytes whichever group's
    view addresses it, so each branch below only has to re-derive the per-group
    shape/stride from that one uniform block stride. The one exception is
    ``contiguous`` + ``kernel_block_size != block_size``, which raises: that
    branch reshapes at virtual-block granularity and has no kernel-block form
    yet.
    """
    is_mla = attention_type == "MLA"
    unified_pool = attention_type == "HYBRID_LINEAR"
    num_k_or_v = 1 if is_mla else 2
    if kernel_block_size is None:
        kernel_block_size = block_size
    if block_size % kernel_block_size != 0:
        raise ValueError(
            f"block_size ({block_size}) must be a multiple of "
            f"kernel_block_size ({kernel_block_size})")
    ratio = block_size // kernel_block_size

    if is_mla:
        blocks_dim_idx = 0
        permute_order = list(range(len(kvcache_shape)))
    elif kvcache_shape[0] == num_k_or_v:
        blocks_dim_idx = 1
        permute_order = [1, 0] + list(range(2, len(kvcache_shape)))
    elif kvcache_shape[1] == num_k_or_v:
        blocks_dim_idx = 0
        permute_order = [0, 1] + list(range(2, len(kvcache_shape)))
    else:
        raise ValueError(f"Unsupported kv cache shape: {kvcache_shape}")

    actual_kvcache_shape: List[int] = list(kvcache_shape)
    actual_kvcache_shape[blocks_dim_idx] = num_blocks_per_layer

    page_size_bytes = math.prod(
        actual_kvcache_shape[:blocks_dim_idx] + actual_kvcache_shape[blocks_dim_idx + 1:]
    ) * dtype.itemsize

    kernel_kvcache_shape: List[int] = list(actual_kvcache_shape)
    if ratio > 1:
        kernel_kvcache_shape[blocks_dim_idx] = num_blocks_per_layer * ratio
        token_dim_idx = 2 if not is_mla else 1
        kernel_kvcache_shape[token_dim_idx] = kernel_block_size

    if not _contiguous_layout:
        kv_tensors: List[torch.Tensor] = []
        if is_mla:
            num_eles = math.prod(kernel_kvcache_shape)
            kv_tensors = [
                t.view(dtype=dtype)[:num_eles].view(kernel_kvcache_shape)
                for t in raw_kv_tensors
            ]
        else:
            shape = list(kernel_kvcache_shape)
            strides = [0] * len(shape)
            strides[-1] = 1
            for i in range(len(shape) - 2, 1, -1):
                strides[i] = strides[i + 1] * shape[i + 1]
            hidden_size_eles = strides[2] * shape[2]
            if unified_pool:
                if blocks_dim_idx == 1:
                    strides[1] = 2 * hidden_size_eles
                    strides[0] = hidden_size_eles
                else:
                    strides[0] = 2 * hidden_size_eles
                    strides[1] = hidden_size_eles
            else:
                v_offset_eles = gpu_mem_bytes_per_layer_k_or_v // dtype.itemsize
                if blocks_dim_idx == 1:
                    strides[1] = hidden_size_eles
                    strides[0] = v_offset_eles
                else:
                    strides[0] = hidden_size_eles
                    strides[1] = v_offset_eles
            for t in raw_kv_tensors:
                kv_tensors.append(
                    torch.as_strided(t.view(dtype=dtype), shape, strides))
    # NOTE: contiguous + HYBRID_LINEAR never reaches build_kv_views (hybrid-linear
    # is excluded from heterogeneous grouping); the kernel-block-granular
    # contiguous view for the unified pool lives in alloc_kv_cache.
    else:
        if ratio > 1:
            # The branch below reshapes at VIRTUAL block granularity: it never
            # consults kernel_kvcache_shape, so with ratio > 1 the views would
            # disagree with the kernel's kernel_bs-token indexing and read
            # silently wrong KV. The non-contiguous branch above does handle
            # this. No engine config has produced ratio > 1 here yet (Gemma 3/4
            # report kernel_block_size == block_size for every group), so rather
            # than ship an unexercised stride derivation, fail loud.
            raise NotImplementedError(
                "kvcached: heterogeneous attention KV groups on the contiguous "
                f"layout do not support kernel_block_size ({kernel_block_size}) "
                f"!= block_size ({block_size}). Re-launch with "
                "KVCACHED_CONTIGUOUS_LAYOUT=false.")
        layer_elem_shape = actual_kvcache_shape[:blocks_dim_idx] + actual_kvcache_shape[blocks_dim_idx + 1:]
        contiguous_shape = [num_blocks_per_layer, num_layers] + layer_elem_shape
        num_eles = math.prod(contiguous_shape)
        contiguous_tensor = raw_kv_tensors[0].view(dtype=dtype)[:num_eles].view(contiguous_shape)
        kv_tensors = [
            contiguous_tensor[:, i].permute(*permute_order) for i in range(num_layers)
        ]

    return kv_tensors, page_size_bytes


def observability_snapshot():
    """Return a read-only snapshot of the vLLM integration state."""
    return build_runtime_snapshot(
        engine="vllm",
        initialized=_kvcached_initialized,
        device=_kvcached_device,
        world_size=_world_size,
        pp_rank=_pp_rank,
        async_sched=_async_sched,
        contiguous_layout=_contiguous_layout,
        is_worker=_is_worker,
    )


def observability_snapshot_dict() -> Dict[str, Any]:
    """Return a JSON-serializable snapshot of the vLLM integration state."""
    return observability_snapshot().to_dict()


def kv_cache_pool_snapshots():
    """Return read-only snapshots for all live vLLM KV pools."""
    return get_registered_kv_cache_pool_snapshots(integration="vllm")


def kv_cache_pool_snapshot_dicts() -> List[Dict[str, Any]]:
    """Return JSON-serializable snapshots for all live vLLM KV pools."""
    return get_registered_kv_cache_pool_snapshot_dicts(integration="vllm")


def kv_cache_pool_operation_snapshots():
    """Return operation snapshots for all live vLLM KV pools."""
    return get_registered_kv_cache_pool_operation_snapshots(integration="vllm")


def kv_cache_pool_operation_snapshot_dicts() -> List[Dict[str, Any]]:
    """Return JSON-serializable operation snapshots for live vLLM KV pools."""
    return get_registered_kv_cache_pool_operation_snapshot_dicts(integration="vllm")


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
    return_meta: bool = False,
) -> List[torch.Tensor]:
    """Allocate KV cache tensors for all supported attention types.

    When ``return_meta`` is True, the physical-allocation metadata needed to
    rebuild per-group views (for heterogeneous hybrid models like Gemma) is
    appended to the return value as a final ``meta`` dict with keys:
    ``raw_kv_tensors``, ``num_blocks_per_layer``, ``gpu_mem_bytes_per_layer_k_or_v``,
    ``num_layers``, ``dtype``.

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
          buffers            - flat int8 tensors: one per pool in the
                               non-contiguous layout, or a single shared base
                               buffer ([base]) in the contiguous layout
          num_blocks         - number of blocks per pool
          page_size_bytes    - uniform page size (bytes) shared by all groups
          block_stride_bytes - byte stride between consecutive blocks of the
                               same pool (page_size_bytes when non-contiguous,
                               num_layers*page_size_bytes when contiguous)
          num_pools          - number of pools (== num_layers)
          is_contiguous      - whether the contiguous layout is in use
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

    # Hybrid linear-attention (HYBRID_LINEAR) supports BOTH the contiguous and
    # non-contiguous KV layouts. In contiguous layout the attention view
    # (contiguous_tensor[:, i].permute, below) is already K/V-interleaved per
    # block, and the mamba state view is rebuilt with a num_layers-scaled block

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

    # Contiguous + hybrid linear-attention supports any ratio (= block_size /
    # kernel_block_size): attention-owned virtual blocks are viewed with
    # kernel-block-outermost striding (globally linear in the kernel-block id,
    # see the unified_pool contiguous branch below), while mamba-owned virtual
    # blocks keep the slot-sequential layout. Both views alias the same bytes;
    # a virtual block is owned by exactly one KV-cache group at a time (vLLM's
    # groups draw disjoint ids from one shared BlockPool), so only one
    # interpretation is ever live for a given block.

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

    assert torch.cuda.is_available(), "GPU backend is not available via torch.cuda."
    device = normalize_gpu_device(device)

    # --- Compute per-layer memory budget and number of blocks ---
    gpu_mem_bytes = torch.cuda.get_device_properties(device).total_memory
    gpu_mem_bytes_per_layer_k_or_v = gpu_mem_bytes // num_layers // num_k_or_v
    # Round down to 2 * PAGE_SIZE for MLA backend.
    # The get_v_base_offset() requires the ftensor size (which equals
    # gpu_mem_bytes_per_layer_k_or_v * num_k_or_v) to be a multiple of
    # 2 * PAGE_SIZE. When num_k_or_v == 1 (MLA), we must align this value
    # to 2 * PAGE_SIZE directly. For MHA/GQA (num_k_or_v == 2), aligning
    # to PAGE_SIZE suffices because ftensor_bytes = 2 * aligned_value is
    # automatically 2*PAGE_SIZE-aligned.
    alignment = 2 * PAGE_SIZE if is_mla else PAGE_SIZE
    gpu_mem_bytes_per_layer_k_or_v = (gpu_mem_bytes_per_layer_k_or_v // alignment) * alignment

    num_blocks_per_layer = gpu_mem_bytes_per_layer_k_or_v // block_mem_bytes
    if requested_num_blocks > num_blocks_per_layer:
        logger.warning(
            f"Requested {requested_num_blocks} blocks, but only {num_blocks_per_layer} blocks are available."
        )

    ftensor_bytes_per_layer = gpu_mem_bytes_per_layer_k_or_v * num_k_or_v

    # For the unified (hybrid) pool, K and V are interleaved into a single
    # buffer per layer, so the KVCacheManager / PageAllocator use
    # num_kv_buffers=1 (see _get_kv_cache_params in patches.py). The C++
    # contiguous layout sizes its compound page as
    # kPageSize*num_layers*num_kv_buffers, so it MUST see the same 1 here --
    # otherwise the compound page is 2x too large and FTensor::map's
    # offset-alignment assert fails on every odd page id (and the compound-page
    # count no longer matches the manager's block count). In non-contiguous
    # layout create_kv_tensors ignores num_kv_buffers, so this is a no-op there.
    # The FTensor byte size is unaffected: it comes from ftensor_bytes_per_layer
    # (= gpu_mem_bytes_per_layer_k_or_v * num_k_or_v) above, which already
    # accounts for both K and V.
    compound_num_kv_buffers = 1 if unified_pool else num_k_or_v
    raw_kv_tensors = create_kv_tensors(
        ftensor_bytes_per_layer, dtype.itemsize, device, num_layers,
        num_kv_buffers=compound_num_kv_buffers, group_id=group_id,
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
    elif unified_pool:
        # Contiguous HYBRID_LINEAR: kernel-block-granular attention views (see
        # the identical branch in build_kv_views for the layout derivation).
        # Slot i's kernel block kb lives at element kb*(num_layers*2h) + i*2h,
        # globally linear in kb; at ratio==1 this is byte-for-byte the
        # historical [slot0: K|V][slot1: K|V]... per-block layout. Mamba-owned
        # blocks are read slot-sequentially by _reshape_mamba_contiguous over
        # the same bytes -- valid because a virtual block belongs to exactly
        # one KV-cache group at a time.
        shape = list(kernel_kvcache_shape)
        strides = [0] * len(shape)
        strides[-1] = 1
        for i in range(len(shape) - 2, 1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        hidden_size_eles = strides[2] * shape[2]  # = kernel_bs * H * D
        if blocks_dim_idx == 1:          # FlashAttn (2, N*ratio, ...)
            strides[1] = num_layers * 2 * hidden_size_eles
            strides[0] = hidden_size_eles
        else:                             # FlashInfer (N*ratio, 2, ...)
            strides[0] = num_layers * 2 * hidden_size_eles
            strides[1] = hidden_size_eles
        flat = raw_kv_tensors[0].view(dtype=dtype)
        kv_tensors = [
            torch.as_strided(flat, shape, strides,
                             storage_offset=i * 2 * hidden_size_eles)
            for i in range(num_layers)
        ]
    else:
        layer_elem_shape = actual_kvcache_shape[:blocks_dim_idx] + actual_kvcache_shape[blocks_dim_idx + 1:]
        contiguous_shape = [num_blocks_per_layer, num_layers] + layer_elem_shape
        num_eles = math.prod(contiguous_shape)
        contiguous_tensor = raw_kv_tensors[0].view(dtype=dtype)[:num_eles].view(contiguous_shape)
        kv_tensors = [
            contiguous_tensor[:, i].permute(*permute_order) for i in range(num_layers)
        ]

    meta = {
        "raw_kv_tensors": raw_kv_tensors,
        "num_blocks_per_layer": num_blocks_per_layer,
        "gpu_mem_bytes_per_layer_k_or_v": gpu_mem_bytes_per_layer_k_or_v,
        "num_layers": num_layers,
        "dtype": dtype,
    }

    if not unified_pool:
        if return_meta:
            return kv_tensors, meta  # type: ignore[return-value]
        return kv_tensors

    # --- Build raw int8 buffers for hybrid model (mamba) support ---
    # Non-contiguous: one compact flat buffer per pool; consecutive blocks of a
    # pool are page_size_bytes apart. Contiguous: a single interleaved base
    # buffer shared by all pools; block N of pool L sits at
    # (N*num_pools + L)*page_size_bytes, so the per-pool block stride is
    # num_layers*page_size_bytes. _reshape_mamba_{non_,}contiguous consume these.
    if not _contiguous_layout:
        pool_bytes = num_blocks_per_layer * page_size_bytes
        raw_int8 = [t.view(torch.int8)[:pool_bytes] for t in raw_kv_tensors]
        block_stride_bytes = page_size_bytes
    else:
        raw_int8 = [raw_kv_tensors[0].view(torch.int8)]
        block_stride_bytes = num_layers * page_size_bytes

    raw_info = {
        "buffers": raw_int8,
        "num_blocks": num_blocks_per_layer,
        "page_size_bytes": page_size_bytes,
        "block_stride_bytes": block_stride_bytes,
        "num_pools": num_layers,
        "is_contiguous": _contiguous_layout,
    }
    if return_meta:
        return kv_tensors, raw_info, meta  # type: ignore[return-value]
    return kv_tensors, raw_info  # type: ignore[return-value]


def get_kv_cache_manager(
    num_blocks: int,
    block_size: int,
    cell_size: int,
    num_layers: int,
    num_kv_buffers: int = 2,
    group_id: int = 0,
    pool_name: Optional[str] = None,
) -> KVCacheManager:
    if not _kvcached_initialized:
        raise RuntimeError("kvcached is not initialized. Please call init_kvcached() first.")

    manager = KVCacheManager(
        num_blocks,
        block_size,
        cell_size,
        num_layers,
        _world_size,
        pp_rank=_pp_rank,
        async_sched=_async_sched,
        num_kv_buffers=num_kv_buffers,
        group_id=group_id,
        reserve_null_block=True,
        pool_name=pool_name,
        defer_physical_release=_async_sched,
    )
    register_kv_cache_pool(
        manager,
        integration="vllm",
    )
    return manager
