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
    get_registered_kv_cache_pools,
    register_kv_cache_pool,
)
from kvcached.tp_ipc_util import (
    resolve_gpu_device_index,
    start_worker_listener_thread,
    stop_worker_listener_threads,
)
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
# Geometry of the KV cache tensors created by alloc_kv_cache() in this
# process, keyed by group_id: the per-layer capacity actually reserved
# (num_blocks, ftensor_bytes_per_layer) and the num_layers / num_kv_buffers
# the FTensors were created with. Whichever call comes second,
# get_kv_cache_manager() or alloc_kv_cache(), validates the manager against
# this record so a manager can never address page ids beyond the FTensor's
# reserved virtual range or at a different compound page stride (issue #437).
_created_kv_tensor_capacity: Dict[int, Dict[str, int]] = {}

# Single source of truth for what this shim accepts. The capability record
# in kvcached.observability reports these, so the guards below and the
# reported record cannot drift apart.
SUPPORTED_ATTENTION_TYPES = ("MHA", "GQA", "MLA", "HYBRID_LINEAR")
SUPPORTED_KV_LAYOUTS = ("NHD", "HND")


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


def shutdown_kvcached() -> bool:
    """Release KV resources, or return False if an active listener or an
    unstopped pool needs a retry."""
    global _kvcached_initialized, _kvcached_device, _async_sched, _is_worker
    _created_kv_tensor_capacity.clear()
    if not _kvcached_initialized:
        clear_registered_kv_cache_pools(integration="vllm")
        return True

    if not stop_worker_listener_threads():
        logger.warning("KV shutdown deferred: a worker IPC listener is still active")
        return False
    # Pools first: each unlinks its /dev/shm segment while the process is
    # still alive (issue #477), then the allocator.
    pools_stopped = True
    for manager, _ in get_registered_kv_cache_pools(integration="vllm"):
        try:
            if manager.shutdown() is False:
                pools_stopped = False
        except Exception as e:
            pools_stopped = False
            logger.warning("Failed to shut down KV cache pool %s: %s",
                           getattr(manager, "pool_name", None), e)
    if not pools_stopped:
        # Keep failed pools reachable for retry, and do not release mappings
        # while a preallocation thread may still be running.
        return False
    _shutdown_kvcached_impl()
    clear_registered_kv_cache_pools(integration="vllm")
    _kvcached_initialized = False
    _kvcached_device = None
    _async_sched = False
    _is_worker = False
    return True


def _set_block_copy_view(
    tensor: torch.Tensor, backing: torch.Tensor, num_blocks: int, block_bytes: int,
) -> None:
    """Record logical block boundaries without using the reserved storage size."""
    raw = backing.view(torch.uint8).view(-1)
    pages = raw[:num_blocks * block_bytes].view(num_blocks, block_bytes)
    setattr(tensor, "_kvcached_block_copy_view", pages)


def _copy_kv_cache_blocks(
    kv_caches, num_blocks: int, block_copies,
) -> None:
    """Copy shared backing pools once, using their actual logical page stride."""
    if not block_copies:
        return
    pairs = list(block_copies)
    if num_blocks <= 0 or any(
        src < 0 or dst < 0 or src >= num_blocks or dst >= num_blocks
        for src, dst in pairs
    ):
        raise ValueError("KV block copy index is outside the logical block pool")
    if len({dst for _, dst in pairs}) != len(pairs):
        raise ValueError("KV block copies must have distinct destinations")

    views = []
    seen = set()
    for entry in kv_caches:
        for tensor in entry if isinstance(entry, (list, tuple)) else (entry,):
            pages = getattr(tensor, "_kvcached_block_copy_view", None)
            if pages is None:
                raise ValueError("KV block copy requires kvcached logical page metadata")
            if pages.ndim < 2 or pages.dtype != torch.uint8 or pages.shape[0] < num_blocks:
                raise ValueError("Invalid kvcached logical block copy view")
            key = (pages.data_ptr(), tuple(pages.shape), tuple(pages.stride()))
            if key not in seen:
                seen.add(key)
                views.append(pages)
    if not views:
        return
    device = views[0].device
    if any(pages.device != device for pages in views):
        raise ValueError("KV block copy views must belong to the same device")
    if device.type == "cuda":
        import numpy as np
        from vllm.v1.worker.utils import async_tensor_h2d

        indices = async_tensor_h2d(np.array(pairs, dtype=np.int64), device=device)
    else:
        indices = torch.tensor(pairs, dtype=torch.int64, device=device)
    src_indices, dst_indices = indices.unbind(dim=1)
    for pages in views:
        # Advanced indexing snapshots the sources, including overlapping copies.
        pages[dst_indices] = pages[src_indices]


def _set_legacy_block_copy_views(
    tensors: List[torch.Tensor], raw_tensors: List[torch.Tensor],
    num_blocks: int, page_bytes: int, num_layers: int,
    unified: bool, split_offset: int,
) -> None:
    for layer, tensor in enumerate(tensors):
        backing = raw_tensors[0 if _contiguous_layout else layer]
        if _contiguous_layout or unified:
            _set_block_copy_view(
                tensor, backing, num_blocks,
                page_bytes * (num_layers if _contiguous_layout else 1),
            )
        else:
            # Preserve the reserved gap between K and V instead of copying it.
            raw_bytes = backing.view(torch.uint8)
            half_bytes = page_bytes // 2
            setattr(tensor, "_kvcached_block_copy_view", torch.as_strided(
                raw_bytes, (num_blocks, 2, half_bytes),
                (half_bytes, split_offset, 1),
                storage_offset=raw_bytes.storage_offset(),
            ))


def _build_packed_kv_views(
    raw_kv_tensors: List[torch.Tensor],
    shape: Tuple[int, ...],
    num_blocks: int,
    block_size: int,
    kernel_block_size: int,
    dtype: torch.dtype,
    num_layers: int,
    kv_layout: str,
) -> List[torch.Tensor]:
    """View a unified pool as vLLM's (blocks, heads, tokens, K+V) cache."""
    if (len(shape) != 4 or shape[2] != block_size
            or any(dim <= 0 for dim in shape) or shape[3] % 2):
        raise ValueError(f"Unsupported packed KV cache shape: {shape}")
    heads, width = shape[1], shape[3]
    if kv_layout == "NHD":
        inner_strides = (width, heads * width, 1)
    elif kv_layout == "HND":
        inner_strides = (kernel_block_size * width, width, 1)
    else:
        raise ValueError(f"Unsupported packed KV layout: {kv_layout}")
    kernel_elements = heads * kernel_block_size * width
    layer_stride = kernel_elements if _contiguous_layout else 0
    block_stride = kernel_elements * (num_layers if _contiguous_layout else 1)
    view_shape = (num_blocks * (block_size // kernel_block_size),
                  heads, kernel_block_size, width)
    tensors = [
        torch.as_strided(
            raw_kv_tensors[0 if _contiguous_layout else layer].view(dtype),
            view_shape, (block_stride, *inner_strides),
            storage_offset=layer * layer_stride,
        )
        for layer in range(num_layers)
    ]
    block_bytes = heads * block_size * width * dtype.itemsize
    for layer, tensor in enumerate(tensors):
        _set_block_copy_view(
            tensor, raw_kv_tensors[0 if _contiguous_layout else layer],
            num_blocks, block_bytes * (num_layers if _contiguous_layout else 1),
        )
    return tensors


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
    kv_layout: str = "NHD",
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

    if not is_mla and len(kvcache_shape) == 4:
        views = _build_packed_kv_views(
            raw_kv_tensors, kvcache_shape, num_blocks_per_layer, block_size,
            kernel_block_size, dtype, num_layers, kv_layout,
        )
        return views, math.prod(kvcache_shape[1:]) * dtype.itemsize

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

    _set_legacy_block_copy_views(
        kv_tensors, raw_kv_tensors, num_blocks_per_layer, page_size_bytes,
        num_layers, is_mla or unified_pool, gpu_mem_bytes_per_layer_k_or_v,
    )
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
      - Packed KV: (num_blocks, head_num, block_size, 2 * head_dim)
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

    if attention_type not in SUPPORTED_ATTENTION_TYPES:
        raise ValueError(f"Attention type {attention_type} is not supported.")

    is_mla = attention_type == "MLA"
    packed_kv = not is_mla and len(kvcache_shape) == 4
    if kv_layout not in SUPPORTED_KV_LAYOUTS or (kv_layout == "HND" and not packed_kv):
        raise ValueError(f"KV layout {kv_layout} is not supported.")

    is_hybrid_linear = attention_type == "HYBRID_LINEAR"
    unified_pool = is_hybrid_linear or packed_kv

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
    if packed_kv:
        if (kvcache_shape[2] != block_size
                or any(dim <= 0 for dim in kvcache_shape)
                or kvcache_shape[3] % 2):
            raise ValueError(f"Unsupported packed KV cache shape: {kvcache_shape}")
        blocks_dim_idx = 0
        permute_order = [0, 1, 2, 3]
        block_mem_bytes = math.prod(kvcache_shape[1:]) * dtype.itemsize // 2
    elif is_mla:
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
    created_capacity = {
        "num_blocks": num_blocks_per_layer,
        "ftensor_bytes_per_layer": ftensor_bytes_per_layer,
        "num_layers": num_layers,
        "num_kv_buffers": compound_num_kv_buffers,
    }
    # Manager-first order: a KVCacheManager for this group may already exist,
    # polling kv_tensors_created() from its _post_init thread and mapping
    # pages as soon as it flips. Validate it against the geometry about to be
    # created before create_kv_tensors() makes the tensors available for
    # mapping (issue #437).
    _validate_registered_managers(group_id, created_capacity)
    raw_kv_tensors = create_kv_tensors(
        ftensor_bytes_per_layer, dtype.itemsize, device, num_layers,
        num_kv_buffers=compound_num_kv_buffers, group_id=group_id,
        unified_pool=unified_pool,
    )

    # Record the geometry actually created for this group so that
    # get_kv_cache_manager() can refuse (or derive) a manager configuration
    # that does not match it (issue #437).
    _created_kv_tensor_capacity[group_id] = created_capacity

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
    kv_tensors: List[torch.Tensor]
    if packed_kv:
        kv_tensors = _build_packed_kv_views(
            raw_kv_tensors, kvcache_shape, num_blocks_per_layer, block_size,
            kernel_block_size, dtype, num_layers, kv_layout,
        )
    elif not _contiguous_layout:
        kv_tensors = []
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

    if not packed_kv:
        _set_legacy_block_copy_views(
            kv_tensors, raw_kv_tensors, num_blocks_per_layer, page_size_bytes,
            num_layers, is_mla or unified_pool, gpu_mem_bytes_per_layer_k_or_v,
        )

    meta = {
        "raw_kv_tensors": raw_kv_tensors,
        "num_blocks_per_layer": num_blocks_per_layer,
        "gpu_mem_bytes_per_layer_k_or_v": gpu_mem_bytes_per_layer_k_or_v,
        "num_layers": num_layers,
        "dtype": dtype,
    }

    if not is_hybrid_linear:
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


def _created_capacity_num_blocks(
    record: Dict[str, int], block_mem_size: int, num_kv_buffers: int
) -> int:
    """Per-layer block capacity of ``record``'s tensors in a manager's geometry.

    A manager's page-id space covers ``num_blocks * block_mem_size`` bytes per
    layer per KV buffer, while each created FTensor reserves
    ``ftensor_bytes_per_layer`` bytes for all ``num_kv_buffers`` of a layer.
    """
    return record["ftensor_bytes_per_layer"] // num_kv_buffers // block_mem_size


def _validate_manager_against_created_tensors(
    record: Dict[str, int],
    num_blocks: int,
    block_mem_size: int,
    num_layers: int,
    num_kv_buffers: int,
    group_id: int,
) -> None:
    """Raise ValueError unless a manager of this geometry fits ``record``.

    ``record`` is the ``_created_kv_tensor_capacity`` entry of ``group_id``,
    already created or about to be. Shared by both construction orders:
    ``get_kv_cache_manager`` after ``alloc_kv_cache`` and ``alloc_kv_cache``
    after ``get_kv_cache_manager``.

    ``num_layers`` and ``num_kv_buffers`` must match exactly. In the
    contiguous layout the FTensor and the PageAllocator both stride compound
    pages by ``page_size * num_layers * num_kv_buffers``, so a manager built
    with different values maps pages at the wrong offsets even when its block
    count fits; in the per-layer layout they size the manager's
    physical-memory accounting.

    ``num_blocks`` must not exceed the created capacity. ``alloc_kv_cache``
    sizes the tensors from device memory and can create fewer blocks than
    requested (it warns and clamps); a manager configured with the original,
    larger count exposes page ids beyond the FTensor's reserved virtual
    range, and the first map past the reservation fails
    ``cuMemUnmap``/``cuMemMap`` and aborts the process inside
    ``FTensor::map`` (issue #437, diagnosed by @rob-9).
    """
    for name, value in (("num_layers", num_layers), ("num_kv_buffers", num_kv_buffers)):
        if value != record[name]:
            raise ValueError(
                f"{name}={value} does not match {name}={record[name]} of the KV "
                f"cache tensors alloc_kv_cache() creates for group {group_id}. "
                "In the contiguous layout the FTensor and the PageAllocator "
                "both stride compound pages by "
                "page_size * num_layers * num_kv_buffers, so a manager built "
                "with different values maps pages at the wrong offsets "
                "(issue #437). Pass get_kv_cache_manager() the num_layers the "
                "tensors were created with, and num_kv_buffers=2 for MHA/GQA "
                "or 1 for MLA/HYBRID_LINEAR."
            )

    capacity_num_blocks = _created_capacity_num_blocks(record, block_mem_size, num_kv_buffers)
    if num_blocks > capacity_num_blocks:
        raise ValueError(
            f"num_blocks={num_blocks} exceeds the capacity of the KV cache "
            f"tensors alloc_kv_cache() creates for group {group_id}: "
            f"{capacity_num_blocks} blocks of {block_mem_size} bytes "
            f"({record['ftensor_bytes_per_layer']} reserved bytes per layer, "
            f"{num_kv_buffers} KV buffers). A manager configured beyond the "
            "created tensors maps pages outside the reserved virtual range "
            "and aborts in FTensor::map (issue #437). Create the manager "
            "after alloc_kv_cache() with num_blocks=None to derive the "
            "capacity, or pass the clamped block count logged by "
            "alloc_kv_cache()."
        )


def _validate_registered_managers(group_id: int, record: Dict[str, int]) -> None:
    """Manager-first order: refuse tensors an existing manager cannot address.

    ``get_kv_cache_manager`` registers every manager it builds in
    ``kvcached.pool_registry`` (weak references, so a collected manager does
    not count). A manager built for ``group_id`` before ``alloc_kv_cache``
    ran had nothing to validate against; check it here against the geometry
    ``record`` about to be created.
    """
    for manager, _ in get_registered_kv_cache_pools(integration="vllm"):
        if manager.group_id != group_id:
            continue
        _validate_manager_against_created_tensors(
            record,
            manager.num_blocks,
            manager.block_mem_size,
            manager.num_layers,
            manager.num_kv_buffers,
            group_id,
        )


def _resolve_manager_num_blocks(
    num_blocks: Optional[int],
    block_size: int,
    cell_size: int,
    num_layers: int,
    num_kv_buffers: int,
    group_id: int,
) -> int:
    """Validate or derive the manager's block capacity for ``group_id``.

    When ``alloc_kv_cache`` has recorded tensors for ``group_id`` in this
    process, the manager must fit them (see
    ``_validate_manager_against_created_tensors``) and ``num_blocks=None``
    derives their capacity directly. When no allocation was recorded (e.g.
    the manager lives in the engine process while ``alloc_kv_cache`` runs in
    the worker process, as in the vLLM integration, or the manager is created
    first), an explicit ``num_blocks`` is returned unchanged and ``None`` is
    rejected; a manager created first is validated by ``alloc_kv_cache``.
    """
    record = _created_kv_tensor_capacity.get(group_id)
    if record is None:
        if num_blocks is None:
            raise ValueError(
                "num_blocks=None requires KV cache tensors created by "
                f"alloc_kv_cache() in this process for group {group_id}; "
                "no allocation is recorded to derive the capacity from."
            )
        return num_blocks

    block_mem_size = block_size * cell_size
    if num_blocks is None:
        num_blocks = _created_capacity_num_blocks(record, block_mem_size, num_kv_buffers)
    _validate_manager_against_created_tensors(
        record, num_blocks, block_mem_size, num_layers, num_kv_buffers, group_id
    )
    return num_blocks


def get_kv_cache_manager(
    num_blocks: Optional[int],
    block_size: int,
    cell_size: int,
    num_layers: int,
    num_kv_buffers: int = 2,
    group_id: int = 0,
    pool_name: Optional[str] = None,
) -> KVCacheManager:
    """Create and register the KVCacheManager for one KV cache group.

    When ``alloc_kv_cache`` already created the tensors for ``group_id`` in
    this process, ``num_layers`` and ``num_kv_buffers`` must match them and
    ``num_blocks`` must fit their capacity (``None`` derives it). When the
    manager is created first, ``alloc_kv_cache`` runs the same validation
    before creating the tensors.
    """
    if not _kvcached_initialized:
        raise RuntimeError("kvcached is not initialized. Please call init_kvcached() first.")

    num_blocks = _resolve_manager_num_blocks(
        num_blocks, block_size, cell_size, num_layers, num_kv_buffers, group_id
    )

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
    )
    register_kv_cache_pool(
        manager,
        integration="vllm",
    )
    return manager
