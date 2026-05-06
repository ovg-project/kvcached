# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, Dict, List, Optional, Tuple, Union

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
_world_size: int = 1
_pp_rank: int = 0


def init_kvcached(
    tp_rank: int = 0,
    world_size: int = 1,
    pp_rank: int = 0,
    device: Optional[str] = None,
    async_sched: bool = False,
) -> None:
    global _kvcached_initialized, _kvcached_device, _async_sched, _world_size, _pp_rank
    if _kvcached_initialized:
        return

    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"

    _init_kvcached_impl(device, PAGE_SIZE, _contiguous_layout)
    _kvcached_initialized = True
    _kvcached_device = device
    _async_sched = async_sched
    _world_size = world_size
    _pp_rank = pp_rank

    if world_size > 1:
        # start the listener thread for tensor parallel kv cache management
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
    dtype: torch.dtype,
    device: str,
    num_layers: int,
    page_size: int = 1,
    attention_type: str = "MHA",  # MHA, GQA, or MLA.
    kv_layout: str = "NHD",  # NHD: (num_tokens, head_num, head_dim)
    group_id: int = 0,
) -> Union[Tuple[List[torch.Tensor], List[torch.Tensor]], List[torch.Tensor]]:
    """Allocate KV cache tensors for MHA/GQA (separate K/V) or MLA (single buffer per layer).

    For MHA/GQA, returns ``(k_tensors, v_tensors)``. For MLA, returns a single list of
    per-layer buffers of shape ``(num_tokens, 1, kv_cache_dim)``.
    """
    if not _kvcached_initialized:
        raise RuntimeError("kvcached is not initialized. Please call init_kvcached() first.")

    if attention_type not in ["MHA", "GQA", "MLA"]:
        raise ValueError(f"Attention type {attention_type} is not supported.")

    is_mla = attention_type == "MLA"
    if not is_mla and kv_layout != "NHD":
        raise ValueError(f"KV layout {kv_layout} is not supported.")

    num_k_or_v = 1 if is_mla else 2
    requested_num_tokens = kvcache_shape[0]

    if len(kvcache_shape) <= 2:
        raise ValueError(f"Unsupported kv cache shape: {kvcache_shape}")

    assert torch.cuda.is_available(), "CUDA is not available."

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
        num_kv_buffers=num_k_or_v, group_id=group_id,
    )

    num_blocks_per_layer = gpu_mem_bytes_per_layer_k_or_v // block_mem_size
    num_tokens = num_blocks_per_layer * block_size
    if requested_num_tokens > num_tokens:
        logger.warning(
            f"Requested {requested_num_tokens} tokens, but only {num_tokens} tokens are available."
        )

    actual_kvcache_shape: List[int] = list(kvcache_shape)
    actual_kvcache_shape[0] = block_size * num_blocks_per_layer

    if is_mla:
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

    k_tensors, v_tensors = [], []

    if not _contiguous_layout:
        # In the per-layer FTensors, K occupies [0, v_offset) and V
        # occupies [v_offset, 2*v_offset) where v_offset equals
        # gpu_mem_bytes_per_layer_k_or_v.  A plain contiguous view would
        # place V at num_blocks_per_layer * block_mem_size, which differs
        # from v_offset when block_mem_size does not evenly divide
        # gpu_mem_bytes_per_layer_k_or_v.  Use as_strided so the K/V
        # dimension stride matches the real V offset.
        v_offset_eles = gpu_mem_bytes_per_layer_k_or_v // dtype.itemsize
        kv_shape = [num_k_or_v] + list(actual_kvcache_shape)
        strides = [0] * len(kv_shape)
        strides[-1] = 1
        for i in range(len(kv_shape) - 2, 0, -1):
            strides[i] = strides[i + 1] * kv_shape[i + 1]
        strides[0] = v_offset_eles
        for t in raw_kv_tensors:
            kv = torch.as_strided(t.view(dtype=dtype), kv_shape, strides)
            k_tensors.append(kv[0])
            v_tensors.append(kv[1])
    else:
        contiguous_shape = (num_tokens, num_layers, num_k_or_v, *actual_kvcache_shape[1:])
        num_eles = math.prod(contiguous_shape)
        contiguous_tensor = raw_kv_tensors[0].view(dtype=dtype)[:num_eles].view(contiguous_shape)
        for i in range(num_layers):
            k_tensors.append(contiguous_tensor[:, i, 0, :, :])
            v_tensors.append(contiguous_tensor[:, i, 1, :, :])

    return k_tensors, v_tensors


def alloc_mamba_states(
    *,
    num_slots: int,
    num_mamba_layers: int,
    cache_params: Any,
    device: str,
    group_id: int = 0,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Allocate kvcached-backed mamba conv + temporal state tensors with a
    super-cell layout.

    Per-slot byte layout within each (slot, layer) super-cell:

        [ conv[0] bytes | conv[1] bytes | ... | temporal bytes ]

    One mamba slot == one kvcached block; a single ``map_to_kv_tensors`` call
    backs every state kind for that slot across all layers simultaneously.

    Two layouts are supported:

    * Contiguous (``KVCACHED_CONTIGUOUS_LAYOUT=true``): all mamba layers
      share a single VM reservation, so the returned ``conv_state``
      entries and ``temporal_state`` are single ``(num_mamba_layers,
      num_slots, *shape)`` tensors — identical in shape to the native
      ``MambaPool`` buffers.

    * Non-contiguous (``KVCACHED_CONTIGUOUS_LAYOUT=false``): each layer
      has its own VM reservation, so a single tensor that spans all
      layers is impossible.  ``conv_state`` is a ``List[List[Tensor]]``
      (outer = conv shape, inner = layer; each inner tensor has shape
      ``(num_slots, *shape)``) and ``temporal_state`` is a
      ``List[Tensor]`` (one per layer, shape ``(num_slots,
      *temporal_shape)``).  The caller is responsible for adapting the
      ``MambaPool.State`` API; SGLang's ``at_layer_idx`` path works
      naturally because it only requests per-layer slices, but
      multi-layer code paths (e.g. ``update_mamba_state_after_mtp_verify``
      in the speculative-decode backend) require contiguous layout.

    Returns:
        (conv_state, temporal_state, layout_info) where layout_info
        carries an ``is_contiguous`` flag so callers can branch on the
        returned shapes.
    """
    if not _kvcached_initialized:
        raise RuntimeError(
            "kvcached is not initialized. Please call init_kvcached() first.")

    assert torch.cuda.is_available(), "CUDA is not available."

    conv_shapes = [tuple(s) for s in cache_params.shape.conv]
    temporal_shape = tuple(cache_params.shape.temporal)
    conv_dtype = cache_params.dtype.conv
    ssm_dtype = cache_params.dtype.temporal

    def _align_up(x: int, a: int) -> int:
        return (x + a - 1) // a * a

    # Pack state kinds end-to-end per slot; track per-kind byte offsets.
    conv_offsets: List[int] = []
    offset = 0
    for shape in conv_shapes:
        offset = _align_up(offset, conv_dtype.itemsize)
        conv_offsets.append(offset)
        offset += int(math.prod(shape)) * conv_dtype.itemsize
    offset = _align_up(offset, ssm_dtype.itemsize)
    temporal_offset = offset
    offset += int(math.prod(temporal_shape)) * ssm_dtype.itemsize

    max_align = max(conv_dtype.itemsize, ssm_dtype.itemsize)
    raw_cell_size = _align_up(offset, max_align)

    if raw_cell_size > PAGE_SIZE:
        raise RuntimeError(
            f"Mamba per-slot super-cell ({raw_cell_size} bytes) exceeds "
            f"kvcached PAGE_SIZE ({PAGE_SIZE} bytes). Raise "
            "KVCACHED_PAGE_SIZE_MB so a single physical page can back at "
            "least one slot.")

    # Pad cell_size up to a divisor of PAGE_SIZE so blocks_per_page is an
    # integer with no straddle-skipping.  Reason: kvcached's PageAllocator
    # delivers floor(page_size / block_mem_size) blocks per page when
    # block_mem_size does not divide page_size evenly (straddling block IDs
    # are dropped by Page.get_block_range).  Passing num_blocks=num_slots to
    # KVCacheManager over-promises capacity — the manager then tracks
    # mem_size = num_slots * raw_cell_size per layer but PageAllocator can
    # only supply floor(mem_size / page_size) blocks, which falls short of
    # num_slots.  The scheduler then hits "Not enough space for mamba
    # cache" long before the real pool is full.
    #
    # Padding to the smallest divisor of PAGE_SIZE >= raw_cell_size makes
    # blocks_per_page a clean integer; num_slots blocks are actually
    # deliverable.  Raising KVCACHED_PAGE_SIZE_MB exposes finer divisors
    # (e.g. 6 MB has a 1.5 MB divisor) and reduces the virtual overhead.
    def _smallest_divisor_ge(n: int, lower: int) -> int:
        best = n
        i = 1
        while i * i <= n:
            if n % i == 0:
                if i >= lower:
                    best = min(best, i)
                j = n // i
                if j >= lower:
                    best = min(best, j)
            i += 1
        return best

    cell_size = _smallest_divisor_ge(PAGE_SIZE, raw_cell_size)
    if cell_size != raw_cell_size:
        overhead = (cell_size - raw_cell_size) * num_mamba_layers * num_slots
        logger.info(
            f"[kvcached] Elastic mamba cell padded: "
            f"raw={raw_cell_size}B -> {cell_size}B "
            f"(divisor of PAGE_SIZE={PAGE_SIZE}B). Virtual overhead: "
            f"{overhead / (1024**3):.2f} GB. Raise KVCACHED_PAGE_SIZE_MB "
            "for finer divisors if needed.")

    # create_kv_tensors expects per-layer size aligned to PAGE_SIZE.
    per_layer_bytes = num_slots * cell_size
    per_layer_bytes_aligned = _align_up(per_layer_bytes, PAGE_SIZE)

    # Non-contiguous layout requires unified_pool=True so the per-layer
    # FTensor map path skips the K/V split (which assumes a 2-buffer layout
    # via get_v_base_offset and asserts num_eles % (2*kPageSize) == 0).
    # Each FTensor here is a single-buffer mamba state, so unified_pool=True
    # makes map_to_kv_tensors map exactly one VMM page per offset per
    # layer.  In contiguous mode, the C++ ignores unified_pool and uses
    # the compound-page path instead, so the flag is harmless there.
    raw_tensors = create_kv_tensors(
        per_layer_bytes_aligned, torch.int8.itemsize, device,
        num_mamba_layers, num_kv_buffers=1, group_id=group_id,
        unified_pool=not _contiguous_layout,
    )

    layout_info: Dict[str, Any] = {
        "cell_size": cell_size,
        "num_slots": num_slots,
        "num_mamba_layers": num_mamba_layers,
        "conv_offsets": conv_offsets,
        "temporal_offset": temporal_offset,
        "is_contiguous": _contiguous_layout,
    }

    if _contiguous_layout:
        # Contiguous layout hands back a single flat int8 FTensor covering
        # all mamba layers: bytes laid out as [slot][layer][cell].
        flat_int8 = raw_tensors[0]

        def _view_state_contig(shape: Tuple[int, ...], dtype: torch.dtype,
                               offset_bytes: int) -> torch.Tensor:
            dtype_size = dtype.itemsize
            assert offset_bytes % dtype_size == 0
            assert cell_size % dtype_size == 0

            inner_stride: List[int] = []
            running = 1
            for d in reversed(shape):
                inner_stride.insert(0, running)
                running *= d

            size = (num_mamba_layers, num_slots, *shape)
            strides = [
                cell_size // dtype_size,
                (num_mamba_layers * cell_size) // dtype_size,
                *inner_stride,
            ]
            return torch.as_strided(
                flat_int8.view(dtype=dtype),
                size=size,
                stride=strides,
                storage_offset=offset_bytes // dtype_size,
            )

        conv_state: Any = [
            _view_state_contig(shape, conv_dtype, conv_offsets[i])
            for i, shape in enumerate(conv_shapes)
        ]
        temporal_state: Any = _view_state_contig(temporal_shape, ssm_dtype,
                                                 temporal_offset)
        return conv_state, temporal_state, layout_info

    # Non-contiguous layout: one int8 FTensor per layer, each laid out as
    # [slot][cell].  We build per-layer (num_slots, *shape) views — no single
    # tensor can span layers because each layer has its own VM reservation.
    def _view_state_per_layer(layer_ftensor: torch.Tensor,
                              shape: Tuple[int, ...], dtype: torch.dtype,
                              offset_bytes: int) -> torch.Tensor:
        dtype_size = dtype.itemsize
        assert offset_bytes % dtype_size == 0
        assert cell_size % dtype_size == 0

        inner_stride: List[int] = []
        running = 1
        for d in reversed(shape):
            inner_stride.insert(0, running)
            running *= d

        size = (num_slots, *shape)
        strides = [
            cell_size // dtype_size,
            *inner_stride,
        ]
        return torch.as_strided(
            layer_ftensor.view(dtype=dtype),
            size=size,
            stride=strides,
            storage_offset=offset_bytes // dtype_size,
        )

    # conv_state_per_layer[shape_idx][layer_idx] -> (num_slots, *shape)
    conv_state_per_layer: List[List[torch.Tensor]] = [
        [
            _view_state_per_layer(raw_tensors[layer_idx], shape, conv_dtype,
                                  conv_offsets[shape_idx])
            for layer_idx in range(num_mamba_layers)
        ] for shape_idx, shape in enumerate(conv_shapes)
    ]
    # temporal_state_per_layer[layer_idx] -> (num_slots, *temporal_shape)
    temporal_state_per_layer: List[torch.Tensor] = [
        _view_state_per_layer(raw_tensors[layer_idx], temporal_shape,
                              ssm_dtype, temporal_offset)
        for layer_idx in range(num_mamba_layers)
    ]
    return conv_state_per_layer, temporal_state_per_layer, layout_info


def get_kv_cache_manager(
    num_blocks: int,
    block_size: int,
    cell_size: int,
    num_layers: int,
    reserve_null_block: bool = True,
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
        world_size=_world_size,
        pp_rank=_pp_rank,
        async_sched=_async_sched,
        reserve_null_block=reserve_null_block,
        num_kv_buffers=num_kv_buffers,
        group_id=group_id,
    )
