# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""SGLang HiCache host copies over kvcached's elastic pools.

kvcached's default (contiguous) layout interleaves all layers (and, for MHA,
K with V) in one buffer: a layer's MHA (or MLA) view has a token stride that
spans every layer, and a Mamba state view has a slot stride that spans every
layer's states. HiCache's all-layer backups address the device through
per-layer pointer tables with dense rows, and its Mamba copies assume dense
slots, so they would copy the wrong bytes. Its one-layer JIT kernel reads each
view's own stride instead, and the MHA and MLA loads already use it. The host
pools below route the remaining copies through that kernel: MHA and MLA
backups one layer at a time, Mamba states as rows of kvcached's flat state
buffer.

With KVCACHED_CONTIGUOUS_LAYOUT=false the MHA and MLA views are dense and keep
SGLang's copies. Each Mamba layer then has a buffer of its own, while the Mamba
host pool sizes itself from one tensor of all layers, so it is shown the layers
stacked, and the states are copied as rows one layer at a time.
"""

import functools
import math
import types
from typing import Any, List, NamedTuple, Optional, Tuple

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.version_utils import VersionAwarePatch, version_range
from kvcached.utils import KVCachedConfigError, get_kvcached_logger

logger = get_kvcached_logger()

HICACHE_RANGE = ">=0.5.18,<0.5.21.dev0"
_REMEDY = "or disable --enable-hierarchical-cache or kvcached (ENABLE_KVCACHED=false)"


def _interleaved_mha(device_pool: Any) -> bool:
    # Only kvcached's elastic MHA pools define the check.
    check = getattr(device_pool, "_kv_buffers_independently_contiguous", None)
    return check is not None and not check()


def _interleaved_mla(device_pool: Any) -> bool:
    # kvcached's elastic MLA pools carry their allocator; their per-layer views
    # are interleaved but still publish data_ptrs, so a dense copy would not
    # fail, it would copy other layers' rows.
    return (getattr(device_pool, "kvcached_allocator", None) is not None
            and not all(t.is_contiguous() for t in device_pool.kv_buffer))


def _elastic_mamba(device_pool: Any) -> bool:
    return getattr(device_pool, "_kvcached_layout", None) is not None


def _indices_like(indices: Any, like: Any) -> Any:
    # The kernel reads both index tensors on the GPU with one dtype; HiCache
    # can keep host indices on the CPU.
    return indices.to(device=like.device, dtype=like.dtype, non_blocking=True)


def _require_jit(element_size: int, what: str) -> None:
    from sglang.kernels.ops.kvcache.hicache import can_use_hicache_jit_kernel

    if not can_use_hicache_jit_kernel(element_size=element_size):
        raise KVCachedConfigError(
            f"kvcached HiCache copies need SGLang's JIT HiCache kernel, which "
            f"does not support {what} rows of {element_size} bytes; {_REMEDY}")


def check_kv_host_pool(host: Any, device_pool: Any, backup_marker: str) -> None:
    """Reject host KV pools whose copies would assume dense device rows."""
    if host.layout not in ("layer_first", "page_first"):
        raise KVCachedConfigError(
            "kvcached supports HiCache with --hicache-mem-layout page_first or "
            f"layer_first, not {host.layout}; {_REMEDY}")
    if not getattr(type(host).backup_from_device_all_layer, backup_marker, False):
        raise KVCachedConfigError(
            f"kvcached does not support HiCache with {type(host).__name__}; {_REMEDY}")
    sharded = getattr(host, "_is_device_layer_sharded", None)
    if (getattr(host, "mtp_draft_device_pools", ()) or getattr(host, "layer_shard_enabled", False)
            or (sharded is not None and sharded(device_pool))
            or getattr(host, "dcp_size", 1) != 1):
        raise KVCachedConfigError(
            "kvcached does not support HiCache with MTP draft pools, layer-sharded "
            f"host pools or decode context parallelism; {_REMEDY}")
    if not getattr(host, "can_use_jit", False):
        # The loads would fall back to dense-row kernels too.
        raise KVCachedConfigError(
            "kvcached HiCache copies need SGLang's JIT HiCache kernel, which this "
            f"KV row size cannot use; {_REMEDY}")


def backup_mha(host: Any, device_pool: Any, host_indices: Any, device_indices: Any) -> None:
    from sglang.kernels.ops.kvcache.hicache import transfer_hicache_one_layer

    if device_indices.numel() == 0:
        return
    dst_indices = _indices_like(host_indices, device_indices)
    for layer in range(host.layer_num):
        transfer_hicache_one_layer(
            k_cache_dst=host.k_data_refs[layer],
            v_cache_dst=host.v_data_refs[layer],
            indices_dst=dst_indices,
            k_cache_src=device_pool.k_buffer[layer],
            v_cache_src=device_pool.v_buffer[layer],
            indices_src=device_indices,
            element_dim=host.element_dim,
        )


def backup_mla(host: Any, device_pool: Any, host_indices: Any, device_indices: Any) -> None:
    from sglang.kernels.ops.kvcache.hicache import transfer_hicache_one_layer_mla

    if device_indices.numel() == 0:
        return
    dst_indices = _indices_like(host_indices, device_indices)
    for layer in range(host.layer_num):
        transfer_hicache_one_layer_mla(
            cache_dst=host.data_refs[layer],
            indices_dst=dst_indices,
            cache_src=device_pool.kv_buffer[layer],
            indices_src=device_indices,
            element_dim=host.kv_cache_dim,
        )


# A Mamba state is up to megabytes per slot, and the JIT kernel is compiled per
# row size, so state copies move rows of at most this many bytes.
_MAMBA_ROW_BYTES = 4096


class _MambaRows(NamedTuple):
    """One state kind of layers ``first_layer`` onward as rows of a dense 2-D
    view on each side.

    The device state (layers, slots, *shape) is a strided view of kvcached's
    flat buffer (one layer's buffer in the per-layer layout), and the host
    state (slots, host_layers, 1, *shape) is dense. Rows of a size that
    divides every offset and stride tile both buffers, so a (slot, layer)
    state is ``per_state`` consecutive rows on either side.
    """

    device: Any
    host: Any
    elements: int  # per row
    per_state: int
    device_base: int
    device_layer: int
    device_slot: int
    first_layer: int
    layers: int
    host_layers: int


class _LayerStack:
    """Per-layer (slots, *shape) states as the (layers, slots, *shape) tensor
    that MambaPoolHost.__init__ reads its shape, dtype and layers from."""

    def __init__(self, layers: List[Any]) -> None:
        import torch

        self._layers = layers
        self.shape = torch.Size((len(layers), *layers[0].shape))
        self.dtype = layers[0].dtype

    def __getitem__(self, layer: int) -> Any:
        return self._layers[layer]


class _StackedMambaPool:
    """A per-layer elastic Mamba pool, seen by MambaPoolHost.__init__."""

    def __init__(self, pool: Any) -> None:
        cache = pool.mamba_cache
        self._pool = pool
        self.mamba_cache = types.SimpleNamespace(
            conv=[_LayerStack(layers) for layers in cache.conv_per_layer],
            temporal=_LayerStack(cache.temporal_per_layer))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)


def _mamba_rows(host_buffer: Any, state: Any, elements: int, first_layer: int,
                host_layers: int) -> _MambaRows:
    import torch

    dtype_size = state.element_size()
    state_bytes = elements * dtype_size
    offset, layer_stride, slot_stride = (
        x * dtype_size for x in (state.storage_offset(), state.stride(0), state.stride(1)))
    common = math.gcd(state_bytes, offset, layer_stride, slot_stride)
    row = 1
    while row < _MAMBA_ROW_BYTES and common % (row * 2) == 0:
        row *= 2
    if not state[0, 0].is_contiguous() or row % 128:
        raise KVCachedConfigError(
            f"kvcached cannot copy Mamba states with {state_bytes}-byte slots at "
            f"offset {offset} for HiCache; {_REMEDY}")
    _require_jit(row, "Mamba state")
    row_elements = row // dtype_size
    device = torch.as_strided(
        state, (state.untyped_storage().nbytes() // row, row_elements), (row_elements, 1), 0)
    return _MambaRows(device=device, host=host_buffer.view(-1, row_elements),
                      elements=row_elements, per_state=state_bytes // row,
                      device_base=offset // row, device_layer=layer_stride // row,
                      device_slot=slot_stride // row, first_layer=first_layer,
                      layers=state.shape[0], host_layers=host_layers)


def _mamba_row_plans(host: Any) -> List[_MambaRows]:
    cache = host.device_pool.mamba_cache
    if host.device_pool._is_contiguous:
        temporal, conv = [cache.temporal], [[state] for state in cache.conv]
    else:
        # Each layer's states are in a buffer of their own: one plan per layer.
        temporal = [state.unsqueeze(0) for state in cache.temporal_per_layer]
        conv = [[state.unsqueeze(0) for state in states] for states in cache.conv_per_layer]
    plans = []

    def add(host_buffer: Any, states: List[Any], elements: int) -> None:
        first = 0
        for state in states:
            plans.append(_mamba_rows(host_buffer, state, elements, first,
                                     host.num_mamba_layers))
            first += state.shape[0]

    if host.temporal_state_elem_size > 0:
        add(host.temporal_buffer, temporal, host.temporal_state_elem_size)
    for index, elements in enumerate(host.conv_state_elem_sizes):
        add(host.conv_buffer[index], conv[index], elements)
    return plans


def check_mamba_host_pool(host: Any) -> None:
    host._kvcached_mamba_rows = _mamba_row_plans(host)


def _row_indices(plan: _MambaRows, slots: Any, host_slots: Any, layers: Any) -> Tuple[Any, Any]:
    import torch

    within = torch.arange(plan.per_state, device=slots.device)
    device = (plan.device_base + slots[:, None, None] * plan.device_slot
              + (layers[None, :, None] - plan.first_layer) * plan.device_layer + within)
    host = ((host_slots[:, None, None] * plan.host_layers + layers[None, :, None])
            * plan.per_state + within)
    return device.reshape(-1), host.reshape(-1)


def copy_mamba(host: Any, host_indices: Any, device_indices: Any, layer: Optional[int],
               to_host: bool) -> None:
    """Copy the states of ``layer`` (every layer if None) between the pools."""
    import torch
    from sglang.kernels.ops.kvcache.hicache import transfer_hicache_one_layer_mla

    if device_indices.numel() == 0:
        return
    slots = device_indices.to(torch.int64)
    host_slots = host_indices.to(device=slots.device, dtype=torch.int64, non_blocking=True)
    for plan in host._kvcached_mamba_rows:
        if layer is None:
            layers = torch.arange(plan.first_layer, plan.first_layer + plan.layers,
                                  device=slots.device)
        elif plan.first_layer <= layer < plan.first_layer + plan.layers:
            layers = torch.tensor([layer], device=slots.device)
        else:
            continue
        device_rows, host_rows = _row_indices(plan, slots, host_slots, layers)
        if to_host:
            transfer_hicache_one_layer_mla(
                cache_dst=plan.host, indices_dst=host_rows,
                cache_src=plan.device, indices_src=device_rows, element_dim=plan.elements)
        else:
            transfer_hicache_one_layer_mla(
                cache_dst=plan.device, indices_dst=device_rows,
                cache_src=plan.host, indices_src=host_rows, element_dim=plan.elements)


def device_pool_bytes(pool: Any) -> float:
    """A device pool's size for splitting --hicache-size between pools."""
    size = pool.get_kv_size_bytes()
    total = sum(size) if isinstance(size, tuple) else size
    buffers = getattr(pool, "k_buffer", None) or getattr(pool, "kv_buffer", None)
    if getattr(pool, "kvcached_allocator", None) is None or not buffers:
        return total
    # Elastic buffers span kvcached's whole virtual reservation, about a GPU
    # per pool, which would starve the other pools (e.g. Mamba states) of
    # host memory. Count the pool's logical capacity instead.
    return total * min(1.0, (pool.size + pool.page_size) / buffers[0].shape[0])


class HiCacheSizeSplitPatch(VersionAwarePatch, BasePatch):
    """Split --hicache-size by the pools' logical sizes, not their reservations."""

    library = "sglang"
    target_module = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler"
    patch_name = "hicache_size_split"

    def apply(self, target_module: types.ModuleType) -> bool:
        if not self.initialize_version_info():
            return False
        return self.patch_split(target_module)

    @version_range(HICACHE_RANGE)
    def patch_split(self, target_module: types.ModuleType) -> bool:
        original = getattr(target_module, "_split_hicache_size", None)
        if original is None:
            return False
        if self._is_already_patched(original):
            return True

        @functools.wraps(original)
        def _split_hicache_size(hicache_size: int, kv_pools: Tuple[Any, ...]) -> Tuple[float, ...]:
            if not enable_kvcached():
                return original(hicache_size, kv_pools)
            sizes = [device_pool_bytes(pool) for pool in kv_pools]
            return tuple(hicache_size * size / sum(sizes) for size in sizes)

        self._mark_as_patched(_split_hicache_size)
        setattr(target_module, "_split_hicache_size", _split_hicache_size)
        return True


class _HostKVBackupPatch(VersionAwarePatch, BasePatch):
    """Back up interleaved elastic KV one layer at a time."""

    library = "sglang"
    backup_marker = "__kvcached_hicache_kv_backup__"

    @staticmethod
    def interleaved(device_pool: Any) -> bool:
        raise NotImplementedError

    @staticmethod
    def backup(host: Any, device_pool: Any, host_indices: Any, device_indices: Any) -> None:
        raise NotImplementedError

    def apply(self, target_module: types.ModuleType) -> bool:
        if not self.initialize_version_info():
            return False
        return self.patch_host_pool(target_module)

    @version_range(HICACHE_RANGE)
    def patch_host_pool(self, target_module: types.ModuleType) -> bool:
        host_cls = self._get_target_class(target_module)
        if host_cls is None:
            return False
        original_backup = host_cls.backup_from_device_all_layer
        if self._is_already_patched(original_backup, self.backup_marker):
            return True
        original_init = host_cls.__init__
        backup_marker, interleaved, backup = self.backup_marker, self.interleaved, self.backup

        @functools.wraps(original_init)
        def _patched_init(host: Any, device_pool: Any, *args: Any, **kwargs: Any) -> None:
            original_init(host, device_pool, *args, **kwargs)
            if enable_kvcached() and interleaved(device_pool):
                check_kv_host_pool(host, device_pool, backup_marker)

        @functools.wraps(original_backup)
        def _patched_backup(host: Any, device_pool: Any, host_indices: Any,
                            device_indices: Any, *args: Any, **kwargs: Any) -> Any:
            if enable_kvcached() and interleaved(device_pool):
                return backup(host, device_pool, host_indices, device_indices)
            return original_backup(host, device_pool, host_indices, device_indices,
                                   *args, **kwargs)

        self._mark_as_patched(_patched_backup, self.backup_marker)
        host_cls.__init__ = _patched_init
        host_cls.backup_from_device_all_layer = _patched_backup
        return True


class HiCacheKVHostPoolPatch(_HostKVBackupPatch):
    target_module = "sglang.srt.mem_cache.pool_host.mha"
    target_class = "MHATokenToKVPoolHost"
    patch_name = "hicache_kv_host_pool"
    interleaved = staticmethod(_interleaved_mha)
    backup = staticmethod(backup_mha)


class HiCacheMLAHostPoolPatch(_HostKVBackupPatch):
    target_module = "sglang.srt.mem_cache.pool_host.mla"
    target_class = "MLATokenToKVPoolHost"
    patch_name = "hicache_mla_host_pool"
    interleaved = staticmethod(_interleaved_mla)
    backup = staticmethod(backup_mla)


class HiCacheMambaHostPoolPatch(VersionAwarePatch, BasePatch):
    """Copy elastic Mamba states as rows of kvcached's flat state buffer."""

    library = "sglang"
    target_module = "sglang.srt.mem_cache.pool_host.mamba"
    target_class = "MambaPoolHost"
    patch_name = "hicache_mamba_host_pool"

    def apply(self, target_module: types.ModuleType) -> bool:
        if not self.initialize_version_info():
            return False
        return self.patch_host_pool(target_module)

    @version_range(HICACHE_RANGE)
    def patch_host_pool(self, target_module: types.ModuleType) -> bool:
        host_cls = self._get_target_class(target_module)
        if host_cls is None:
            return False
        original_backup = host_cls.backup_from_device_all_layer
        if self._is_already_patched(original_backup):
            return True
        original_init = host_cls.__init__
        original_load = host_cls.load_to_device_per_layer

        @functools.wraps(original_init)
        def _patched_init(host: Any, device_pool: Any, *args: Any, **kwargs: Any) -> None:
            if not (enable_kvcached() and _elastic_mamba(device_pool)):
                original_init(host, device_pool, *args, **kwargs)
                return
            original_init(host, device_pool if device_pool._is_contiguous
                          else _StackedMambaPool(device_pool), *args, **kwargs)
            host.device_pool = device_pool
            check_mamba_host_pool(host)

        @functools.wraps(original_backup)
        def _patched_backup(host: Any, device_pool: Any, host_indices: Any,
                            device_indices: Any, *args: Any, **kwargs: Any) -> Any:
            if not (enable_kvcached() and _elastic_mamba(device_pool)):
                return original_backup(host, device_pool, host_indices, device_indices,
                                       *args, **kwargs)
            copy_mamba(host, host_indices, device_indices, None, to_host=True)
            return None

        @functools.wraps(original_load)
        def _patched_load(host: Any, device_pool: Any, host_indices: Any,
                          device_indices: Any, layer_id: int, *args: Any,
                          **kwargs: Any) -> Any:
            if not (enable_kvcached() and _elastic_mamba(device_pool)):
                return original_load(host, device_pool, host_indices, device_indices,
                                     layer_id, *args, **kwargs)
            copy_mamba(host, host_indices, device_indices, layer_id, to_host=False)
            return None

        self._mark_as_patched(_patched_backup)
        host_cls.__init__ = _patched_init
        host_cls.backup_from_device_all_layer = _patched_backup
        host_cls.load_to_device_per_layer = _patched_load
        return True
