# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.29 raw KV allocation, retaining native view and binding semantics."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.utils import CONTIGUOUS_LAYOUT, PAGE_SIZE, KVCachedConfigError

_persistent_allocation: ContextVar[bool] = ContextVar("kvcached_mrv2_allocation", default=False)


class _OwnerSpecLookup(dict):
    """Resolve consumer indexing without including aliases in storage accounting."""

    def __init__(self, specs, shared_layers):
        super().__init__(specs)
        self._consumer_specs = {
            consumer: specs[owner]
            for consumer, owner in shared_layers.items()
            if owner in specs
        }

    def __missing__(self, name):
        return self._consumer_specs[name]


def _with_owner_spec_lookup(config, shared_layers):
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    groups = []
    changed = False
    for group in config.kv_cache_groups:
        spec = group.kv_cache_spec
        if (isinstance(spec, UniformTypeKVCacheSpecs)
                and any(owner in spec.kv_cache_specs for owner in shared_layers.values())):
            spec = replace(spec, kv_cache_specs=_OwnerSpecLookup(spec.kv_cache_specs, shared_layers))
            # Preserve native layer-list updates for subsequent binding, while
            # keeping the owner's original spec dictionary entirely untouched.
            group = replace(group, kv_cache_spec=spec)
            changed = True
        groups.append(group)
    return replace(config, kv_cache_groups=groups) if changed else config


@dataclass(frozen=True)
class CacheGeometry:
    block_size: int
    page_bytes: int
    num_pools: int

    @property
    def cell_size(self) -> int:
        return self.page_bytes // self.block_size


def cache_geometry(config: Any) -> CacheGeometry:
    """Derive the same physical block units in coordinator and worker."""
    from vllm.v1.kv_cache_interface import (
        AttentionSpec,
        FullAttentionSpec,
        MambaSpec,
        MLAAttentionSpec,
        SlidingWindowSpec,
        UniformTypeKVCacheSpecs,
    )

    specs = []
    for group in config.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            specs.extend(group.kv_cache_spec.kv_cache_specs.values())
        else:
            specs.append(group.kv_cache_spec)
    # Specialized subclasses can have different manager ownership semantics;
    # inheritance alone does not qualify them for the elastic block pool.
    supported_specs = (FullAttentionSpec, SlidingWindowSpec, MLAAttentionSpec, MambaSpec)
    if not specs or any(type(spec) not in supported_specs for spec in specs):
        raise KVCachedConfigError(
            "kvcached MRV2 supports FullAttentionSpec, SlidingWindowSpec, "
            "MLAAttentionSpec and MambaSpec only"
        )
    first_attention = next((spec for spec in specs if isinstance(spec, AttentionSpec)), None)
    if first_attention is None:
        raise KVCachedConfigError("kvcached MRV2 requires an attention group")
    block_size = first_attention.block_size
    # Attention discovery appends borrowers to groups without allocating storage.
    # Count only descriptor-backed owners, across all spec regions in each
    # group. Groups reuse the backing, so their counts must not be summed.
    owners = {layer for tensor in config.kv_cache_tensors for layer in tensor.layers}
    num_pools = max(len(owners.intersection(group.layer_names)) for group in config.kv_cache_groups)
    if CONTIGUOUS_LAYOUT:
        # The scheduler collapses UniformTypeKVCacheSpecs to a representative
        # layer, but retains the backing placement. Derive allocation units
        # from that shared placement so worker and scheduler cannot disagree.
        sizes = {tensor.size for tensor in config.kv_cache_tensors}
        if len(sizes) != 1 or config.num_blocks <= 0 or num_pools <= 0:
            raise KVCachedConfigError("KV packing requires one backing and positive block/pool counts")
        backing_size = sizes.pop()
        packed_bytes, remainder = divmod(backing_size, config.num_blocks)
        if remainder:
            raise KVCachedConfigError("KV backing size must divide exactly into configured blocks")
        page_bytes, remainder = divmod(packed_bytes, num_pools)
        if remainder or page_bytes <= 0:
            raise KVCachedConfigError("Packed KV block bytes must divide exactly into native pool units")
        for tensor in config.kv_cache_tensors:
            if (tensor.block_stride != packed_bytes or tensor.offset < 0
                    or tensor.layer_stride <= 0
                    or tensor.offset + len(tensor.layers) * tensor.layer_stride > packed_bytes):
                raise KVCachedConfigError("KV layer placement exceeds its packed block or uses a different stride")
    else:
        page_sizes = {spec.page_size_bytes for spec in specs}
        if len(page_sizes) != 1:
            raise KVCachedConfigError(
                "kvcached MRV2 noncontiguous pools require uniform pages: "
                f"per-layer page sizes={sorted(page_sizes)}"
            )
        page_bytes = page_sizes.pop()
        expected_size = config.num_blocks * num_pools * page_bytes
        if any(tensor.size != expected_size for tensor in config.kv_cache_tensors):
            raise KVCachedConfigError("KV backing size disagrees with the uniform physical pool geometry")
    if page_bytes % block_size:
        raise KVCachedConfigError("KV allocation unit bytes must divide exactly by the attention block size")
    if page_bytes > PAGE_SIZE:
        raise KVCachedConfigError(
            f"KV allocation unit ({page_bytes} bytes) exceeds the native page ({PAGE_SIZE} bytes)"
        )
    return CacheGeometry(block_size, page_bytes, num_pools)


def allocate_kv_cache(config: Any, device: Any, layout: Any, kernel_block_sizes=None):
    """Reserve raw native storage and let vLLM construct its configured views."""
    import torch
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs, create_kv_cache_views

    from kvcached.integration.vllm import interfaces as kvi

    if not kvi._kvcached_initialized:
        raise RuntimeError("kvcached MRV2 worker was not initialized")
    geometry = cache_geometry(config)
    allowed = ("BLNHC", "BLHNC") if CONTIGUOUS_LAYOUT else ("LBNHC", "LBHNC")
    if layout.name not in allowed:
        raise KVCachedConfigError(f"KV layout {layout.name} is incompatible with kvcached {allowed}")
    per_pool_bytes = torch.cuda.get_device_properties(device).total_memory // geometry.num_pools
    per_pool_bytes = per_pool_bytes // PAGE_SIZE * PAGE_SIZE
    if config.num_blocks * geometry.page_bytes > per_pool_bytes:
        raise KVCachedConfigError("Configured KV blocks exceed the native virtual reservation")
    raw = kvi.create_kv_tensors(
        per_pool_bytes,
        1,
        str(device),
        geometry.num_pools,
        num_kv_buffers=1,
        unified_pool=True,
    )

    caches: dict[str, Any] = {}
    for tensor in config.kv_cache_tensors:
        group_id, group = next(
            (index, group)
            for index, group in enumerate(config.kv_cache_groups)
            if tensor.layers[0] in group.layer_names
        )
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[tensor.layers[0]]
        kernel_size = kernel_block_sizes[group_id] if kernel_block_sizes is not None else None
        if CONTIGUOUS_LAYOUT:
            views = create_kv_cache_views(
                raw[0], spec, config.num_blocks, layout, tensor, kernel_size
            )
            caches.update(zip(tensor.layers, views))
        else:
            pool_span = config.num_blocks * geometry.page_bytes
            for layer_index, layer_name in enumerate(tensor.layers):
                offset = tensor.offset + layer_index * tensor.layer_stride
                pool_index, remainder = divmod(offset, pool_span)
                if remainder or pool_index >= geometry.num_pools:
                    raise KVCachedConfigError("KV layer placement does not align with a native pool")
                placement = replace(
                    tensor, layers=[layer_name], offset=0,
                    layer_stride=pool_span, block_stride=geometry.page_bytes,
                )
                caches[layer_name] = create_kv_cache_views(
                    raw[pool_index], spec, config.num_blocks, layout, placement, kernel_size
                )[0]
    kvi.logger.info(
        "kvcached MRV2 persistent allocation: pools=%d page_bytes=%d blocks=%d layout=%s",
        geometry.num_pools, geometry.page_bytes, config.num_blocks, layout.name,
    )
    return caches


class ModelRunnerV2Patch(BasePatch):
    library = "vllm"
    target_module = "vllm.v1.worker.gpu.model_runner"
    target_class = "GPUModelRunner"
    patch_name = "model_runner_v2"

    def apply(self, module) -> bool:
        from vllm.v1.worker.gpu import attn_utils

        from kvcached.integration.vllm.patches import _should_enable_async_sched

        runner = self._get_target_class(module)
        if runner is None:
            return False
        if self._is_already_patched(runner.initialize_kv_cache):
            return True
        original_init = runner.__init__
        original_initialize = runner.initialize_kv_cache
        original_allocate = attn_utils.allocate_kv_cache
        original_discovery = module.init_attn_backend

        @wraps(original_discovery)
        def discover_attention(kv_cache_config, vllm_config, *args, **kwargs):
            if enable_kvcached():
                shared_layers = attn_utils.get_shared_kv_cache_layers(vllm_config)
                if shared_layers:
                    kv_cache_config = _with_owner_spec_lookup(kv_cache_config, shared_layers)
            return original_discovery(kv_cache_config, vllm_config, *args, **kwargs)

        @wraps(original_init)
        def initialize_worker(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if enable_kvcached():
                from vllm.distributed.parallel_state import (
                    get_pp_group,
                    get_tensor_model_parallel_rank,
                    get_tensor_model_parallel_world_size,
                )

                from kvcached.integration.vllm import interfaces as kvi

                kvi.init_kvcached(
                    tp_rank=get_tensor_model_parallel_rank(),
                    world_size=get_tensor_model_parallel_world_size(),
                    pp_rank=get_pp_group().rank_in_group,
                    is_worker=True,
                    device=str(self.device),
                    async_sched=_should_enable_async_sched(self.vllm_config),
                )

        @wraps(original_initialize)
        def initialize_cache(self, kv_cache_config, is_profiling=False,
                             kv_cache_allocation_context=None):
            # Graph profiling creates a temporary cache and tears it down. It
            # must not register a process-lifetime native pool with that geometry.
            token = _persistent_allocation.set(enable_kvcached() and not is_profiling)
            try:
                return original_initialize(
                    self, kv_cache_config, is_profiling=is_profiling,
                    kv_cache_allocation_context=kv_cache_allocation_context,
                )
            finally:
                _persistent_allocation.reset(token)

        @wraps(original_allocate)
        def scoped_allocate(*args, **kwargs):
            if _persistent_allocation.get():
                return allocate_kv_cache(*args, **kwargs)
            return original_allocate(*args, **kwargs)

        self._mark_as_patched(initialize_cache)
        runner.__init__ = initialize_worker
        runner.initialize_kv_cache = initialize_cache
        module.init_attn_backend = discover_attention
        attn_utils.allocate_kv_cache = scoped_allocate
        return True


class KVLayoutV2Patch(BasePatch):
    library = "vllm"
    target_module = "vllm.v1.worker.gpu_worker"
    target_class = "Worker"
    patch_name = "kv_layout_v2"

    def apply(self, module) -> bool:
        worker = self._get_target_class(module)
        if worker is None:
            return False
        original = worker.get_supported_kv_cache_layouts
        if self._is_already_patched(original):
            return True

        @wraps(original)
        def supported_layouts(self):
            layouts = original(self)
            if not enable_kvcached():
                return layouts
            allowed = ("BLNHC", "BLHNC") if CONTIGUOUS_LAYOUT else ("LBNHC", "LBHNC")
            compatible = [layout for layout in layouts if layout in allowed]
            if not compatible:
                raise KVCachedConfigError("No backend KV layout matches the kvcached physical layout")
            return compatible

        self._mark_as_patched(supported_layouts)
        worker.get_supported_kv_cache_layouts = supported_layouts
        return True
