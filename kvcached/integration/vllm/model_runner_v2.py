# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.29 raw KV allocation, retaining native view and binding semantics."""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import partial, wraps
from math import prod
from typing import Any

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.kv_geometry import (
    check_page_geometry,
    has_zero_capacity_pages,
    select_page_size,
    whole_blocks_in_pages,
)
from kvcached.utils import (
    CONTIGUOUS_LAYOUT,
    PAGE_SIZE,
    KVCachedConfigError,
    get_page_size_for_block,
)

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
    physical_page_size: int
    has_qsa_ring: bool = False

    @property
    def cell_size(self) -> int:
        return self.page_bytes // self.block_size

    def usable_blocks(self, backing_blocks: int) -> int:
        pages = backing_blocks * self.page_bytes // self.physical_page_size
        return whole_blocks_in_pages(pages, self.page_bytes, self.physical_page_size)


def cache_geometry(config: Any) -> CacheGeometry:
    """Derive the same physical block units in coordinator and worker."""
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
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
    token_attention_specs = (FullAttentionSpec, SlidingWindowSpec, MLAAttentionSpec)
    # vLLM allocates a single raw-key ring block per QSA request.
    # Its block_size is the ring capacity, independent of token block size.
    supported_specs = (*token_attention_specs, MambaSpec, CircularBufferSpec)
    if not specs or any(type(spec) not in supported_specs for spec in specs):
        raise KVCachedConfigError(
            "kvcached MRV2 supports FullAttentionSpec, SlidingWindowSpec, "
            "MLAAttentionSpec, MambaSpec and CircularBufferSpec only"
        )
    first_attention = next((spec for spec in specs if type(spec) in token_attention_specs), None)
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
    page_size = get_page_size_for_block(page_bytes, PAGE_SIZE)
    # QSA packs the compressed index beside main K/V. Only the final
    # descriptors give the size of a block in the elastic pool.
    has_qsa_ring = CircularBufferSpec in {type(spec) for spec in specs}
    if (has_qsa_ring
            and os.getenv("KVCACHED_PAGE_SIZE_MB") is None
            and has_zero_capacity_pages(page_bytes, page_size)):
        page_size = select_page_size(page_bytes)
    geometry_error = check_page_geometry(page_bytes, page_size, block_size)
    if geometry_error is not None:
        raise KVCachedConfigError(geometry_error)
    return CacheGeometry(block_size, page_bytes, num_pools, page_size, has_qsa_ring)


def create_cache_manager(num_blocks, geometry):
    from kvcached.integration.vllm.interfaces import _create_cache_manager

    return _create_cache_manager(num_blocks, geometry.block_size, geometry.cell_size,
                                 geometry.num_pools, 1, 0, "block_pool", geometry.physical_page_size)


def patch_qsa_watermark(manager_cls) -> None:
    from inspect import signature

    original = manager_cls.__init__
    parameters = signature(original)
    if "watermark" not in parameters.parameters:
        return

    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        capacity = getattr(self.block_pool, "_qsa_whole_blocks", None)
        if capacity is not None and self.watermark_blocks:
            watermark = parameters.bind(self, *args, **kwargs).arguments["watermark"]
            self.watermark_blocks = int(watermark * capacity)

    manager_cls.__init__ = initialize


def validate_qsa_cache_plan(vllm_config, configs) -> None:
    from vllm.v1.core import kv_cache_utils

    # IPC translates physical strides per worker. A shared block ID must
    # still belong to the same logical page on every pipeline stage.
    geometries = [cache_geometry(config) for config in configs]
    mappings = {(g.block_size, Fraction(g.page_bytes, g.physical_page_size)) for g in geometries}
    if len(mappings) > 1:
        raise KVCachedConfigError("QSA pipeline stages have different block-to-page mappings")

    # Leave the worker descriptors at their full extent so that high IDs
    # after a page boundary still address the original native views.
    groups = [config.kv_cache_groups for config in configs]
    budgets = [
        (geometry.usable_blocks(config.num_blocks) - 1) * geometry.page_bytes * geometry.num_pools
        for geometry, config in zip(geometries, configs)
    ]
    if min(budgets) <= 0:
        raise KVCachedConfigError("QSA pool needs a whole block in addition to its reserved null block")
    if vllm_config.model_config.original_max_model_len == -1:
        kv_cache_utils._auto_fit_max_model_len(vllm_config, groups, budgets)
    else:
        for worker_groups, budget in zip(groups, budgets):
            kv_cache_utils._check_enough_kv_cache_memory(
                budget, partial(kv_cache_utils._max_memory_usage_bytes_from_groups, vllm_config, worker_groups),
                vllm_config.model_config.max_model_len,
                partial(kv_cache_utils._estimate_max_model_len_from_groups, vllm_config, worker_groups),
            )


def allocate_kv_cache(config: Any, device: Any, layout: Any, kernel_block_sizes=None):
    """Reserve raw native storage and let vLLM construct its configured views."""
    import torch
    from vllm.v1.kv_cache_interface import (
        UniformTypeKVCacheSpecs,
        compute_layer_kv_cache_shape_bytes,
        create_kv_cache_views,
    )

    from kvcached.integration.vllm import interfaces as kvi

    if not kvi._kvcached_initialized:
        raise RuntimeError("kvcached MRV2 worker was not initialized")
    geometry = cache_geometry(config)
    allowed = ("BLNHC", "BLHNC") if CONTIGUOUS_LAYOUT else ("LBNHC", "LBHNC")
    if layout.name not in allowed:
        raise KVCachedConfigError(f"KV layout {layout.name} is incompatible with kvcached {allowed}")
    placements = []
    for tensor in config.kv_cache_tensors:
        group_id, group = next(
            (index, group)
            for index, group in enumerate(config.kv_cache_groups)
            if tensor.layers[0] in group.layer_names
        )
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[tensor.layers[0]]
        if getattr(tensor, "host_resident", False):
            raise KVCachedConfigError("kvcached does not support host-resident KV tensors")
        kernel_size = kernel_block_sizes[group_id] if kernel_block_sizes is not None else None
        if getattr(spec, "storage_block_size", None) is not None:
            # vLLM 0.30 MLA specs can fix the stored kernel block size.
            kernel_size = spec.storage_block_size
        # Match create_kv_cache_views: splitting requires dense, unpadded
        # manager blocks. page_size_bytes includes padding and is not the
        # dense size. Layer-compact storage removes interleaving, not padding.
        shape_bytes = compute_layer_kv_cache_shape_bytes(spec, 1, kernel_size)
        dense_page_size = prod(compute_layer_kv_cache_shape_bytes(spec, 1)[1:])
        block_stride = tensor.block_stride if CONTIGUOUS_LAYOUT else geometry.page_bytes
        if shape_bytes[0] > 1 and block_stride != dense_page_size:
            layout_advice = (
                "For uniform, unpadded per-layer pages, "
                "KVCACHED_CONTIGUOUS_LAYOUT=false can remove layer interleaving. "
                if CONTIGUOUS_LAYOUT and spec.page_size_bytes == dense_page_size else
                "Changing the KV layout does not remove per-layer page padding. "
            )
            raise KVCachedConfigError(
                f"vLLM cannot split {spec.block_size}-token blocks into the "
                f"{kernel_size}-token kernel blocks this attention backend reads: "
                f"the {layout.name} block stride is {block_stride} bytes but a dense "
                f"page is {dense_page_size} bytes. {layout_advice}"
                f"Pick an attention backend that accepts {spec.block_size}-token "
                "blocks (e.g. --attention-backend TRITON_ATTN)."
            )
        placements.append((tensor, spec, kernel_size))
    per_pool_bytes = torch.cuda.get_device_properties(device).total_memory // geometry.num_pools
    page_size = geometry.physical_page_size
    per_pool_bytes = per_pool_bytes // page_size * page_size
    if config.num_blocks * geometry.page_bytes > per_pool_bytes:
        raise KVCachedConfigError("Configured KV blocks exceed the native virtual reservation")
    raw = kvi.create_kv_tensors(
        per_pool_bytes,
        1,
        str(device),
        geometry.num_pools,
        num_kv_buffers=1,
        unified_pool=True,
        page_size=page_size,
    )

    caches: dict[str, Any] = {}
    for tensor, spec, kernel_size in placements:
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
