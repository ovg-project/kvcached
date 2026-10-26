# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Configure rank-private native CPU offload before worker KV initialization."""

from __future__ import annotations

from functools import wraps
from math import isfinite
from typing import Any

from kvcached.integration.patch_base import enable_kvcached
from kvcached.utils import KVCachedConfigError

SPEC_NAME = "KVCachedCPUOffloadingSpec"
SPEC_MODULE = "kvcached.integration.vllm.offload_spec"
PLAN_KEY = "kvcached_cpu_offload"


def positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise KVCachedConfigError(f"kvcached offloading requires positive integer {name}")
    return value


def declared_payload(config: Any) -> int:
    """Match native unique canonical tensor payload, without VMM storage sizes.

    Descriptors share one backing, so summing their size double counts it.
    Native registration deduplicates equal-address, equal-stride views; packed
    attention is represented by one complete packed block.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec

    blocks = positive_int(config.num_blocks, "GPU block count")
    if not config.kv_cache_tensors:
        raise KVCachedConfigError("kvcached CPU offloading does not support a rank without KV tensors")
    specs = {}
    for group in config.kv_cache_groups:
        per_layer = getattr(group.kv_cache_spec, "kv_cache_specs", {})
        for name in group.layer_names:
            specs[name] = per_layer.get(name, group.kv_cache_spec)
    sizes = {tensor.size for tensor in config.kv_cache_tensors}
    if len(sizes) != 1:
        raise KVCachedConfigError("kvcached offloading requires one declared KV backing per rank")
    backing = positive_int(sizes.pop(), "declared KV backing size")
    views: dict[tuple[int, int], int] = {}
    packed_strides = set()
    strided_states = False
    owners = set()
    for tensor in config.kv_cache_tensors:
        stride = positive_int(tensor.block_stride, "KV block stride")
        layer_stride = positive_int(tensor.layer_stride, "KV layer stride")
        for index, name in enumerate(tensor.layers):
            if name not in specs or name in owners:
                raise KVCachedConfigError(f"Invalid or duplicate KV descriptor owner: {name}")
            owners.add(name)
            page = positive_int(specs[name].page_size_bytes, "KV page size")
            offset = tensor.offset + index * layer_stride
            if offset < 0 or stride < page or offset + (blocks - 1) * stride + page > backing:
                raise KVCachedConfigError(f"KV descriptor exceeds its logical backing: {name}")
            if isinstance(specs[name], AttentionSpec) and stride != page:
                packed_strides.add(stride)
            elif not isinstance(specs[name], AttentionSpec) and stride != page:
                strided_states = True
            native_stride = stride if isinstance(specs[name], AttentionSpec) else page
            key = (offset, native_stride)
            if key in views and views[key] != page:
                raise KVCachedConfigError("Aliased KV views disagree on their page size")
            views[key] = page
    if not views:
        raise KVCachedConfigError("kvcached CPU offloading does not support a rank without KV tensors")
    if packed_strides:
        if len(packed_strides) != 1:
            raise KVCachedConfigError("Packed KV views disagree on their block stride")
        payload = packed_strides.pop()
        if payload * blocks != backing:
            raise KVCachedConfigError("Packed KV stride disagrees with the logical backing")
        return payload
    if strided_states:
        # Native registration hardcodes page-contiguous views for state layers;
        # only its packed-attention branch replaces these with a packed view.
        raise KVCachedConfigError(
            "Native CPU offloading needs page-contiguous state-only ranks; "
            "use a layer-compact KV layout for this pipeline partition")
    return sum(views.values())


def build_offload_plan(vllm_config: Any, configs: list[Any]) -> dict[str, Any]:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        build_offloading_config,
    )

    parallel = vllm_config.parallel_config
    workers = positive_int(parallel.tensor_parallel_size, "TP") * positive_int(
        parallel.pipeline_parallel_size, "PP")
    if len(configs) != workers:
        raise KVCachedConfigError("KV cache config count differs from TP * PP")
    extra = vllm_config.kv_transfer_config.kv_connector_extra_config
    if extra.get("spec_name", "CPUOffloadingSpec") not in ("CPUOffloadingSpec", SPEC_NAME):
        raise KVCachedConfigError("kvcached offloading requires the native CPU offloading spec")
    if extra.get("canonical_layout", False):
        raise KVCachedConfigError("kvcached rank-private offloading does not support canonical_layout")
    # The CLI converts a float GiB value to bytes without changing its type.
    # Native CPUOffloadingSpec truncates fractional bytes before budgeting.
    budget = extra.get("cpu_bytes_to_use")
    if isinstance(budget, float) and isfinite(budget):
        budget = int(budget)
    budget = positive_int(budget, "cpu_bytes_to_use")
    payloads = [declared_payload(config) for config in configs]
    if len({config.num_blocks for config in configs}) != 1:
        raise KVCachedConfigError("Logical GPU block counts differ between ranks")
    chunk_sizes = {build_offloading_config(vllm_config, config).cache.blocks_per_chunk
                   for config in configs}
    if len(chunk_sizes) != 1:
        raise KVCachedConfigError("Offload chunk granularity differs between ranks")
    blocks_per_chunk = positive_int(chunk_sizes.pop(), "blocks_per_chunk")
    groups = [[bool(group.layer_names) for group in config.kv_cache_groups]
              for config in configs]
    if len({len(group) for group in groups}) != 1:
        raise KVCachedConfigError("KV cache group indices differ between ranks")
    num_chunks = budget // (blocks_per_chunk * sum(payloads))
    if num_chunks == 0:
        raise KVCachedConfigError("CPU offload budget cannot hold one chunk across all ranks")
    return dict(num_chunks=num_chunks, blocks_per_chunk=blocks_per_chunk,
                payloads=payloads, groups=groups,
                gpu_blocks=[config.num_blocks for config in configs])


def _set_offload_plan(vllm_config: Any, plan: dict[str, Any]) -> None:
    transfer = vllm_config.kv_transfer_config
    transfer.kv_connector_extra_config = dict(
        transfer.kv_connector_extra_config,
        spec_name=SPEC_NAME, spec_module_path=SPEC_MODULE,
        **{PLAN_KEY: plan},
    )


def _configure_offload_worker(worker: Any, plan: dict[str, Any]) -> int:
    """Runs on every existing worker before its connector/spec is created."""
    config = worker.vllm_config
    rank = config.parallel_config.rank
    if not isinstance(rank, int) or not 0 <= rank < len(plan["payloads"]):
        raise KVCachedConfigError("Invalid worker rank for CPU offload capacity plan")
    _set_offload_plan(config, plan)
    return rank


def patch_offload_initialization(engine_cls: Any) -> None:
    """Intercept the native KV initialization call, retaining native setup."""
    original = getattr(engine_cls, "_initialize_kv_caches", None)
    if original is None or getattr(original, "_kvcached_offload_init", False):
        return

    @wraps(original)
    def prepare(engine: Any, config: Any, *args: Any, **kwargs: Any) -> Any:
        transfer = getattr(config, "kv_transfer_config", None)
        if (not enable_kvcached() or transfer is None
                or getattr(transfer, "kv_connector", None) != "OffloadingConnector"):
            return original(engine, config, *args, **kwargs)
        executor = engine.model_executor
        initialize = executor.initialize_from_config
        had_override = "initialize_from_config" in vars(executor)
        previous_override = vars(executor).get("initialize_from_config")
        previous_extra = transfer.kv_connector_extra_config

        def initialize_with_plan(configs: list[Any]) -> Any:
            plan = build_offload_plan(config, configs)
            workers = len(plan["payloads"])
            if workers > 1 and getattr(executor, "world_size", None) != workers:
                raise KVCachedConfigError("Executor world size differs from CPU offload plan")
            _set_offload_plan(config, plan)
            ranks = executor.collective_rpc(_configure_offload_worker, args=(plan,))
            if sorted(ranks) != list(range(workers)):
                raise KVCachedConfigError("CPU offload capacity plan did not reach every worker")
            return initialize(configs)

        executor.initialize_from_config = initialize_with_plan
        try:
            return original(engine, config, *args, **kwargs)
        except Exception:
            transfer.kv_connector_extra_config = previous_extra
            raise
        finally:
            if had_override:
                executor.initialize_from_config = previous_override
            else:
                del executor.initialize_from_config

    prepare._kvcached_offload_init = True  # type: ignore[attr-defined]
    engine_cls._initialize_kv_caches = prepare
