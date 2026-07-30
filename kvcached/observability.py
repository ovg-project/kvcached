# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Read-only observability snapshots for kvcached integrations.

The structures in this module are intentionally policy-free.  They expose
allocator and runtime state so monitoring systems or external control planes
can consume kvcached status without depending on private patch details.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from kvcached.pool_registry import get_registered_kv_cache_pools

SCHEMA_VERSION = "kvcached.observability.v1"


def _call_int(obj: Any, name: str) -> Optional[int]:
    method = getattr(obj, name, None)
    if method is None:
        return None
    return int(method())


def _int_attr(obj: Any, name: str) -> Optional[int]:
    value = getattr(obj, name, None)
    if value is None:
        return None
    return int(value)


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Runtime integration state for an engine shim."""

    schema_version: str
    engine: str
    initialized: bool
    device: Optional[str]
    world_size: int
    pp_rank: int
    async_sched: bool
    contiguous_layout: bool
    is_worker: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KVCachePoolSnapshot:
    """Read-only state of one kvcached-backed KV pool."""

    schema_version: str
    pool_type: str
    integration: Optional[str]
    pool_name: Optional[str]
    group_id: int
    num_layers: int
    num_kv_buffers: int
    page_size_bytes: int
    block_size_bytes: int
    total_blocks: int
    available_blocks: int
    allocated_blocks: int
    reserved_blocks: int
    available_bytes: int
    allocated_bytes: int
    reserved_bytes: int
    null_block_reserved: bool
    virtual_per_layer_bytes: int
    virtual_total_bytes: int
    mapped_bytes: int
    total_pages: Optional[int]
    free_pages: Optional[int]
    inuse_pages: Optional[int]
    reserved_pages: Optional[int]
    available_physical_pages: Optional[int]
    effective_free_pages: Optional[int]
    in_shrink: bool
    shrink_target_blocks: Optional[int]
    resize_target_bytes: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KVCachePoolOperationSnapshot:
    """Monotonic operation counters for one kvcached-backed KV pool."""

    schema_version: str
    pool_type: str
    integration: Optional[str]
    pool_name: Optional[str]
    group_id: int
    allocation_requests_total: int
    allocation_successes_total: int
    allocation_failures_total: int
    capacity_exhausted_total: int
    allocated_blocks_total: int
    free_requests_total: int
    free_successes_total: int
    free_failures_total: int
    freed_blocks_total: int
    physical_page_allocations_total: int
    physical_page_allocation_failures_total: int
    physical_page_frees_total: int
    resize_requests_total: int
    resize_successes_total: int
    resize_deferred_total: int
    resize_completions_total: int
    trim_requests_total: int
    trim_successes_total: int
    clear_requests_total: int
    clear_successes_total: int
    operation_errors_total: int
    post_init_errors_total: int
    allocation_errors_total: int
    free_errors_total: int
    resize_errors_total: int
    trim_errors_total: int
    clear_errors_total: int
    state_inconsistency_errors_total: int
    last_error_code: Optional[str]
    last_error_timestamp_ns: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_capabilities() -> Dict[str, Any]:
    """Return the stable observability surface currently exposed by kvcached."""

    return {
        "schema_version": SCHEMA_VERSION,
        "features": {
            "runtime_snapshot": True,
            "kv_cache_pool_snapshot": True,
            "kv_cache_pool_operation_snapshot": True,
            "registered_kv_cache_pool_snapshots": True,
            "registered_kv_cache_pool_operation_snapshots": True,
            "read_only": True,
            "policy_control": False,
        },
        "pool_snapshot_fields": list(KVCachePoolSnapshot.__dataclass_fields__.keys()),
        "pool_operation_snapshot_fields": list(
            KVCachePoolOperationSnapshot.__dataclass_fields__.keys()
        ),
        "runtime_snapshot_fields": list(RuntimeSnapshot.__dataclass_fields__.keys()),
    }


def build_runtime_snapshot(
    *,
    engine: str,
    initialized: bool,
    device: Optional[str],
    world_size: int,
    pp_rank: int,
    async_sched: bool,
    contiguous_layout: bool,
    is_worker: Optional[bool] = None,
) -> RuntimeSnapshot:
    return RuntimeSnapshot(
        schema_version=SCHEMA_VERSION,
        engine=engine,
        initialized=initialized,
        device=device,
        world_size=world_size,
        pp_rank=pp_rank,
        async_sched=async_sched,
        contiguous_layout=contiguous_layout,
        is_worker=is_worker,
    )


def build_kv_cache_pool_snapshot(
    manager: Any,
    *,
    integration: Optional[str] = None,
) -> KVCachePoolSnapshot:
    """Build a read-only snapshot from a ``KVCacheManager``-like object."""

    allocator = manager.page_allocator
    free_pages = _call_int(allocator, "get_num_free_pages")
    reserved_pages = _call_int(allocator, "get_num_reserved_pages")
    available_physical_pages = _call_int(allocator, "get_avail_physical_pages")
    if free_pages is None or reserved_pages is None or available_physical_pages is None:
        effective_free_pages = None
    else:
        effective_free_pages = min(free_pages, available_physical_pages + reserved_pages)

    mapped_bytes = int(manager.get_mapped_memory_size("bytes"))
    virtual_per_layer_bytes = _int_attr(manager, "mem_size") or 0
    num_layers = _int_attr(manager, "num_layers") or 0
    num_kv_buffers = _int_attr(manager, "num_kv_buffers") or 0
    block_size_bytes = _int_attr(manager, "block_mem_size") or 0
    bytes_per_block = block_size_bytes * num_layers * num_kv_buffers
    reserved_blocks = len(getattr(manager, "reserved_blocks", []))
    available_blocks = max(int(manager.available_size()) - reserved_blocks, 0)
    allocated_blocks = max(int(manager._get_num_alloced_blocks()) - reserved_blocks, 0)

    return KVCachePoolSnapshot(
        schema_version=SCHEMA_VERSION,
        pool_type="kv_cache",
        integration=integration,
        pool_name=getattr(manager, "pool_name", None),
        group_id=_int_attr(manager, "group_id") or 0,
        num_layers=num_layers,
        num_kv_buffers=num_kv_buffers,
        page_size_bytes=_int_attr(manager, "page_size") or 0,
        block_size_bytes=block_size_bytes,
        total_blocks=_int_attr(manager, "num_blocks") or 0,
        available_blocks=available_blocks,
        allocated_blocks=allocated_blocks,
        reserved_blocks=reserved_blocks,
        available_bytes=available_blocks * bytes_per_block,
        allocated_bytes=allocated_blocks * bytes_per_block,
        reserved_bytes=reserved_blocks * bytes_per_block,
        null_block_reserved=getattr(manager, "null_block", None) is not None,
        virtual_per_layer_bytes=virtual_per_layer_bytes,
        virtual_total_bytes=virtual_per_layer_bytes * num_layers * num_kv_buffers,
        mapped_bytes=mapped_bytes,
        total_pages=_call_int(allocator, "get_num_total_pages"),
        free_pages=free_pages,
        inuse_pages=_call_int(allocator, "get_num_inuse_pages"),
        reserved_pages=reserved_pages,
        available_physical_pages=available_physical_pages,
        effective_free_pages=effective_free_pages,
        in_shrink=bool(getattr(manager, "in_shrink", False)),
        shrink_target_blocks=getattr(manager, "target_num_blocks", None),
        resize_target_bytes=_call_int(allocator, "get_resize_target"),
    )


def build_kv_cache_pool_operation_snapshot(
    manager: Any,
    *,
    integration: Optional[str] = None,
) -> KVCachePoolOperationSnapshot:
    """Build an exporter-neutral operation snapshot from a manager-like object."""

    get_state = getattr(manager, "_get_operation_observability_state", None)
    if get_state is None:
        counters = dict(getattr(manager, "_operation_counters", {}))
        last_error_code = getattr(manager, "_last_error_code", None)
        last_error_timestamp_ns = getattr(manager, "_last_error_timestamp_ns", None)
    else:
        counters, last_error_code, last_error_timestamp_ns = get_state()

    def counter(name: str) -> int:
        return int(counters.get(name, 0))

    return KVCachePoolOperationSnapshot(
        schema_version=SCHEMA_VERSION,
        pool_type="kv_cache",
        integration=integration,
        pool_name=getattr(manager, "pool_name", None),
        group_id=_int_attr(manager, "group_id") or 0,
        allocation_requests_total=counter("allocation_requests_total"),
        allocation_successes_total=counter("allocation_successes_total"),
        allocation_failures_total=counter("allocation_failures_total"),
        capacity_exhausted_total=counter("capacity_exhausted_total"),
        allocated_blocks_total=counter("allocated_blocks_total"),
        free_requests_total=counter("free_requests_total"),
        free_successes_total=counter("free_successes_total"),
        free_failures_total=counter("free_failures_total"),
        freed_blocks_total=counter("freed_blocks_total"),
        physical_page_allocations_total=counter("physical_page_allocations_total"),
        physical_page_allocation_failures_total=counter(
            "physical_page_allocation_failures_total"
        ),
        physical_page_frees_total=counter("physical_page_frees_total"),
        resize_requests_total=counter("resize_requests_total"),
        resize_successes_total=counter("resize_successes_total"),
        resize_deferred_total=counter("resize_deferred_total"),
        resize_completions_total=counter("resize_completions_total"),
        trim_requests_total=counter("trim_requests_total"),
        trim_successes_total=counter("trim_successes_total"),
        clear_requests_total=counter("clear_requests_total"),
        clear_successes_total=counter("clear_successes_total"),
        operation_errors_total=counter("operation_errors_total"),
        post_init_errors_total=counter("post_init_errors_total"),
        allocation_errors_total=counter("allocation_errors_total"),
        free_errors_total=counter("free_errors_total"),
        resize_errors_total=counter("resize_errors_total"),
        trim_errors_total=counter("trim_errors_total"),
        clear_errors_total=counter("clear_errors_total"),
        state_inconsistency_errors_total=counter(
            "state_inconsistency_errors_total"
        ),
        last_error_code=last_error_code,
        last_error_timestamp_ns=last_error_timestamp_ns,
    )


def get_registered_kv_cache_pool_snapshots(
    *,
    integration: Optional[str] = None,
) -> List[KVCachePoolSnapshot]:
    """Return snapshots of currently live registered KV pools."""

    snapshots = []
    for manager, registered_integration in get_registered_kv_cache_pools(
        integration=integration
    ):
        snapshots.append(
            build_kv_cache_pool_snapshot(
                manager,
                integration=registered_integration,
            )
        )
    return sorted(
        snapshots,
        key=lambda snapshot: (
            snapshot.integration or "",
            snapshot.pool_name or "",
            snapshot.group_id,
        ),
    )


def get_registered_kv_cache_pool_snapshot_dicts(
    *,
    integration: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return JSON-serializable snapshots of currently live registered KV pools."""

    return [
        snapshot.to_dict()
        for snapshot in get_registered_kv_cache_pool_snapshots(integration=integration)
    ]


def get_registered_kv_cache_pool_operation_snapshots(
    *,
    integration: Optional[str] = None,
) -> List[KVCachePoolOperationSnapshot]:
    """Return operation snapshots of currently live registered KV pools."""

    snapshots = []
    for manager, registered_integration in get_registered_kv_cache_pools(
        integration=integration
    ):
        snapshots.append(
            build_kv_cache_pool_operation_snapshot(
                manager,
                integration=registered_integration,
            )
        )
    return sorted(
        snapshots,
        key=lambda snapshot: (
            snapshot.integration or "",
            snapshot.pool_name or "",
            snapshot.group_id,
        ),
    )


def get_registered_kv_cache_pool_operation_snapshot_dicts(
    *,
    integration: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return JSON-serializable operation snapshots of registered KV pools."""

    return [
        snapshot.to_dict()
        for snapshot in get_registered_kv_cache_pool_operation_snapshots(
            integration=integration
        )
    ]
