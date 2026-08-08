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
    """Read-only state of one kvcached-backed KV pool.

    Block accounting (these fields are NOT mutually exclusive -- do not sum
    them):

    * ``allocated_blocks`` -- blocks currently handed out of their pages.
      Includes ``reserved_blocks``.
    * ``reserved_blocks`` -- the subset of ``allocated_blocks`` held on the
      pool's reserve ledger. ``alloc()`` drains this ledger first.
    * ``available_blocks`` -- what the next request could obtain. Because
      ``alloc()`` drains the reserve ledger first, this also includes
      ``reserved_blocks``.

    ``reserved_blocks`` counts *blocks* on that ledger and is unrelated to
    ``reserved_pages``, which counts *physical pages* held by the background
    pre-allocation thread.

    Values are best-effort rather than a consistent point-in-time cut: the
    C++ pre-allocation thread mutates page counters without taking the
    manager lock, so fields may come from marginally different instants.
    """

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


def get_capabilities() -> Dict[str, Any]:
    """Return the stable observability surface currently exposed by kvcached."""

    return {
        "schema_version": SCHEMA_VERSION,
        "features": {
            "runtime_snapshot": True,
            "kv_cache_pool_snapshot": True,
            "registered_kv_cache_pool_snapshots": True,
            "read_only": True,
            "policy_control": False,
        },
        "pool_snapshot_fields": list(KVCachePoolSnapshot.__dataclass_fields__.keys()),
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
    # available_size() can transiently go negative: it derives from
    # PageAllocator::get_num_free_pages(), which reads a mutable counter
    # without holding the allocator mutex, so a concurrent snapshot may
    # observe an intermediate value. A negative gauge is never meaningful to
    # an exporter, so clamp at zero.
    available_blocks = max(int(manager.available_size()), 0)
    allocated_blocks = max(int(manager._get_num_alloced_blocks()), 0)
    reserved_blocks = len(getattr(manager, "reserved_blocks", []))

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


def _snapshot_one_pool(
    manager: Any,
    integration: Optional[str],
) -> KVCachePoolSnapshot:
    """Snapshot one pool through its own synchronized entry point.

    ``KVCacheManager.observability_snapshot()`` is ``@synchronized``, so the
    whole snapshot is taken under a single hold of the manager lock (the
    nested per-accessor acquisitions are reentrant). Calling
    ``build_kv_cache_pool_snapshot()`` directly instead takes the lock once
    per accessor, which lets a writer commit in the gaps and stitches the
    result together from several different instants.

    Fall back to the module-level builder for the duck-typed manager-likes
    that ``build_kv_cache_pool_snapshot`` is documented to accept.
    """
    take_snapshot = getattr(manager, "observability_snapshot", None)
    if take_snapshot is None:
        return build_kv_cache_pool_snapshot(manager, integration=integration)
    return take_snapshot(integration=integration)


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
            _snapshot_one_pool(manager, registered_integration)
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
