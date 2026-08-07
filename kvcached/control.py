# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Provider-owned physical memory controls for live kvcached pools."""

from __future__ import annotations

from typing import Any, Dict

from kvcached.pool_registry import get_registered_kv_cache_pools


def set_instance_physical_memory_limit(
    limit_bytes: int,
    *,
    revision: int,
) -> Dict[str, Any]:
    """Split and apply one instance cap across all live KV pools."""
    limit_bytes = int(limit_bytes)
    revision = int(revision)
    if limit_bytes < 0:
        raise ValueError("limit_bytes must be non-negative")
    if revision < 0:
        raise ValueError("revision must be non-negative")

    managers = [
        manager for manager, _ in get_registered_kv_cache_pools()
    ]
    if not managers:
        return {
            "status": "unavailable",
            "reason": "no_registered_kv_cache_pool",
            "limit_bytes": limit_bytes,
            "effective_limit_bytes": 0,
            "revision": revision,
            "mapped_bytes": 0,
            "pools": [],
        }
    managers.sort(
        key=lambda manager: (
            str(getattr(manager, "pool_name", "") or ""),
            int(getattr(manager, "group_id", 0)),
        )
    )
    capacities = [
        max(
            0,
            int(manager.mem_size)
            * int(manager.num_layers)
            * int(manager.num_kv_buffers),
        )
        for manager in managers
    ]
    total_capacity = sum(capacities)
    if total_capacity <= 0:
        raise ValueError("registered KV pools have no virtual capacity")

    remaining = limit_bytes
    pool_states = []
    for index, (manager, capacity) in enumerate(zip(managers, capacities)):
        if index == len(managers) - 1:
            share = remaining
        else:
            share = min(remaining, limit_bytes * capacity // total_capacity)
        pool_states.append(
            manager.set_physical_memory_limit(share, revision=revision)
        )
        remaining -= share

    mapped = sum(int(state.get("mapped_bytes") or 0) for state in pool_states)
    effective = sum(
        int(state.get("effective_limit_bytes") or 0) for state in pool_states
    )
    deferred = any(state.get("status") == "deferred" for state in pool_states)
    stale = all(state.get("status") == "stale" for state in pool_states)
    return {
        "status": "stale" if stale else ("deferred" if deferred else "applied"),
        "reason": "mapped_usage_above_limit" if deferred else "",
        "limit_bytes": limit_bytes,
        "effective_limit_bytes": effective,
        "revision": revision,
        "mapped_bytes": mapped,
        "remaining_bytes": max(0, effective - mapped),
        "overage_bytes": max(0, mapped - effective),
        "pools": pool_states,
    }
