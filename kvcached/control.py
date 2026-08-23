# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Revisioned memory controls for live kvcached KV pools."""

from __future__ import annotations

from typing import Any, Dict

from kvcached.pool_registry import get_registered_kv_cache_pools


def set_instance_memory_limit(
    limit_bytes: int,
    *,
    revision: int,
) -> Dict[str, Any]:
    """Split and apply one instance limit across all live KV pools."""
    limit_bytes = int(limit_bytes)
    revision = int(revision)
    if limit_bytes < 0:
        raise ValueError("limit_bytes must be non-negative")
    if revision < 0:
        raise ValueError("revision must be non-negative")

    managers = [manager for manager, _ in get_registered_kv_cache_pools()]
    if not managers:
        return {
            "status": "unavailable",
            "reason": "no_registered_kv_cache_pool",
            "limit_bytes": limit_bytes,
            "effective_limit_bytes": 0,
            "revision": revision,
            "mapped_bytes": 0,
            "remaining_bytes": 0,
            "overage_bytes": 0,
            "pools": [],
        }

    managers.sort(
        key=lambda manager: (
            str(manager.pool_name or ""),
            int(manager.group_id),
        )
    )
    capacities = [
        max(
            0,
            int(manager.mem_size) * int(manager.num_layers) * int(manager.num_kv_buffers),
        )
        for manager in managers
    ]
    total_capacity = sum(capacities)
    if total_capacity <= 0:
        raise ValueError("registered KV pools have no virtual capacity")

    remaining = min(limit_bytes, total_capacity)
    pool_states = []
    for index, (manager, capacity) in enumerate(zip(managers, capacities)):
        share = (
            remaining
            if index == len(managers) - 1
            else min(
                remaining,
                min(limit_bytes, total_capacity) * capacity // total_capacity,
            )
        )
        pool_states.append(manager.set_memory_limit(share, revision=revision))
        remaining -= share

    mapped = sum(int(state["mapped_bytes"]) for state in pool_states)
    effective = sum(int(state["effective_limit_bytes"] or 0) for state in pool_states)
    statuses = {str(state["status"]) for state in pool_states}
    if "conflict" in statuses:
        status = "conflict"
    elif "stale" in statuses:
        status = "stale"
    elif "deferred" in statuses:
        status = "deferred"
    else:
        status = "applied"
    return {
        "status": status,
        "reason": {
            "deferred": "inuse_capacity_above_limit",
            "conflict": "revision_reused_with_different_limit",
        }.get(status, ""),
        "limit_bytes": limit_bytes,
        "effective_limit_bytes": effective,
        "revision": revision,
        "mapped_bytes": mapped,
        "remaining_bytes": max(0, effective - mapped),
        "overage_bytes": max(0, mapped - effective),
        "pools": pool_states,
    }
