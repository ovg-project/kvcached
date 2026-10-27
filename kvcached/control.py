# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Post-initialization controls for physical KV and prefix retention."""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping
from typing import Any


def get_prefix_cache_retention_limit() -> int:
    """Return this engine process's idle prefix retention limit in tokens."""
    from kvcached.prefix_cache_retention import (
        _get_prefix_cache_retention_limit,
    )

    return _get_prefix_cache_retention_limit()


def set_prefix_cache_retention_limit(limit_tokens: int) -> None:
    """Update prefix retention from the engine or scheduler thread."""
    from kvcached.prefix_cache_retention import (
        _set_prefix_cache_retention_limit,
    )

    _set_prefix_cache_retention_limit(limit_tokens)


def get_instance_memory_limit(instance_name: str | None = None) -> dict[str, Any]:
    """Return a bounded observational snapshot, never an admission decision."""
    return _get_instance_memory_limit(instance_name, observe=True)


def _get_instance_memory_limit(instance_name: str | None, *, observe: bool) -> dict[str, Any]:
    from kvcached import _C  # type: ignore[attr-defined]
    from kvcached.utils import DEFAULT_IPC_NAME

    name = DEFAULT_IPC_NAME if instance_name is None else instance_name
    directory = os.getenv("KVCACHED_PHYSICAL_GROWTH_LOCK_DIR", "/dev/shm")
    query = _C.observe_instance_control if observe else _C.query_instance_control
    states = []
    errors = []
    for path in sorted(glob.glob(os.path.join(directory, "kvcached-physical-growth-*.lock.budget-v2"))):
        try:
            state = dict(query(path, name))
        except (OSError, RuntimeError) as error:
            errors.append({"gpu": path, "error": str(error)})
            continue
        if state["generation"]:
            states.append(dict(state, gpu=path))
    expected = {state["expected_participants"] for state in states}
    complete = (len(expected) == 1 and next(iter(expected)) > 0
                and sum(state["registered_participants"] for state in states) == next(iter(expected)))
    converged = complete and all(state["convergence"] == "converged" for state in states)
    unsupported = any(state["convergence"] == "unsupported" for state in states)
    unavailable = any(state["convergence"] == "unavailable" for state in states)
    failed = any(state["convergence"] == "failed" for state in states)
    return {"instance": name, "status": "unavailable" if not complete or unavailable else
            "failed" if failed else "unsupported" if unsupported else "converged" if converged else "pending", "complete_membership": complete,
            "converged": converged, "gpus": states, "discovery_errors": errors}


def set_instance_memory_limit(
    limit_bytes: int, *, revision: int | Mapping[str, int] | None = None,
    instance_name: str | None = None,
    generations: Mapping[str, int] | None = None,
    gpu_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Publish one max on each selected GPU; callers repair partial failures.

    Explicit revisions require generations from get_instance_memory_limit(),
    so a retry cannot accidentally target a replacement instance. No revision
    allocates a new one atomically, even when the byte value is unchanged.
    """
    if not isinstance(limit_bytes, int) or not 0 <= limit_bytes <= (1 << 64) - 1:
        raise ValueError("limit_bytes must be a uint64 byte count")
    if revision is not None and generations is None:
        raise ValueError("explicit revision requires per-GPU generations from get_instance_memory_limit()")
    from kvcached import _C  # type: ignore[attr-defined]

    snapshot = _get_instance_memory_limit(instance_name, observe=False)
    if not snapshot["complete_membership"]:
        return dict(snapshot, accepted=False)
    selected = set(gpu_ids) if gpu_ids is not None else {state["gpu"] for state in snapshot["gpus"]}
    known = {state["gpu"] for state in snapshot["gpus"]}
    if not selected or not selected <= known:
        raise ValueError("gpu_ids must select live GPUs from the instance snapshot")
    requests = []
    for state in snapshot["gpus"]:
        gpu = state["gpu"]
        if gpu not in selected:
            continue
        requested_revision = revision[gpu] if isinstance(revision, Mapping) else revision
        generation = state["generation"] if generations is None else generations[gpu]
        if requested_revision is not None and (not isinstance(requested_revision, int)
                                              or not 0 <= requested_revision <= (1 << 64) - 1):
            raise ValueError("revision must be a uint64")
        if not isinstance(generation, int) or not 0 <= generation <= (1 << 64) - 1:
            raise ValueError("generation must be a uint64")
        requests.append((gpu, generation, requested_revision))
    results = []
    for gpu, generation, requested_revision in requests:
        try:
            state = dict(_C.publish_instance_control(
                gpu, snapshot["instance"], limit_bytes, generation, requested_revision))
        except (OSError, RuntimeError) as error:
            state = {"status": "error", "convergence": "unknown", "error": str(error)}
        results.append(dict(state, gpu=gpu))
    accepted = [state["status"] in ("accepted", "idempotent") for state in results]
    return {"instance": snapshot["instance"], "limit_bytes": limit_bytes,
            "status": "accepted" if all(accepted) else "partial" if any(accepted) else "rejected",
            "accepted": all(accepted), "converged": False, "gpus": results}
