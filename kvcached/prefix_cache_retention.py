# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Read idle prefix retention and enforce it only on engine threads."""

from __future__ import annotations

import time
import weakref
from typing import Any

from kvcached.utils import MAX_CACHED_TOKENS


class _PrefixCacheRetentionPolicy:
    def __init__(self, limit_tokens: int) -> None:
        self.limit_tokens = -1 if limit_tokens < 0 else limit_tokens
        self.enforcers: weakref.WeakSet[Any] = weakref.WeakSet()
        self.instance_name: str | None = None
        self.applied_revision: tuple | None = None
        self.last_poll = 0.0

    def register(self, enforcer: Any) -> None:
        self.enforcers.add(enforcer)

    def set_limit(self, limit_tokens: int) -> None:
        """Update and enforce on the engine or scheduler thread.

        Returning means currently evictable excess has passed through the
        caches' normal eviction paths, not that GPU memory has been released.
        """
        if (not isinstance(limit_tokens, int) or isinstance(limit_tokens, bool)
                or limit_tokens < -1):
            raise ValueError("limit_tokens must be -1 or a non-negative integer")
        # The policy remains in force even if an adapter cannot finish its
        # immediate trim. Eviction is irreversible, so rollback would be
        # misleading; later cache transitions retry against this value.
        self.limit_tokens = limit_tokens
        first_error: Exception | None = None
        for enforcer in list(self.enforcers):
            try:
                enforcer._enforce_prefix_cache_retention(limit_tokens)
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


_policy = _PrefixCacheRetentionPolicy(MAX_CACHED_TOKENS)


def _read_prefix_cache_retention_update(
    *, snapshot: dict[str, Any] | None = None, throttle: bool = False,
) -> dict[str, Any] | None:
    """Read a complete assignment; throttle engine-loop reads, not idle watchers."""
    from kvcached.control import _get_instance_memory_limit

    name = _policy.instance_name
    if name is None:
        return None
    if throttle:
        now = time.monotonic()
        if now - _policy.last_poll < 0.1:
            return None
        _policy.last_poll = now
    if snapshot is None:
        snapshot = _get_instance_memory_limit(name, observe=False)
    if not snapshot["complete_membership"]:
        return None
    states = snapshot["gpus"]
    if not states:
        return None
    revision = states[0]["retention_revision"]
    limit_tokens = states[0]["retention_tokens"]
    if not revision or not all(
        state["retention_revision"] == revision and state["retention_tokens"] == limit_tokens
        for state in states[1:]
    ):
        return None
    signature = tuple((state["gpu"], state["generation"], revision) for state in states)
    if signature == _policy.applied_revision:
        return None
    return {"limit_tokens": limit_tokens, "revision": signature}


def _apply_prefix_cache_retention_update(assignment: dict[str, Any] | None) -> None:
    if assignment is not None and assignment["revision"] != _policy.applied_revision:
        _policy.set_limit(assignment["limit_tokens"])
        _policy.applied_revision = assignment["revision"]
