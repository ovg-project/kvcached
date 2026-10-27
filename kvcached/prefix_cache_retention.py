# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Process-local control of idle prefix-cache retention."""

from __future__ import annotations

import weakref
from typing import Any

from kvcached.utils import MAX_CACHED_TOKENS


class _PrefixCacheRetentionPolicy:
    def __init__(self, limit_tokens: int) -> None:
        self.limit_tokens = -1 if limit_tokens < 0 else limit_tokens
        self.enforcers: weakref.WeakSet[Any] = weakref.WeakSet()

    def register(self, enforcer: Any) -> None:
        self.enforcers.add(enforcer)

    def set_limit(self, limit_tokens: int) -> None:
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


def _get_prefix_cache_retention_limit() -> int:
    """Return this engine process's current logical token limit."""
    return _policy.limit_tokens


def _set_prefix_cache_retention_limit(limit_tokens: int) -> None:
    """Update and synchronously enforce the process-local retention policy.

    The caller must run this on the engine or scheduler thread. Returning
    means currently evictable excess has passed through each engine's normal
    eviction path; it does not mean physical GPU memory has been released.
    """
    _policy.set_limit(limit_tokens)


def _register_prefix_cache_retention_enforcer(enforcer: Any) -> None:
    _policy.register(enforcer)
