# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest

import kvcached.prefix_cache_retention as retention
from kvcached import control


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    policy = retention._PrefixCacheRetentionPolicy(16000)
    monkeypatch.setattr(retention, "_policy", policy)
    return policy


class Enforcer:
    def __init__(self, error=None):
        self.limits = []
        self.error = error

    def _enforce_prefix_cache_retention(self, limit):
        self.limits.append(limit)
        if self.error is not None:
            raise self.error


@pytest.mark.parametrize("limit", [-1, 0, 1, 16000])
def test_setter_updates_and_enforces_registered_caches(limit):
    cache = Enforcer()
    retention._register_prefix_cache_retention_enforcer(cache)

    retention._set_prefix_cache_retention_limit(limit)

    assert retention._get_prefix_cache_retention_limit() == limit
    assert cache.limits == [limit]


@pytest.mark.parametrize("limit", [-2, 1.5, "1", True, None])
def test_setter_rejects_invalid_limits(limit):
    with pytest.raises(ValueError, match="-1 or a non-negative integer"):
        retention._set_prefix_cache_retention_limit(limit)


def test_registration_is_weak():
    cache = Enforcer()
    retention._register_prefix_cache_retention_enforcer(cache)
    assert len(retention._policy.enforcers) == 1

    del cache
    gc.collect()

    assert len(retention._policy.enforcers) == 0


def test_policy_stays_updated_when_enforcement_fails():
    healthy = Enforcer()
    broken = Enforcer(RuntimeError("eviction failed"))
    retention._register_prefix_cache_retention_enforcer(healthy)
    retention._register_prefix_cache_retention_enforcer(broken)

    with pytest.raises(RuntimeError, match="eviction failed"):
        retention._set_prefix_cache_retention_limit(100)

    assert retention._get_prefix_cache_retention_limit() == 100
    assert healthy.limits == [100]
    assert broken.limits == [100]


def test_public_control_interface_delegates_to_process_policy():
    control.set_prefix_cache_retention_limit(123)

    assert control.get_prefix_cache_retention_limit() == 123
