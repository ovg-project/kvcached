# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
    sys.modules.setdefault("torch", types.ModuleType("torch"))

from kvcached.integration.sglang.dsv4_pool_stats import DeepSeekV4PoolStatsPatch  # noqa: E402


def make_observer(monkeypatch, *, managed=True, physical=0, full_capacity=1000,
                  logical_full=700, logical_swa=80, hisparse=False):
    calls: list[str] = []

    class Observer:
        def _get_swa_token_info(self):
            calls.append("native")
            return SimpleNamespace(
                full_available_size=physical if full_capacity else 0,
                full_evictable_size=100, full_num_used=900 - physical if full_capacity else 0,
                full_token_usage=(900 - physical) / full_capacity if full_capacity else 0,
                swa_available_size=physical, swa_evictable_size=10,
                swa_num_used=90 - physical, swa_token_usage=(90 - physical) / 100,
                native_extra="preserved",
            )

    module = SimpleNamespace(SchedulerPoolStatsObserver=Observer)
    patch = DeepSeekV4PoolStatsPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    assert patch.apply(module)
    observer: Any = Observer()
    observer.token_to_kv_pool_allocator = SimpleNamespace(
        _kvcached_dsv4_backing=object() if managed else None,
        _conserve_full_available_size=lambda: logical_full,
        _conserve_swa_available_size=lambda: logical_swa,
        full_available_size=lambda: physical,
        swa_available_size=lambda: physical,
    )
    observer.full_tokens_per_layer = full_capacity
    observer.swa_tokens_per_layer = 100
    observer.enable_hisparse = hisparse
    return observer, patch, module, calls


@pytest.mark.parametrize("physical", [0, 5, 40])
def test_physical_pressure_does_not_inflate_logical_usage(monkeypatch, physical):
    observer, _, _, calls = make_observer(monkeypatch, physical=physical)
    stats = observer._get_swa_token_info()
    assert (stats.full_available_size, stats.full_num_used, stats.full_token_usage) == (700, 200, 0.2)
    assert (stats.swa_available_size, stats.swa_num_used, stats.swa_token_usage) == (80, 10, 0.1)
    assert stats.native_extra == "preserved" and calls == ["native"]
    # Allocation still sees the same physical limit after a statistics read.
    assert observer.token_to_kv_pool_allocator.full_available_size() == physical
    assert observer.token_to_kv_pool_allocator.swa_available_size() == physical


def test_native_fallback_statistics_are_unchanged(monkeypatch):
    observer, _, _, _ = make_observer(monkeypatch, managed=False, physical=5)
    stats = observer._get_swa_token_info()
    assert (stats.full_num_used, stats.swa_num_used) == (895, 85)


def test_zero_full_capacity_preserves_native_zero_fields(monkeypatch):
    observer, _, _, _ = make_observer(monkeypatch, full_capacity=0)
    stats = observer._get_swa_token_info()
    assert (stats.full_available_size, stats.full_num_used, stats.full_token_usage) == (0, 0, 0)
    assert stats.swa_num_used == 10


@pytest.mark.parametrize("hisparse,expected", [(False, -10), (True, 0)])
def test_only_native_hisparse_conservation_clamp_is_preserved(monkeypatch, hisparse, expected):
    observer, _, _, _ = make_observer(monkeypatch, logical_swa=100, hisparse=hisparse)
    stats = observer._get_swa_token_info()
    assert stats.swa_num_used == expected


def test_missing_conservation_method_retains_native_component(monkeypatch):
    observer, _, _, _ = make_observer(monkeypatch, physical=5)
    del observer.token_to_kv_pool_allocator._conserve_swa_available_size
    stats = observer._get_swa_token_info()
    assert stats.full_num_used == 200 and stats.swa_num_used == 85


@pytest.mark.parametrize("managed", [False, True])
def test_hisparse_wrapper_uses_inner_backing_without_changing_device_headroom(monkeypatch, managed):
    observer, _, _, _ = make_observer(monkeypatch, managed=managed, physical=5, hisparse=True)
    inner = observer.token_to_kv_pool_allocator
    del inner._conserve_full_available_size
    observer.token_to_kv_pool_allocator = SimpleNamespace(
        logical_attn_allocator=inner,
        full_available_size=lambda: 5,
        swa_available_size=inner.swa_available_size,
    )
    stats = observer._get_swa_token_info()
    assert stats.full_num_used == 895
    assert stats.swa_num_used == (10 if managed else 85)
    assert observer.token_to_kv_pool_allocator.full_available_size() == 5
    assert observer.token_to_kv_pool_allocator.swa_available_size() == 5


def test_conservation_error_is_not_hidden(monkeypatch):
    observer, _, _, _ = make_observer(monkeypatch)
    def fail():
        raise RuntimeError("backing failed")
    observer.token_to_kv_pool_allocator._conserve_swa_available_size = fail
    with pytest.raises(RuntimeError, match="backing failed"):
        observer._get_swa_token_info()


def test_patch_is_idempotent(monkeypatch):
    observer, patch, module, calls = make_observer(monkeypatch)
    before = type(observer)._get_swa_token_info
    assert patch.apply(module)
    assert type(observer)._get_swa_token_info is before
    observer._get_swa_token_info()
    assert calls == ["native"]


def test_missing_native_api_or_unknown_version_is_not_patched(monkeypatch):
    patch = DeepSeekV4PoolStatsPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    assert not patch.apply(SimpleNamespace())
    monkeypatch.setattr(patch, "initialize_version_info", lambda: False)
    assert not patch.apply(SimpleNamespace())
