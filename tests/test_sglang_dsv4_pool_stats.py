# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
    sys.modules.setdefault("torch", types.ModuleType("torch"))

from kvcached.integration.sglang.dsv4_pool_stats import (  # noqa: E402
    DeepSeekV4FullPoolCheckPatch,
    DeepSeekV4PoolStatsPatch,
)


def make_observer(monkeypatch, *, managed=True, physical=0, full_capacity=1000,
                  logical_full=700, logical_swa=80, hisparse=False):
    calls: list[str] = []

    class Observer:
        token_to_kv_pool_allocator: Any

        def _get_swa_token_info(self):
            calls.append("native")
            full_available = self.token_to_kv_pool_allocator.full_available_size() if full_capacity else 0
            return SimpleNamespace(
                full_available_size=full_available,
                full_evictable_size=100, full_num_used=900 - full_available if full_capacity else 0,
                full_token_usage=(900 - full_available) / full_capacity if full_capacity else 0,
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
@pytest.mark.parametrize(("full_headroom", "device_slots", "expected_available"), [
    (700, 10, 40),  # Device slots limit FULL even without physical pressure.
    (0, 200, 700),  # Physical pressure alone does not occupy logical slots.
])
def test_hisparse_full_stats_preserve_device_limit(monkeypatch, managed, full_headroom, device_slots,
                                                  expected_available):
    observer, _, _, _ = make_observer(monkeypatch, managed=managed, physical=5, hisparse=True)
    inner = observer.token_to_kv_pool_allocator
    inner.full_available_size = lambda: full_headroom
    wrapper = SimpleNamespace(
        logical_attn_allocator=inner,
        hisparse_attn_allocator=SimpleNamespace(available_size=lambda: device_slots),
        compress_ratio=4,
        swa_available_size=inner.swa_available_size,
    )
    wrapper.full_available_size = lambda: min(
        inner.full_available_size(), wrapper.hisparse_attn_allocator.available_size() * wrapper.compress_ratio)
    observer.token_to_kv_pool_allocator = wrapper
    stats = observer._get_swa_token_info()
    native_available = min(full_headroom, device_slots * 4)
    available = expected_available if managed else native_available
    assert stats.full_available_size == available
    assert stats.full_num_used == 900 - available
    assert stats.full_token_usage == (900 - available) / 1000
    assert stats.swa_num_used == (10 if managed else 85)
    assert wrapper.full_available_size() == native_available
    assert wrapper.swa_available_size() == 5


@pytest.mark.parametrize("managed", [False, True])
def test_hisparse_full_check_counts_logical_slots_without_changing_report(monkeypatch, managed):
    @dataclass
    class Stats:
        full_available_size: int

    class Checker:
        full_tokens_per_layer = 1000
        token_to_kv_pool_allocator = SimpleNamespace(
            logical_attn_allocator=SimpleNamespace(
                _kvcached_dsv4_backing=object() if managed else None,
                _conserve_full_available_size=lambda: 700,
            ),
            hisparse_attn_allocator=object(),
        )

        def _check_full_pool(self, ps, uncached=0):
            return ps.full_available_size + uncached != self.full_tokens_per_layer

    patch = DeepSeekV4FullPoolCheckPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    assert patch.apply(SimpleNamespace(SchedulerInvariantChecker=Checker))
    stats = Stats(full_available_size=40)
    checker = Checker()
    assert checker._check_full_pool(stats, uncached=300) is (not managed)
    assert checker._check_full_pool(stats, uncached=299)  # One slot really is missing.
    assert stats.full_available_size == 40


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
