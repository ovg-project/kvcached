# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""KVCACHED_MAX_CACHED_TOKENS must also bound the prefix caches that are not
RadixCache subclasses.

SGLang uses SWARadixCache / MambaRadixCache for hybrid models up to 0.5.15,
UnifiedRadixCache for them from 0.5.16 and for every radix-cache model from
0.5.19. The RadixCache-only cap left an idle server holding its whole prefix
cache mapped.
"""

import dataclasses
import sys
import types
from typing import Any

import pytest

import kvcached.integration.sglang.patches as sgl_patches


@dataclasses.dataclass
class EvictParams:
    num_tokens: int = 0
    swa_num_tokens: int = 0
    mamba_num: int = 0


PATCHES = [
    ("UnifiedRadixCacheLimitPatch", "sglang.srt.mem_cache.unified_radix_cache", "UnifiedRadixCache"),
    ("SWARadixCacheLimitPatch", "sglang.srt.mem_cache.swa_radix_cache", "SWARadixCache"),
    ("MambaRadixCacheLimitPatch", "sglang.srt.mem_cache.mamba_radix_cache", "MambaRadixCache"),
]


@pytest.fixture(params=PATCHES, ids=[p[2] for p in PATCHES])
def target(request):
    return request.param


def _modules(monkeypatch, full_evictable, target, swa_evictable=0):
    base: Any = types.ModuleType("sglang.srt.mem_cache.base_prefix_cache")
    base.EvictParams = EvictParams
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.base_prefix_cache", base)

    class Cache:
        def __init__(self):
            self.finished = []
            self.evicted = []

        def cache_finished_req(self, req, is_insert=True, **kwargs):
            self.finished.append(req)

        def full_evictable_size(self):
            return full_evictable

        def swa_evictable_size(self):
            return swa_evictable

        def evict(self, params):
            self.evicted.append(params)

    _, module_name, class_name = target
    mod: Any = types.ModuleType(module_name)
    setattr(mod, class_name, Cache)
    return mod, Cache


def _apply(monkeypatch, mod, cap, target):
    monkeypatch.setattr(sgl_patches, "MAX_CACHED_TOKENS", cap)
    patch = getattr(sgl_patches, target[0])()
    monkeypatch.setattr(patch.version_manager, "detect_version", lambda _name: "0.5.20")
    assert patch.apply(mod)
    return patch


def test_finished_request_trims_the_cache_to_the_cap(monkeypatch, target):
    mod, cache_cls = _modules(monkeypatch, 20000, target)
    _apply(monkeypatch, mod, 16000, target)

    cache = cache_cls()
    cache.cache_finished_req("req", kv_len_to_handle=8)

    assert cache.finished == ["req"]
    assert cache.evicted == [EvictParams(num_tokens=4000)]


@pytest.mark.parametrize("evictable", [0, 16000])
def test_cache_within_the_cap_is_left_alone(monkeypatch, target, evictable):
    mod, cache_cls = _modules(monkeypatch, evictable, target)
    _apply(monkeypatch, mod, 16000, target)

    cache = cache_cls()
    cache.cache_finished_req("req", kv_len_to_handle=8)
    assert cache.evicted == []


def test_unlimited_cap_leaves_the_class_unpatched(monkeypatch, target):
    mod, cache_cls = _modules(monkeypatch, 20000, target)
    original = cache_cls.cache_finished_req
    _apply(monkeypatch, mod, -1, target)
    assert cache_cls.cache_finished_req is original


def test_patch_is_idempotent(monkeypatch, target):
    mod, cache_cls = _modules(monkeypatch, 20000, target)
    _apply(monkeypatch, mod, 16000, target)
    once = cache_cls.cache_finished_req
    _apply(monkeypatch, mod, 16000, target)
    assert cache_cls.cache_finished_req is once


@pytest.mark.parametrize("target", PATCHES[:2], ids=[p[2] for p in PATCHES[:2]])
@pytest.mark.parametrize("full_cap,swa_cap,full,swa,expected", [
    (-1, 4, 20000, 12, EvictParams(swa_num_tokens=8)),
    (4, -1, 12, 20000, EvictParams(num_tokens=8)),
    (4, 4, 12, 12, EvictParams(num_tokens=8, swa_num_tokens=8)),
    (0, 0, 12, 8, EvictParams(num_tokens=12, swa_num_tokens=8)),
    (-1, 0, 20000, 12, EvictParams(swa_num_tokens=12)),
    (4, 4, 4, 4, None),
    (4, 4, 0, 12, EvictParams(swa_num_tokens=8)),
    (4, 4, 12, 0, EvictParams(num_tokens=8)),
])
def test_component_limits_are_independent(
    monkeypatch, target, full_cap, swa_cap, full, swa, expected,
):
    mod, cache_cls = _modules(monkeypatch, full, target, swa)
    monkeypatch.setattr(sgl_patches, "MAX_CACHED_SWA_TOKENS", swa_cap)
    _apply(monkeypatch, mod, full_cap, target)
    cache = cache_cls()
    cache.cache_finished_req("req", kv_len_to_handle=8)
    assert cache.finished == ["req"]
    # One native call lets FULL eviction contribute to the SWA request instead
    # of issuing a second stale SWA request that can evict too much.
    assert cache.evicted == ([] if expected is None else [expected])


def test_swa_setting_does_not_enable_full_only_mamba_cap(monkeypatch):
    target = PATCHES[2]
    mod, cache_cls = _modules(monkeypatch, 20000, target, 20000)
    original = cache_cls.cache_finished_req
    monkeypatch.setattr(sgl_patches, "MAX_CACHED_SWA_TOKENS", 0)
    _apply(monkeypatch, mod, -1, target)
    assert cache_cls.cache_finished_req is original
    _apply(monkeypatch, mod, 16000, target)
    cache = cache_cls()
    cache.cache_finished_req("req")
    assert cache.evicted == [EvictParams(num_tokens=4000)]


def test_cap_reads_accounting_only_after_native_completion(monkeypatch):
    target = PATCHES[0]
    mod, cache_cls = _modules(monkeypatch, 0, target)
    native_finished = cache_cls.cache_finished_req

    def finished(self, *args, **kwargs):
        native_finished(self, *args, **kwargs)
        self.swa_evictable_size = lambda: 12

    cache_cls.cache_finished_req = finished
    monkeypatch.setattr(sgl_patches, "MAX_CACHED_SWA_TOKENS", 4)
    _apply(monkeypatch, mod, -1, target)
    cache = cache_cls()
    cache.cache_finished_req("req", is_insert=False, kv_len_to_handle=8)
    assert cache.evicted == [EvictParams(swa_num_tokens=8)]


def test_native_completion_failure_does_not_start_eviction(monkeypatch):
    target = PATCHES[0]
    mod, cache_cls = _modules(monkeypatch, 20, target, 20)

    def fail(*args, **kwargs):
        raise RuntimeError("native completion failed")

    cache_cls.cache_finished_req = fail
    monkeypatch.setattr(sgl_patches, "MAX_CACHED_SWA_TOKENS", 0)
    _apply(monkeypatch, mod, 0, target)
    cache = cache_cls()
    with pytest.raises(RuntimeError, match="native completion failed"):
        cache.cache_finished_req("req")
    assert cache.evicted == []


@pytest.mark.parametrize("missing", ["accounting", "parameters"])
def test_unsupported_swa_contract_rejected_before_wrapping(monkeypatch, missing):
    target = PATCHES[0]
    mod, cache_cls = _modules(monkeypatch, 20, target, 20)
    original = cache_cls.cache_finished_req
    if missing == "accounting":
        del cache_cls.swa_evictable_size
    else:
        @dataclasses.dataclass
        class FullOnlyParams:
            num_tokens: int = 0

        monkeypatch.setattr(sys.modules["sglang.srt.mem_cache.base_prefix_cache"],
                            "EvictParams", FullOnlyParams)
    monkeypatch.setattr(sgl_patches, "MAX_CACHED_SWA_TOKENS", 4)
    with pytest.raises(ValueError, match="set it to -1"):
        _apply(monkeypatch, mod, 16, target)
    assert cache_cls.cache_finished_req is original
    # The suggested compatible setting actually retains the old FULL limit.
    monkeypatch.setattr(sgl_patches, "MAX_CACHED_SWA_TOKENS", -1)
    _apply(monkeypatch, mod, 16, target)
    cache = cache_cls()
    cache.cache_finished_req("req")
    assert cache.evicted[0].num_tokens == 4
