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


def _modules(monkeypatch, full_evictable, target):
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
