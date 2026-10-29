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
import kvcached.prefix_cache_retention as retention


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
    monkeypatch.setattr(
        retention, "_policy", retention._PrefixCacheRetentionPolicy(cap))
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


def test_unlimited_cap_installs_runtime_hook_but_does_not_evict(
        monkeypatch, target):
    mod, cache_cls = _modules(monkeypatch, 20000, target)
    original = cache_cls.cache_finished_req
    _apply(monkeypatch, mod, -1, target)
    assert cache_cls.cache_finished_req is not original

    cache = cache_cls()
    cache.cache_finished_req("req")
    assert cache.evicted == []

    retention._policy.set_limit(16000)
    assert cache.evicted == [EvictParams(num_tokens=4000)]


def test_patch_is_idempotent(monkeypatch, target):
    mod, cache_cls = _modules(monkeypatch, 20000, target)
    _apply(monkeypatch, mod, 16000, target)
    once = cache_cls.cache_finished_req
    _apply(monkeypatch, mod, 16000, target)
    assert cache_cls.cache_finished_req is once


@pytest.fixture(params=["HiRadixCache", "UnifiedRadixCache"])
def hicache(monkeypatch, request):
    # HiRadixCache inherits the patched RadixCache constructor; Unified's
    # constructor is patched directly. Exercise both registration paths.
    target = PATCHES[0] if request.param == "UnifiedRadixCache" else (
        "RadixCacheLimitPatch", "sglang.srt.mem_cache.radix_cache", "RadixCache"
    )
    mod, base_cls = _modules(monkeypatch, 0, target)
    mod.EvictParams = EvictParams
    base_cls.reset = lambda self: None

    class HiCache(base_cls):  # type: ignore[valid-type, misc]
        def __init__(self):
            super().__init__()
            self.nodes = []
            self.events = []
            self.write_back = False
            self.ready_count = 0

        @property
        def evictable_size_(self):
            return self.full_evictable_size()

        def full_evictable_size(self):
            return sum(n.tokens for n in self.nodes if n.device and n.locks == 0)

        def cache_finished_req(self, req, **kwargs):
            return super().cache_finished_req(req, **kwargs)

        def _finish_copies(self, kind, finish_count):
            if finish_count == 0:
                return
            for node in self.nodes:
                if node.pending != kind:
                    continue
                self.events.append("synchronize")
                if kind == "write":
                    node.host = True
                node.locks -= 1
                # Unified finishes load-back unpinning after dec_lock_ref.
                self.events.append("unlock")
                node.pending = None
                self.events.append("finish-copy")

        def writing_check(self, write_back=False, finish_count=None):
            self._finish_copies("write", finish_count)
            return "write-result"

        def loading_check(self, finish_count=None):
            self._finish_copies("load", finish_count)
            return "load-result"

        def check_hicache_events(self):
            self.writing_check(finish_count=self.ready_count)
            self.loading_check(finish_count=self.ready_count)
            return "events-result"

        def evict(self, params):
            if self.write_back:
                # Native write-back eviction waits for its own staged copies.
                # A nested retention check here would recursively call evict.
                self.writing_check(write_back=True)
            self.events.append("evict")
            self.evicted.append(params)
            remaining = params.num_tokens
            for node in self.nodes:
                if remaining <= 0:
                    break
                if node.device and node.locks == 0:
                    assert node.pending is None
                    node.device = False
                    remaining -= node.tokens

    if request.param == "UnifiedRadixCache":
        setattr(mod, target[2], HiCache)
    _apply(monkeypatch, mod, -1, target)
    return HiCache()


def _pending_copy(cache, kind, *, active=False):
    node = types.SimpleNamespace(
        tokens=8, device=True, host=kind == "load",
        locks=2 if active else 1, pending=kind,
    )
    cache.nodes.append(node)
    return node


@pytest.mark.parametrize("kind", ["write", "load"])
def test_hicache_enforces_new_limit_after_copy_completion(hicache, kind, monkeypatch):
    import kvcached
    from kvcached import control

    state = dict(gpu="gpu-0", generation=42, retention_tokens=-1, retention_revision=0)

    def publish(gpu, instance, limit, generation, revision):
        assert (gpu, instance, generation) == ("gpu-0", "model", 42)
        state.update(retention_tokens=limit, retention_revision=revision)
        return dict(state, status="accepted")

    monkeypatch.setattr(kvcached, "_C", types.SimpleNamespace(
        publish_prefix_cache_retention=publish), raising=False)
    monkeypatch.setattr(control, "_get_instance_memory_limit", lambda name, observe: {
        "instance": name, "complete_membership": True, "gpus": [dict(state)]})
    retention._policy.instance_name = "model"
    node = _pending_copy(hicache, kind)
    hicache.cache_finished_req("finished-request")

    assert control.set_prefix_cache_retention_limit(0, instance_name="model")["accepted"]
    assert control.get_prefix_cache_retention_limit() == -1  # Publishing only updates the ledger.
    retention._apply_prefix_cache_retention_update(
        retention._read_prefix_cache_retention_update())
    assert control.get_prefix_cache_retention_limit() == 0
    assert hicache.evicted == []  # The copy still protects its device slots.

    assert hicache.check_hicache_events() == "events-result"
    assert hicache.evicted == []
    assert node.device

    hicache.ready_count = 1
    assert hicache.check_hicache_events() == "events-result"
    assert hicache.events == ["synchronize", "unlock", "finish-copy", "evict"]
    assert hicache.evicted == [EvictParams(num_tokens=8)]
    assert not node.device
    assert node.host  # Retention trims device KV through native eviction.


@pytest.mark.parametrize("kind", ["write", "load"])
def test_hicache_copy_completion_preserves_active_request(hicache, kind):
    node = _pending_copy(hicache, kind, active=True)
    retention._policy.set_limit(0)
    hicache.ready_count = 1
    hicache.check_hicache_events()
    assert node.locks == 1
    assert node.device
    assert hicache.evicted == []


@pytest.mark.parametrize("kind", ["write", "load"])
def test_hicache_unlimited_retains_completed_copy(hicache, kind):
    node = _pending_copy(hicache, kind)
    hicache.ready_count = 1
    hicache.check_hicache_events()
    assert node.locks == 0
    assert node.device
    assert hicache.evicted == []


def test_hicache_write_back_eviction_does_not_reenter(hicache):
    node = _pending_copy(hicache, "write")
    hicache.writing_check(finish_count=1)
    hicache.write_back = True

    retention._policy.set_limit(0)

    assert hicache.evicted == [EvictParams(num_tokens=8)]
    assert not node.device


def test_hicache_completion_hook_is_installed_once_per_class(hicache):
    cache_cls = type(hicache)
    check_events = cache_cls.check_hicache_events
    loading_check = cache_cls.loading_check
    cache_cls()
    assert cache_cls.check_hicache_events is check_events
    assert cache_cls.loading_check is loading_check


def test_hicache_direct_loading_check_enforces_after_unpin(hicache):
    node = _pending_copy(hicache, "load")
    retention._policy.set_limit(0)

    assert hicache.loading_check(finish_count=1) == "load-result"

    assert hicache.events == ["synchronize", "unlock", "finish-copy", "evict"]
    assert not node.device


def test_hicache_positive_cap_keeps_remaining_device_cache(hicache):
    first = _pending_copy(hicache, "write")
    second = _pending_copy(hicache, "write")
    retention._policy.set_limit(8)

    hicache.ready_count = 1
    hicache.check_hicache_events()

    assert hicache.evicted == [EvictParams(num_tokens=8)]
    assert not first.device
    assert second.device
    assert first.host and second.host
    assert hicache.full_evictable_size() == 8
