# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""HiCache retention runs after copy completion, through native eviction."""

import types

import pytest

from kvcached import prefix_cache_retention as retention
from kvcached.integration.sglang.hicache import (
    HiRadixCacheRetentionPatch,
    UnifiedHiCacheRetentionPatch,
)
from kvcached.integration.sglang.patches import _install_prefix_cache_retention_hooks


@pytest.fixture(params=[HiRadixCacheRetentionPatch, UnifiedHiCacheRetentionPatch])
def adapter(monkeypatch, request):
    class Cache:
        def __init__(self):
            self.nodes = []
            self.events = []
            self.ready_count = 0

        def cache_finished_req(self, req):
            pass

        def _finish_copies(self, kind, finish_count):
            if not finish_count:
                return
            for node in self.nodes:
                if node.pending == kind:
                    node.host = True
                    node.locks -= 1
                    node.pending = None
                    self.events.append(f"{kind}-finished")

        def writing_check(self, write_back=False, finish_count=None):
            self._finish_copies("write", finish_count)

        def loading_check(self, finish_count=None):
            self._finish_copies("load", finish_count)
            # Unified unpins the loaded nodes after releasing their locks.
            if finish_count:
                self.events.append("unpin-finished")
            return "load-result"

        def check_hicache_events(self):
            self.writing_check(finish_count=self.ready_count)
            self.loading_check(finish_count=self.ready_count)
            self.events.append("poll-finished")
            return "poll-result"

        def evict(self, num_tokens):
            # Native write-back eviction calls this while device KV is still
            # evictable. Wrapping writing_check would re-enter eviction.
            self.writing_check(write_back=True)
            self.events.append("evict")
            for node in self.nodes:
                if node.device and node.locks == 0 and num_tokens > 0:
                    assert node.pending is None
                    node.device = False
                    num_tokens -= 8

    def enforce(cache, limit):
        if limit >= 0:
            excess = sum(8 for node in cache.nodes if node.device and node.locks == 0) - limit
            if excess > 0:
                cache.evict(excess)

    patch = request.param()
    monkeypatch.setattr(patch.version_manager, "detect_version", lambda _: "0.5.20")
    monkeypatch.setattr(retention, "_policy", retention._PrefixCacheRetentionPolicy(-1))
    _install_prefix_cache_retention_hooks(patch, Cache, Cache.cache_finished_req, enforce)
    module = types.ModuleType(patch.target_module)
    setattr(module, patch.target_class, Cache)
    return patch, module, Cache


def _pending_copy(cache, kind, *, active=False):
    node = types.SimpleNamespace(
        pending=kind, locks=2 if active else 1, device=True, host=kind == "load",
    )
    cache.nodes.append(node)
    return node


@pytest.mark.parametrize("version,supported", [
    ("0.5.17", False),
    ("0.5.18", True),
    ("0.5.20.post1", True),
    ("0.5.21.dev0", False),
])
def test_completion_patch_respects_hicache_range(adapter, monkeypatch, version, supported):
    patch, module, cache_cls = adapter
    monkeypatch.setattr(patch.version_manager, "detect_version", lambda _: version)
    original_init = cache_cls.__init__
    original_loading = cache_cls.loading_check
    original_events = cache_cls.check_hicache_events

    assert patch.apply(module) is supported
    cache = cache_cls()
    if supported:
        assert cache_cls.loading_check is not original_loading
        assert cache_cls.check_hicache_events is not original_events
    else:
        assert cache_cls.__init__ is original_init
        assert cache_cls.loading_check is original_loading
        assert cache_cls.check_hicache_events is original_events
    assert cache.loading_check() == "load-result"


def test_completion_trims_after_the_whole_poll_and_preserves_active_kv(adapter):
    patch, module, cache_cls = adapter
    assert patch.apply(module)
    cache = cache_cls()
    writing = _pending_copy(cache, "write")
    loading = _pending_copy(cache, "load")
    active = _pending_copy(cache, "load", active=True)
    retention._policy.set_limit(0)
    cache.check_hicache_events()
    assert writing.device and loading.device and active.device
    cache.events.clear()

    cache.ready_count = 1
    assert cache.check_hicache_events() == "poll-result"

    assert cache.events == [
        "write-finished", "load-finished", "load-finished",
        "unpin-finished", "poll-finished", "evict",
    ]
    assert not writing.device and not loading.device
    assert writing.host and loading.host
    assert active.device and active.locks == 1


def test_direct_loading_check_trims_after_unpin(adapter):
    patch, module, cache_cls = adapter
    assert patch.apply(module)
    cache = cache_cls()
    node = _pending_copy(cache, "load")
    retention._policy.set_limit(0)

    assert cache.loading_check(finish_count=1) == "load-result"

    assert cache.events == ["load-finished", "unpin-finished", "evict"]
    assert not node.device
