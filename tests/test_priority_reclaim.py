# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Releasing memory for a higher priority's demand, tier by tier (#585).

Every instance reads the same plan from its GPU's ledger and releases its own
part: pre-mapped pages that hold no data first, then whole pages of prefix
cache. What is left belongs to running requests, which only the engine can
preempt. Each step the instance also posts what it could release, so the
next plan picks the cheapest tier.
"""
from __future__ import annotations

import sys
import threading
import types
from typing import Any
from unittest import mock

import pytest


class _FakeInternalPage:
    @staticmethod
    def get_num_blocks(page_size: int, block_mem_size: int) -> int:
        return page_size // block_mem_size


class _PageAllocator:
    """Reserve of pre-mapped pages; eviction may add pages back to it."""

    def __init__(self, reserved: int) -> None:
        self.reserved = reserved
        self.trimmed: list[int] = []

    def get_num_reserved_pages(self) -> int:
        return self.reserved

    def trim(self, max_pages: int = -1) -> int:
        count = self.reserved if max_pages < 0 else min(self.reserved, max_pages)
        self.reserved -= count
        self.trimmed.append(count)
        return count


@pytest.fixture
def manager_module(monkeypatch):
    try:
        import kvcached.vmm_ops  # noqa: F401
    except ImportError:
        stub = types.ModuleType("kvcached.vmm_ops")
        stub.PageAllocator = object  # type: ignore[attr-defined]
        stub.InternalPage = _FakeInternalPage  # type: ignore[attr-defined]
        stub.kv_tensors_created = lambda group_id=0: True  # type: ignore[attr-defined]
        stub.map_to_kv_tensors = lambda *a, **k: None  # type: ignore[attr-defined]
        stub.unmap_from_kv_tensors = lambda *a, **k: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", stub)
    import kvcached.kv_cache_manager as module

    ops = sys.modules["kvcached.vmm_ops"]
    reclaimable = mock.MagicMock()
    monkeypatch.setattr(ops, "set_kv_reclaimable", reclaimable, raising=False)
    plan = {"empty_bytes": 0, "cache_bytes": 0, "running_bytes": 0}
    monkeypatch.setattr(ops, "kv_reclaim", lambda group_id=0: dict(plan),
                        raising=False)
    monkeypatch.setattr(module, "InternalPage", _FakeInternalPage)
    return module, reclaimable, plan


def _manager(module: Any, reserved: int = 0) -> Any:
    from kvcached.locks import NoOpLock

    manager = object.__new__(module.KVCacheManager)
    # A page bundle is 4 bytes x 2 layers x K and V = 16 bytes.
    manager.page_size = 4
    manager.block_mem_size = 1
    manager.num_layers = 2
    manager.num_kv_buffers = 2
    manager.world_size = 1
    manager.group_id = 0
    manager._uses_worker_ipc = False
    manager.page_allocator = _PageAllocator(reserved)
    manager._lock = NoOpLock()
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()
    manager._operation_counters = None
    manager._retired_pages = []
    return manager


class _Cache:
    """A prefix cache whose evicted pages land on the pool's reserve."""

    def __init__(self, allocator: _PageAllocator, pages: int) -> None:
        self.allocator = allocator
        self.pages = pages
        self.evicted: list[int] = []

    def count_pages(self) -> int:
        return self.pages

    def evict_pages(self, pages: int) -> int:
        pages = min(pages, self.pages)
        self.pages -= pages
        self.allocator.reserved += pages
        self.evicted.append(pages)
        return pages


def test_empty_pages_then_cache_are_released_and_levels_posted(manager_module):
    module, reclaimable, plan = manager_module
    manager = _manager(module, reserved=3)
    cache = _Cache(manager.page_allocator, pages=5)
    manager._register_cache_owner(cache.count_pages, cache.evict_pages)
    # The ledger asks for 2 empty pages, 3 of cache and 4 of running
    # requests (16 bytes a page, rounded up).
    plan.update(empty_bytes=32, cache_bytes=40, running_bytes=64)
    assert manager.reclaim_for_priority() == 4
    assert cache.evicted == [3]
    # The 3 evicted pages went to the reserve, so 2 + 3 pages are unmapped.
    assert manager.page_allocator.trimmed == [5]
    assert manager.page_allocator.reserved == 1
    # What is left to release is posted at once: 1 empty page, 2 of cache.
    reclaimable.assert_called_once_with(16, 32, 0, 0)


def test_nothing_is_released_without_a_plan(manager_module):
    module, reclaimable, plan = manager_module
    manager = _manager(module, reserved=3)
    cache = _Cache(manager.page_allocator, pages=5)
    manager._register_cache_owner(cache.count_pages, cache.evict_pages)
    plan.update(running_bytes=16)
    assert manager.reclaim_for_priority() == 1
    assert cache.evicted == [] and manager.page_allocator.trimmed == []
    assert not reclaimable.called


def test_levels_are_written_only_when_they_change(manager_module):
    module, reclaimable, _ = manager_module
    manager = _manager(module, reserved=2)
    manager._retired_pages = [(1, [7, 8]), (2, [9])]
    cache = _Cache(manager.page_allocator, pages=4)
    manager._register_cache_owner(cache.count_pages, cache.evict_pages)
    manager.post_reclaimable_levels()
    manager.post_reclaimable_levels()
    cache.pages = 0
    manager.post_reclaimable_levels()
    assert reclaimable.call_args_list == [mock.call(32, 64, 48, 0),
                                          mock.call(32, 0, 48, 0)]


def test_a_dead_cache_owner_is_dropped(manager_module):
    module, _, plan = manager_module
    manager = _manager(module)
    cache = _Cache(manager.page_allocator, pages=4)
    manager._register_cache_owner(cache.count_pages, cache.evict_pages)
    del cache
    plan.update(cache_bytes=16)
    assert manager.reclaim_for_priority() == 0
    assert manager._live_cache_owners() == []


class _WorkerControl:
    """Runtime control over two workers whose pages are 16 and 24 bytes."""

    worker_ipc = True
    sources = ({"gpu": "gpu0", "instance": "model", "generation": 7},
               {"gpu": "gpu1", "instance": "model", "generation": 8})

    def poll(self) -> tuple:
        return ({"page_bundle_bytes": 16, "reclaim_empty_bytes": 16,
                 "reclaim_cache_bytes": 33, "reclaim_running_bytes": 0},
                {"page_bundle_bytes": 24, "reclaim_empty_bytes": 48,
                 "reclaim_cache_bytes": 24, "reclaim_running_bytes": 1})


@pytest.fixture
def native(monkeypatch):
    import kvcached

    fake = mock.MagicMock()
    monkeypatch.setattr(kvcached, "_C", fake, raising=False)
    return fake


def test_workers_plan_in_their_own_page_size(manager_module, native):
    module, reclaimable, _ = manager_module
    manager = _manager(module)
    manager.world_size = 2
    assert manager.reclaim_plan_pages() == {"empty": 0, "cache": 0, "running": 0}
    manager.post_reclaimable(1, 2, 0)  # workers not registered yet
    assert not native.publish_instance_reclaimable.called
    manager._runtime_control = _WorkerControl()
    # Every page spans both GPUs, so the larger ask wins.
    assert manager.reclaim_plan_pages() == {"empty": 2, "cache": 3, "running": 1}
    manager.post_reclaimable(1, 2, 0)
    assert native.publish_instance_reclaimable.call_args_list == [
        mock.call("gpu0", "model", 7, 16, 32, 0),
        mock.call("gpu1", "model", 8, 24, 48, 0)]
    assert not reclaimable.called


def test_page_index_counts_pages_of_only_evictable_blocks():
    from kvcached.integration.vllm.page_eviction import PageEvictionIndex

    occupancy = {0: 2, 1: 3}
    manager = types.SimpleNamespace(
        block_mem_size=1, page_size=4,
        get_page_occupancy=lambda pages: {p: occupancy[p] for p in pages})
    index = PageEvictionIndex(manager)
    for block in (0, 1, 4, 5):  # page 1 also holds a running request's block
        index.add(block)
    assert index.whole_pages() == 1
    index.add(6)
    assert index.whole_pages() == 2
    assert sorted(index.victims(10, max_pages=1)) == [0, 1]


def test_sglang_cache_owner_counts_and_evicts_pages(monkeypatch):
    from kvcached.integration.sglang import patches

    evicted: list[int] = []

    class Cache:
        page_size = 2
        tokens = 20

        def evictable_size(self) -> int:
            return self.tokens

    cache = Cache()

    def evict(target: Any, num_tokens: int) -> None:
        evicted.append(num_tokens)
        target.tokens -= min(target.tokens, num_tokens)

    class Scheduler:
        tree_cache = cache

    monkeypatch.setattr(patches, "_evict_prefix_cache", evict)
    scheduler = Scheduler()
    manager = types.SimpleNamespace(page_size=8, block_mem_size=2)
    reclaimer = patches._PrefixCacheShrinkReclaimer(scheduler, manager)
    # 4 blocks of 2 tokens a page: 20 evictable tokens fill 2 pages.
    assert reclaimer.count_pages() == 2
    assert reclaimer.evict_pages(1) == 1 and evicted == [8]
    assert reclaimer.evict_pages(0) == 0 and evicted == [8]


@pytest.fixture
def vllm_patches(monkeypatch):
    import importlib

    torch = mock.MagicMock()
    torch.__version__ = "2.6.0"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", torch.cuda)
    monkeypatch.setitem(sys.modules, "torch.utils", torch.utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension",
                        torch.utils.cpp_extension)
    monkeypatch.setitem(sys.modules, "posix_ipc", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())
    monkeypatch.delitem(sys.modules, "kvcached.integration.vllm.patches",
                        raising=False)
    patches = importlib.import_module("kvcached.integration.vllm.patches")
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    return patches


def test_vllm_step_releases_before_scheduling_and_posts_after(
        vllm_patches, monkeypatch):
    events: list[str] = []

    class KV:
        def reclaim_for_priority(self) -> int:
            events.append("reclaim")
            return 3

        def post_reclaimable_levels(self) -> None:
            events.append("levels")

    class Pool:
        kv_cache_manager = KV()

        def begin_demand_step(self) -> None:
            events.append("begin")

        def post_demand_blocks(self, unfit: int) -> None:
            events.append("demand")

    class Scheduler:
        def __init__(self) -> None:
            self.kv_cache_manager = types.SimpleNamespace(block_pool=Pool())

        def schedule(self) -> str:
            events.append("schedule")
            return "output"

    module = types.ModuleType("vllm.v1.core.sched.scheduler")
    module.Scheduler = Scheduler  # type: ignore[attr-defined]
    patch = vllm_patches.SchedulerDemandPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    patch.detected_version = None
    assert patch.patch_schedule(module) is True
    scheduler = Scheduler()
    assert scheduler.schedule() == "output"
    assert events == ["begin", "reclaim", "schedule", "demand", "levels"]
    # The running-request part is left for the engine to preempt.
    assert getattr(scheduler, "_kvcached_reclaim_running_pages") == 3


def test_a_failed_release_does_not_fail_the_step(vllm_patches, monkeypatch):
    class KV:
        def reclaim_for_priority(self) -> int:
            raise RuntimeError("ledger unavailable")

    class Scheduler:
        def __init__(self) -> None:
            pool = types.SimpleNamespace(kv_cache_manager=KV())
            self.kv_cache_manager = types.SimpleNamespace(block_pool=pool)

        def schedule(self) -> str:
            return "output"

    module = types.ModuleType("vllm.v1.core.sched.scheduler")
    module.Scheduler = Scheduler  # type: ignore[attr-defined]
    patch = vllm_patches.SchedulerDemandPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    patch.detected_version = None
    assert patch.patch_schedule(module) is True
    assert Scheduler().schedule() == "output"
