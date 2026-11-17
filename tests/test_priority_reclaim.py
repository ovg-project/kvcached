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
from dataclasses import dataclass
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

    def free_pages(self, ids) -> None:
        self.reserved += len(ids)

    def _free_pages_with_receipt(self, ids) -> int:
        before = self.reserved
        self.free_pages(ids)
        return min(len(ids), self.reserved - before)

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
    plan = {"limit_bytes": 1600, "empty_bytes": 0, "cache_bytes": 0, "running_bytes": 0}
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
    from test_prefix_cache_shrink import FakePage

    manager.in_shrink = False
    manager.reserved_blocks = []
    manager.num_avail_blocks = 0
    manager.avail_pages = {}
    manager.full_pages = {}
    manager.page_allocator.group_indices_by_page = lambda indices, block_size: {
        pid: [i for i in indices if i // 4 == pid] for pid in {i // 4 for i in indices}}
    for pid in range(1, 6):
        page = FakePage(pid)
        page.init(1)
        page.alloc(4)
        manager.full_pages[pid] = page

    def evict(self, count):
        first = 6 - cache.pages
        cache.pages -= count
        cache.evicted.append(count)
        manager.free(list(range(first * 4, (first + count) * 4)))
        return count

    manager._register_cache_owner(cache.count_pages, types.MethodType(evict, cache))
    # The ledger asks for 2 empty pages, 3 of cache and 4 of running
    # requests (16 bytes a page, rounded up).
    plan.update(empty_bytes=32, cache_bytes=40, running_bytes=64)
    assert manager.reclaim_for_priority() == 4
    assert cache.evicted == [3]
    # The 3 evicted pages went to the reserve, so 2 + 3 pages are unmapped.
    assert manager.page_allocator.trimmed == [3, 2]
    assert manager.page_allocator.reserved == 1
    # What is left to release is posted at once: 1 empty page, 2 of cache.
    assert reclaimable.call_args == mock.call(16, 32, 0, 0)


def test_nothing_is_released_without_a_plan(manager_module):
    module, reclaimable, plan = manager_module
    manager = _manager(module, reserved=3)
    cache = _Cache(manager.page_allocator, pages=5)
    manager._register_cache_owner(cache.count_pages, cache.evict_pages)
    plan.update(running_bytes=16)
    assert manager.reclaim_for_priority() == 1
    assert cache.evicted == [] and manager.page_allocator.trimmed == []



def test_levels_are_written_only_when_they_change(manager_module):
    module, reclaimable, _ = manager_module
    manager = _manager(module, reserved=2)
    manager._retired_pages = [(1, [7, 8]), (2, [9])]
    manager._priority_retired_ids = {7, 8, 9}
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

    def poll(self, *, force=False) -> tuple:
        return ({"page_bundle_bytes": 16, "reclaim_limit_bytes": 1600, "reclaim_empty_bytes": 16,
                 "reclaim_cache_bytes": 33, "reclaim_running_bytes": 0},
                {"page_bundle_bytes": 24, "reclaim_limit_bytes": 2400, "reclaim_empty_bytes": 48,
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


@pytest.mark.parametrize("swa", [False, True], ids=["full", "swa"])
def test_sglang_cache_owner_counts_and_evicts_pages(monkeypatch, swa):
    from kvcached.integration.sglang import patches

    @dataclass
    class EvictParams:
        # SGLang v0.5.11 requires an explicit FULL demand, even for SWA-only requests.
        num_tokens: int
        swa_num_tokens: int = 0
        mamba_num: int = 0

    monkeypatch.setattr(patches, "_sglang_evict_params_cls", lambda: EvictParams)
    evicted: list[int] = []

    class Cache:
        page_size = 2
        tokens = 20

        def evictable_size(self) -> int:
            return self.tokens

        def swa_evictable_size(self) -> int:
            return self.tokens

    cache = Cache()

    def evict(target: Any, params: Any) -> None:
        tokens = params.swa_num_tokens if swa else params.num_tokens
        assert (params.num_tokens if swa else params.swa_num_tokens) == 0
        assert params.mamba_num == 0
        evicted.append(tokens)
        target.tokens -= min(target.tokens, tokens)

    class Scheduler:
        tree_cache = cache
        token_to_kv_pool_allocator: Any = None

    monkeypatch.setattr(patches, "evict_prefix_cache", evict)
    scheduler = Scheduler()
    manager = types.SimpleNamespace(page_size=8, block_mem_size=2)
    if swa:
        scheduler.token_to_kv_pool_allocator = types.SimpleNamespace(
            swa_attn_allocator=types.SimpleNamespace(kvcached_allocator=manager))
    reclaimer = patches._PrefixCacheShrinkReclaimer(scheduler, manager)
    # Eight tokens per page: 20 tokens fill two whole pages; the partial
    # third page is not counted.
    assert reclaimer.count_pages() == 2
    assert reclaimer.evict_pages(1) == 1 and evicted == [8]
    assert reclaimer.evict_pages(0) == 0 and evicted == [8]
    # Fewer tokens than a page count as none, so a requester's few cached
    # tokens cannot cover a page of its demand.
    cache.tokens = 5
    assert reclaimer.count_pages() == 0


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


def test_pages_retired_for_a_priority_are_not_reused_meanwhile(manager_module):
    module, _, plan = manager_module
    manager = _manager(module)
    manager.in_shrink = False
    manager._retired_pages = [(3, [5, 6])]
    plan.update(running_bytes=32)
    # Without a plan, a retired page may be reused (its unmap is cancelled).
    assert manager._unreusable_page_ids() == frozenset()
    manager.reclaim_for_priority()
    # Pages emptied for the plan wait for the release barrier instead.
    assert manager._unreusable_page_ids() == frozenset({5, 6})
    plan.update(running_bytes=0)
    manager.reclaim_for_priority()
    assert manager._unreusable_page_ids() == frozenset({5, 6})


@pytest.mark.parametrize("empty,cache,running", [(5, 0, 0), (0, 5, 0), (0, 0, 5)])
def test_subpage_excess_cannot_cross_min(manager_module, empty, cache, running):
    module, _, plan = manager_module
    manager = _manager(module)
    plan.update(limit_bytes=5, empty_bytes=empty, cache_bytes=cache, running_bytes=running)
    assert manager.reclaim_plan_pages() == {"empty": 0, "cache": 0, "running": 0}


def test_tier_rounding_shares_one_floor_limit(manager_module):
    module, _, plan = manager_module
    manager = _manager(module)
    plan.update(limit_bytes=32, empty_bytes=1, cache_bytes=1, running_bytes=1)
    assert manager.reclaim_plan_pages() == {"empty": 1, "cache": 1, "running": 0}


def test_worker_with_no_ask_still_limits_other_workers(manager_module):
    module, _, _ = manager_module
    manager = _manager(module)
    manager.world_size = 2
    manager._runtime_control = types.SimpleNamespace(worker_ipc=True, poll=lambda **kw: (
        {"page_bundle_bytes": 16, "reclaim_limit_bytes": 160, "reclaim_empty_bytes": 160},
        {"page_bundle_bytes": 24, "reclaim_limit_bytes": 47},
    ))
    assert manager.reclaim_plan_pages() == {"empty": 1, "cache": 0, "running": 0}


def test_pinned_reclaim_survives_zero_plan_until_native_refund(manager_module):
    module, reclaimable, plan = manager_module
    manager = _manager(module)
    manager.in_shrink = False
    manager.avail_pages = {7: types.SimpleNamespace(num_free_blocks=lambda: 4)}
    manager.num_avail_blocks = 4
    manager._retired_pages = [(1, [7])]
    manager._copy_pins = {7: 1}
    manager.page_allocator.free_pages = lambda ids: setattr(
        manager.page_allocator, "reserved", manager.page_allocator.reserved + len(ids))
    manager._maybe_finish_shrink = lambda: None
    plan.update(cache_bytes=16)
    manager.reclaim_for_priority()
    plan.update(cache_bytes=0)
    manager.reclaim_for_priority()
    manager.release_retired_pages_through(1)
    assert manager._unreusable_page_ids() == frozenset({7})
    assert manager.page_allocator.trimmed == []
    manager._copy_pins.clear()
    manager.release_retired_pages_through(1)
    assert manager._unreusable_page_ids() == frozenset()
    assert manager.page_allocator.trimmed == [1]
    assert manager.page_allocator.reserved == 0
    assert reclaimable.call_args == mock.call(0, 0, 0, 0)


@pytest.mark.parametrize("reserve", [False, True])
def test_only_promised_pages_are_guarded_and_refunded(manager_module, reserve):
    module, _, _ = manager_module
    manager = _manager(module)
    manager.in_shrink = False
    manager._priority_release_allowance = 1
    manager._retired_pages = [(1, [7, 8, 9])]
    manager._claim_priority_retired_pages([7, 8, 9])
    manager.avail_pages = {
        pid: types.SimpleNamespace(num_free_blocks=lambda: 4) for pid in (7, 8, 9)}
    manager.num_avail_blocks = 12
    manager._maybe_finish_shrink = lambda: None
    manager.page_allocator.free_pages = lambda ids: setattr(
        manager.page_allocator, "reserved", len(ids) if reserve else 0)
    assert manager._unreusable_page_ids() == frozenset({7})
    manager.release_retired_pages_through(1)
    assert manager.page_allocator.reserved == (2 if reserve else 0)
    assert manager.page_allocator.trimmed == ([1] if reserve else [])
    assert manager._priority_reserve_debt == 0
    assert manager._unreusable_page_ids() == frozenset()


def test_failed_trim_keeps_refund_debt_without_repeating_free(manager_module):
    module, _, _ = manager_module
    manager = _manager(module)
    manager.in_shrink = False
    manager._priority_retired_ids = {7}
    manager._retired_pages = [(1, [7])]
    manager.avail_pages = {7: types.SimpleNamespace(num_free_blocks=lambda: 4)}
    manager.num_avail_blocks = 4
    manager._maybe_finish_shrink = lambda: None
    manager._record_native_fatal = mock.Mock()
    freed = []

    def free(ids):
        freed.extend(ids)
        manager.page_allocator.reserved += len(ids)

    manager.page_allocator.free_pages = free
    trim = manager.page_allocator.trim
    manager.page_allocator.trim = mock.Mock(side_effect=RuntimeError("trim failed"))
    with pytest.raises(RuntimeError, match="trim failed"):
        manager.release_retired_pages_through(1)
    assert manager._retired_pages == []
    assert manager._priority_reserve_debt == 1
    manager.page_allocator.trim = trim
    manager.release_retired_pages_through(1)
    assert freed == [7]
    assert manager.page_allocator.reserved == 0
    assert manager._priority_reserve_debt == 0


def test_cache_eviction_claims_new_retirement_not_next_step_ask(manager_module):
    module, reclaimable, plan = manager_module
    manager = _manager(module)
    manager.in_shrink = False

    class AsyncCache:
        pages = 1

        def count_pages(self):
            return self.pages

        def evict_pages(self, count):
            self.pages = 0
            manager._retired_pages = [(1, [7])]
            manager._claim_priority_retired_pages([7])
            return 1

    cache = AsyncCache()
    manager._register_cache_owner(cache.count_pages, cache.evict_pages)
    plan.update(cache_bytes=16)
    manager.reclaim_for_priority()
    assert manager._unreusable_page_ids() == frozenset({7})
    assert reclaimable.call_args == mock.call(0, 0, 16, 0)
    plan.update(cache_bytes=0)
    manager.reclaim_for_priority()
    assert manager._unreusable_page_ids() == frozenset({7})
    assert reclaimable.call_args == mock.call(0, 0, 16, 0)


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("reserve_limit", [0, 10])
@pytest.mark.parametrize("deferred", [False, True])
def test_whole_victim_respects_min_before_native_reserve_trim(
        manager_module, reserve_limit, deferred, cache):
    from kvcached.kv_cache_manager import KVCacheManager
    from kvcached.locks import NoOpLock

    class Page:
        def __init__(self, page_id):
            self.page_id = page_id
            self.free_blocks = 0

        def free_batch(self, indices):
            self.free_blocks += len(indices)

        def num_free_blocks(self):
            return self.free_blocks

        def empty(self):
            return self.free_blocks == 1

    class NativeReserve(_PageAllocator):
        # Model PageAllocator::free_pages: overflow unmaps immediately,
        # before the engine's final trim call.
        mapped = 21

        def group_indices_by_page(self, indices, block_mem_size):
            return {index: [index] for index in indices}

        def free_pages(self, pages):
            retained = min(len(pages), max(0, reserve_limit - self.reserved))
            self.reserved += retained
            self.mapped -= len(pages) - retained

        def trim(self, max_pages: int = -1) -> int:
            released = self.reserved if max_pages < 0 else min(max_pages, self.reserved)
            self.reserved -= released
            self.mapped -= released
            return released

    manager: Any = object.__new__(KVCacheManager)
    manager._lock = NoOpLock()
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()
    initial_reserve = int(cache and reserve_limit > 0)
    manager.page_allocator = NativeReserve(initial_reserve)
    manager.page_allocator.mapped += initial_reserve
    manager.block_mem_size = 1
    manager.defer_physical_release = deferred
    manager._physical_release_epoch = 0
    manager._retired_pages = []
    manager.in_shrink = False
    manager.target_num_pages = None
    manager.reserved_blocks = []
    manager.avail_pages = {}
    manager.full_pages = {index: Page(index) for index in range(1, 22)}
    manager.num_avail_blocks = 0
    manager._operation_counters = None
    manager.reclaim_plan_pages = lambda: {
        "empty": initial_reserve, "cache": int(cache), "running": int(not cache)}
    manager.post_reclaimable_levels = lambda: None

    class WholeCache:
        def count_pages(self):
            return 20

        def evict_pages(self, count):
            manager.free(list(range(1, 21)))
            return 20

    owner = WholeCache()
    if cache:
        manager._register_cache_owner(owner.count_pages, owner.evict_pages)
    ask = manager.reclaim_for_priority()
    if not cache:
        reserved = manager.page_allocator.reserved
        manager.free(list(range(1, 21)))
        manager.release_empty_pages(min(ask, manager.page_allocator.reserved - reserved))
    if deferred:
        assert manager.page_allocator.mapped == 21
        manager.release_retired_pages_through(manager.capture_physical_release_marker())
    assert manager.page_allocator.mapped == 20
    # Protected empty pages remain backed and usable by this instance.
    assert manager.num_avail_blocks + manager.page_allocator.reserved == 19
    # Reusing one retained page must not attach its old protection to the
    # next request's ordinary free, even when the native page ID is reused.
    page_id = next(iter(manager._priority_retained_ids))
    page = manager.avail_pages.pop(page_id)
    page.free_blocks = 0
    manager.num_avail_blocks -= 1
    manager.full_pages[page_id] = page
    manager._priority_reclaim_active = False
    manager.free([page_id])
    if deferred:
        manager.release_retired_pages_through(manager.capture_physical_release_marker())
    assert page_id not in manager._priority_retained_ids
    assert page_id not in manager.avail_pages


def test_new_ask_can_claim_pinned_surplus_before_barrier(manager_module):
    module, _, plan = manager_module
    manager = _manager(module)
    manager.in_shrink = False
    manager._retired_pages = [(1, [7, 8, 9])]
    manager._priority_retired_ids = {7}
    manager._priority_retained_ids = {8, 9}
    manager._copy_pins = {7: 1, 8: 1, 9: 1}
    manager.avail_pages = {
        pid: types.SimpleNamespace(empty=lambda: True, num_free_blocks=lambda: 4)
        for pid in (7, 8, 9)}
    manager.num_avail_blocks = 12
    manager._maybe_finish_shrink = lambda: None
    freed = []

    def free(ids):
        freed.extend(ids)
        manager.page_allocator.reserved += len(ids)

    manager.page_allocator.free_pages = free
    plan.update(running_bytes=16)
    manager.reclaim_for_priority()
    assert len(manager._priority_retired_ids) == 2
    assert not manager._priority_retired_ids.intersection(manager._priority_retained_ids)
    manager.release_retired_pages_through(1)
    assert freed == []
    manager._copy_pins.clear()
    manager.release_retired_pages_through(1)
    assert len(freed) == 2
    assert manager._priority_retired_ids == set()
    assert len(manager.avail_pages) == 1
    assert manager._unreusable_page_ids() == frozenset()
    # The remaining, barrier-safe empty page can satisfy a later ask.
    assert manager.release_empty_pages(1) == 1
    assert manager.avail_pages == {}


def test_surplus_only_batch_still_waits_for_release_barrier(manager_module):
    module, _, _ = manager_module
    manager = _manager(module)
    manager.in_shrink = False
    manager._retired_pages = [(1, [7])]
    manager._priority_retained_ids = {7}
    manager.avail_pages = {
        7: types.SimpleNamespace(empty=lambda: True, num_free_blocks=lambda: 4)}
    manager.num_avail_blocks = 4
    manager._maybe_finish_shrink = lambda: None
    manager._record_native_fatal = mock.Mock()
    manager.physical_release_barrier = mock.Mock(side_effect=RuntimeError("barrier failed"))
    with pytest.raises(RuntimeError, match="barrier failed"):
        manager.release_retired_pages_through(1)
    assert manager._retired_pages == [(1, [7])]
    assert manager._priority_empty_ids() == []
    manager.physical_release_barrier.side_effect = None
    manager.release_retired_pages_through(1)
    assert manager.physical_release_barrier.call_count == 2
    assert manager._retired_pages == []
    assert manager._priority_empty_ids() == [7]


def test_explicit_shrink_releases_safe_retained_empty_pages(manager_module):
    module, _, _ = manager_module
    manager = _manager(module)
    manager.in_shrink = True
    manager.target_num_pages = 0
    manager.full_pages = {}
    manager._priority_retained_ids = {7}
    manager.avail_pages = {
        7: types.SimpleNamespace(empty=lambda: True, num_free_blocks=lambda: 4)}
    manager.num_avail_blocks = 4
    manager._get_num_alloced_blocks = lambda: 0
    manager.page_allocator.free_pages = lambda ids: setattr(
        manager.page_allocator, "reserved", manager.page_allocator.reserved + len(ids))
    manager.page_allocator.resize = lambda size: not manager.avail_pages
    manager._maybe_finish_shrink()
    assert not manager.in_shrink
    assert manager.target_num_pages is None
    assert manager.avail_pages == {}
    assert manager.page_allocator.reserved == 0


def _coupled_managers(monkeypatch, *, deferred=False, swa_layers=1):
    from test_prefix_cache_shrink import FakePage, _make_manager

    from kvcached import kv_cache_manager
    from kvcached.priority_reclaim import PriorityReclaimGroup

    members: tuple[Any, ...] = (_make_manager(deferred), _make_manager(deferred))
    ops = sys.modules.get("kvcached.vmm_ops")
    if ops is None:
        ops = types.ModuleType("kvcached.vmm_ops")
        monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", ops)

    def publish_pools(path, name, generation, pools):
        # Keep the existing aggregate assertions while recording every pool.
        post = getattr(ops, "set_kv_reclaimable", None)
        if post is not None:
            post(*(sum(pool[i] for pool in pools) for i in (2, 3, 4)), 0)
        return True

    monkeypatch.setattr(ops, "_publish_pool_reclaimable", publish_pools, raising=False)
    monkeypatch.setattr(kv_cache_manager, "InternalPage", FakePage)
    for i, member in enumerate(members):
        member.group_id = i
        member.world_size = 1
        member._uses_worker_ipc = False
        member._runtime_control = types.SimpleNamespace(
            worker_ipc=False, sources=({"gpu": "gpu", "instance": "model", "generation": 1,
                                      "pool_slot": i, "pool_owner": i + 1},),
            poll=lambda **kw: (), apply=lambda: None)
        assert member.alloc(7) == list(range(1, 8))
    members[1].num_layers = swa_layers
    return members, PriorityReclaimGroup(members)


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("swa_layers", [1, 2])
def test_coupled_victim_uses_one_byte_allowance(manager_module, monkeypatch, deferred, swa_layers):
    _, posted, plan = manager_module
    (full, swa), group = _coupled_managers(monkeypatch, deferred=deferred, swa_layers=swa_layers)
    plan.update(limit_bytes=4, running_bytes=1)  # A sub-page ask can release one fitting bundle.
    assert full.reclaim_for_priority() == 1
    before = sum(len(m.page_allocator.inuse) * m._memory_limit_page_bundle_bytes() for m in (full, swa))
    # Native hybrid-SWA free_segment frees SWA before full.
    swa.free([4, 5, 6, 7])
    full.free([4, 5, 6, 7])
    assert not group.pending()
    if deferred:
        assert sum(len(getattr(m, "_priority_retired_ids", ())) for m in (full, swa)) == 1
        for m in (full, swa):
            m.release_retired_pages_through(m.capture_physical_release_marker())
    after = sum(len(m.page_allocator.inuse) * m._memory_limit_page_bundle_bytes() for m in (full, swa))
    assert before - after == 4
    retained = full if swa_layers == 1 else swa
    assert retained._priority_empty_ids() == [1]
    retained.post_reclaimable_levels()
    assert posted.call_args == mock.call(retained._memory_limit_page_bundle_bytes(), 0, 0, 0)
    # A subsequent native empty-tier ask must reach the other pool too.
    bundle = retained._memory_limit_page_bundle_bytes()
    plan.update(limit_bytes=bundle, running_bytes=0, empty_bytes=bundle)
    full.reclaim_for_priority()
    assert not retained._priority_empty_ids()
    assert len(retained.page_allocator.inuse) == 1


def test_coupled_concurrent_frees_share_claim_and_keep_pool_ids_distinct(manager_module, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    _, _, plan = manager_module
    members, group = _coupled_managers(monkeypatch, deferred=True)
    plan.update(limit_bytes=4, running_bytes=4)
    members[0].reclaim_for_priority()
    start = threading.Barrier(2)

    def free(member):
        start.wait(timeout=5)
        member.free([4, 5, 6, 7])

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(free, members))
    assert sum(len(getattr(m, "_priority_retired_ids", ())) for m in members) == 1
    assert group.limit == 0
    for member in members:
        member.release_retired_pages_through(member.capture_physical_release_marker())
    assert sum(len(m.page_allocator.inuse) for m in members) == 3


def test_coupled_zero_plan_preserves_pinned_promises(manager_module, monkeypatch):
    _, posted, plan = manager_module
    (full, swa), group = _coupled_managers(monkeypatch, deferred=True)
    plan.update(limit_bytes=4, running_bytes=4)
    full.reclaim_for_priority()
    swa.free([4, 5, 6, 7])
    swa._copy_pins = {1: 1}
    plan.update(limit_bytes=0, running_bytes=0)
    full.reclaim_for_priority()
    swa.release_retired_pages_through(1)
    assert swa._unreusable_page_ids() == frozenset({1})
    group.publish()
    assert posted.call_args == mock.call(0, 0, 4, 0)
    swa._copy_pins.clear()
    swa.release_retired_pages_through(1)
    assert swa._unreusable_page_ids() == frozenset()
    assert posted.call_args == mock.call(0, 0, 0, 0)


def test_coupled_native_failure_does_not_replenish_allowance(manager_module, monkeypatch):
    _, _, plan = manager_module
    (full, swa), group = _coupled_managers(monkeypatch)
    plan.update(limit_bytes=4, running_bytes=4)
    full.reclaim_for_priority()
    monkeypatch.setattr(swa.page_allocator, "free_pages", mock.Mock(side_effect=RuntimeError("native failure")))
    with pytest.raises(RuntimeError, match="native failure"):
        swa.free([4, 5, 6, 7])
    assert group.failed and group.limit == 0
    full.free([4, 5, 6, 7])
    assert full._priority_empty_ids() == [1]
    with pytest.raises(RuntimeError, match="unconfirmed"):
        full.reclaim_for_priority()


def test_coupled_identity_rejects_different_instances(manager_module, monkeypatch):
    from kvcached.priority_reclaim import PriorityReclaimGroup

    members, _ = _coupled_managers(monkeypatch)
    members[1]._runtime_control.sources[0]["instance"] = "other"
    with pytest.raises(ValueError, match="one instance"):
        PriorityReclaimGroup(members)


def test_coupled_failed_trim_retries_debt_without_freeing_twice(manager_module, monkeypatch):
    _, _, plan = manager_module
    (full, swa), group = _coupled_managers(monkeypatch)
    plan.update(limit_bytes=4, running_bytes=4)
    full.reclaim_for_priority()
    allocator = swa.page_allocator
    allocator.reserved = 0
    allocator.get_num_reserved_pages = lambda: allocator.reserved
    freed = []

    def free(ids):
        freed.extend(ids)
        allocator.reserved += len(ids)

    allocator.free_pages = free
    allocator._free_pages_with_receipt = lambda ids: (free(ids), len(ids))[1]
    allocator.trim = mock.Mock(side_effect=RuntimeError("trim failed"))
    with pytest.raises(RuntimeError, match="trim failed"):
        swa.free([4, 5, 6, 7])
    assert swa._priority_reserve_debt == 1
    assert not group.failed and not group.pending()

    def trim(count):
        count = min(count, allocator.reserved)
        allocator.reserved -= count
        return count

    allocator.trim = trim
    group.finish()
    assert freed == [1] and swa._priority_reserve_debt == 0


def test_coupled_partial_allocation_rollback_keeps_shared_allowance(manager_module, monkeypatch):
    _, _, plan = manager_module
    (full, swa), group = _coupled_managers(monkeypatch)
    plan.update(limit_bytes=4, running_bytes=4)
    full.reclaim_for_priority()
    original = full.page_allocator.alloc_page
    attempts = []

    def allocate():
        attempts.append(1)
        if len(attempts) == 2:
            raise RuntimeError("physical pool exhausted")
        return original()

    monkeypatch.setattr(full.page_allocator, "alloc_page", allocate)
    assert full.alloc(8) is None
    assert len(full.page_allocator.inuse) == 2
    assert not group.pending() and group.limit == 0
    # The rollback already consumed the allowance. A later coupled free is retained.
    swa.free([4, 5, 6, 7])
    assert swa._priority_empty_ids() == [1]


def test_sglang_partial_cache_page_is_not_offered_but_evicts_when_asked(manager_module, monkeypatch):
    from test_prefix_cache_shrink import FakePage, _make_manager

    from kvcached import kv_cache_manager
    from kvcached.integration.sglang import patches

    _, posted, plan = manager_module
    monkeypatch.setattr(kv_cache_manager, "InternalPage", FakePage)
    manager = _make_manager(False)
    assert manager.alloc(5) == [1, 2, 3, 4, 5]
    manager.free([1, 2, 3])
    manager.world_size = 1
    manager._uses_worker_ipc = False

    class Cache:
        page_size = 1
        tokens = 2

        def evictable_size(self):
            return self.tokens

        def evict(self, params):
            manager.free([4, 5])
            self.tokens = 0

    class Scheduler:
        tree_cache = Cache()

    scheduler = Scheduler()
    reclaimer = patches._PrefixCacheShrinkReclaimer(scheduler, manager)
    manager._register_cache_owner(reclaimer.count_pages, reclaimer.evict_pages)
    # Two cached tokens do not fill a page, so none is offered.
    assert reclaimer.count_pages() == 0
    manager.post_reclaimable_levels()
    posted.assert_not_called()  # all levels are still zero
    # A plan made while the cache still held a page evicts what is left.
    plan.update(limit_bytes=4, cache_bytes=4)
    manager.reclaim_for_priority()
    assert len(manager.page_allocator.inuse) == 1
    assert scheduler.tree_cache.tokens == 0


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("coupled", [False, True])
def test_release_receipt_excludes_concurrent_preallocation(manager_module, monkeypatch, deferred, coupled):
    _, _, plan = manager_module
    (manager, sibling), group = _coupled_managers(monkeypatch, deferred=deferred)
    if not coupled:
        del manager._priority_reclaim_group
        del sibling._priority_reclaim_group
    allocator = manager.page_allocator
    allocator.reserved = 0
    allocator.get_num_reserved_pages = lambda: allocator.reserved
    trim = mock.Mock(return_value=0)
    allocator.trim = trim
    native_free = allocator.free_pages

    def receipt(ids):
        native_free(ids)  # All input pages unmap; none land in the reserve.
        allocator.reserved += 2  # An already-charged preallocation finishes.
        return 0

    allocator._free_pages_with_receipt = receipt
    plan.update(limit_bytes=4, running_bytes=4)
    manager.reclaim_for_priority()
    manager.free([4, 5, 6, 7])
    if deferred:
        manager.release_retired_pages_through(manager.capture_physical_release_marker())
    assert allocator.reserved == 2
    trim.assert_not_called()
    assert manager._priority_reserve_debt == 0
    assert not manager._priority_retired_ids
    if coupled:
        assert not group.pending()
    else:
        assert manager._priority_release_allowance == 0


def test_cache_reclaim_does_not_trim_preallocation_completion(manager_module):
    module, _, plan = manager_module
    manager = _manager(module)

    class Cache:
        def count_pages(self):
            return 1

        def evict_pages(self, count):
            manager.page_allocator.reserved += 2
            return 0  # No cache page was released.

    cache = Cache()
    manager._register_cache_owner(cache.count_pages, cache.evict_pages)
    plan.update(limit_bytes=16, cache_bytes=16)
    assert manager.reclaim_for_priority() == 0
    assert manager.page_allocator.reserved == 2
    assert manager.page_allocator.trimmed == []


def test_pages_freed_after_the_step_in_flight_post_as_releasing(manager_module):
    module, reclaimable, _ = manager_module
    manager = _manager(module)
    manager._priority_release_allowance = 2
    manager.note_pending_release(2)
    manager.post_reclaimable_levels()
    reclaimable.assert_called_once_with(0, 0, 32, 0)
    assert manager._priority_release_allowance == 0


def test_coupled_publishes_pool_tiers_when_aggregate_does_not_change(manager_module, monkeypatch):
    members, group = _coupled_managers(monkeypatch)
    ops = sys.modules["kvcached.vmm_ops"]
    published = mock.Mock(return_value=True)
    monkeypatch.setattr(ops, "_publish_pool_reclaimable", published)
    levels = [[1, 0, 0], [0, 0, 0]]
    for i, member in enumerate(members):
        monkeypatch.setattr(member, "_reclaimable_levels", lambda i=i: levels[i])
    group.publish()
    first = published.call_args.args[-1]
    levels.reverse()
    group.publish()
    second = published.call_args.args[-1]
    assert published.call_count == 2
    assert first != second
    assert sum(pool[2] for pool in first) == sum(pool[2] for pool in second)
    assert [pool[:2] for pool in second] == [(0, 1), (1, 2)]
    group.publish()
    assert published.call_count == 2


def test_coupled_rejected_pool_publication_keeps_retryable_snapshot(manager_module, monkeypatch):
    _, group = _coupled_managers(monkeypatch)
    published = mock.Mock(return_value=False)
    monkeypatch.setattr(sys.modules["kvcached.vmm_ops"], "_publish_pool_reclaimable", published)
    with pytest.raises(RuntimeError, match="participants changed"):
        group.publish()
    assert group.posted is None
    published.return_value = True
    group.publish()
    assert group.posted is not None


def test_coupled_reclaim_refreshes_pool_metadata_after_legacy_publication(manager_module, monkeypatch):
    _, _, plan = manager_module
    _, group = _coupled_managers(monkeypatch)
    published = mock.Mock(return_value=True)
    monkeypatch.setattr(sys.modules["kvcached.vmm_ops"], "_publish_pool_reclaimable", published)
    group.publish()
    before = published.call_count
    plan.update(limit_bytes=0)
    group.reclaim()
    assert published.call_count == before + 1
