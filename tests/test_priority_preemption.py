# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Preempting running requests for a higher priority's demand (#585, tier 3).

When a lower priority's empty pages and prefix cache cannot cover a higher
priority's demand, the ledger asks it for memory that running requests hold.
vLLM preempts them in its own order; SGLang retracts them through its decode
capacity gate. Only pages left empty count as released.
"""
from __future__ import annotations

import importlib
import sys
import threading
import types
from typing import Any
from unittest import mock

import pytest


class _Reserve:
    def __init__(self) -> None:
        self.reserved = 0
        self.trimmed: list[int] = []

    def get_num_reserved_pages(self) -> int:
        return self.reserved


class _KV:
    """A kvcached manager stand-in: each request owns whole pages, and the
    pages a preempted request frees land on the reserve."""

    def __init__(self, occupied: int) -> None:
        self.page_allocator = _Reserve()
        self.occupied = occupied
        self.cache_asks: list[int] = []

    def occupied_pages(self) -> int:
        return self.occupied

    def evict_cache_pages(self, pages: int) -> int:
        self.cache_asks.append(pages)
        return 0

    def release_empty_pages(self, pages: int) -> int:
        self.page_allocator.trimmed.append(pages)
        self.page_allocator.reserved -= pages
        return pages


class _Request:
    def __init__(self, request_id: str, pages: int, priority: int = 0,
                 arrival_time: float = 0.0, freeable: bool = True) -> None:
        self.request_id = request_id
        self.pages = pages
        self.priority = priority
        self.arrival_time = arrival_time
        self.freeable = freeable


class _Scheduler:
    def __init__(self, kv: _KV, running: list[_Request], policy: str = "FCFS") -> None:
        self.kv = kv
        self.running = running
        self.policy = types.SimpleNamespace(name=policy)
        self.requires_kv_delivery = False
        self.prev_step_scheduled_req_ids = {r.request_id for r in running}
        self.preempted: list[tuple[str, bool]] = []

    def _request_blocks_can_be_freed(self, request: _Request) -> bool:
        return request.freeable

    def _preempt_request(self, request: _Request, timestamp: float,
                         drop_stale_output: bool = False) -> None:
        self.preempted.append((request.request_id, drop_stale_output))
        self.kv.occupied -= request.pages
        self.kv.page_allocator.reserved += request.pages


@pytest.fixture
def vllm_patches(monkeypatch):
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
    return importlib.import_module("kvcached.integration.vllm.patches")


def test_vllm_preempts_the_last_admitted_until_enough_pages_empty(vllm_patches):
    kv = _KV(occupied=10)
    running = [_Request("a", 3), _Request("b", 1), _Request("c", 2)]
    scheduler = _Scheduler(kv, running)
    assert len(vllm_patches._kvcached_preempt_for_priority(scheduler, kv, 3)) == 2
    assert scheduler.preempted == [("c", False), ("b", False)]
    assert [r.request_id for r in scheduler.running] == ["a"]
    # Preempted requests may resume this step, so their tokens are resent.
    assert scheduler.prev_step_scheduled_req_ids == {"a"}
    # Blocks the preemption left in the prefix cache are evicted as well,
    # and pages that landed on the reserve are unmapped.
    assert kv.cache_asks == [1, 0]
    assert kv.page_allocator.trimmed == [3]


def test_vllm_priority_policy_preempts_the_lowest_priority(vllm_patches):
    kv = _KV(occupied=10)
    running = [_Request("low", 2, priority=5), _Request("high", 2, priority=0),
               _Request("late", 2, priority=5, arrival_time=1.0)]
    scheduler = _Scheduler(kv, running, policy="PRIORITY")
    vllm_patches._kvcached_preempt_for_priority(scheduler, kv, 3)
    assert [rid for rid, _ in scheduler.preempted] == ["late", "low"]


def test_vllm_stops_at_a_request_an_in_flight_step_still_writes(vllm_patches):
    kv = _KV(occupied=10)
    running = [_Request("a", 2), _Request("busy", 2, freeable=False)]
    scheduler = _Scheduler(kv, running)
    assert vllm_patches._kvcached_preempt_for_priority(scheduler, kv, 3) == []
    assert len(scheduler.running) == 2


def test_vllm_preempts_nothing_without_an_ask(vllm_patches):
    kv = _KV(occupied=10)
    scheduler = _Scheduler(kv, [_Request("a", 2)])
    assert vllm_patches._kvcached_preempt_for_priority(scheduler, kv, 0) == []
    assert scheduler.preempted == []


def test_vllm_step_preempts_after_releasing_empty_pages_and_cache(
        vllm_patches, monkeypatch):
    monkeypatch.setattr(vllm_patches, "enable_kvcached", lambda: True)
    calls: list[Any] = []
    monkeypatch.setattr(vllm_patches, "_kvcached_preempt_for_priority",
                        lambda scheduler, kv, pages: calls.append(pages))

    class KV:
        def reclaim_for_priority(self) -> int:
            return 4

    class Scheduler:
        def __init__(self) -> None:
            pool = types.SimpleNamespace(kv_cache_manager=KV())
            self.kv_cache_manager = types.SimpleNamespace(block_pool=pool)

        def schedule(self) -> str:
            calls.append("schedule")
            return "output"

    module = types.ModuleType("vllm.v1.core.sched.scheduler")
    module.Scheduler = Scheduler  # type: ignore[attr-defined]
    patch = vllm_patches.SchedulerDemandPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    patch.detected_version = None
    assert patch.patch_schedule(module) is True
    assert Scheduler().schedule() == "output"
    assert calls == [4, "schedule"]


def test_manager_counts_retired_pages_as_released(monkeypatch):
    try:
        import kvcached.vmm_ops  # noqa: F401
    except ImportError:
        stub = types.ModuleType("kvcached.vmm_ops")
        stub.PageAllocator = object  # type: ignore[attr-defined]
        stub.InternalPage = object  # type: ignore[attr-defined]
        stub.kv_tensors_created = lambda group_id=0: True  # type: ignore[attr-defined]
        stub.map_to_kv_tensors = lambda *a, **k: None  # type: ignore[attr-defined]
        stub.unmap_from_kv_tensors = lambda *a, **k: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", stub)
    import kvcached.kv_cache_manager as module
    from kvcached.locks import NoOpLock

    manager = object.__new__(module.KVCacheManager)
    manager._lock = NoOpLock()
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()
    manager.full_pages = {1: None, 2: None}
    manager.avail_pages = {3: None, 4: None}
    manager._retired_pages = [(7, [4])]
    assert manager.occupied_pages() == 3


class _GateManager:
    def __init__(self, occupied: int) -> None:
        self.page_allocator = _Reserve()
        self.occupied = occupied
        self.reserves: list[int] = []
        self.released: list[int] = []

    def occupied_pages(self) -> int:
        return self.occupied

    def try_to_reserve(self, need: int, packed: bool = False) -> bool:
        self.reserves.append(need)
        return True

    def release_empty_pages(self, pages: int) -> int:
        self.released.append(pages)
        return pages


def test_sglang_decode_gate_retracts_until_enough_pages_empty(monkeypatch):
    from kvcached.integration.sglang import patches

    manager = _GateManager(occupied=10)
    monkeypatch.setattr(patches, "_sglang_demand_manager", lambda scheduler: manager)
    scheduler = types.SimpleNamespace(token_to_kv_pool_allocator=None)
    allocator = types.SimpleNamespace(kvcached_allocator=manager)
    patches._sglang_begin_preemption(scheduler, 3)
    try:
        # SGLang's retract loop asks again after each retracted request.
        assert not patches._sglang_reserve_decode_capacity(allocator, None, 8, 4)
        manager.occupied = 8
        assert not patches._sglang_reserve_decode_capacity(allocator, None, 8, 4)
        manager.occupied = 7
        manager.page_allocator.reserved = 2
        assert patches._sglang_reserve_decode_capacity(allocator, None, 8, 4)
        assert manager.reserves == [4]
    finally:
        patches._sglang_end_preemption()
    # The emptied pages on the reserve are unmapped when the round ends.
    assert manager.released == [2]
    # No pending ask: the gate only reserves.
    assert patches._sglang_reserve_decode_capacity(allocator, None, 8, 4)


def test_sglang_does_not_retract_while_frees_are_grouped(monkeypatch):
    from kvcached.integration.sglang import patches

    manager = _GateManager(occupied=10)
    monkeypatch.setattr(patches, "_sglang_demand_manager", lambda scheduler: manager)
    monkeypatch.setattr(patches, "_frees_are_deferred", lambda allocator: True)
    patches._sglang_begin_preemption(types.SimpleNamespace(), 3)
    allocator = types.SimpleNamespace(kvcached_allocator=manager)
    try:
        assert patches._sglang_reserve_decode_capacity(allocator, None, 8, 4)
    finally:
        patches._sglang_end_preemption()


def test_vllm_preempts_the_owners_of_the_cheapest_pages_first(vllm_patches):
    # 4 blocks a page. Page 0 holds a's and b's blocks, page 1 only c's,
    # page 2 c's and d's; page 3 also holds a cached block nobody runs.
    blocks = {"a": [1, 12], "b": [2, 3], "c": [4, 5, 9], "d": [10]}
    occupancy = {0: 3, 1: 2, 2: 2, 3: 2}

    class KV(_KV):
        block_mem_size = 1
        page_size = 4

        def get_page_occupancy(self, pages: list[int]) -> dict[int, int]:
            return {page: occupancy[page] for page in pages}

    kv = KV(occupied=10)
    running = [_Request(rid, 1) for rid in "abcd"]
    scheduler = _Scheduler(kv, running)
    setattr(scheduler, "kv_cache_manager", types.SimpleNamespace(
        get_block_ids=lambda request_id: (blocks[request_id],)))
    victims = vllm_patches._kvcached_page_victims(scheduler, kv, running, 2)
    # Page 1 has one owner (c), then page 2 two (c, d); page 3 cannot empty.
    assert [r.request_id for r in victims] == ["c", "d"]
    vllm_patches._kvcached_preempt_for_priority(scheduler, kv, 2)
    assert [rid for rid, _ in scheduler.preempted] == ["c", "d"]



def test_vllm_step_does_not_resume_its_victims(vllm_patches, monkeypatch):
    # vLLM schedules no waiting request in a step that preempted one; a KV
    # connector asserts a preempted request has no load in flight.
    monkeypatch.setattr(vllm_patches, "enable_kvcached", lambda: True)
    victim = _Request("victim", 1)

    class Waiting(list):
        def remove_requests(self, requests: Any) -> None:
            for request in requests:
                self.remove(request)

        def prepend_request(self, request: Any) -> None:
            self.insert(0, request)

    def preempt(scheduler: Any, kv: Any, pages: int) -> list:
        scheduler.waiting.prepend_request(victim)
        return [victim]

    monkeypatch.setattr(vllm_patches, "_kvcached_preempt_for_priority", preempt)

    class KV:
        def reclaim_for_priority(self) -> int:
            return 1

    seen: list[list[str]] = []

    class Scheduler:
        def __init__(self) -> None:
            pool = types.SimpleNamespace(kv_cache_manager=KV())
            self.kv_cache_manager = types.SimpleNamespace(block_pool=pool)
            self.waiting = Waiting([_Request("queued", 1)])

        def schedule(self) -> str:
            seen.append([r.request_id for r in self.waiting])
            return "output"

    module = types.ModuleType("vllm.v1.core.sched.scheduler")
    module.Scheduler = Scheduler  # type: ignore[attr-defined]
    patch = vllm_patches.SchedulerDemandPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    patch.detected_version = None
    assert patch.patch_schedule(module) is True
    scheduler = Scheduler()
    assert scheduler.schedule() == "output"
    assert seen == [["queued"]]
    assert [r.request_id for r in scheduler.waiting] == ["victim", "queued"]
