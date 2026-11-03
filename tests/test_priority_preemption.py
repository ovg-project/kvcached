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

    def note_pending_release(self, pages: int) -> None:
        self.pending = pages

    def _refund_priority_reserve(self) -> None:
        pass  # The manager refunds successful frees from their native receipts.

    def post_reclaimable_levels(self) -> None:
        pass


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
        self.deferred_frees: list = []
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
        if request.freeable:  # otherwise vLLM frees after the step in flight
            self.kv.occupied -= request.pages
            self.kv.page_allocator.trimmed.append(request.pages)
        else:
            self.deferred_frees.append((1, [request.request_id]))


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
    # Blocks the preemption left in the prefix cache are evicted as well,
    # and pages that landed on the reserve are unmapped.
    assert kv.cache_asks == [1, 0]
    assert kv.page_allocator.trimmed == [2, 1]


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
        self.refund_debt = 0

    def occupied_pages(self) -> int:
        return self.occupied

    def try_to_reserve(self, need: int, packed: bool = False) -> bool:
        self.reserves.append(need)
        return True

    def release_empty_pages(self, pages: int) -> int:
        self.released.append(pages)
        return pages

    def _refund_priority_reserve(self) -> None:
        self.released.append(self.refund_debt)
        self.page_allocator.reserved -= self.refund_debt
        self.refund_debt = 0

    def post_reclaimable_levels(self) -> None:
        pass


def test_sglang_decode_gate_retracts_until_enough_pages_empty(monkeypatch):
    from kvcached.integration.sglang import patches

    manager = _GateManager(occupied=10)
    monkeypatch.setattr(patches, "_sglang_demand_manager", lambda scheduler: manager)
    scheduler = types.SimpleNamespace(token_to_kv_pool_allocator=None)
    allocator = types.SimpleNamespace(kvcached_allocator=manager)
    patches._sglang_begin_preemption(scheduler, 3)
    try:
        # SGLang's retract loop asks again after each retracted request.
        assert not patches._sglang_reserve_decode_capacity(allocator, None, 8, 4, requests=[object(), object()])
        manager.occupied = 8
        assert not patches._sglang_reserve_decode_capacity(allocator, None, 8, 4, requests=[object(), object()])
        manager.occupied = 7
        manager.page_allocator.reserved = 3  # Includes one concurrent preallocation.
        manager.refund_debt = 2
        assert patches._sglang_reserve_decode_capacity(allocator, None, 8, 4, requests=[object(), object()])
        assert manager.reserves == [4]
    finally:
        patches._sglang_end_preemption()
    # The emptied pages on the reserve are unmapped when the round ends.
    assert manager.released == [2]
    assert manager.page_allocator.reserved == 1
    # No pending ask: the gate only reserves.
    assert patches._sglang_reserve_decode_capacity(allocator, None, 8, 4, requests=[object(), object()])


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


@pytest.mark.parametrize("released_pool", [0, 1])
@pytest.mark.parametrize("deferred", [False, True])
def test_sglang_coupled_preemption_stops_on_either_pools_claim(monkeypatch, released_pool, deferred):
    from test_priority_reclaim import _coupled_managers

    from kvcached.integration.sglang import patches

    members, group = _coupled_managers(monkeypatch, deferred=deferred)
    ops = sys.modules["kvcached.vmm_ops"]
    monkeypatch.setattr(ops, "set_kv_reclaimable", lambda *a: None, raising=False)
    monkeypatch.setattr(ops, "kv_reclaim", lambda *a: dict(
        limit_bytes=4, empty_bytes=0, cache_bytes=0, running_bytes=4), raising=False)
    full = members[0]
    allocator = types.SimpleNamespace(kvcached_allocator=full)
    scheduler = types.SimpleNamespace(token_to_kv_pool_allocator=allocator)
    monkeypatch.setattr(patches, "_sglang_demand_manager", lambda _: full)
    pages = full.reclaim_for_priority()
    patches._sglang_begin_preemption(scheduler, pages)
    try:
        assert patches._sglang_preemption_pending(full)
        occupied = full.occupied_pages()
        members[released_pool].free([4, 5, 6, 7])
        if released_pool == 1:
            assert full.occupied_pages() == occupied
        assert not patches._sglang_preemption_pending(full)
        assert not group.pending()
    finally:
        patches._sglang_end_preemption()
    # A deferred claim remains guarded until its acknowledged batch completes.
    member = members[released_pool]
    if deferred:
        assert member._unreusable_page_ids() == frozenset({1})
        member.release_retired_pages_through(member.capture_physical_release_marker())
    assert len(member.page_allocator.inuse) == 1


def test_sglang_coupled_grouped_frees_keep_session_without_forced_retraction(monkeypatch):
    from test_priority_reclaim import _coupled_managers

    from kvcached.integration.sglang import patches

    (full, _), group = _coupled_managers(monkeypatch)
    group.active, group.limit, group.asks = True, 4, [0, 0, 4]
    monkeypatch.setattr(patches, "_sglang_demand_manager", lambda _: full)
    monkeypatch.setattr(patches, "_frees_are_deferred", lambda _: True)
    scheduler = types.SimpleNamespace(token_to_kv_pool_allocator=None)
    assert patches._sglang_reclaim_for_priority(scheduler) == 0
    patches._sglang_begin_preemption(scheduler, 1)
    assert not patches._sglang_preemption_pending(full)
    assert group.pending() and group.limit == 4


def _paged(blocks: dict, occupancy: dict, freeable: bool = True) -> tuple:
    """Four blocks a page; `blocks` maps a request to its block ids."""

    class KV(_KV):
        block_mem_size = 1
        page_size = 4

        def get_page_occupancy(self, pages: list[int]) -> dict[int, int]:
            return {page: occupancy[page] for page in pages}

    kv = KV(occupied=10)
    running = [_Request(rid, 1, freeable=freeable) for rid in blocks]
    scheduler = _Scheduler(kv, running)
    references: dict[int, int] = {}
    for ids in blocks.values():
        for bid in ids:
            references[bid] = references.get(bid, 0) + 1
    setattr(scheduler, "kv_cache_manager", types.SimpleNamespace(
        get_block_ids=lambda request_id: (blocks[request_id],),
        block_pool=types.SimpleNamespace(blocks={
            bid: types.SimpleNamespace(ref_cnt=count) for bid, count in references.items()})))
    return kv, running, scheduler


def test_vllm_preempts_the_owners_of_the_cheapest_pages_first(vllm_patches):
    # Page 0 holds a's and b's blocks, page 1 only c's, page 2 c's and d's;
    # page 3 also holds a cached block nobody runs, so it cannot empty.
    kv, running, scheduler = _paged(
        {"a": [1, 12], "b": [2, 3], "c": [4, 5, 9], "d": [10]},
        {0: 3, 1: 2, 2: 2, 3: 2})
    targets = vllm_patches._kvcached_target_pages(scheduler, kv, running)
    assert [(page, [r.request_id for r in owners]) for page, owners in targets] == [
        (1, ["c"]), (2, ["c", "d"]), (0, ["a", "b"])]
    vllm_patches._kvcached_preempt_for_priority(scheduler, kv, 2)
    assert [rid for rid, _ in scheduler.preempted] == ["c", "d"]


def test_vllm_page_selection_preserves_request_priority(vllm_patches):
    kv, running, scheduler = _paged(
        {"low": [4], "late": [5], "high": [8]}, {1: 2, 2: 1})
    scheduler.policy = types.SimpleNamespace(name="PRIORITY")
    for request in running[:2]:
        request.priority = 5
    running[1].arrival_time = 1.0
    # The high-priority request owns the cheapest page, but native priority
    # order must still choose the latest of the lowest-priority requests.
    vllm_patches._kvcached_preempt_for_priority(scheduler, kv, 1)
    assert [rid for rid, _ in scheduler.preempted] == ["late"]
    assert [request.request_id for request in scheduler.running] == ["low", "high"]


def test_vllm_counts_pages_freed_after_the_step_in_flight(vllm_patches):
    # With an asynchronous KV connector vLLM frees preempted requests' blocks
    # after the step in flight: their pages count as released, and the
    # ledger hears of them as releasing until then.
    kv, running, scheduler = _paged(
        {"a": [1, 12], "b": [2, 3], "c": [4, 5, 9], "d": [10]},
        {0: 3, 1: 2, 2: 2, 3: 2}, freeable=False)
    vllm_patches._kvcached_preempt_for_priority(scheduler, kv, 2)
    assert [rid for rid, _ in scheduler.preempted] == ["c", "d"]
    assert kv.occupied == 10 and getattr(kv, "pending") == 2


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


def test_vllm_preemption_does_not_trim_concurrent_preallocation(vllm_patches, monkeypatch):
    kv = _KV(occupied=10)
    scheduler = _Scheduler(kv, [_Request("victim", 1)])
    native_preempt = scheduler._preempt_request

    def preempt(request, timestamp, drop_stale_output=False):
        native_preempt(request, timestamp, drop_stale_output)
        kv.page_allocator.reserved += 2

    monkeypatch.setattr(scheduler, "_preempt_request", preempt)
    vllm_patches._kvcached_preempt_for_priority(scheduler, kv, 1)
    assert kv.page_allocator.reserved == 2
    assert kv.page_allocator.trimmed == [1]


@pytest.mark.parametrize("requests", [None, [], [types.SimpleNamespace()],
                                     [types.SimpleNamespace(beam_group=object()), types.SimpleNamespace()]])
def test_sglang_priority_does_not_force_a_terminal_or_beam_abort(monkeypatch, requests):
    from kvcached.integration.sglang import patches

    manager = _GateManager(10)
    allocator = types.SimpleNamespace(kvcached_allocator=manager)
    monkeypatch.setattr(patches, "_sglang_preemption_pending", lambda _: True)
    monkeypatch.setattr(patches._SGLANG_DEMAND, "preempt", (manager, 9), raising=False)
    monkeypatch.setattr(patches, "_sglang_group_capacity_miss", lambda miss: miss)
    assert patches._sglang_reserve_decode_capacity(allocator, None, 1, 1, requests=requests)
    assert manager.reserves == [1]
    monkeypatch.setattr(manager, "try_to_reserve", lambda *a, **kw: False)
    # Actual OOM still follows the native failure path.
    assert not patches._sglang_reserve_decode_capacity(allocator, None, 1, 1, requests=requests)


def test_vllm_target_pages_exclude_references_outside_running(vllm_patches):
    kv, _, scheduler = _paged({"a": [4], "b": [8]}, {1: 1, 2: 1})
    scheduler.kv_cache_manager.block_pool.blocks[4].ref_cnt = 2  # A waiting KV load holds it too.
    targets = vllm_patches._kvcached_target_pages(scheduler, kv, scheduler.running)
    assert [(page, [req.request_id for req in owners]) for page, owners in targets] == [(2, ["b"])]
    vllm_patches._kvcached_preempt_for_priority(scheduler, kv, 1)
    assert scheduler.preempted == [("b", False)]
    assert kv.occupied == 9


@pytest.mark.parametrize("connector", [False, True])
def test_vllm_external_preemption_reaches_output_and_connector(vllm_patches, monkeypatch, connector):
    monkeypatch.setattr(vllm_patches, "enable_kvcached", lambda: True)
    victim = _Request("victim", 1)

    class Waiting(list):
        def remove_requests(self, requests):
            for request in requests:
                self.remove(request)

        def prepend_request(self, request):
            self.insert(0, request)

    def preempt(scheduler, kv, pages):
        scheduler.waiting.prepend_request(victim)
        return [victim]

    monkeypatch.setattr(vllm_patches, "_kvcached_preempt_for_priority", preempt)
    seen = []

    class Scheduler:
        def __init__(self):
            self.kv_cache_manager = types.SimpleNamespace(block_pool=types.SimpleNamespace(
                kv_cache_manager=types.SimpleNamespace(reclaim_for_priority=lambda: 1)))
            self.waiting = Waiting()

        def _build_kv_connector_meta(self, connector, output):
            seen.append(set(output.preempted_req_ids))
            return None

        def schedule(self):
            output = types.SimpleNamespace(preempted_req_ids={"native-victim"})
            if connector:
                self._build_kv_connector_meta(None, output)
            return output

    module = types.ModuleType("vllm.v1.core.sched.scheduler")
    module.Scheduler = Scheduler  # type: ignore[attr-defined]
    assert vllm_patches.SchedulerDemandPatch().patch_schedule(module)
    scheduler = Scheduler()
    output = scheduler.schedule()
    assert output.preempted_req_ids == {"native-victim", "victim"}
    assert seen == ([{"native-victim", "victim"}] if connector else [])
    assert scheduler.waiting == [victim]
    assert not getattr(scheduler, "_kvcached_priority_preempted_ids")


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("unlocked", [False, True])
def test_sglang_reclaims_unlocked_cache_before_next_victim(monkeypatch, composite, unlocked):
    from kvcached.integration.sglang import patches

    manager = _GateManager(10)
    evictions = []

    def evict(pages):
        evictions.append(pages)
        if unlocked:
            manager.occupied -= pages
        return pages if unlocked else 0

    monkeypatch.setattr(manager, "evict_cache_pages", evict, raising=False)
    monkeypatch.setattr(patches._SGLANG_DEMAND, "preempt", (manager, 9), raising=False)
    monkeypatch.setattr(patches, "_sglang_group_capacity_miss", lambda miss: miss)
    requests = [types.SimpleNamespace(), types.SimpleNamespace()]
    if composite:
        calls = []

        class Base:
            def check_decode_capacity(self, **kwargs):
                calls.append(kwargs)
                return True

        class Composite(Base):
            full_attn_allocator = types.SimpleNamespace(kvcached_allocator=manager)
            swa_attn_allocator = None

        module = types.ModuleType("swa")
        module.SWATokenToKVPoolAllocator = Composite  # type: ignore[attr-defined]
        assert patches.ElasticSWAAllocatorPatch().patch_swa_composite_misses(module)
        result = Composite().check_decode_capacity(num_tokens=2, tree_cache=None, requests=requests)
        assert len(calls) == int(unlocked)
    else:
        allocator = types.SimpleNamespace(kvcached_allocator=manager)
        result = patches._sglang_reserve_decode_capacity(allocator, None, 2, 2, requests=requests)
    assert result is unlocked
    assert evictions == [1]


@pytest.mark.parametrize("ordinary_first", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_priority_engine_deferred_allowance_survives_zero_plans(monkeypatch, ordinary_first, deferred):
    from test_prefix_cache_shrink import FakePage, _make_manager

    from kvcached import kv_cache_manager

    monkeypatch.setattr(kv_cache_manager, "InternalPage", FakePage)
    manager = _make_manager(deferred)
    assert manager.alloc(11) == list(range(1, 12))  # Null + three native pages.
    monkeypatch.setattr(manager, "post_reclaimable_levels", lambda: None)
    plan = dict(empty=0, cache=0, running=1)
    monkeypatch.setattr(manager, "reclaim_plan_pages", lambda: plan.copy())
    assert manager.reclaim_for_priority() == 1
    manager.note_pending_release(1)
    assert manager._pending_release_pages == 1
    plan["running"] = 0
    for _ in range(2):
        manager.reconcile_pending_release(True)
        assert manager.reclaim_for_priority() == 0
    if ordinary_first:
        manager.free([8, 9, 10, 11])
        assert manager._pending_release_pages == 0
        manager.reconcile_pending_release(True)
        assert manager.reclaim_for_priority() == 0
    # A single deferred request can empty two pages after its one-page ask.
    manager.free([4, 5, 6, 7] if ordinary_first else list(range(4, 12)))
    if deferred:
        manager.release_retired_pages_through(manager.capture_physical_release_marker())
    assert len(manager.page_allocator.inuse) == 2
    assert manager._pending_release_pages == 0
    assert len(manager._priority_empty_ids()) == 1
    manager.reconcile_pending_release(False)
    manager.reclaim_for_priority()
    assert not manager._priority_reclaim_active


def test_vllm_only_priority_deferred_tickets_keep_guard(vllm_patches, monkeypatch):
    monkeypatch.setattr(vllm_patches, "enable_kvcached", lambda: True)
    priority, ordinary = [object()], [object()]
    seen: list[bool] = []

    class Scheduler:
        def __init__(self):
            kv = types.SimpleNamespace(reclaim_for_priority=lambda: 0,
                                       reconcile_pending_release=seen.append)
            self.kv_cache_manager = types.SimpleNamespace(block_pool=types.SimpleNamespace(kv_cache_manager=kv))
            self._kvcached_priority_deferred = [priority]
            self.deferred_frees = [(1, priority), (2, ordinary)]

        def schedule(self):
            return None

    module = types.ModuleType("vllm.v1.core.sched.scheduler")
    module.Scheduler = Scheduler  # type: ignore[attr-defined]
    assert vllm_patches.SchedulerDemandPatch().patch_schedule(module)
    scheduler = Scheduler()
    scheduler.schedule()
    scheduler.deferred_frees.pop(0)
    scheduler.schedule()
    assert seen == [True, False]
    assert scheduler._kvcached_priority_deferred == []


def test_vllm_deferred_guard_survives_later_eviction_error(vllm_patches, monkeypatch):
    kv, _, scheduler = _paged({"a": [4]}, {1: 1}, freeable=False)
    monkeypatch.setattr(kv, "evict_cache_pages", mock.Mock(side_effect=RuntimeError("eviction failed")))
    with pytest.raises(RuntimeError, match="eviction failed"):
        vllm_patches._kvcached_preempt_for_priority(scheduler, kv, 1)
    assert getattr(scheduler, "_kvcached_priority_deferred") == [scheduler.deferred_frees[0][1]]
    assert kv.pending == 0  # Guard established; no unproven page promise published.


@pytest.mark.parametrize("composite", [False, True])
def test_sglang_legacy_decode_context_preserves_selected_requests(monkeypatch, composite):
    from kvcached.integration.sglang import patches

    manager = _GateManager(10)
    sub = types.SimpleNamespace(kvcached_allocator=manager)
    monkeypatch.setattr(patches._SGLANG_DEMAND, "preempt", (manager, 9), raising=False)
    monkeypatch.setattr(patches._SGLANG_DEMAND, "decode_requests", None, raising=False)
    monkeypatch.setattr(patches, "_sglang_group_capacity_miss", lambda miss: miss)

    class Allocator:
        def check_decode_capacity(self, *, num_tokens, tree_cache):
            if composite:
                return True  # The native SWA base gate has no requests argument.
            return patches._sglang_reserve_decode_capacity(sub, tree_cache, num_tokens, num_tokens)

    if composite:
        Allocator.full_attn_allocator = sub  # type: ignore[attr-defined]
        Allocator.swa_attn_allocator = None  # type: ignore[attr-defined]
        module = types.ModuleType("swa")
        module.SWATokenToKVPoolAllocator = Allocator  # type: ignore[attr-defined]
        assert patches.ElasticSWAAllocatorPatch().patch_swa_composite_misses(module)
    allocator = Allocator()

    class Batch:
        reqs = [types.SimpleNamespace(beam_group=None) for _ in range(2)]

        def check_decode_mem(self, selected_indices=None):
            return allocator.check_decode_capacity(num_tokens=2, tree_cache=None)

    module = types.ModuleType("sglang.srt.managers.schedule_batch")
    module.ScheduleBatch = Batch  # type: ignore[attr-defined]
    assert patches.ScheduleBatchDecodeContextPatch().patch_check_decode_mem(module)
    batch = Batch()
    assert not batch.check_decode_mem()
    assert batch.check_decode_mem([1])  # Native retraction's final survivor stays healthy.
    batch.reqs[0].beam_group = object()
    assert batch.check_decode_mem()
    assert patches._SGLANG_DEMAND.decode_requests is None
    assert allocator.check_decode_capacity(num_tokens=2, tree_cache=None)


def test_sglang_legacy_decode_context_restored_after_error(monkeypatch):
    from kvcached.integration.sglang import patches

    previous = [object()]
    monkeypatch.setattr(patches._SGLANG_DEMAND, "decode_requests", previous, raising=False)

    class Batch:
        reqs = [object(), object()]

        def check_decode_mem(self, selected_indices=None):
            assert patches._SGLANG_DEMAND.decode_requests == [self.reqs[1]]
            raise RuntimeError("native gate failed")

    module = types.ModuleType("sglang.srt.managers.schedule_batch")
    module.ScheduleBatch = Batch  # type: ignore[attr-defined]
    assert patches.ScheduleBatchDecodeContextPatch().patch_check_decode_mem(module)
    with pytest.raises(RuntimeError, match="native gate failed"):
        Batch().check_decode_mem([1])
    assert patches._SGLANG_DEMAND.decode_requests is previous
