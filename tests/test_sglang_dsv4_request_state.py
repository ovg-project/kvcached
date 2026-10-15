# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for physical request-state admission and native row ownership."""

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest
from test_sglang_dsv4_native_backing import Manager, SWABacking


@pytest.fixture
def request_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "kvcached.integration.sglang.dsv4_backing",
                        SimpleNamespace(SWABacking=SWABacking))
    path = Path(__file__).parents[1] / "kvcached/integration/sglang/dsv4_request_state.py"
    spec = importlib.util.spec_from_file_location("request_state_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeRows:
    """The public 0.5.20 row API, including ordered tail allocation."""

    _on_alloc_rows: Any = None

    def __init__(self, size=8):
        self.size = size
        self.free_slots = list(range(1, size + 1))
        self.generation = [0] * (size + 1)
        self.freed = []

    def available_size(self):
        return len(self.free_slots)

    def alloc_rows(self, n):
        if n > len(self.free_slots):
            return None
        if not n:
            return []
        selected = self.free_slots[-n:]
        del self.free_slots[-n:]
        for row in selected:
            self.generation[row] += 1
        if self._on_alloc_rows:
            self._on_alloc_rows(selected)
        return selected

    def free_rows(self, rows):
        self.freed.append(list(rows))
        self.free_slots.extend(rows)

    def clear(self):
        self.free_slots = list(range(1, self.size + 1))
        self.generation = [0] * (self.size + 1)


def bind(module, *, banks=1, second=True):
    pool = NativeRows()
    groups = []
    for _ in range(2 if second else 1):
        backing = SWABacking(Manager(30), native_pages=32, native_page_bytes=2,
                             physical_page_bytes=8)
        groups.append(dict(backing=backing, bank_stride=10 if banks > 1 else 0,
                           bank_count=banks))
    owner = module.RequestStateBacking(pool, groups)
    owner.bind()
    return pool, owner, [g["backing"] for g in groups]


def test_admission_preserves_native_order_and_prevents_late_capacity_miss(request_module):
    pool, owner, backings = bind(request_module)
    before = list(pool.free_slots)
    assert owner.reserve(owner.admission_rows) == 1
    assert pool.free_slots == before and owner.standby == {8}
    assert pool.generation[8] == 1
    for backing in backings:
        backing.manager.budget = len(backing.manager.live)
    assert pool.alloc_rows(1) == [8]
    assert pool.generation[8] == 2
    assert owner.active == {8} and not owner.standby
    assert pool.alloc_rows(0) == []
    pool.free_rows([8])
    assert pool.free_slots == before and owner.standby == {8}
    assert owner.reserve(owner.admission_rows) == 1


@pytest.mark.parametrize("banks", [1, 3])
def test_multi_pool_and_bank_miss_rolls_back_without_consuming_rows(request_module, banks):
    pool, owner, backings = bind(request_module, banks=banks)
    backings[-1].manager.budget = 0
    before = list(pool.free_slots)
    assert owner.reserve(owner.admission_rows) == 0
    assert pool.free_slots == before and not owner.active and not owner.standby
    assert all(not b.live and not b.refs and not b.manager.live for b in backings)
    backings[-1].manager.budget = 30
    assert owner.reserve(owner.admission_rows) == 1
    assert backings[0].live == ({8, 18, 28} if banks == 3 else {8})


def test_free_vector_keeps_native_tail_and_reclaims_surplus(request_module):
    pool, owner, backings = bind(request_module)
    owner.admission_rows = 4
    assert owner.reserve(owner.admission_rows) == 4
    rows = pool.alloc_rows(4)
    assert rows == [5, 6, 7, 8]
    owner.admission_rows = 1
    pool.free_rows([8, 6, 5, 7])
    assert pool.free_slots == [1, 2, 3, 4, 8, 6, 5, 7]
    assert owner.standby == {7} and not owner.active
    assert all(b.live == {7} for b in backings)
    assert pool.alloc_rows(1) == [7]
    assert owner.active == {7}
    pool.clear()
    assert pool.free_slots == list(range(1, 9))
    assert not owner.active and not owner.standby
    assert all(not b.live and not b.refs for b in backings)


def test_failed_larger_admission_preserves_existing_standby(request_module):
    pool, owner, backings = bind(request_module)
    assert owner.reserve(owner.admission_rows) == 1
    for b in backings:
        b.manager.budget = len(b.manager.live)
    owner.admission_rows = 5
    assert owner.reserve(owner.admission_rows) == 1
    assert owner.standby == {8} and all(b.live == {8} for b in backings)
    owner.admission_rows = 1
    assert owner.reserve(owner.admission_rows) == 1
    assert pool.alloc_rows(1) == [8]


def test_direct_unadmitted_allocation_miss_restores_native_rows(request_module):
    pool, owner, backings = bind(request_module)
    backings[-1].manager.budget = 0
    assert pool.alloc_rows(2) is None
    assert pool.free_slots == list(range(1, 9))
    assert not owner.active and all(not b.live for b in backings)


def test_backend_exception_poisoning_preserves_native_rollback(request_module):
    pool, owner, backings = bind(request_module)
    backings[-1].manager.fault = RuntimeError("map failure")
    with pytest.raises(RuntimeError, match="map failure"):
        owner.reserve(owner.admission_rows)
    assert pool.free_slots == list(range(1, 9))
    assert not backings[0].live and owner.failed
    with pytest.raises(RuntimeError, match="after a backend failure"):
        pool.alloc_rows(1)


def test_new_row_callback_is_rejected_before_preflight(request_module):
    pool, owner, _ = bind(request_module)
    calls: list[Any] = []
    pool._on_alloc_rows = calls.append
    with pytest.raises(RuntimeError, match="row callbacks"):
        owner.reserve(owner.admission_rows)
    assert not calls and pool.free_slots == list(range(1, 9))


def test_beam_admission_reserves_pending_rows_before_native_gate(request_module):
    pool, owner, backings = bind(request_module)

    class Scheduler:
        req_to_token_pool = pool
        running_batch = object()
        beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 2)

        def get_num_allocatable_reqs(self, running_bs, beam_width=None, running_batch=None):
            available = max(0, self.req_to_token_pool.available_size() - 2)
            return min(8 - running_bs, available, available // beam_width if beam_width else available)

    assert request_module.install_request_admission(Scheduler)
    assert request_module.install_request_admission(Scheduler)
    assert Scheduler().get_num_allocatable_reqs(0, 3) == 2
    assert owner.standby == set(range(1, 9))
    for b in backings:
        b.manager.budget = len(b.manager.live)
    assert pool.alloc_rows(2) == [7, 8]  # Two new leaders.
    assert pool.alloc_rows(2) == [5, 6]  # Previously admitted pending members.
    assert pool.alloc_rows(4) == [1, 2, 3, 4]  # Both leaders' members.
    assert owner.active == set(range(1, 9))


@pytest.mark.parametrize("running,expected", [(0, 8), (3, 5), (7, 1), (8, 0)])
def test_native_batch_limit_preserved_with_sufficient_backing(request_module, running, expected):
    pool, owner, _ = bind(request_module)

    class Scheduler:
        req_to_token_pool = pool
        running_batch = object()
        beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)

        def get_num_allocatable_reqs(self, running_bs, beam_width=None, running_batch=None):
            return min(8 - running_bs, self.req_to_token_pool.available_size())

    request_module.install_request_admission(Scheduler)
    before = list(pool.free_slots)
    assert Scheduler().get_num_allocatable_reqs(running) == expected
    assert pool.free_slots == before and len(owner.standby) == expected
    assert request_module.request_admission_context.get() is None
    assert pool.alloc_rows(expected) == (before[-expected:] if expected else [])


@pytest.mark.parametrize("banks", [1, 3])
@pytest.mark.parametrize("capacity", [0, 1, 3, 5, 8])
def test_admission_reports_largest_backed_native_tail(request_module, monkeypatch, banks, capacity):
    pool, owner, backings = bind(request_module, banks=banks)
    owner.admission_rows = pool.size
    original = backings[-1].acquire
    monkeypatch.setattr(backings[-1], "acquire", lambda ids: len(ids) <= capacity * banks and original(ids))
    before = list(pool.free_slots)
    assert owner.reserve(owner.admission_rows) == capacity
    selected = set(before[-capacity:]) if capacity else set()
    assert pool.free_slots == before and owner.standby == selected
    assert all(b.live == owner._ids(selected, 10 if banks > 1 else 0, banks) for b in backings)
    assert pool.alloc_rows(capacity) == (before[-capacity:] if capacity else [])


def test_native_admission_query_restores_context_after_error(request_module):
    pool, owner, _ = bind(request_module)

    class Scheduler:
        req_to_token_pool = pool
        running_batch = object()
        beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)

        def get_num_allocatable_reqs(self, *args):
            raise ValueError("native gate failure")

    request_module.install_request_admission(Scheduler)
    with pytest.raises(ValueError, match="native gate failure"):
        Scheduler().get_num_allocatable_reqs(0)
    assert request_module.request_admission_context.get() is None
    assert owner.reserve(owner.admission_rows) == 1


def test_partial_preflight_exception_releases_earlier_probe_leases(request_module, monkeypatch):
    pool, owner, backings = bind(request_module)
    owner.admission_rows = pool.size
    original = backings[-1].acquire

    def acquire(ids):
        if len(ids) == 8:
            return False
        if len(ids) == 6:
            raise RuntimeError("partial probe failure")
        return original(ids)

    monkeypatch.setattr(backings[-1], "acquire", acquire)
    with pytest.raises(RuntimeError, match="partial probe failure"):
        owner.reserve(owner.admission_rows)
    assert owner.failed and not owner.active and not owner.standby
    assert pool.free_slots == list(range(1, 9))
    assert all(not b.live and not b.refs and not b.manager.live for b in backings)


def test_logical_request_capacity_is_independent_of_admission_pressure(request_module):
    pool, owner, backings = bind(request_module)

    class Scheduler:
        req_to_token_pool = pool
        running_batch = object()
        beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)

        def get_num_allocatable_reqs(self, running_bs, beam_width=None, running_batch=None):
            return min(8 - running_bs, self.req_to_token_pool.available_size())

    request_module.install_request_admission(Scheduler)
    assert pool.available_size() == pool.size and not owner.standby
    backings[-1].manager.budget = 0
    assert Scheduler().get_num_allocatable_reqs(0) == 0
    assert pool.available_size() == pool.size and not owner.standby
    backings[-1].manager.budget = 30
    assert Scheduler().get_num_allocatable_reqs(0) == pool.size
    rows = pool.alloc_rows(3)
    assert pool.available_size() == pool.size - 3
    pool.free_rows(rows)
    pool.clear()
    assert pool.available_size() == pool.size
    assert all(not b.live and not b.refs for b in backings)


@pytest.mark.parametrize("operation", ["reserve", "idle"])
@pytest.mark.parametrize("waiting_index", [0, 1])
def test_request_state_waits_for_every_manager_before_resize(request_module, monkeypatch, operation, waiting_index):
    pool, owner, backings = bind(request_module)
    waiting, ready = Event(), Event()
    calls = []

    def wait_ready():
        waiting.set()
        assert ready.wait(3)

    monkeypatch.setattr(backings[waiting_index].manager, "wait_ready", wait_ready)
    for b in backings:
        b.manager.in_shrink = False
        monkeypatch.setattr(b.manager, "apply_resize_target", lambda: calls.append("resize"))
    before = list(pool.free_slots)
    call = (lambda: owner.reserve(owner.admission_rows)) if operation == "reserve" else lambda: owner.reclaim_on_idle(keep_standby=False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(call)
        try:
            assert waiting.wait(1) and not result.done()
            assert not calls and pool.free_slots == before
        finally:
            ready.set()
        result.result(timeout=1)
    assert calls == ["resize", "resize"]


def test_request_state_shrink_preserves_prefixes_and_active_or_promised_rows(request_module, monkeypatch):
    from test_sglang_dsv4_kv_pool_patch import _load_modules

    _interfaces, patches, _torch = _load_modules(monkeypatch)
    pool, owner, backings = bind(request_module)
    assert owner.reserve(owner.admission_rows) == 1
    active = pool.alloc_rows(1)
    assert owner.reserve(owner.admission_rows) == 1
    assert owner.active == {8} and owner.standby == {7}
    counters = {"limits": 0, "reclaim": 0, "prefix": 0}

    def apply():
        counters["limits"] += 1

    def reclaim():
        counters["reclaim"] += 1

    def evict(_):
        counters["prefix"] += 1
        raise AssertionError("Request-state pressure must not evict token prefixes")

    for backing in backings:
        manager = backing.manager
        manager.in_shrink = True
        manager.apply_resize_target = apply
        manager.reclaim_for_shrink = reclaim
    scheduler = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(_kvcached_dsv4_request_state=owner),
        req_to_token_pool=pool, running_batch=object(), tree_cache=SimpleNamespace(evict=evict),
        beam_coordinator=SimpleNamespace(pending_member_rows=lambda batch: 1),
    )
    assert patches._token_kv_managers(scheduler.token_to_kv_pool_allocator) == []
    assert patches._register_shrink_reclaimers(scheduler) == []
    patches._apply_kvcached_limits_on_idle(scheduler)
    assert owner.standby == {7} and all(b.live == {7, 8} for b in backings)
    scheduler.beam_coordinator.pending_member_rows = lambda batch: 0
    patches._apply_kvcached_limits_on_idle(scheduler)
    assert not owner.standby and owner.active == set(active)
    assert all(b.live == {8} for b in backings)
    assert counters == {"limits": 4, "reclaim": 4, "prefix": 0}
    assert len(pool.free_slots) == pool.size - 1


@pytest.mark.parametrize("unsupported", [None, "draft", "decode", "priority", "dllm", "callback"])
def test_factory_context_gates_unaudited_admission_paths(request_module, unsupported):
    pool = NativeRows()
    observed = []

    class Configurator:
        is_draft_worker = unsupported == "draft"

        def _build_token_to_kv_pool(self, *, req_to_token_pool):
            observed.append(request_module.request_pool_context.get())
            return SimpleNamespace()

    if unsupported == "callback":
        pool._on_alloc_rows = lambda rows: None
    module = SimpleNamespace(
        KVCacheConfigurator=Configurator, ReqToTokenPool=NativeRows,
        get_schedule=lambda: SimpleNamespace(enable_priority_scheduling=unsupported == "priority",
                                              disable_priority_preemption=False),
        get_disagg=lambda: SimpleNamespace(disaggregation_mode="decode" if unsupported == "decode" else "null"),
        get_exec=lambda: SimpleNamespace(dllm=SimpleNamespace(dllm_algorithm="test" if unsupported == "dllm" else None)),
    )
    assert request_module.install_request_state_context(module)
    Configurator()._build_token_to_kv_pool(req_to_token_pool=pool)
    assert observed == [pool if unsupported is None else None]
    assert request_module.request_pool_context.get() is None
