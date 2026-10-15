# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Backing and transaction contracts; no GPU, Triton, or model execution."""

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, RLock
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "dsv4_backing_under_test",
    Path(__file__).parents[1] / "kvcached/integration/sglang/dsv4_backing.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SWABacking = MODULE.SWABacking


class Manager:
    def __init__(self, budget=20):
        self.budget = budget
        self.live = set()
        self.claims = []
        self.fault = None
        self.init_failure = None

    def wait_ready(self):
        if self.init_failure:
            raise self.init_failure

    def apply_resize_target(self):
        self.wait_ready()

    def reclaim_for_shrink(self):
        pass

    def alloc_exact(self, ids):
        if self.fault:
            raise self.fault
        assert not self.live.intersection(ids)
        if len(ids) > self.available_size():
            return None
        self.live.update(ids)
        self.claims.append(list(ids))
        return ids

    def free(self, ids):
        assert set(ids) <= self.live
        self.live.difference_update(ids)

    def available_size(self):
        return self.budget - len(self.live)


def make_backing(manager=None):
    return SWABacking(manager or Manager(), native_pages=12,
                      native_page_bytes=6, physical_page_bytes=8)


def test_padding_shared_acquire_waits_for_null_initialization(monkeypatch):
    manager = Manager()
    backing = SWABacking(manager, native_pages=12, native_page_bytes=2,
                         physical_page_bytes=8)
    entered, ready = Event(), Event()

    def wait_ready():
        entered.set()
        assert ready.wait(2), "test initialization was not released"

    monkeypatch.setattr(manager, "wait_ready", wait_ready)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(backing.acquire, {1})
        try:
            assert entered.wait(1)
            assert not result.done()
            assert not backing.live and not manager.claims
        finally:
            ready.set()
        assert result.result(timeout=1) is True
    assert backing.live == {1}
    assert not manager.claims  # Native page 1 needs only permanent physical page 0.


def test_backing_covers_crossing_and_shared_physical_pages():
    backing = make_backing()
    assert backing.acquire({1, 2})
    assert backing.manager.claims == [[1, 2]]
    assert dict(backing.refs) == {1: 2, 2: 1}
    assert backing.acquire({2, 3})
    assert backing.manager.claims == [[1, 2]]
    backing.release({1})
    assert backing.manager.live == {1, 2}
    backing.release({2})
    assert backing.manager.live == {2}
    backing.release({3})
    assert not backing.manager.live
    assert not backing.live and not backing.refs
    assert all(0 not in claim for claim in backing.manager.claims)


def test_rejected_acquire_preserves_live_prefix_and_can_retry():
    backing = make_backing(Manager(budget=1))
    assert backing.acquire({1})
    assert not backing.acquire({1, 2})
    assert backing.live == {1}
    assert dict(backing.refs) == {1: 1}
    backing.manager.budget = 2
    assert backing.acquire({1, 2})
    assert dict(backing.refs) == {1: 2, 2: 1}


@pytest.mark.parametrize("page", [-1, 0, 12])
def test_bad_native_ids_fail_before_backend_mutation(page):
    backing = make_backing()
    with pytest.raises(ValueError, match="outside"):
        backing.acquire({page})
    assert backing.manager.claims == []


class Tensor:
    """Only the tensor operations used by the Python boundary adapter."""
    def __init__(self, values):
        self.values = list(values)

    def __floordiv__(self, other):
        return Tensor(value // other for value in self.values)

    def unique(self):
        return Tensor(sorted(set(self.values)))

    def tolist(self):
        return list(self.values)


class NativeChild:
    def __init__(self, page_size, prefix, output):
        self.page_size = page_size
        self.device = "cpu"
        self.size = 12 * page_size
        self.live = {prefix} if prefix else set()
        self.output = output
        self.free_group = None
        self.free_page_ids_group = []
        self.failure = None

    def available_size(self):
        return self.size - len(self.live) * self.page_size

    def alloc(self, need_size=0):
        if self.failure:
            raise self.failure
        if self.output is None:
            return None
        self.live.update(v // self.page_size for v in self.output)
        return Tensor(self.output)

    def alloc_extend(self, last_loc):
        return self._allocate_result()

    def alloc_decode(self, last_loc):
        return self._allocate_result()

    # Native paged methods do not call alloc; keep those independent hooks.
    _allocate_result = alloc

    def free(self, indices):
        ids = (indices // self.page_size).unique()
        if self.free_group is None:
            self._release(ids)
        else:
            self.free_group.append(ids)

    def free_page_ids(self, ids):
        if self.page_size == 1:
            self.free(ids)  # Token allocator delegates to public free().
        elif self.free_group is None:
            self._release(ids)
        else:
            self.free_page_ids_group.append(ids)

    def _release(self, ids):
        pages = set(ids.tolist())
        assert pages <= self.live
        self.live.difference_update(pages)

    def free_group_end(self):
        group, self.free_group = self.free_group, None
        for ids in group or []:
            self.free(Tensor(v * self.page_size for v in ids.tolist()))
        for ids in self.free_page_ids_group:
            self._release(ids)
        self.free_page_ids_group = []

    def clear(self):
        self.live.clear()
        self.free_group = None
        self.free_page_ids_group = []


class Composite:
    def __init__(self, page_size):
        self.page_size = page_size
        self.full_attn_allocator = NativeChild(page_size, 2, [5 * page_size])
        self.swa_attn_allocator = NativeChild(page_size, 1, [3 * page_size])
        self.mapping = {2 * page_size: page_size}

    def _allocate(self, method):
        full = self.full_attn_allocator
        swa = self.swa_attn_allocator
        if method == "alloc":
            f, s = full.alloc(1), swa.alloc(1)
        else:
            f = getattr(full, method)(last_loc=Tensor([2 * self.page_size]))
            s = getattr(swa, method)(last_loc=Tensor([self.page_size]))
        assert f is not None and s is not None
        self.mapping.update(zip(f.tolist(), s.tolist()))
        return f

    def alloc(self):
        return self._allocate("alloc")

    def alloc_extend(self):
        return self._allocate("alloc_extend")

    def alloc_decode(self):
        return self._allocate("alloc_decode")

    def alloc_extend_swa_tail(self):
        return self._allocate("alloc_extend")

    def resize(self, config):
        self.full_attn_allocator.size = config.full_max_total_num_tokens
        self.swa_attn_allocator.size = config.swa_max_total_num_tokens
        self.full_attn_allocator.clear()
        self.swa_attn_allocator.clear()


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        int64=object(), tensor=lambda values, **_: Tensor(values)))

    def create(page_size=4):
        backing = make_backing()
        assert backing.acquire({1})
        composite = Composite(page_size)
        MODULE.bind_native_swa(composite, backing)
        return composite, backing
    return create


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("method", ["alloc", "alloc_extend", "alloc_decode", "alloc_extend_swa_tail"])
@pytest.mark.parametrize("miss", ["full", "swa", "backing"])
def test_miss_returns_only_new_native_pages_before_mapping(setup, page_size, method, miss):
    composite, backing = setup(page_size)
    full, swa = composite.full_attn_allocator, composite.swa_attn_allocator
    if method != "alloc":
        full.output.insert(0, 2 * page_size + (page_size > 1))
        swa.output.insert(0, page_size + (page_size > 1))
    if miss == "backing":
        backing.manager.budget = 1
    else:
        getattr(composite, miss + "_attn_allocator").output = None
    before = dict(composite.mapping)
    assert getattr(composite, method)() is None
    assert composite.mapping == before
    assert full.live == {2} and swa.live == {1}
    assert backing.live == {1} and backing.manager.live == {1}
    full.output, swa.output = [5 * page_size], [3 * page_size]
    backing.manager.budget = 20
    assert getattr(composite, method)().tolist() == [5 * page_size]
    assert composite.mapping[5 * page_size] == 3 * page_size
    assert full.live == {2, 5} and swa.live == {1, 3}
    assert backing.live == {1, 3}


@pytest.mark.parametrize("method", ["alloc", "alloc_extend", "alloc_decode", "alloc_extend_swa_tail"])
def test_fatal_mapping_error_requires_teardown(setup, method):
    composite, backing = setup()
    failure = RuntimeError("mapping is inconsistent")
    backing.manager.fault = failure
    with pytest.raises(RuntimeError) as caught:
        getattr(composite, method)()
    assert caught.value is failure
    assert composite.mapping == {8: 4}
    with pytest.raises(RuntimeError, match="tear down"):
        composite.alloc()
    with pytest.raises(RuntimeError, match="tear down"):
        composite.swa_attn_allocator.clear()


def test_null_initialization_failure_prevents_mapping_publication(setup):
    manager = Manager()
    failure = RuntimeError("null backing initialization failed")
    manager.init_failure = failure
    backing = SWABacking(manager, native_pages=12, native_page_bytes=2,
                         physical_page_bytes=8)
    composite = Composite(4)
    composite.full_attn_allocator.live.clear()
    composite.swa_attn_allocator.live.clear()
    composite.mapping.clear()
    MODULE.bind_native_swa(composite, backing)
    # The newly selected native SWA page 3 fits entirely in physical page zero.
    with pytest.raises(RuntimeError) as caught:
        composite.alloc()
    assert caught.value is failure
    assert not composite.mapping and not backing.live and not manager.claims
    with pytest.raises(RuntimeError, match="tear down"):
        composite.alloc()


@pytest.mark.parametrize("method", ["alloc", "alloc_extend", "alloc_decode", "alloc_extend_swa_tail"])
def test_first_allocation_allows_null_initialization_to_take_manager_lock(setup, monkeypatch, method):
    composite, backing = setup()
    manager = backing.manager
    lock, waiting, ready = RLock(), Event(), Event()

    def wait_ready():
        waiting.set()
        assert ready.wait(3), "test initialization was not released"

    def apply_resize_target():
        # Match KVCacheManager's synchronized resize and post-init wait.
        with lock:
            wait_ready()

    monkeypatch.setattr(manager, "wait_ready", wait_ready)
    monkeypatch.setattr(manager, "apply_resize_target", apply_resize_target)
    before = dict(composite.mapping)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(getattr(composite, method))
        try:
            assert waiting.wait(1)
            assert not result.done() and composite.mapping == before
            # The background null reservation must acquire this lock before
            # it can signal readiness. Waiting inside resize deadlocks it.
            acquired = lock.acquire(timeout=1)
            try:
                assert acquired, "allocation held the lock needed by initialization"
            finally:
                if acquired:
                    lock.release()
        finally:
            ready.set()
        assert result.result(timeout=1) is not None
    assert composite.mapping[20] == 12


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("method", ["free", "free_page_ids"])
def test_deferred_free_keeps_backing_until_native_commit(setup, page_size, method):
    composite, backing = setup(page_size)
    assert composite.alloc() is not None
    swa = composite.swa_attn_allocator
    swa.free_group = []
    ids = [3 * page_size] if method == "free" else [3]
    getattr(swa, method)(Tensor(ids))
    assert 3 in swa.live and 3 in backing.live
    swa.free_group_end()
    assert swa.live == {1} and backing.live == {1}
    assert backing.manager.live == {1}


def test_clear_releases_backing_and_growth_rejects_before_native_mutation(setup):
    composite, backing = setup()
    assert composite.alloc() is not None
    with pytest.raises(ValueError, match="exceeds"):
        composite.resize(SimpleNamespace(full_max_total_num_tokens=49,
                                         swa_max_total_num_tokens=48))
    assert composite.full_attn_allocator.size == 48
    assert backing.live == {1, 3}
    composite.resize(SimpleNamespace(full_max_total_num_tokens=24,
                                     swa_max_total_num_tokens=24))
    assert not backing.live and not backing.manager.live


def test_capacity_reports_backing_limit_but_slot_conservation_is_native(setup):
    composite, backing = setup()
    swa = composite.swa_attn_allocator
    backing.manager.budget = 1
    assert swa.available_size() == 0
    assert composite._conserve_swa_available_size() == 44
    swa.free_page_ids(Tensor([1]))
    assert swa.available_size() == 4
    assert composite._conserve_swa_available_size() == 48


@pytest.mark.parametrize("method", ["alloc", "alloc_extend", "alloc_decode", "alloc_extend_swa_tail"])
def test_shrink_reclaims_before_native_ids_and_backing_refs_are_selected(setup, monkeypatch, method):
    composite, backing = setup()
    assert composite._kvcached_dsv4_backing is backing
    # These IDs have different units; do not expose a token allocator manager.
    assert not hasattr(composite, "kvcached_allocator")
    assert not hasattr(composite.swa_attn_allocator, "kvcached_allocator")
    events = []

    def apply():
        events.append("apply")

    def reclaim():
        events.append("reclaim")
        assert composite.full_attn_allocator.live == {2}
        assert composite.swa_attn_allocator.live == {1}
        composite.swa_attn_allocator.free_page_ids(Tensor([1]))
        assert not backing.live and not backing.refs

    monkeypatch.setattr(backing.manager, "apply_resize_target", apply)
    monkeypatch.setattr(backing.manager, "reclaim_for_shrink", reclaim)
    assert getattr(composite, method)() is not None
    assert events == ["apply", "reclaim"]
    assert backing.live == {3}
    assert dict(backing.refs) == {2: 1}
    assert backing.manager.live == {2}
