# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Backing and transaction contracts; no GPU, Triton, or model execution."""

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, RLock
from types import SimpleNamespace
from typing import Any

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
        self.page_allocator = SimpleNamespace(get_num_reserved_pages=lambda: 0)
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

    def can_alloc_exact(self, ids):
        if self.fault:
            raise self.fault
        return not self.live.intersection(ids) and len(ids) <= self.available_size()

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

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def __len__(self):
        return len(self.values)


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

    def alloc_decode(self, last_loc, seq_lens=None, seq_lens_cpu=None):
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
        int64=object(), tensor=lambda values, **_: Tensor(values),
        device=lambda value: SimpleNamespace(type=str(value).split(":")[0])))

    def create(page_size=4, on_capacity_miss=None):
        backing = make_backing()
        assert backing.acquire({1})
        composite = Composite(page_size)
        MODULE.bind_native_swa(composite, backing, on_capacity_miss=on_capacity_miss)
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


@pytest.mark.parametrize("page_size", [1, 4])
def test_prepare_rollback_returns_only_new_native_pages(setup, page_size):
    allocator, backing = setup(page_size)
    journal = allocator._kvcached_prepare_transaction()
    with journal:
        assert allocator.alloc_extend() is not None
    assert allocator.full_attn_allocator.live == {2, 5}
    assert allocator.swa_attn_allocator.live == {1, 3}
    journal.rollback()
    assert allocator.full_attn_allocator.live == {2}
    assert allocator.swa_attn_allocator.live == {1}
    assert backing.live == {1}


def test_prepare_journal_does_not_return_a_page_already_freed(setup):
    allocator, backing = setup()
    journal = allocator._kvcached_prepare_transaction()
    with journal:
        allocator.alloc_extend()
        allocator.swa_attn_allocator.free_page_ids(Tensor([3]))
    journal.rollback()
    assert allocator.full_attn_allocator.live == {2}
    assert allocator.swa_attn_allocator.live == {1}
    assert backing.live == {1}


def test_prepare_miss_keeps_previous_success_in_journal(setup):
    allocator, backing = setup()
    journal = allocator._kvcached_prepare_transaction()
    with journal:
        allocator.alloc_extend()
        allocator.full_attn_allocator.output = [24]
        allocator.swa_attn_allocator.output = None
        assert allocator.alloc_extend() is None
    journal.rollback()
    assert allocator.full_attn_allocator.live == {2}
    assert allocator.swa_attn_allocator.live == {1}
    assert backing.live == {1}


def test_prepare_rollback_waits_for_index_writes_before_returning_pages(setup, monkeypatch):
    allocator, backing = setup()
    allocator.full_attn_allocator.device = "cuda:0"
    calls = []
    monkeypatch.setattr(sys.modules["torch"], "cuda", SimpleNamespace(
        current_stream=lambda device: SimpleNamespace(synchronize=lambda: calls.append("ready"))),
        raising=False)
    original = backing.release
    def release(pages):
        assert calls == ["ready"]
        return original(pages)
    monkeypatch.setattr(backing, "release", release)
    journal = allocator._kvcached_prepare_transaction()
    with journal:
        allocator.alloc_extend()
    journal.rollback()


def test_prepare_rollback_failure_poisoned_adapter_cannot_retry(setup, monkeypatch):
    allocator, backing = setup()
    journal = allocator._kvcached_prepare_transaction()
    with journal:
        allocator.alloc_extend()
    def fail(pages):
        raise RuntimeError("rollback failure")
    monkeypatch.setattr(backing, "release", fail)
    with pytest.raises(RuntimeError, match="rollback failure"):
        journal.rollback()
    with pytest.raises(RuntimeError, match="tear down"):
        allocator.alloc_extend()



def test_round_budget_is_shared_and_tracks_native_alloc_free(setup):
    allocator, backing = setup()
    full, swa = allocator.full_attn_allocator, allocator.swa_attn_allocator
    with allocator._kvcached_budget_context((20, 12)):
        assert full.available_size() == 20
        assert swa.available_size() == 12
        journal = allocator._kvcached_prepare_transaction()
        with journal:
            allocator.alloc_extend()
        assert (full.available_size(), swa.available_size()) == (16, 8)
        # A newly observed local headroom drop must not split an open round.
        backing.manager.budget = len(backing.manager.live)
        assert swa.available_size() == 8
        journal.rollback()
        assert (full.available_size(), swa.available_size()) == (20, 12)


@pytest.mark.parametrize("page_size", [1, 4])
def test_grouped_free_credits_round_budget_once(setup, page_size):
    allocator, backing = setup(page_size)
    child = allocator.swa_attn_allocator
    with allocator._kvcached_budget_context((10 * page_size, 6 * page_size)):
        allocator.alloc_extend()
        assert child.available_size() == 5 * page_size
        child.free_group = []
        child.free_page_ids(Tensor([3]))
        assert child.available_size() == 5 * page_size
        child.free_group_end()
        assert child.available_size() == 6 * page_size
        assert backing.live == {1}


def test_capacity_translation_happens_after_rollback_without_poisoning(setup):
    class Retry(Exception):
        pass

    def miss(*args):
        raise Retry()

    allocator, backing = setup(on_capacity_miss=miss)
    allocator.swa_attn_allocator.output = None
    with pytest.raises(Retry):
        allocator.alloc_extend()
    assert allocator.full_attn_allocator.live == {2}
    assert backing.live == {1}
    allocator.swa_attn_allocator.output = [12]
    assert allocator.alloc_extend().tolist() == [20]


class DecodeChild(NativeChild):
    def __init__(self, prefix, free):
        super().__init__(4, prefix, None)
        self.choices = list(free)
        self.released = []
        self.need_sort = True

    @property
    def free_pages(self):
        return Tensor(self.choices)

    def merge_and_sort_free(self):
        self.choices = sorted(self.choices + self.released)
        self.released = []

    def get_all_free_pages(self):
        return Tensor(self.choices + self.released)

    def alloc_decode(self, last_loc, seq_lens=None, seq_lens_cpu=None):
        # Native 0.5.20 may sort for bs even when fewer new pages are needed.
        if self.need_sort:
            self.merge_and_sort_free()
        result = []
        for length, previous in zip(seq_lens_cpu.tolist(), last_loc.tolist()):
            if (length - 1) % self.page_size == 0:
                if not self.choices:
                    return None
                page = self.choices.pop(0)
                self.live.add(page)
                result.append(page * self.page_size)
            else:
                result.append(previous + 1)
        return Tensor(result)

    def _release(self, ids):
        super()._release(ids)
        if self.need_sort:
            self.released.extend(ids.tolist())
        else:
            self.choices = ids.tolist() + self.choices


class DecodeComposite(Composite):
    def __init__(self, swa_choices):
        super().__init__(4)
        self.full_attn_allocator = DecodeChild(2, [5, 7, 9])
        self.swa_attn_allocator = DecodeChild(1, swa_choices)

    def check_decode_capacity(self, *, num_tokens, **kwargs):
        return min(self.full_attn_allocator.available_size(),
                   self.swa_attn_allocator.available_size()) >= num_tokens

    def alloc_decode(self, next_length=5):
        lengths = [next_length] if isinstance(next_length, int) else next_length
        offsets = [(length - 2) % 4 for length in lengths]
        full = self.full_attn_allocator.alloc_decode(
            last_loc=Tensor([8 + offset for offset in offsets]), seq_lens=Tensor(lengths),
            seq_lens_cpu=Tensor(lengths))
        swa = self.swa_attn_allocator.alloc_decode(
            last_loc=Tensor([4 + offset for offset in offsets]), seq_lens=Tensor(lengths),
            seq_lens_cpu=Tensor(lengths))
        assert full is not None and swa is not None
        self.mapping.update(zip(full.tolist(), swa.tolist()))
        return full


def decode_setup(swa_choices=(3, 7, 9), *, paged_v0520=None):
    backing = make_backing()
    assert backing.acquire({1})
    allocator = DecodeComposite(swa_choices)
    MODULE.bind_native_swa(allocator, backing, paged_v0520=paged_v0520)
    return allocator, backing


@pytest.mark.parametrize("lengths,crosses", [([4], False), ([4, 5], True)])
@pytest.mark.parametrize("audited", [False, True])
def test_decode_reads_touched_pages_and_journals_only_new(setup, monkeypatch, lengths, crosses, audited):
    allocator, backing = decode_setup(paged_v0520=DecodeChild if audited else None)
    full, swa = allocator.full_attn_allocator, allocator.swa_attn_allocator
    assert allocator.check_decode_capacity(num_tokens=4 if crosses else 0)
    reads = []
    original_unique = Tensor.unique
    def read_ids(tensor):
        reads.append(tensor.tolist())
        return original_unique(tensor)

    monkeypatch.setattr(Tensor, "unique", read_ids)
    acquire = backing.acquire
    touched = []
    def acquire_touched(pages):
        touched.append(pages)
        return acquire(pages)

    monkeypatch.setattr(backing, "acquire", acquire_touched)
    journal = allocator._kvcached_prepare_transaction()
    with allocator._kvcached_budget_context((8, 8)):
        with journal:
            allocator.alloc_decode(lengths)
        if audited:
            assert reads == ([[1, 3]] if crosses else [[1]])
        else:
            assert len(reads) == 4
        assert touched == [{1, 3} if crosses else {1}]
        assert journal.pages == {id(full): {5} if crosses else set(), id(swa): {3} if crosses else set()}
        assert full.available_size() == swa.available_size() == (4 if crosses else 8)
        journal.rollback()
        assert full.available_size() == swa.available_size() == 8
    assert full.live == {2} and swa.live == backing.live == {1}


def test_reserved_staged_only_prefix_is_not_committed_on_miss(setup):
    allocator, backing = decode_setup((3,), paged_v0520=DecodeChild)
    full, swa = allocator.full_attn_allocator, allocator.swa_attn_allocator
    assert allocator.check_decode_capacity(num_tokens=4)
    swa.released, swa.choices = swa.choices, []
    journal = allocator._kvcached_prepare_transaction()
    with journal:
        assert allocator.alloc_decode() is None
    assert not any(journal.pages.values())
    assert full.live == {2} and swa.live == backing.live == {1}


def test_decode_gate_checks_placement_before_native_allocation(setup):
    allocator, backing = decode_setup((10,))
    assert backing.acquire({2})
    allocator.swa_attn_allocator.live.add(2)
    backing.manager.budget = len(backing.manager.live)
    assert allocator.swa_attn_allocator.available_size() >= 4
    assert not allocator.check_decode_capacity(num_tokens=4)
    assert allocator.swa_attn_allocator.choices == [10]
    assert backing.live == {1, 2}


def test_failed_placement_suppresses_admission_until_exact_capacity_recovers(setup):
    allocator, backing = decode_setup((10,))
    child = allocator.swa_attn_allocator
    assert backing.acquire({2})
    child.live.add(2)
    child.need_sort = False
    backing.manager.budget = len(backing.manager.live)
    assert child.available_size() > 0
    assert allocator.alloc_decode() is None
    claims = list(backing.manager.claims)
    for _ in range(3):
        assert child.available_size() == 0
        assert child.choices == [10]
        assert backing.manager.claims == claims
    backing.manager.budget += 2
    assert child.available_size() > 0
    assert backing.manager.claims == claims
    assert allocator.alloc_decode().tolist() == [20]
    assert allocator.mapping[20] == 40


def test_parked_prefill_growth_is_bounded_after_large_miss(setup):
    allocator, backing = decode_setup((3, 7, 9))
    manager = backing.manager
    manager.budget = len(manager.live)
    assert allocator.alloc_decode([5, 5, 5]) is None
    assert allocator._kvcached_backing_demand(1) == [(manager, 4)]

    # A failed large prefix still belongs in demand reporting. The smaller
    # retry can proceed with one bundle even though the bound covers two.
    manager.in_shrink = False
    manager.page_allocator = SimpleNamespace(get_num_free_pages=lambda: 2, get_num_reserved_pages=lambda: 0)
    manager._memory_limit_page_bundle_bytes = lambda: 20
    manager.budget += 1
    assert manager.available_size() == 1
    assert backing.can_acquire({3})
    assert allocator._kvcached_parked_prefill_growth_costs() == {20, 40}
    manager.page_allocator.get_num_reserved_pages = lambda: 1
    assert allocator._kvcached_parked_prefill_growth_costs() == {20}
    manager.page_allocator.get_num_free_pages = lambda: 1
    assert allocator._kvcached_parked_prefill_growth_costs() == set()
    manager.page_allocator.get_num_free_pages = lambda: 2
    manager.in_shrink = True
    assert allocator._kvcached_parked_prefill_growth_costs() == set()


def test_parked_prefill_counts_straddling_full_page_and_shared_driver_budget(setup):
    allocator: Any = DecodeComposite([3, 7])
    allocator.full_attn_allocator.choices = [2]
    full = SWABacking(Manager(), native_pages=13, native_page_bytes=6, physical_page_bytes=8)
    swa = make_backing()
    MODULE.bind_native_swa(allocator, swa, full)
    for item, bundle in ((full, 20), (swa, 10)):
        item.manager.in_shrink = False
        item.manager.page_allocator = SimpleNamespace(get_num_free_pages=lambda: 2, get_num_reserved_pages=lambda: 0)
        item.manager._memory_limit_page_bundle_bytes = lambda bundle=bundle: bundle

    # Both pools can map fresh pages from the same driver. Each has its
    # own bundle size.
    assert allocator._kvcached_parked_prefill_growth_costs() == {10, 20, 30, 40, 50, 60}

    # Native eviction may satisfy the full side. It cannot veto a useful
    # release for SWA, even if disaggregated prefill will merge staged IDs.
    full.manager.page_allocator.get_num_free_pages = lambda: 0
    allocator.swa_attn_allocator.released = [1]
    assert allocator._kvcached_parked_prefill_growth_costs() == {10, 20}
    assert allocator.swa_attn_allocator.choices == [3, 7]
    assert allocator.swa_attn_allocator.released == [1]


def test_retry_prefix_counts_shared_headroom_after_compressed_rollback(setup):
    capacity = [1]
    pools: list[Any] = []

    class SharedManager(Manager):
        def available_size(self):
            return capacity[0] - sum(len(pool.live) for pool in pools)

        def exact_recovery_quote(self, ids):
            return len(ids) * 20, [], 0

    pools.extend([SharedManager(), SharedManager()])
    for pool in pools:
        pool.page_allocator = SimpleNamespace(get_avail_physical_bytes=lambda: pools[0].available_size() * 20,
                                              get_num_reserved_pages=lambda: 0)
    full, indexer = (SWABacking(pool, native_pages=13, native_page_bytes=8, physical_page_bytes=8)
                     for pool in pools)
    allocator = DecodeComposite([3, 7, 9])
    MODULE.bind_native_swa(allocator, make_backing(), full, full_backings=(indexer,))
    assert allocator.alloc_decode() is None
    assert all(not pool.live for pool in pools)
    assert allocator.full_attn_allocator.available_size() == 0
    capacity[0] = 2
    assert allocator.full_attn_allocator.available_size() >= allocator.page_size
    assert allocator.alloc_decode() is not None
    assert all(pool.live == {5} for pool in pools)


@pytest.mark.parametrize("initial", [1, 2])
def test_trim_on_second_kv_map_can_fund_the_indexer(setup, initial):
    kv: Any = Manager(0)
    index: Any = Manager(0)
    allocator = DecodeComposite([3, 7, 9])
    full = SWABacking(kv, native_pages=13, native_page_bytes=8, physical_page_bytes=8)
    indexer = SWABacking(index, native_pages=13, native_page_bytes=2, physical_page_bytes=8)
    MODULE.bind_native_swa(allocator, make_backing(), full, full_backings=(indexer,))
    assert allocator.alloc_decode([5] * initial) is None
    kv.budget = index.budget = 8
    samples = []
    room = [20]

    def sample():
        samples.append(room[0])
        return room[0]

    kv.page_allocator = SimpleNamespace(get_avail_physical_bytes=sample, get_num_reserved_pages=lambda: 2)
    index.page_allocator = SimpleNamespace(get_num_reserved_pages=lambda: 0)
    kv.exact_recovery_quote = lambda ids: (20 * len(ids), [10, 11], 0)
    kv._memory_limit_page_bundle_bytes = lambda: 20
    index.exact_recovery_quote = lambda ids: (20 * len(ids), [], 0)
    assert allocator.full_attn_allocator.available_size() == 2 * allocator.page_size
    assert samples == [20]
    # The old one-page miss now fits directly, but the longest prefix still
    # needs one idle reserve. Two polls must report the same fundable prefix.
    room[0] = 40
    assert allocator.full_attn_allocator.available_size() == 2 * allocator.page_size
    assert allocator.full_attn_allocator.available_size() == 2 * allocator.page_size
    assert samples == [20, 40, 40]
    room[0] = 100
    assert allocator.full_attn_allocator.available_size() >= 2 * allocator.page_size
    assert allocator.full_attn_allocator.available_size() >= 2 * allocator.page_size
    assert samples == [20, 40, 40, 100]


@pytest.mark.parametrize("local", [0, 1])
def test_exact_recovery_mixed_bundles_keeps_only_required_reserves(local):
    small = SimpleNamespace(exact_recovery_quote=lambda ids: (4, [10], local),
                            _memory_limit_page_bundle_bytes=lambda: 4)
    large = SimpleNamespace(exact_recovery_quote=lambda ids: (16, [11], 0),
                            _memory_limit_page_bundle_bytes=lambda: 16)
    pools = tuple(SWABacking(manager, native_pages=13, native_page_bytes=8, physical_page_bytes=8)
                  for manager in (small, large))
    expected = [(small, [10])] if local else []
    assert MODULE._exact_recovery_plan(pools, {5}, 10) == [*expected, (large, [11])]


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("shorter_retry", [False, True])
def test_shared_miss_trims_only_idle_ids_needed_to_fund_retry(setup, gate, shorter_retry):
    room = [-2 if shorter_retry else 1]

    class Pool(Manager):
        def __init__(self, reserves):
            super().__init__()
            self.reserves = set(reserves)
            self.page_allocator = SimpleNamespace(get_num_reserved_pages=lambda: len(self.reserves),
                                                  get_avail_physical_bytes=lambda: room[0])

        def available_size(self):
            return max(room[0], 0) + len(self.reserves)

        def alloc_exact(self, ids):
            fresh = set(ids) - self.reserves
            if fresh and len(fresh) > room[0]:
                return None
            room[0] -= len(fresh)
            self.reserves.difference_update(ids)
            self.live.update(ids)
            return ids

        def free(self, ids):
            self.live.difference_update(ids)
            self.reserves.update(ids)

        def exact_recovery_quote(self, ids):
            return len(set(ids) - self.reserves), list(self.reserves - set(ids)), 0

        def trim_selected_reserves(self, ids):
            self.reserves.difference_update(ids)
            room[0] += len(ids)

        def _memory_limit_page_bundle_bytes(self):
            return 1

        def invalidate_physical_capacity(self):
            pass

    kv, index = Pool([5, 7, 11] if shorter_retry else [9, 10]), Pool([])
    full, indexer = [SWABacking(pool, native_pages=13, native_page_bytes=8, physical_page_bytes=8)
                     for pool in (kv, index)]
    allocator = DecodeComposite([3, 7, 9])
    MODULE.bind_native_swa(allocator, make_backing(), full, full_backings=(indexer,))
    count = 2 if shorter_retry else 1
    if gate and not shorter_retry:
        assert not allocator.check_decode_capacity(num_tokens=4 * count)
    else:
        assert allocator.alloc_decode([5] * count) is None
    if shorter_retry:
        assert kv.reserves == {5, 7, 11}
        assert allocator.full_attn_allocator.available_size() == 0
        room[0] = 0
        assert allocator.full_attn_allocator.available_size() == allocator.page_size
        assert kv.reserves == {5, 7, 11}  # Capacity queries do not trim.
    else:
        assert room == [1] and 5 in kv.reserves and len(kv.reserves) == 2
    if not gate:
        assert allocator.alloc_decode().tolist() == [20]
        return
    assert allocator.check_decode_capacity(num_tokens=4)
    assert allocator.alloc_decode().tolist() == [20]


def test_decode_gate_retains_backing_until_the_promised_native_ids_are_used(setup):
    allocator, backing = decode_setup()
    assert allocator.check_decode_capacity(num_tokens=4)
    claims = list(backing.manager.claims)
    backing.manager.budget = len(backing.manager.live)
    assert allocator.alloc_decode().tolist() == [20]
    assert allocator.mapping[20] == 12
    assert backing.manager.claims == claims
    assert backing.live == {1, 3}


def test_decode_reservation_survives_native_window_frees_and_sort_hint(setup):
    allocator, backing = decode_setup()
    assert backing.acquire({2})
    child = allocator.swa_attn_allocator
    child.live.add(2)
    assert allocator.check_decode_capacity(num_tokens=4)
    child.free_page_ids(Tensor([2]))
    assert child.released == [2]
    allocator.alloc_decode()
    assert allocator.mapping[20] == 12  # The reserved page, not newly freed 2.
    assert backing.live == {1, 3}


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("free_tokens", [False, True])
def test_unsorted_window_frees_retain_backing_until_decode_chooses_ids(
    setup, grouped, free_tokens
):
    allocator, backing = decode_setup()
    assert backing.acquire({2})
    child = allocator.swa_attn_allocator
    child.need_sort = False
    child.live.add(2)
    assert allocator.check_decode_capacity(num_tokens=4)
    claims = list(backing.manager.claims)
    backing.manager.budget = len(backing.manager.live)
    if grouped:
        child.free_group = []
    if free_tokens:
        child.free(Tensor([8, 9, 10, 11]))
    else:
        child.free_page_ids(Tensor([2]))
    if grouped:
        child.free_group_end()
    assert child.choices[0] == 2
    assert backing.live == {1, 2, 3}
    allocator.alloc_decode()
    assert allocator.mapping[20] == 8
    assert backing.manager.claims == claims
    assert backing.live == {1, 2}  # Unused original reservation is released.


def test_outer_decode_gate_keeps_native_limit_and_checks_logical_placement(setup):
    logical, backing = decode_setup((10,))
    assert backing.acquire({2})
    logical.swa_attn_allocator.live.add(2)
    backing.manager.budget = len(backing.manager.live)

    class Outer:
        logical_attn_allocator = logical
        device_tokens = 4

        def check_decode_capacity(self, *, num_tokens, **kwargs):
            return min(self.device_tokens, logical.swa_attn_allocator.available_size()) >= num_tokens

    outer = Outer()
    assert outer.check_decode_capacity(num_tokens=4)
    MODULE.bind_native_decode_wrapper(outer)
    bound = outer.check_decode_capacity
    MODULE.bind_native_decode_wrapper(outer)
    assert outer.check_decode_capacity is bound
    assert not outer.check_decode_capacity(num_tokens=4)
    assert backing.live == {1, 2}
    logical.swa_attn_allocator.choices = [3, 7, 9]
    assert outer.check_decode_capacity(num_tokens=4)
    assert backing.live == {1, 2, 3}
    outer.device_tokens = 0
    assert not outer.check_decode_capacity(num_tokens=4)
    assert backing.live == {1, 2}
    outer.device_tokens = 4
    assert outer.check_decode_capacity(num_tokens=4)
    assert logical.alloc_decode().tolist() == [20]


def test_peer_decode_rejection_returns_only_reservations(setup):
    allocator, backing = decode_setup()
    assert allocator.check_decode_capacity(num_tokens=8)
    assert backing.live == {1, 3, 7}
    allocator._kvcached_release_decode_reservation()
    assert backing.live == {1}
    assert allocator.swa_attn_allocator.choices == [3, 7, 9]
    assert allocator.check_decode_capacity(num_tokens=4)
    assert backing.live == {1, 3}


def test_decode_inside_existing_pages_needs_no_new_backing(setup):
    allocator, backing = decode_setup(())
    allocator.full_attn_allocator.choices = []
    backing.manager.budget = len(backing.manager.live)
    assert allocator.check_decode_capacity(num_tokens=0)
    assert allocator.alloc_decode(next_length=4).tolist() == [11]
    assert backing.live == {1}


def test_decode_gate_backend_error_is_fatal_not_a_capacity_miss(setup):
    allocator, backing = decode_setup()
    backing.manager.fault = RuntimeError("mapping outcome unknown")
    with pytest.raises(RuntimeError, match="mapping outcome unknown"):
        allocator.check_decode_capacity(num_tokens=4)
    with pytest.raises(RuntimeError, match="tear down"):
        allocator.check_decode_capacity(num_tokens=4)


@pytest.fixture
def composite_backings(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        int64=object(), tensor=lambda values, **_: Tensor(values),
        device=lambda value: SimpleNamespace(type=value)))
    composite = Composite(256)
    swa = make_backing()
    full = SWABacking(Manager(), native_pages=13,
                      native_page_bytes=3, physical_page_bytes=8)
    assert swa.acquire({1}) and full.acquire({2})
    MODULE.bind_native_swa(composite, swa, full)
    return composite, swa, full


@pytest.mark.parametrize("method", ["alloc_extend", "alloc_decode", "alloc_extend_swa_tail"])
@pytest.mark.parametrize("miss", ["full", "swa"])
def test_c128_all_child_rollback_preserves_prefix(composite_backings, method, miss):
    composite, swa, full = composite_backings
    composite.full_attn_allocator.output.insert(0, 513)
    composite.swa_attn_allocator.output.insert(0, 257)
    failed = full if miss == "full" else swa
    failed.manager.budget = len(failed.manager.live)
    before = [(set(b.live), dict(b.refs), set(b.manager.live)) for b in (swa, full)]
    assert getattr(composite, method)() is None
    assert composite.mapping == {512: 256}
    assert composite.full_attn_allocator.live == {2}
    assert composite.swa_attn_allocator.live == {1}
    assert before == [(set(b.live), dict(b.refs), set(b.manager.live)) for b in (swa, full)]
    failed.manager.budget = 20
    assert getattr(composite, method)() is not None
    assert full.live == {2, 5} and swa.live == {1, 3}


def test_c128_prepare_rollback_preserves_prefix_leases(composite_backings):
    composite, swa, full = composite_backings
    journal = composite._kvcached_prepare_transaction()
    with journal:
        assert composite.alloc_extend() is not None
    # New native page 5 shares physical page 1 with the prefix and also needs page 2.
    assert full.live == {2, 5}
    assert dict(full.refs) == {1: 2, 2: 1}
    assert full.manager.live == {1, 2}

    # A peer rejected this preparation after the local allocation succeeded.
    journal.rollback()
    assert composite.full_attn_allocator.live == full.live == {2}
    assert composite.swa_attn_allocator.live == swa.live == {1}
    assert dict(full.refs) == {1: 1}
    assert full.manager.live == {1}
    assert composite.alloc_extend() is not None
    assert full.manager.live == {1, 2}


def test_c128_full_pages_outlive_swa_and_deferred_full_free(composite_backings):
    composite, swa, full = composite_backings
    assert composite.alloc() is not None
    composite.swa_attn_allocator.free_page_ids(Tensor([3]))
    assert swa.live == {1} and full.live == {2, 5}
    native_full = composite.full_attn_allocator
    native_full.free_group = []
    native_full.free_page_ids(Tensor([5]))
    assert full.live == {2, 5}
    native_full.free_group_end()
    assert full.live == {2}
    native_full.clear()
    assert not full.live and not full.manager.live


def test_c128_limit_and_fatal_failure_are_composite_wide(composite_backings):
    composite, swa, full = composite_backings
    full.manager.budget = 0
    assert composite.full_attn_allocator.available_size() < composite._conserve_full_available_size()
    full.manager.fault = RuntimeError("C128 map failed")
    with pytest.raises(RuntimeError, match="C128 map failed"):
        composite.alloc()
    for child in (composite.full_attn_allocator, composite.swa_attn_allocator):
        with pytest.raises(RuntimeError, match="tear down"):
            child.clear()


def test_c128_capacity_rejected_before_binding(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    composite = Composite(256)
    original_alloc = composite.full_attn_allocator.alloc
    with pytest.raises(ValueError, match="cannot cover"):
        MODULE.bind_native_swa(composite, make_backing(), make_backing())
    assert composite.full_attn_allocator.alloc == original_alloc


@pytest.fixture
def all_backings(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        int64=object(), tensor=lambda values, **_: Tensor(values)))
    composite = Composite(256)
    swa = make_backing()
    compressed = tuple(SWABacking(Manager(), native_pages=13,
                                 native_page_bytes=width, physical_page_bytes=8)
                       for width in (3, 6, 4))
    assert swa.acquire({1})
    for backing in compressed:
        assert backing.acquire({2})
    MODULE.bind_native_swa(composite, swa, compressed[0], full_backings=compressed[1:])
    return composite, swa, compressed


@pytest.mark.parametrize("method", ["alloc_extend", "alloc_decode", "alloc_extend_swa_tail"])
@pytest.mark.parametrize("miss_index", [0, 1, 2, 3])
def test_compressed_transaction_rolls_back_every_pool(all_backings, method, miss_index):
    composite, swa, compressed = all_backings
    pools = (*compressed, swa)
    composite.full_attn_allocator.output.insert(0, 513)
    composite.swa_attn_allocator.output.insert(0, 257)
    failed = pools[miss_index]
    failed.manager.budget = len(failed.manager.live)
    before = [(set(b.live), dict(b.refs), set(b.manager.live)) for b in pools]
    assert getattr(composite, method)() is None
    assert composite.mapping == {512: 256}
    assert composite.full_attn_allocator.live == {2}
    assert composite.swa_attn_allocator.live == {1}
    assert before == [(set(b.live), dict(b.refs), set(b.manager.live)) for b in pools]
    failed.manager.budget = 20
    assert getattr(composite, method)() is not None
    assert all(b.live == {2, 5} for b in compressed)
    assert swa.live == {1, 3}


def test_compressed_lifetimes_and_minimum_capacity(all_backings):
    composite, swa, compressed = all_backings
    assert composite.alloc() is not None
    composite.swa_attn_allocator.free_page_ids(Tensor([3]))
    assert swa.live == {1}
    assert all(b.live == {2, 5} for b in compressed)
    native = composite.full_attn_allocator
    for backing in compressed:
        backing.manager.budget = len(backing.manager.live)
    assert native.available_size() == min(b.available_pages() * 256 for b in compressed)
    native.free_group = []
    native.free_page_ids(Tensor([5]))
    assert all(b.live == {2, 5} for b in compressed)
    native.free_group_end()
    assert all(b.live == {2} for b in compressed)
    native.clear()
    assert all(not b.live and not b.manager.live for b in compressed)


def test_late_compressed_backend_error_poisons_all_children(all_backings):
    composite, _swa, compressed = all_backings
    compressed[-1].manager.fault = RuntimeError("indexer map failed")
    with pytest.raises(RuntimeError, match="indexer map failed"):
        composite.alloc()
    assert composite.mapping == {512: 256}
    for child in (composite.full_attn_allocator, composite.swa_attn_allocator):
        with pytest.raises(RuntimeError, match="tear down"):
            child.clear()


def _assert_all_backings_ready_before_resize(monkeypatch, composite, pools, waiting_pool, method):
    lock, waiting, ready = RLock(), Event(), Event()
    operations = []

    def wait_ready():
        waiting.set()
        assert ready.wait(3), "test initialization was not released"

    monkeypatch.setattr(waiting_pool.manager, "wait_ready", wait_ready)
    for pool in pools:
        original = pool.manager.apply_resize_target

        def apply(original=original, pool=pool):
            operations.append(pool)
            with lock:
                original()

        monkeypatch.setattr(pool.manager, "apply_resize_target", apply)
    before = dict(composite.mapping)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(getattr(composite, method))
        try:
            assert waiting.wait(1)
            assert not result.done() and composite.mapping == before
            acquired = lock.acquire(timeout=1)
            try:
                assert acquired, "a side pool held the initialization lock"
                assert not operations, "resize started before all side pools were ready"
            finally:
                if acquired:
                    lock.release()
        finally:
            ready.set()
        assert result.result(timeout=1) is not None
    assert set(operations) == set(pools)


@pytest.mark.parametrize("method", ["alloc", "alloc_extend", "alloc_decode", "alloc_extend_swa_tail"])
@pytest.mark.parametrize("waiting_side", ["swa", "full"])
def test_c128_waits_for_both_managers_before_any_resize(composite_backings, monkeypatch, method, waiting_side):
    composite, swa, full = composite_backings
    _assert_all_backings_ready_before_resize(
        monkeypatch, composite, (swa, full), swa if waiting_side == "swa" else full, method)


@pytest.mark.parametrize("method", ["alloc", "alloc_extend", "alloc_decode", "alloc_extend_swa_tail"])
@pytest.mark.parametrize("waiting_index", [1, 2])
def test_additional_compressed_pools_settle_before_resize(all_backings, monkeypatch, method, waiting_index):
    composite, swa, compressed = all_backings
    _assert_all_backings_ready_before_resize(
        monkeypatch, composite, (swa, *compressed), compressed[waiting_index], method)
