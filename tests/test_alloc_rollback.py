# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KVCacheManager partial-allocation rollback (issue #364).

GPU-free: ``kvcached.vmm_ops`` is stubbed with pure-Python fakes when the
compiled extension is unavailable, so these tests run without a GPU or a
built C++ extension. They cover the multi-instance race where
``available_size()`` reports enough logical capacity but the shared physical
pool is exhausted by the time ``PageAllocator.alloc_page()`` runs: ``alloc()``
must roll back partially consumed reserved blocks and page blocks and return
``None`` (allocation miss) instead of leaking them.
"""
import sys
import threading
import types
from typing import Any, Dict, List, Optional

import pytest

BLOCKS_PER_PAGE = 4


class FakePage:
    """Pure-Python stand-in for kvcached_cpp.InternalPage."""

    def __init__(self, page_id: int):
        self.page_id = page_id
        self.free_list: List[int] = []

    def init(self, block_mem_size: int) -> None:
        start = self.page_id * BLOCKS_PER_PAGE
        self.free_list = list(range(start, start + BLOCKS_PER_PAGE))

    def num_free_blocks(self) -> int:
        return len(self.free_list)

    def alloc(self, num: int) -> List[int]:
        out, self.free_list = self.free_list[:num], self.free_list[num:]
        return out

    def free_batch(self, idxs: List[int]) -> None:
        self.free_list.extend(idxs)

    def full(self) -> bool:
        return not self.free_list

    def empty(self) -> bool:
        return len(self.free_list) == BLOCKS_PER_PAGE

    @staticmethod
    def get_num_blocks(page_size: int, block_mem_size: int) -> int:
        return page_size // block_mem_size

    @staticmethod
    def get_block_range(page_id: int, page_size: int,
                        block_mem_size: int) -> tuple[int, int]:
        return page_id * BLOCKS_PER_PAGE, (page_id + 1) * BLOCKS_PER_PAGE


class FakePageAllocator:
    """Reports ample capacity but fails alloc_page() after ``fail_after``
    pages, mimicking another instance draining the shared physical pool."""

    def __init__(self, fail_after: int):
        self.fail_after = fail_after
        self.num_allocated = 0
        self.freed_pages: List[int] = []

    def alloc_page(self) -> FakePage:
        if self.num_allocated >= self.fail_after:
            raise RuntimeError("physical page pool exhausted")
        page = FakePage(self.num_allocated)
        self.num_allocated += 1
        return page

    def free_pages(self, page_ids: List[int]) -> None:
        self.freed_pages.extend(page_ids)

    def group_indices_by_page(self, indices: List[int],
                              block_mem_size: int) -> Dict[int, List[int]]:
        grouped: Dict[int, List[int]] = {}
        for idx in indices:
            grouped.setdefault(idx // BLOCKS_PER_PAGE, []).append(idx)
        return grouped

    def get_resize_target(self) -> int:
        return 0

    def get_num_free_pages(self) -> int:
        return 100  # Logical capacity always looks ample (the race).

    def get_avail_physical_pages(self) -> int:
        return 100

    def get_num_reserved_pages(self) -> int:
        return 0


def _install_vmm_ops_stub() -> None:
    stub = types.ModuleType("kvcached.vmm_ops")
    stub.PageAllocator = FakePageAllocator  # type: ignore[attr-defined]
    stub.InternalPage = FakePage  # type: ignore[attr-defined]
    stub.kv_tensors_created = (  # type: ignore[attr-defined]
        lambda group_id=0: True)
    stub.map_to_kv_tensors = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: None)
    stub.unmap_from_kv_tensors = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: None)
    sys.modules["kvcached.vmm_ops"] = stub


try:
    import kvcached.vmm_ops  # noqa: F401
except ImportError:
    _install_vmm_ops_stub()

from kvcached.kv_cache_manager import KVCacheManager  # noqa: E402
from kvcached.lifecycle import LifecycleState  # noqa: E402
from kvcached.locks import NoOpLock  # noqa: E402


def make_manager(fail_after: int,
                 reserved_blocks: Optional[List[int]] = None) -> KVCacheManager:
    """Build a KVCacheManager around fakes without running __init__ (which
    needs the C++ extension, KV tensors, and background threads)."""
    manager = object.__new__(KVCacheManager)
    manager.page_size = BLOCKS_PER_PAGE
    manager.block_mem_size = 1
    manager.page_allocator = FakePageAllocator(fail_after)
    manager.num_avail_blocks = 0
    manager.avail_pages = {}
    manager.full_pages = {}
    manager.reserved_blocks = list(reserved_blocks or [])
    manager.null_block = None
    manager.in_shrink = False
    manager.target_num_blocks = None
    manager._lock = NoOpLock()
    manager._shutdown_lock = threading.Lock()
    manager._shutdown_requested = threading.Event()
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()
    manager._lifecycle = LifecycleState("rollback-test")
    manager._lifecycle.mark_ready()
    return manager


def enable_operation_counters(manager: KVCacheManager) -> None:
    manager._operation_error_lock = threading.RLock()
    manager._operation_error_state = (0, None, None)
    manager._operation_counters = {}
    manager._last_error_code = None
    manager._last_error_timestamp_ns = None


def test_successful_alloc_unchanged():
    manager = make_manager(fail_after=2)
    assert manager.alloc(6) == [0, 1, 2, 3, 4, 5]
    assert manager.num_avail_blocks == 2


@pytest.mark.parametrize("indexed_before_reservation", [False, True])
@pytest.mark.parametrize("release", ["free_reserved", "resize"])
def test_external_release_restores_page_eviction_candidate(
    monkeypatch, indexed_before_reservation, release,
):
    import kvcached.kv_cache_manager as kcm
    from kvcached.integration.vllm.page_eviction import PageEvictionIndex

    monkeypatch.setattr(kcm, "InternalPage", FakePage)
    manager = make_manager(fail_after=2)
    manager.page_allocator.resize = lambda size: False
    index = PageEvictionIndex(manager)
    cached = manager.alloc(1)
    assert cached is not None
    index.add(cached[0])
    if indexed_before_reservation:
        assert index.victims(1) == cached
    assert manager.try_to_reserve(1)
    assert index.victims(1) == []

    if release == "free_reserved":
        manager.free_reserved()
    else:
        assert manager.resize(0) is False

    assert manager.reserved_blocks == []
    assert index.victims(1) == cached
    index.remove(cached[0])
    manager.free(cached)
    assert manager.page_allocator.freed_pages == [0]


def test_external_release_only_refreshes_affected_pages(monkeypatch):
    from unittest.mock import Mock

    import kvcached.kv_cache_manager as kcm
    from kvcached.integration.vllm.page_eviction import PageEvictionIndex

    monkeypatch.setattr(kcm, "InternalPage", FakePage)
    manager = make_manager(fail_after=4096)
    manager.page_allocator.get_num_free_pages = lambda: 4096
    manager.page_allocator.get_avail_physical_pages = lambda: 4096
    allocated = manager.alloc(4096 * BLOCKS_PER_PAGE)
    assert allocated is not None
    index = PageEvictionIndex(manager)
    for block in allocated[::BLOCKS_PER_PAGE]:
        index.add(block)
    assert index.victims(1) == []
    occupancy = Mock(wraps=manager.get_page_occupancy)
    manager.get_page_occupancy = occupancy

    # Release the other occupants of just one of 4096 pinned pages directly.
    manager.free(allocated[1:BLOCKS_PER_PAGE])
    assert index.victims(1) == [allocated[0]]
    assert sum(len(call.args[0]) for call in occupancy.call_args_list) == 2


def test_page_release_watchers_are_weak_and_manager_local(monkeypatch):
    import gc
    import weakref

    import kvcached.kv_cache_manager as kcm
    from kvcached.integration.vllm.page_eviction import PageEvictionIndex

    monkeypatch.setattr(kcm, "InternalPage", FakePage)
    manager = make_manager(fail_after=2)
    other = make_manager(fail_after=2)
    allocated = manager.alloc(2)
    assert allocated is not None
    cached, pinned = allocated
    index = PageEvictionIndex(manager)
    index.add(cached)
    survivor = PageEvictionIndex(manager)
    survivor.add(cached)
    assert index.victims(1) == survivor.victims(1) == []
    owner = weakref.ref(index)
    del index
    gc.collect()
    assert owner() is None

    # A release in another manager with the same page ids cannot notify us.
    other.free(other.alloc(1))
    assert not survivor.dirty
    manager.free([pinned])
    assert survivor.victims(1) == [cached]


def test_manager_page_counters_include_retained_page_reuse():
    class RetainingPageAllocator(FakePageAllocator):
        def __init__(self):
            super().__init__(fail_after=0)
            self.reserved_pages = []
            self.map_count = 0
            self.unmap_count = 0

        def preallocate(self):
            self.reserved_pages.append(FakePage(self.map_count))
            self.map_count += 1

        def alloc_page(self):
            return self.reserved_pages.pop()

        def free_pages(self, page_ids):
            self.freed_pages.extend(page_ids)
            self.reserved_pages.extend(FakePage(page_id) for page_id in page_ids)

        def trim(self):
            self.unmap_count += len(self.reserved_pages)
            self.reserved_pages.clear()

    manager = make_manager(fail_after=0)
    enable_operation_counters(manager)
    allocator = RetainingPageAllocator()
    manager.page_allocator = allocator

    allocator.preallocate()
    assert manager.operation_snapshot_dict()["manager_page_allocations_total"] == 0
    for handoffs in (1, 2):
        blocks = manager.alloc(BLOCKS_PER_PAGE)
        assert blocks == list(range(BLOCKS_PER_PAGE))
        manager.free(blocks)
        data = manager.operation_snapshot_dict()
        assert data["manager_page_allocations_total"] == handoffs
        assert data["manager_page_releases_total"] == handoffs
        assert data["manager_page_allocation_failures_total"] == 0
        assert data["freed_blocks_total"] == handoffs * BLOCKS_PER_PAGE
        assert allocator.map_count == 1
        assert allocator.unmap_count == 0

    manager.trim()
    data = manager.operation_snapshot_dict()
    assert allocator.unmap_count == 1
    assert data["trim_successes_total"] == 1
    assert data["manager_page_allocations_total"] == 2
    assert data["manager_page_releases_total"] == 2


@pytest.mark.parametrize("fail_after", [0, 1])
def test_consistency_error_is_not_an_allocation_miss(monkeypatch, fail_after):
    from kvcached.errors import StateConsistencyError

    manager = make_manager(fail_after=fail_after, reserved_blocks=[10])
    enable_operation_counters(manager)
    alloc_page = manager.page_allocator.alloc_page
    error = StateConsistencyError("unmap commit unconfirmed")

    def fail():
        if manager.page_allocator.num_allocated >= fail_after:
            raise error
        return alloc_page()

    monkeypatch.setattr(manager.page_allocator, "alloc_page", fail)
    with pytest.raises(StateConsistencyError, match="commit unconfirmed") as exc_info:
        manager.alloc(BLOCKS_PER_PAGE + 2)
    assert exc_info.value is error
    assert manager.page_allocator.freed_pages == []
    assert manager.reserved_blocks == []
    assert len(manager.full_pages) == fail_after
    counters = manager._operation_counters
    assert counters["allocation_requests_total"] == 1
    assert counters["allocation_failures_total"] == 1
    assert counters["allocation_errors_total"] == 1
    assert counters["operation_errors_total"] == 1
    assert counters["manager_page_allocation_failures_total"] == 1
    assert counters.get("manager_page_allocations_total", 0) == fail_after
    assert counters.get("capacity_exhausted_total", 0) == 0
    assert counters.get("allocated_blocks_total", 0) == 0
    assert counters.get("free_requests_total", 0) == 0
    assert manager._last_error_code == "allocation_failed"


def test_quarantined_map_returns_miss_without_handing_out_blocks(monkeypatch):
    from kvcached.errors import MapQuarantinedError

    manager = make_manager(fail_after=0, reserved_blocks=[10, 11])
    enable_operation_counters(manager)

    def fail():
        raise MapQuarantinedError("unpublished page quarantined")

    monkeypatch.setattr(manager.page_allocator, "alloc_page", fail)
    assert manager.alloc(4) is None
    assert manager.reserved_blocks == [10, 11]
    counters = manager._operation_counters
    assert counters["manager_page_allocation_failures_total"] == 1
    assert counters["allocation_failures_total"] == 1
    assert counters["capacity_exhausted_total"] == 1
    assert counters.get("operation_errors_total", 0) == 0
    assert counters.get("free_requests_total", 0) == 0


@pytest.mark.parametrize("rejected", [False, True])
@pytest.mark.parametrize("defer_release", [False, True])
def test_deferred_resize_result_is_not_reported_as_applied(monkeypatch, rejected, defer_release):
    from kvcached.errors import QuarantinedResizeError

    manager = make_manager(fail_after=2)
    enable_operation_counters(manager)
    manager.defer_physical_release = defer_release
    blocks = manager.alloc(BLOCKS_PER_PAGE)
    manager.in_shrink = True
    manager.target_num_blocks = BLOCKS_PER_PAGE

    def resize(_size):
        if rejected:
            raise QuarantinedResizeError("quarantined pages")
        return False

    monkeypatch.setattr(manager.page_allocator, "resize", resize, raising=False)
    manager.free(blocks)
    if defer_release:
        assert manager.operation_snapshot_dict()["resize_errors_total"] == 0
        manager.release_retired_pages_through(manager.capture_physical_release_marker())
    assert manager._operation_counters.get("resize_completions_total", 0) == 0
    assert manager._operation_counters["free_successes_total"] == 1
    assert manager._operation_counters["freed_blocks_total"] == BLOCKS_PER_PAGE
    data = manager.operation_snapshot_dict()
    assert data["resize_errors_total"] == data["operation_errors_total"] == int(rejected)
    assert data["free_errors_total"] == data["free_failures_total"] == 0
    if rejected:
        assert data["last_error_code"] == "resize_failed"
        manager.free([])
        assert manager.operation_snapshot_dict()["resize_errors_total"] == 1
        assert manager._resize_rejected
        assert not manager.in_shrink
        assert manager.target_num_blocks is None
        assert manager.alloc(1) is not None
    else:
        assert manager.in_shrink
        assert manager.target_num_blocks == BLOCKS_PER_PAGE


def test_deferred_resize_completion_is_counted_once(monkeypatch):
    manager = make_manager(fail_after=2)
    enable_operation_counters(manager)
    blocks = manager.alloc(BLOCKS_PER_PAGE)
    manager.in_shrink = True
    manager.target_num_blocks = BLOCKS_PER_PAGE
    resize_calls = []

    def resize(size):
        resize_calls.append(size)
        return True

    monkeypatch.setattr(manager.page_allocator, "resize", resize, raising=False)
    manager.free(blocks)
    manager.free([])

    assert resize_calls == [BLOCKS_PER_PAGE * manager.block_mem_size]
    assert not manager.in_shrink
    assert manager.target_num_blocks is None
    assert manager._operation_counters["resize_completions_total"] == 1


@pytest.mark.parametrize("failure_stage", ["free_pages", "resize"])
@pytest.mark.parametrize("fatal", [False, True])
def test_caller_free_progress_survives_later_allocator_failure(
        monkeypatch, failure_stage, fatal):
    from kvcached.errors import StateConsistencyError

    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)
    blocks = manager.alloc(BLOCKS_PER_PAGE)
    page = manager.full_pages[0]
    manager.in_shrink = failure_stage == "resize"
    manager.target_num_blocks = BLOCKS_PER_PAGE if manager.in_shrink else None
    error_type = StateConsistencyError if fatal else RuntimeError
    error = error_type("injected allocator failure")

    def fail(_value):
        assert page.empty()
        assert manager._operation_counters["freed_blocks_total"] == BLOCKS_PER_PAGE
        raise error

    monkeypatch.setattr(manager.page_allocator, failure_stage, fail, raising=False)
    with pytest.raises(error_type, match="injected allocator failure") as exc_info:
        manager.free(blocks)

    assert exc_info.value is error
    data = manager.operation_snapshot_dict()
    assert data["freed_blocks_total"] == BLOCKS_PER_PAGE
    assert data["free_requests_total"] == 1
    assert data["free_failures_total"] == 1
    assert data["free_successes_total"] == 0
    assert data["free_errors_total"] == 1
    assert data["operation_errors_total"] == 1
    assert data["manager_page_releases_total"] == int(failure_stage == "resize")
    assert data["resize_completions_total"] == 0


@pytest.mark.parametrize("failed_page", [0, 1])
def test_caller_free_counts_only_completed_page_batches(monkeypatch, failed_page):
    manager = make_manager(fail_after=2)
    enable_operation_counters(manager)
    blocks = manager.alloc(2 * BLOCKS_PER_PAGE)
    pages = dict(manager.full_pages)

    def fail(_indices):
        assert manager._operation_counters.get("freed_blocks_total", 0) == (
            failed_page * BLOCKS_PER_PAGE)
        raise RuntimeError("injected page free failure")

    monkeypatch.setattr(pages[failed_page], "free_batch", fail)
    with pytest.raises(RuntimeError, match="injected page free failure"):
        manager.free(blocks)

    data = manager.operation_snapshot_dict()
    assert data["freed_blocks_total"] == failed_page * BLOCKS_PER_PAGE
    assert data["free_requests_total"] == 1
    assert data["free_failures_total"] == 1
    assert data["free_successes_total"] == 0
    assert data["manager_page_releases_total"] == 0
    assert manager.page_allocator.freed_pages == []
    for page_id, page in pages.items():
        assert page.num_free_blocks() == (BLOCKS_PER_PAGE if page_id < failed_page else 0)


def test_internal_free_keeps_tuple_contract_without_caller_accounting():
    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)
    blocks = manager.alloc(BLOCKS_PER_PAGE)
    assert blocks is not None

    assert manager._free(blocks) == (BLOCKS_PER_PAGE, False)
    assert manager._free([]) == (0, False)
    data = manager.operation_snapshot_dict()
    assert data["manager_page_releases_total"] == 1
    assert data["freed_blocks_total"] == 0
    assert data["free_requests_total"] == 0
    assert data["free_successes_total"] == 0


def test_deferred_release_counts_only_the_completed_handoff():
    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)
    manager.defer_physical_release = True
    blocks = manager.alloc(BLOCKS_PER_PAGE)
    assert blocks is not None
    manager.free(blocks)
    marker = manager.capture_physical_release_marker()

    manager.release_retired_pages_through(marker - 1)
    data = manager.operation_snapshot_dict()
    assert data["free_successes_total"] == 1
    assert data["freed_blocks_total"] == BLOCKS_PER_PAGE
    assert data["manager_page_releases_total"] == 0
    assert manager.page_allocator.freed_pages == []

    manager.release_retired_pages_through(marker)
    manager.release_retired_pages_through(marker)
    data = manager.operation_snapshot_dict()
    assert data["manager_page_releases_total"] == 1
    assert data["free_requests_total"] == 1
    assert data["freed_blocks_total"] == BLOCKS_PER_PAGE
    assert manager.page_allocator.freed_pages == [0]


def test_reused_retired_page_does_not_count_as_a_new_handoff():
    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)
    manager.defer_physical_release = True
    blocks = manager.alloc(BLOCKS_PER_PAGE)
    assert blocks is not None
    manager.free(blocks)
    old_marker = manager.capture_physical_release_marker()
    assert manager.alloc(BLOCKS_PER_PAGE) == blocks
    manager.release_retired_pages_through(old_marker)
    assert manager.page_allocator.freed_pages == []
    assert manager.operation_snapshot_dict()["manager_page_allocations_total"] == 1
    assert manager.operation_snapshot_dict()["manager_page_releases_total"] == 0

    manager.free(blocks)
    manager.release_retired_pages_through(old_marker)
    assert manager.page_allocator.freed_pages == []
    manager.release_retired_pages_through(manager.capture_physical_release_marker())
    assert manager.operation_snapshot_dict()["manager_page_releases_total"] == 1


@pytest.mark.parametrize("stage", ["barrier", "free_pages", "resize"])
@pytest.mark.parametrize("fatal", [False, True])
def test_deferred_release_failure_keeps_logical_free_accounting(monkeypatch, stage, fatal):
    from kvcached.errors import StateConsistencyError

    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)
    manager.defer_physical_release = True
    blocks = manager.alloc(BLOCKS_PER_PAGE)
    assert blocks is not None
    manager.in_shrink = True
    manager.target_num_blocks = 0
    manager.free(blocks)
    error = (StateConsistencyError if fatal else RuntimeError)("release failed")

    def fail(*args):
        raise error

    if stage == "barrier":
        manager.physical_release_barrier = fail
    else:
        monkeypatch.setattr(manager.page_allocator, stage, fail, raising=False)
    with pytest.raises(type(error)) as caught:
        manager.release_retired_pages_through(manager.capture_physical_release_marker())
    assert caught.value is error
    data = manager.operation_snapshot_dict()
    assert data["free_requests_total"] == data["free_successes_total"] == 1
    assert data["free_failures_total"] == 0
    assert data["freed_blocks_total"] == BLOCKS_PER_PAGE
    assert data["manager_page_releases_total"] == int(stage == "resize")
    assert data["free_errors_total"] == data["operation_errors_total"] == 1
    assert data["last_error_code"] == "deferred_release_failed"
    from kvcached.lifecycle import LifecyclePhase

    assert manager.lifecycle_phase is (
        LifecyclePhase.FAILED if fatal else LifecyclePhase.READY)
    if fatal:
        with pytest.raises(type(error)) as readiness_error:
            manager.wait_ready()
        assert readiness_error.value is error


@pytest.mark.parametrize("typed", [False, True])
def test_fatal_mapping_counts_error_and_closes_readiness(monkeypatch, typed):
    from kvcached.errors import StateConsistencyError
    from kvcached.lifecycle import LifecyclePhase

    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)
    error = (StateConsistencyError if typed else RuntimeError)("unsafe mapping")

    def fail():
        raise error

    monkeypatch.setattr(manager.page_allocator, "alloc_page", fail)
    monkeypatch.setattr(manager.page_allocator, "get_transaction_state",
                        lambda: {"state": "FAILED"}, raising=False)
    with pytest.raises(type(error)) as caught:
        manager.alloc(1)
    assert caught.value is error
    assert manager.page_allocator.freed_pages == []
    data = manager.operation_snapshot_dict()
    assert data["manager_page_allocation_failures_total"] == 1
    assert data["allocation_failures_total"] == 1
    assert data["allocation_errors_total"] == 1
    assert data["capacity_exhausted_total"] == 0
    assert manager.lifecycle_phase is LifecyclePhase.FAILED
    with pytest.raises(type(error)) as readiness_error:
        manager.wait_ready()
    assert readiness_error.value is error


def test_clear_counts_a_page_in_both_retired_and_available_lists_once(monkeypatch):
    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)
    manager.defer_physical_release = True
    manager.reserve_null_block = False
    blocks = manager.alloc(BLOCKS_PER_PAGE)
    assert blocks is not None
    manager.free(blocks)
    assert manager.avail_pages and manager._retired_pages
    for method in ("stop_prealloc_thread", "start_prealloc_thread", "trim", "reset_free_page_order"):
        monkeypatch.setattr(manager.page_allocator, method, lambda: None, raising=False)
    manager.clear()
    assert manager.page_allocator.freed_pages == [0]
    assert manager.operation_snapshot_dict()["manager_page_releases_total"] == 1
    assert not manager._retired_pages


@pytest.mark.parametrize("pending_shrink", [False, True])
def test_rejected_automatic_resize_does_not_block_healthy_allocations(monkeypatch, pending_shrink):
    from kvcached.errors import QuarantinedResizeError

    manager = make_manager(fail_after=4)
    manager.in_shrink = pending_shrink
    manager.target_num_blocks = BLOCKS_PER_PAGE if pending_shrink else None
    target = [1000]
    calls = []

    def resize(size):
        calls.append(size)
        raise QuarantinedResizeError("quarantined pages")

    monkeypatch.setattr(manager.page_allocator, "resize", resize, raising=False)
    monkeypatch.setattr(manager.page_allocator, "get_resize_target", lambda: target[0])
    assert manager.alloc(1) is not None
    assert manager.alloc(1) is not None
    assert calls == [1000]
    assert manager._resize_rejected
    target[0] = 2000
    assert manager.alloc(1) is not None
    assert calls == [1000, 2000]
    assert not manager.in_shrink
    assert manager.target_num_blocks is None
    with pytest.raises(QuarantinedResizeError):
        manager.resize(2000)


def test_miss_with_no_partial_state_is_clean():
    manager = make_manager(fail_after=1)
    assert manager.alloc(BLOCKS_PER_PAGE) == [0, 1, 2, 3]
    # Second alloc needs a fresh page and fails immediately; nothing partial.
    assert manager.alloc(BLOCKS_PER_PAGE) is None
    assert manager.num_avail_blocks == 0
    assert list(manager.full_pages) == [0]
    assert manager.avail_pages == {}


def test_page_blocks_rolled_back_and_reusable():
    manager = make_manager(fail_after=1)
    assert manager.alloc(2) == [0, 1]
    # Consumes the page's remaining [2, 3], then alloc_page() fails.
    assert manager.alloc(4) is None
    # The two page blocks must be back and allocatable again.
    assert manager.num_avail_blocks == 2
    assert 0 in manager.avail_pages
    assert manager.alloc(2) == [2, 3]


def test_partial_alloc_rollback_is_not_counted_as_public_free():
    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)

    # Allocates all four blocks from page 0, then fails to obtain page 1.
    assert manager.alloc(6) is None

    counters = manager._operation_counters
    assert counters["allocation_requests_total"] == 1
    assert counters["allocation_failures_total"] == 1
    assert counters["capacity_exhausted_total"] == 1
    assert counters.get("allocated_blocks_total", 0) == 0
    assert counters.get("free_requests_total", 0) == 0
    assert counters.get("free_successes_total", 0) == 0
    assert counters.get("freed_blocks_total", 0) == 0
    assert counters["manager_page_allocations_total"] == 1
    assert counters["manager_page_allocation_failures_total"] == 1
    assert counters["manager_page_releases_total"] == 1


def test_failed_allocation_rollback_does_not_count_caller_free_progress(monkeypatch):
    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)

    def fail(_page_ids):
        raise RuntimeError("rollback unmap failed")

    monkeypatch.setattr(manager.page_allocator, "free_pages", fail)
    with pytest.raises(RuntimeError, match="rollback unmap failed"):
        manager.alloc(BLOCKS_PER_PAGE + 1)

    data = manager.operation_snapshot_dict()
    assert data["allocation_failures_total"] == 1
    assert data["allocation_errors_total"] == 1
    assert data["freed_blocks_total"] == 0
    assert data["free_requests_total"] == 0
    assert data["free_successes_total"] == 0
    assert data["free_failures_total"] == 0
    assert data["manager_page_releases_total"] == 0


def test_reserved_blocks_restored_on_miss():
    manager = make_manager(fail_after=0, reserved_blocks=[10, 11])
    assert manager.alloc(4) is None
    assert manager.reserved_blocks == [10, 11]


def test_mixed_reserved_and_page_blocks_restored():
    manager = make_manager(fail_after=1, reserved_blocks=[10, 11])
    # Takes 2 reserved + all 4 blocks of page 0, then fails needing a 2nd page.
    assert manager.alloc(8) is None
    assert manager.reserved_blocks == [10, 11]
    # Page 0 went fully free again, so it was returned to the page allocator.
    assert manager.page_allocator.freed_pages == [0]
    assert manager.num_avail_blocks == 0
    assert manager.avail_pages == {}
    assert manager.full_pages == {}


def test_alloc_after_rollback_succeeds_when_pool_recovers():
    manager = make_manager(fail_after=1, reserved_blocks=[10])
    assert manager.alloc(6) is None
    # Another instance released memory: the next attempt must see the
    # restored reserved block and succeed from a clean state.
    manager.page_allocator.fail_after = 10
    result = manager.alloc(5)
    assert result is not None
    assert result[0] == 10  # reserved block reused first
    assert len(result) == 5


def _explode_alloc(num: int) -> List[int]:
    """Stand-in for InternalPage.alloc()'s "Not enough free blocks in page"
    invariant failure (csrc/page_allocator.cpp)."""
    raise RuntimeError("Not enough free blocks in page")


def test_page_alloc_failure_on_avail_page_stays_fail_loud(monkeypatch):
    # The rollback handler is scoped to alloc_page(): by the time page.alloc()
    # runs, _pick_avail_page() has removed the page from avail_pages while its
    # blocks are not yet in ret_index, so rollback could not restore it. The
    # invariant failure must propagate, not degrade into a None miss (#430).
    manager = make_manager(fail_after=1)
    assert manager.alloc(2) == [0, 1]
    monkeypatch.setattr(manager.avail_pages[0], "alloc", _explode_alloc)
    with pytest.raises(RuntimeError, match="Not enough free blocks in page"):
        manager.alloc(2)


def test_page_alloc_failure_on_fresh_page_stays_fail_loud(monkeypatch):
    # The failing page does not exist until alloc() creates it, so patch the
    # class rather than an instance.
    manager = make_manager(fail_after=10)
    monkeypatch.setattr(FakePage, "alloc", lambda self, num: _explode_alloc(num))
    with pytest.raises(RuntimeError, match="Not enough free blocks in page"):
        manager.alloc(2)


def test_page_init_failure_is_not_a_physical_allocation_miss(monkeypatch):
    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)
    manager.available_size()
    assert manager._avail_physical_pages_cache is not None

    def fail_init(self, block_mem_size):
        raise RuntimeError("page initialization failed")

    monkeypatch.setattr(FakePage, "init", fail_init)
    with pytest.raises(RuntimeError, match="page initialization failed"):
        manager.alloc(1)

    assert manager.page_allocator.freed_pages == []
    # A real page was handed out even though its block initialization failed.
    assert manager._avail_physical_pages_cache is None
    counters = manager._operation_counters
    assert counters["manager_page_allocations_total"] == 1
    assert counters.get("manager_page_allocation_failures_total", 0) == 0
    assert counters["allocation_failures_total"] == 1
    assert counters["allocation_errors_total"] == 1
    assert counters.get("capacity_exhausted_total", 0) == 0


@pytest.mark.parametrize("async_sched", [False, True])
@pytest.mark.parametrize("outcome", ["success", "capacity", "error"])
def test_operation_snapshot_reports_inflight_allocation(monkeypatch, async_sched, outcome):
    from concurrent.futures import ThreadPoolExecutor

    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)
    if async_sched:
        manager._lock = threading.RLock()
    entered = threading.Event()
    release = threading.Event()
    error = RuntimeError("injected allocation failure")

    def allocate(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        if outcome == "error":
            raise error
        return [0, 1] if outcome == "success" else None

    monkeypatch.setattr(manager, "_alloc_impl", allocate)
    with ThreadPoolExecutor(max_workers=2) as executor:
        allocation = executor.submit(manager.alloc, 2)
        try:
            assert entered.wait(5)
            data = executor.submit(manager.operation_snapshot_dict).result(timeout=5)
            assert data["allocation_requests_total"] == 1
            assert data["allocation_successes_total"] == data["allocation_failures_total"] == 0
        finally:
            release.set()
        if outcome == "error":
            with pytest.raises(RuntimeError) as raised:
                allocation.result(timeout=5)
            assert raised.value is error
        else:
            assert allocation.result(timeout=5) == ([0, 1] if outcome == "success" else None)
    data = manager.operation_snapshot_dict()
    assert data["allocation_successes_total"] == int(outcome == "success")
    assert data["allocated_blocks_total"] == (2 if outcome == "success" else 0)
    assert data["allocation_failures_total"] == int(outcome != "success")
    assert data["capacity_exhausted_total"] == int(outcome == "capacity")
    assert data["allocation_errors_total"] == data["operation_errors_total"] == int(outcome == "error")


@pytest.mark.parametrize("async_sched", [False, True])
@pytest.mark.parametrize("phase", ["construction", "serialization"])
def test_operation_snapshot_formatting_does_not_block_writers(monkeypatch, async_sched, phase):
    from concurrent.futures import ThreadPoolExecutor

    import kvcached.observability as observability

    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)
    if async_sched:
        manager._lock = threading.RLock()
    entered = threading.Event()
    release = threading.Event()
    snapshot_type = observability.KVCachePoolOperationSnapshot

    if phase == "construction":
        def blocked_constructor(**kwargs):
            entered.set()
            assert release.wait(5)
            return snapshot_type(**kwargs)

        monkeypatch.setattr(observability, "KVCachePoolOperationSnapshot", blocked_constructor)
    else:
        serialize = snapshot_type.to_dict

        def blocked_serialization(snapshot):
            entered.set()
            assert release.wait(5)
            return serialize(snapshot)

        monkeypatch.setattr(snapshot_type, "to_dict", blocked_serialization)

    with ThreadPoolExecutor(max_workers=2) as executor:
        snapshot = executor.submit(manager.operation_snapshot_dict)
        try:
            assert entered.wait(5)
            # Both the allocator and counter locks must be free while formatting.
            assert executor.submit(manager.alloc, 1).result(timeout=5) == [0]
        finally:
            release.set()
        data = snapshot.result(timeout=5)
    assert data["allocation_requests_total"] == 0  # The copied state stays detached.
    assert manager.operation_snapshot_dict()["allocation_requests_total"] == 1


@pytest.mark.parametrize("async_sched", [False, True])
def test_operation_sampling_preserves_concurrent_events(async_sched):
    from concurrent.futures import ThreadPoolExecutor

    from kvcached.observability import get_registered_kv_cache_pool_operation_snapshot_dicts
    from kvcached.pool_registry import clear_registered_kv_cache_pools, register_kv_cache_pool

    manager = make_manager(fail_after=2)
    enable_operation_counters(manager)
    if async_sched:
        manager._lock = threading.RLock()
    # Sync scheduling has one allocator writer; async writers already share
    # the business lock. Background error publication has its own cold lock.
    allocator_writers = 3 if async_sched else 1
    manager.alloc(1)  # Keep a page mapped during repeated block allocation/free.
    integration = "concurrent-operation-test"
    register_kv_cache_pool(manager, integration=integration)
    ready = threading.Event()
    stop = threading.Event()
    rounds = 300

    def allocate_and_free():
        assert ready.wait(5)
        for _ in range(rounds):
            blocks = manager.alloc(1)
            assert blocks is not None
            manager.free(blocks)

    def record_errors():
        assert ready.wait(5)
        for _ in range(rounds):
            manager._record_operation_error("post_init_failed", "post_init_errors_total")

    def sample():
        previous: Dict[str, Any] = {}
        samples = 0
        while not stop.is_set():
            direct = manager.operation_snapshot_dict(integration=integration)
            registered = get_registered_kv_cache_pool_operation_snapshot_dicts(
                integration=integration)[0]
            for data in (direct, registered):
                for name, value in data.items():
                    if name.endswith("_total"):
                        assert value >= previous.get(name, 0)
                # Polling may see related counters at different instants.
                if data["last_error_code"] is not None:
                    assert data["last_error_code"] == "post_init_failed"
                    assert data["last_error_timestamp_ns"] is not None
                previous = data
            samples += 1
            ready.set()
        return samples

    try:
        with ThreadPoolExecutor(max_workers=allocator_writers + 3) as executor:
            sampler = executor.submit(sample)
            writers = [executor.submit(allocate_and_free) for _ in range(allocator_writers)]
            writers.extend(executor.submit(record_errors) for _ in range(2))
            try:
                for writer in writers:
                    writer.result(timeout=10)
            finally:
                stop.set()
            assert sampler.result(timeout=5) > 0
        data = manager.operation_snapshot_dict(integration=integration)
        assert data == get_registered_kv_cache_pool_operation_snapshot_dicts(
            integration=integration)[0]
        assert data["allocation_requests_total"] == rounds * allocator_writers + 1
        assert data["allocation_successes_total"] == rounds * allocator_writers + 1
        assert data["free_requests_total"] == data["free_successes_total"] == rounds * allocator_writers
        assert data["freed_blocks_total"] == rounds * allocator_writers
        assert data["post_init_errors_total"] == data["operation_errors_total"] == rounds * 2
        assert data["free_failures_total"] == data["allocation_failures_total"] == 0
    finally:
        clear_registered_kv_cache_pools(integration=integration)


@pytest.mark.parametrize("async_sched", [False, True])
@pytest.mark.parametrize("outcome", ["success", "error", "recorded_error"])
def test_free_start_preserves_inflight_request_and_error_baseline(monkeypatch, async_sched, outcome):
    from concurrent.futures import ThreadPoolExecutor

    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)
    if async_sched:
        manager._lock = threading.RLock()
    manager._record_operation_error("post_init_failed", "post_init_errors_total")
    entered = threading.Event()
    release = threading.Event()
    error = RuntimeError("injected free failure")

    def free(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        if outcome == "recorded_error":
            manager._record_operation_error("state_inconsistency", "state_inconsistency_errors_total")
        if outcome != "success":
            raise error
        return 0, False

    monkeypatch.setattr(manager, "_free", free)
    with ThreadPoolExecutor(max_workers=2) as executor:
        freeing = executor.submit(manager.free, [])
        try:
            assert entered.wait(5)
            data = executor.submit(manager.operation_snapshot_dict).result(timeout=5)
            assert data["free_requests_total"] == 1
            assert data["free_successes_total"] == data["free_failures_total"] == 0
            assert data["operation_errors_total"] == data["post_init_errors_total"] == 1
        finally:
            release.set()
        if outcome == "success":
            assert freeing.result(timeout=5) is None
        else:
            with pytest.raises(RuntimeError) as raised:
                freeing.result(timeout=5)
            assert raised.value is error
    data = manager.operation_snapshot_dict()
    assert data["free_successes_total"] == int(outcome == "success")
    assert data["free_failures_total"] == int(outcome != "success")
    assert data["operation_errors_total"] == 1 + int(outcome != "success")
    assert data["free_errors_total"] == int(outcome == "error")
    assert data["state_inconsistency_errors_total"] == int(outcome == "recorded_error")
    assert data["freed_blocks_total"] == 0


@pytest.mark.parametrize("async_sched", [False, True])
def test_operation_sampling_and_success_do_not_acquire_error_lock(monkeypatch, async_sched):
    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)
    if async_sched:
        manager._lock = threading.RLock()

    class ForbiddenLock:
        def __enter__(self):
            pytest.fail("normal operations and sampling must not acquire the error lock")

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(manager, "_operation_error_lock", ForbiddenLock())
    blocks = manager.alloc(1)
    assert blocks == [0]
    manager.free(blocks)
    assert manager.alloc(2 * BLOCKS_PER_PAGE) is None
    data = manager.operation_snapshot_dict()
    assert data["allocation_requests_total"] == 2
    assert data["allocation_successes_total"] == data["allocation_failures_total"] == 1
    assert data["free_successes_total"] == data["freed_blocks_total"] == 1
    assert data["operation_errors_total"] == 0


def test_async_counter_updates_use_existing_allocator_writer_lock():
    import time
    from concurrent.futures import ThreadPoolExecutor

    class YieldingCounters(Dict[str, int]):
        def get(self, key, default=None):
            result = super().get(key, default)
            if key in ("allocation_requests_total", "allocation_successes_total",
                       "allocated_blocks_total", "free_requests_total",
                       "free_successes_total", "freed_blocks_total"):
                # Force a thread switch between read and write. Accuracy must
                # come from the allocator lock, not a claim about Python +=.
                time.sleep(0.00001)
            return result

    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)
    manager._lock = threading.RLock()
    manager._operation_counters = YieldingCounters()
    anchor = manager.alloc(1)
    assert anchor is not None

    def worker():
        for _ in range(40):
            blocks = manager.alloc(1)
            assert blocks is not None
            manager.free(blocks)

    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs = [executor.submit(worker) for _ in range(3)]
        for job in jobs:
            job.result(timeout=10)
    data = manager.operation_snapshot_dict()
    assert data["allocation_requests_total"] == data["allocation_successes_total"] == 121
    assert data["free_requests_total"] == data["free_successes_total"] == 120
    assert data["allocated_blocks_total"] == 121
    assert data["freed_blocks_total"] == 120
    manager.free(anchor)


@pytest.mark.parametrize("async_sched", [False, True])
def test_post_init_error_does_not_hide_waiting_foreground_free_error(monkeypatch, async_sched):
    from concurrent.futures import ThreadPoolExecutor

    manager = make_manager(fail_after=1)
    enable_operation_counters(manager)
    if async_sched:
        manager._lock = threading.RLock()
    manager.world_size = 1
    manager.pp_rank = 0
    manager.group_id = 0
    manager._post_init_done.clear()
    entered_init, entered_free, release_init = (threading.Event() for _ in range(3))
    init_error = RuntimeError("injected initialization failure")
    free_error = RuntimeError("independent foreground free failure")
    init_globals = getattr(KVCacheManager._post_init, "__globals__")
    monkeypatch.setitem(init_globals, "kv_tensors_created", lambda **kwargs: True)
    monkeypatch.setitem(init_globals, "broadcast_kv_tensors_created", lambda *args, **kwargs: True)

    def reserve():
        entered_init.set()
        assert release_init.wait(5)
        raise init_error

    def free(*args, **kwargs):
        entered_free.set()
        manager._wait_post_init()
        raise free_error

    monkeypatch.setattr(manager, "_reserve_null_block", reserve)
    monkeypatch.setattr(manager, "_free", free)
    with ThreadPoolExecutor(max_workers=2) as executor:
        initializing = executor.submit(manager._post_init)
        assert entered_init.wait(5)
        freeing = executor.submit(manager.free, [])
        try:
            assert entered_free.wait(5)
            assert manager.operation_snapshot_dict()["free_requests_total"] == 1
        finally:
            release_init.set()
        for future, expected in ((initializing, init_error), (freeing, free_error)):
            with pytest.raises(RuntimeError) as raised:
                future.result(timeout=5)
            assert raised.value is expected
    data = manager.operation_snapshot_dict()
    assert data["operation_errors_total"] == 2
    assert data["post_init_errors_total"] == data["free_errors_total"] == 1
    assert data["free_failures_total"] == 1
    assert data["last_error_code"] == "free_failed"
    assert data["last_error_timestamp_ns"] is not None
