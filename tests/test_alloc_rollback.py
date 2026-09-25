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
from typing import Dict, List, Optional

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
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()
    return manager


def enable_operation_counters(manager: KVCacheManager) -> None:
    manager._operation_lock = threading.RLock()
    manager._operation_counters = {}
    manager._last_error_code = None
    manager._last_error_timestamp_ns = None


def test_successful_alloc_unchanged():
    manager = make_manager(fail_after=2)
    assert manager.alloc(6) == [0, 1, 2, 3, 4, 5]
    assert manager.num_avail_blocks == 2


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
def test_deferred_resize_result_is_not_reported_as_applied(monkeypatch, rejected):
    from kvcached.errors import QuarantinedResizeError

    manager = make_manager(fail_after=2)
    enable_operation_counters(manager)
    blocks = manager.alloc(BLOCKS_PER_PAGE)
    manager.in_shrink = True
    manager.target_num_blocks = BLOCKS_PER_PAGE

    def resize(_size):
        if rejected:
            raise QuarantinedResizeError("quarantined pages")
        return False

    monkeypatch.setattr(manager.page_allocator, "resize", resize, raising=False)
    manager.free(blocks)
    assert manager._operation_counters.get("resize_completions_total", 0) == 0
    assert manager._operation_counters["free_successes_total"] == 1
    assert manager._operation_counters["freed_blocks_total"] == BLOCKS_PER_PAGE
    if rejected:
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


def test_failed_legacy_growth_retires_empty_pages_until_safe_epoch():
    manager = make_manager(fail_after=1)
    manager.defer_physical_release = True
    manager._retired_pages = []
    manager._physical_release_epoch = 8
    assert manager.alloc(6) is None
    assert list(manager.avail_pages) == [0]
    assert manager.avail_pages[0].empty()
    assert manager.page_allocator.freed_pages == []
    assert manager._retired_pages == [(9, [0])]
    manager.release_retired_pages_through(8)
    assert manager.page_allocator.freed_pages == []
    manager.release_retired_pages_through(9)
    assert manager.page_allocator.freed_pages == [0]


def test_reserved_blocks_restored_on_miss():
    manager = make_manager(fail_after=0, reserved_blocks=[10, 11])
    assert manager.alloc(4) is None
    assert manager.reserved_blocks == [10, 11]


def test_deferred_release_counts_only_acknowledged_physical_release(monkeypatch):
    manager = make_manager(fail_after=2)
    enable_operation_counters(manager)
    manager.defer_physical_release = True
    blocks = manager.alloc(BLOCKS_PER_PAGE)
    manager.free(blocks)
    marker = manager.capture_physical_release_marker()
    assert marker > 0
    assert manager.page_allocator.freed_pages == []
    assert manager._get_operation_counter("manager_page_releases_total") == 0

    def fail(_page_ids):
        raise RuntimeError("release not acknowledged")

    release = manager.page_allocator.free_pages
    monkeypatch.setattr(manager.page_allocator, "free_pages", fail)
    with pytest.raises(RuntimeError, match="release not acknowledged"):
        manager.release_retired_pages_through(marker)
    assert manager._get_operation_counter("manager_page_releases_total") == 0
    assert manager._retired_pages

    monkeypatch.setattr(manager.page_allocator, "free_pages", release)
    manager.release_retired_pages_through(marker)
    assert manager.page_allocator.freed_pages == [0]
    assert manager._get_operation_counter("manager_page_releases_total") == 1
    manager.release_retired_pages_through(marker)
    assert manager._get_operation_counter("manager_page_releases_total") == 1


def test_mixed_reserved_and_page_blocks_restored():
    manager = make_manager(fail_after=1, reserved_blocks=[10, 11])
    # Takes 2 reserved + all 4 blocks of page 0, then fails needing a 2nd page.
    assert manager.alloc(8) is None
    assert manager.reserved_blocks == [10, 11]
    # A failed legacy single-page allocation must not pin an idle partial
    # request while another instance is waiting for the same capacity.
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


def test_growth_backoff_does_not_complete_a_nonexistent_shrink(monkeypatch):
    manager = make_manager(fail_after=1)
    manager.defer_physical_release = False
    assert manager.alloc(2) == [0, 1]
    monkeypatch.setattr(manager, "_physical_growth_retry_is_blocked", lambda: True)
    manager.free([0])
    assert manager.target_num_blocks is None
    assert not manager.in_shrink
    assert manager.num_avail_blocks == 3


def test_growth_backoff_still_counts_resident_reserve_pages(monkeypatch):
    manager = make_manager(fail_after=1, reserved_blocks=[90])
    manager.num_avail_blocks = 2
    monkeypatch.setattr(manager, "_physical_growth_retry_is_blocked", lambda: True)
    monkeypatch.setattr(manager.page_allocator, "get_num_reserved_pages", lambda: 3)
    monkeypatch.setattr(manager.page_allocator, "get_avail_physical_pages",
                        lambda: pytest.fail("backoff must not probe physical capacity"))
    assert manager.available_size() == 2 + 1 + 3 * BLOCKS_PER_PAGE
    manager.in_shrink = True
    assert manager.available_size() == 3


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


def test_unknown_outcome_is_not_an_allocation_miss(monkeypatch):
    from kvcached.tp_ipc_util import MapTransactionOutcomeUnknownError

    manager = make_manager(fail_after=0)

    def fail():
        raise MapTransactionOutcomeUnknownError("unconfirmed map")

    monkeypatch.setattr(manager.page_allocator, "alloc_page", fail)
    with pytest.raises(MapTransactionOutcomeUnknownError, match="unconfirmed map"):
        manager.alloc(1)


@pytest.mark.parametrize("legacy_native_wrapper", [False, True])
def test_latched_failure_surfaces_before_capacity_or_miss(monkeypatch, legacy_native_wrapper):
    # Model a background callback failure, and an older extension that wraps
    # the Python exception in std::runtime_error on the foreground path.
    from kvcached.tp_ipc_util import MapTransactionOutcomeUnknownError

    manager = make_manager(fail_after=0, reserved_blocks=[10])
    failed = not legacy_native_wrapper

    def check():
        if failed:
            raise MapTransactionOutcomeUnknownError("restart the engine")

    def fail():
        nonlocal failed
        failed = True
        raise RuntimeError("wrapped callback failure")

    monkeypatch.setitem(
        KVCacheManager.available_size.__wrapped__.__globals__,
        "raise_if_physical_growth_unresolved", check,
    )
    monkeypatch.setattr(manager.page_allocator, "alloc_page", fail)
    if not legacy_native_wrapper:
        with pytest.raises(MapTransactionOutcomeUnknownError):
            manager.available_size()
    with pytest.raises(MapTransactionOutcomeUnknownError, match="restart"):
        manager.alloc(2)
