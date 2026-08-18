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


def test_successful_alloc_unchanged():
    manager = make_manager(fail_after=2)
    assert manager.alloc(6) == [0, 1, 2, 3, 4, 5]
    assert manager.num_avail_blocks == 2


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
