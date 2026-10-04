# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import threading
from unittest import mock

import pytest

_torch_mock = mock.MagicMock()
_torch_mock.__version__ = "2.6.0"
sys.modules.setdefault("torch", _torch_mock)
sys.modules.setdefault("torch.cuda", _torch_mock.cuda)
sys.modules.setdefault("torch.utils", _torch_mock.utils)
sys.modules.setdefault("torch.utils.cpp_extension", _torch_mock.utils.cpp_extension)
sys.modules.setdefault("posix_ipc", mock.MagicMock())
sys.modules.setdefault("kvcached.vmm_ops", mock.MagicMock())


class FakeAllocator:
    def __init__(self):
        self.freed_pages = []
        self.resize = mock.Mock(return_value=True)
        self.get_resize_target = mock.Mock(return_value=0)
        self.alloc_page = mock.Mock(side_effect=AssertionError("Mapped page must be reused"))

    def free_pages(self, page_ids):
        self.freed_pages.extend(page_ids)

    def group_indices_by_page(self, indices, block_mem_size):
        return {FakePage.page_id: list(indices)}


class FakePage:
    page_id = 7

    def __init__(self, page_id=7, free=False):
        self.page_id = page_id
        self.free_blocks = [100] if free else []

    def free_batch(self, indices):
        self.free_blocks.extend(indices)

    def num_free_blocks(self):
        return len(self.free_blocks)

    def empty(self):
        return self.free_blocks == [100]

    def full(self):
        return not self.free_blocks

    def alloc(self, count):
        result, self.free_blocks = self.free_blocks[:count], self.free_blocks[count:]
        return result


def _manager():
    from kvcached.kv_cache_manager import KVCacheManager
    from kvcached.locks import NoOpLock

    manager = object.__new__(KVCacheManager)
    manager._lock = NoOpLock()
    manager.page_allocator = FakeAllocator()
    manager.block_mem_size = 1
    manager.defer_physical_release = True
    manager._physical_release_epoch = 0
    manager._retired_pages = []
    manager.in_shrink = False
    manager.target_num_blocks = None
    manager._avail_physical_pages_cache = mock.sentinel.cached_capacity
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()
    manager.reserved_blocks = []
    manager.avail_pages = {}
    manager.full_pages = {}
    manager.num_avail_blocks = 0
    return manager


def _retire(manager, entries):
    manager._retired_pages = entries
    manager.avail_pages = {pid: FakePage(pid, free=True) for _, ids in entries for pid in ids}
    manager.num_avail_blocks = len(manager.avail_pages)


def test_async_free_retires_empty_page_without_unmapping():
    manager = _manager()
    manager._wait_post_init = mock.Mock()
    manager.block_mem_size = 1
    manager.reserved_blocks = []
    manager.avail_pages = {}
    manager.full_pages = {FakePage.page_id: FakePage()}
    manager.num_avail_blocks = 0

    manager.free([100])

    assert manager.page_allocator.freed_pages == []
    assert manager._retired_pages == [(1, [FakePage.page_id])]
    assert manager.capture_physical_release_marker() == 1
    assert manager.num_avail_blocks == 1
    assert FakePage.page_id in manager.avail_pages
    assert manager._avail_physical_pages_cache is mock.sentinel.cached_capacity


def test_release_retired_pages_honors_marker():
    manager = _manager()
    manager._physical_release_epoch = 3
    _retire(manager, [(1, [10]), (2, [11, 12]), (3, [13])])

    manager.release_retired_pages_through(2)

    assert manager.page_allocator.freed_pages == [10, 11, 12]
    assert manager._retired_pages == [(3, [13])]
    assert manager._avail_physical_pages_cache is None
    assert list(manager.avail_pages) == [13]
    assert manager.num_avail_blocks == 1


def test_release_retired_pages_is_idempotent():
    manager = _manager()
    manager._physical_release_epoch = 1
    _retire(manager, [(1, [10])])

    manager.release_retired_pages_through(1)
    manager.release_retired_pages_through(1)

    assert manager.page_allocator.freed_pages == [10]


@pytest.mark.parametrize("failure_stage", ["barrier", "unmap"])
def test_failed_release_preserves_retirement_for_retry(failure_stage):
    manager = _manager()
    _retire(manager, [(1, [10])])
    events: list[object] = []

    def barrier():
        events.append("barrier")
        if failure_stage == "barrier":
            raise RuntimeError("injected failure")

    def free_pages(_pages):
        events.append("unmap")
        raise RuntimeError("injected failure")

    manager.physical_release_barrier = barrier
    manager.page_allocator.free_pages = free_pages
    with pytest.raises(RuntimeError, match="injected failure"):
        manager.release_retired_pages_through(1)
    assert manager._retired_pages == [(1, [10])]
    assert manager._avail_physical_pages_cache is mock.sentinel.cached_capacity
    assert events == (["barrier"] if failure_stage == "barrier" else ["barrier", "unmap"])

    manager.physical_release_barrier = lambda: events.append("retry_barrier")
    manager.page_allocator.free_pages = lambda pages: events.append(("retry_unmap", pages))
    manager.release_retired_pages_through(1)
    assert events[-2:] == ["retry_barrier", ("retry_unmap", [10])]
    assert manager._retired_pages == []
    assert manager._avail_physical_pages_cache is None


def test_shrink_waits_until_retired_pages_are_physically_released():
    manager = _manager()
    manager.in_shrink = True
    manager.target_num_blocks = 0
    manager._get_num_alloced_blocks = mock.Mock(return_value=0)
    _retire(manager, [(1, [10])])

    manager._maybe_finish_shrink()
    manager.page_allocator.resize.assert_not_called()

    manager.release_retired_pages_through(1)

    manager.page_allocator.resize.assert_called_once_with(0)
    assert manager.in_shrink is False


def test_reuse_before_old_fence_does_not_unmap_reallocated_page(monkeypatch):
    manager = _manager()
    manager.full_pages[7] = FakePage()
    monkeypatch.setattr(manager, "available_size", lambda: manager.num_avail_blocks)
    manager.free([100])
    old_marker = manager.capture_physical_release_marker()
    assert manager.alloc(1) == [100]
    assert manager.num_avail_blocks == 0
    assert manager._retired_pages == []
    assert 7 in manager.full_pages
    manager.release_retired_pages_through(old_marker)
    assert manager.page_allocator.freed_pages == []

    manager.free([100])
    assert manager.capture_physical_release_marker() == old_marker + 1
    manager.release_retired_pages_through(old_marker)
    assert manager.page_allocator.freed_pages == []
    assert manager.num_avail_blocks == 1
    manager.release_retired_pages_through(old_marker + 1)
    assert manager.page_allocator.freed_pages == [7]
    assert manager.num_avail_blocks == 0
    assert manager.avail_pages == {}
    manager.page_allocator.alloc_page.assert_not_called()


def test_reusing_one_page_preserves_other_pending_epochs(monkeypatch):
    manager = _manager()
    _retire(manager, [(1, [7, 8]), (2, [9])])
    monkeypatch.setattr(manager, "available_size", lambda: manager.num_avail_blocks)
    assert manager.alloc(1) == [100]
    assert manager._retired_pages == [(1, [8]), (2, [9])]
    manager.release_retired_pages_through(1)
    assert manager.page_allocator.freed_pages == [8]
    assert 7 in manager.full_pages and 9 in manager.avail_pages
