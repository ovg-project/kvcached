# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for KVCacheManager allocation miss rollback."""

import sys
from unittest import mock

_torch_mock = mock.MagicMock()
_torch_mock.__version__ = "2.6.0"
sys.modules.setdefault("torch", _torch_mock)
sys.modules.setdefault("torch.cuda", _torch_mock.cuda)
sys.modules.setdefault("torch.utils", _torch_mock.utils)
sys.modules.setdefault("torch.utils.cpp_extension", _torch_mock.utils.cpp_extension)
sys.modules.setdefault("posix_ipc", mock.MagicMock())
sys.modules.setdefault("kvcached.vmm_ops", mock.MagicMock())

class FailingPageAllocator:
    def __init__(self):
        self.freed_pages = []

    def get_resize_target(self):
        return 0

    def alloc_page(self):
        raise RuntimeError("physical pool exhausted")

    def group_indices_by_page(self, indices, block_mem_size):
        return {FakePage.page_id: list(indices)}

    def free_pages(self, page_ids):
        self.freed_pages.extend(page_ids)


class FakePage:
    page_id = 7

    def __init__(self, free_blocks=None, capacity=2):
        self.capacity = capacity
        self._free = list(free_blocks if free_blocks is not None else [100, 101])

    def num_free_blocks(self):
        return len(self._free)

    def alloc(self, n):
        out = self._free[:n]
        self._free = self._free[n:]
        return out

    def full(self):
        return not self._free

    def empty(self):
        return len(self._free) == self.capacity

    def free_batch(self, indices):
        self._free.extend(indices)
        self._free.sort()


def _manager_with_allocator(free_size):
    from kvcached.kv_cache_manager import KVCacheManager
    from kvcached.locks import NoOpLock

    manager = object.__new__(KVCacheManager)
    manager._lock = NoOpLock()
    manager.page_allocator = FailingPageAllocator()
    manager.block_mem_size = 1
    manager.reserved_blocks = []
    manager.avail_pages = {}
    manager.full_pages = {}
    manager.num_avail_blocks = 0
    manager.resize = mock.Mock()
    setattr(manager, "available_size", mock.Mock(return_value=free_size))
    manager._wait_post_init = mock.Mock()  # type: ignore[method-assign]
    manager.in_shrink = False
    manager.target_num_blocks = None
    return manager


def test_alloc_page_failure_rolls_back_reserved_blocks():
    manager = _manager_with_allocator(free_size=2)
    page = FakePage(free_blocks=[], capacity=1)
    manager.full_pages = {page.page_id: page}
    manager.reserved_blocks = [10]

    result = manager._alloc(2, _skip_wait=True)

    assert result is None
    assert manager.page_allocator.freed_pages == []
    assert manager.avail_pages[page.page_id].empty()
    assert manager.reserved_blocks == []


def test_alloc_page_failure_rolls_back_blocks_from_existing_page():
    manager = _manager_with_allocator(free_size=3)
    page = FakePage()
    manager.avail_pages = {page.page_id: page}
    manager.num_avail_blocks = page.num_free_blocks()

    result = manager._alloc(3, _skip_wait=True)

    assert result is None
    assert manager.page_allocator.freed_pages == []
    assert manager.avail_pages[page.page_id].empty()
    assert manager.num_avail_blocks == page.capacity


def test_alloc_page_failure_without_partial_allocation_returns_none():
    manager = _manager_with_allocator(free_size=1)

    result = manager._alloc(1, _skip_wait=True)

    assert result is None
    assert manager.page_allocator.freed_pages == []
