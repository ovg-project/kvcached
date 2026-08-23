# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

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


class FakeAllocator:
    def __init__(self):
        self.freed_pages = []
        self.resize = mock.Mock(return_value=True)

    def free_pages(self, page_ids):
        self.freed_pages.extend(page_ids)

    def group_indices_by_page(self, indices, block_mem_size):
        return {FakePage.page_id: list(indices)}


class FakePage:
    page_id = 7

    def __init__(self):
        self.free_blocks = []

    def free_batch(self, indices):
        self.free_blocks.extend(indices)

    def num_free_blocks(self):
        return len(self.free_blocks)

    def empty(self):
        return self.free_blocks == [100]


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
    return manager


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


def test_release_retired_pages_honors_marker():
    manager = _manager()
    manager._physical_release_epoch = 3
    manager._retired_pages = [(1, [10]), (2, [11, 12]), (3, [13])]

    manager.release_retired_pages_through(2)

    assert manager.page_allocator.freed_pages == [10, 11, 12]
    assert manager._retired_pages == [(3, [13])]


def test_release_retired_pages_is_idempotent():
    manager = _manager()
    manager._physical_release_epoch = 1
    manager._retired_pages = [(1, [10])]

    manager.release_retired_pages_through(1)
    manager.release_retired_pages_through(1)

    assert manager.page_allocator.freed_pages == [10]


def test_shrink_waits_until_retired_pages_are_physically_released():
    manager = _manager()
    manager.in_shrink = True
    manager.target_num_blocks = 0
    manager._get_num_alloced_blocks = mock.Mock(return_value=0)
    manager._retired_pages = [(1, [10])]

    manager._maybe_finish_shrink()
    manager.page_allocator.resize.assert_not_called()

    manager.release_retired_pages_through(1)

    manager.page_allocator.resize.assert_called_once_with(0)
    assert manager.in_shrink is False
