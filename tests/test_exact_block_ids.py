# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Exact-ID manager contracts with a deterministic, GPU-free page backend."""
import ast
import functools
import operator
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class UnsafeMapping(RuntimeError):
    pass


class RejectedResize(RuntimeError):
    pass


class Page:
    def __init__(self, page_id, page_size=16):
        self.page_id = page_id
        self.page_size = page_size
        self.free = []

    @staticmethod
    def get_block_range(page_id, page_size, block_size):
        return ((page_id * page_size + block_size - 1) // block_size,
                (page_id + 1) * page_size // block_size)

    def init(self, block_size):
        start, end = self.get_block_range(self.page_id, self.page_size, block_size)
        self.free = list(range(start, end))

    def get_free_blocks(self):
        return list(self.free)

    def num_free_blocks(self):
        return len(self.free)

    def alloc_exact(self, ids):
        if not set(ids) <= set(self.free):
            return False
        self.free = [i for i in self.free if i not in ids]
        return True

    def full(self):
        return not self.free


class Backend:
    def __init__(self):
        self.calls = []
        self.released = []
        self.fail_page = None
        self.failure = None
        self.page_size = 16
        self.resize_target = -1
        self.reserved_pages = 0

    def get_num_reserved_pages(self):
        return self.reserved_pages

    def get_resize_target(self):
        return self.resize_target

    def try_alloc_page_at(self, page_id):
        self.calls.append(page_id)
        if page_id == self.fail_page:
            if self.failure is not None:
                raise self.failure
            return None
        return Page(page_id, self.page_size)

    def free_pages(self, page_ids):
        self.released.extend(page_ids)


def make_manager():
    source = Path(__file__).resolve().parents[1] / 'kvcached/kv_cache_manager.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'KVCacheManager')
    selected = {'alloc_exact', '_alloc_exact_impl', '_release_unclaimed_exact_pages',
                '_apply_resize_target', '_retiring_page_ids'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in selected]
    # Extract the production methods, not copies, without importing torch or
    # loading the compiled GPU extension. Calls to the native layer are fakes.
    decorators = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name in {'synchronized', '_after_post_init'}]
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                        names=[ast.alias(name='annotations')], level=0), *decorators, cls],
                        type_ignores=[])
    namespace: dict[str, Any] = {'operator': operator, 'InternalPage': Page,
                 'StateConsistencyError': UnsafeMapping,
                 'QuarantinedResizeError': RejectedResize,
                 'logger': SimpleNamespace(warning=lambda message: None),
                 'functools': functools}
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    manager = namespace['KVCacheManager']()
    manager._lock = threading.RLock()
    manager.logical_num_blocks = None
    manager._logical_allocated_blocks = 0
    manager._virtual_num_blocks = 32
    manager.reserve_null_block = True
    manager.block_mem_size = 4
    manager.page_size = 16
    manager.reserved_blocks = []
    manager.full_pages = {}
    manager.avail_pages = {}
    manager.num_avail_blocks = 0
    manager._retired_pages = []
    manager.in_shrink = False
    manager.page_allocator = Backend()
    manager._operation_counters = {}
    manager._increment_operation_counter = lambda key, value=1: manager._operation_counters.update(
        {key: manager._operation_counters.get(key, 0) + value})
    manager._record_operation_error = lambda *args, **kwargs: None
    manager._wait_post_init = lambda: None
    manager.available_size = lambda: 32
    manager._get_cached_avail_physical_pages = lambda: 8
    manager.resize = lambda size: None
    manager._record_native_fatal = lambda operation, exc: isinstance(exc, UnsafeMapping)
    manager.failures = []
    manager._lifecycle = SimpleNamespace(mark_failed=lambda reason, exc: manager.failures.append(exc))
    return manager


def page_for(manager, page_id, free):
    page = Page(page_id, manager.page_size)
    page.free = list(free)
    manager.avail_pages[page_id] = page
    manager.num_avail_blocks += len(free)
    return page


@pytest.mark.parametrize('ids,error', [([1, 1], ValueError), ([-1], ValueError),
                                      ([32], ValueError), ([0], ValueError),
                                      ([True], TypeError), ([1.5], TypeError)])
def test_invalid_request_is_rejected_before_mapping(ids, error):
    manager = make_manager()
    with pytest.raises(error):
        manager.alloc_exact(ids)
    assert manager.page_allocator.calls == []
    assert not manager.avail_pages


def test_geometry_rejects_a_boundary_crossing_block_before_mapping():
    manager = make_manager()
    manager.block_mem_size = 6
    with pytest.raises(ValueError, match='boundary'):
        manager.alloc_exact([2])  # bytes 12..17 cross a 16-byte page.
    assert manager.page_allocator.calls == []


def test_claim_preserves_requested_order_and_other_allocations():
    manager = make_manager()
    original = page_for(manager, 0, [2, 3])  # 0 reserved null, 1 live.
    assert manager.alloc_exact([7, 2, 5]) == [7, 2, 5]
    assert original.free == [3]
    assert manager.avail_pages[1].free == [4, 6]
    assert manager.num_avail_blocks == 3
    assert manager.page_allocator.calls == [1]
    assert manager._operation_counters['allocated_blocks_total'] == 3


def test_existing_conflict_preflights_the_whole_batch():
    manager = make_manager()
    original = page_for(manager, 0, [2, 3])
    assert manager.alloc_exact([7, 1]) is None
    assert manager.page_allocator.calls == []
    assert original.free == [2, 3]


@pytest.mark.parametrize('failure', [None, RuntimeError('capacity unavailable')])
def test_later_page_miss_unwinds_only_new_pages(failure):
    manager = make_manager()
    original = page_for(manager, 0, [2, 3])
    manager.reserved_blocks = [1]
    manager.page_allocator.fail_page = 2
    manager.page_allocator.failure = failure
    for _ in range(3):
        assert manager.alloc_exact([2, 4, 8, 1]) is None
        assert original.free == [2, 3]
        assert manager.reserved_blocks == [1]
        assert set(manager.avail_pages) == {0}
        assert manager.num_avail_blocks == 2
    assert manager.page_allocator.released == [1, 1, 1]
    manager.page_allocator.fail_page = None
    assert manager.alloc_exact([2, 4, 8, 1]) == [2, 4, 8, 1]
    assert manager.reserved_blocks == []


def test_fatal_map_failure_is_not_a_retry_or_a_release():
    manager = make_manager()
    manager.page_allocator.fail_page = 2
    manager.page_allocator.failure = UnsafeMapping('mapping outcome unknown')
    with pytest.raises(UnsafeMapping):
        manager.alloc_exact([4, 8])
    assert manager.page_allocator.released == []
    assert manager.failures == [manager.page_allocator.failure]
    assert manager.avail_pages[1].free == [4, 5, 6, 7]


def test_claiming_an_existing_reservation_does_not_allocate_again():
    manager = make_manager()
    page = page_for(manager, 0, [3])
    manager.reserved_blocks = [1, 2]
    assert manager.alloc_exact([2, 1]) == [2, 1]
    assert page.free == [3]
    assert manager.num_avail_blocks == 1
    assert manager.reserved_blocks == []
    assert manager.page_allocator.calls == []


def test_live_reuse_removes_only_its_own_retirement():
    manager = make_manager()
    page_for(manager, 1, [4, 5, 6, 7])
    manager._retired_pages = [(1, [1, 2]), (2, [3])]
    assert manager.alloc_exact([4]) == [4]
    assert manager._retired_pages == [(1, [2]), (2, [3])]


def test_capacity_limit_blocks_even_a_physically_available_exact_page():
    manager = make_manager()
    manager.available_size = lambda: 0
    assert manager.alloc_exact([4]) is None
    assert manager.page_allocator.calls == []


def test_empty_request_does_not_map_or_resize():
    manager = make_manager()
    manager.resize = lambda size: pytest.fail('unexpected resize')
    manager.page_allocator.resize_target = 16
    assert manager.alloc_exact(iter([])) == []
    assert manager.page_allocator.calls == []


def test_zero_resize_target_is_applied_before_an_exact_claim():
    manager = make_manager()
    page = page_for(manager, 0, [2, 3])
    applied = []

    def resize(size):
        applied.append(size)
        manager.available_size = lambda: 0

    manager.resize = resize
    manager.page_allocator.resize_target = 0
    assert manager.alloc_exact([2]) is None
    assert applied == [0]
    assert page.free == [2, 3]
    assert manager.page_allocator.calls == []


@pytest.mark.parametrize('ids', [[4], [8], [2, 4], [2, 8]])
def test_pending_shrink_rejects_retired_or_unmapped_ids_atomically(ids):
    manager = make_manager()
    active = page_for(manager, 0, [2, 3])
    retired = page_for(manager, 1, [4, 5, 6, 7])
    manager._retired_pages = [(9, [1])]
    manager.in_shrink = True
    # Eligible free blocks elsewhere cannot satisfy a claim for these IDs.
    manager.available_size = lambda: 2
    assert manager.alloc_exact(ids) is None
    assert active.free == [2, 3]
    assert retired.free == [4, 5, 6, 7]
    assert manager._retired_pages == [(9, [1])]
    assert manager.page_allocator.calls == []


def test_pending_shrink_can_claim_a_nonretiring_resident_page():
    manager = make_manager()
    page_for(manager, 0, [2, 3])
    retired = page_for(manager, 1, [4, 5, 6, 7])
    manager._retired_pages = [(9, [1])]
    manager.in_shrink = True
    assert manager.alloc_exact([2]) == [2]
    assert retired.free == [4, 5, 6, 7]
    assert manager._retired_pages == [(9, [1])]
    assert manager.page_allocator.calls == []


def test_sparse_ids_require_physical_pages_not_just_free_blocks():
    manager = make_manager()
    manager.available_size = lambda: 4
    manager._get_cached_avail_physical_pages = lambda: 1
    assert manager.alloc_exact([4, 8, 12, 16]) is None
    assert manager.page_allocator.calls == []
    assert not manager.avail_pages


def test_exact_claim_counts_only_pages_that_are_not_already_resident():
    manager = make_manager()
    page_for(manager, 0, [2, 3])
    manager._get_cached_avail_physical_pages = lambda: 1
    assert manager.alloc_exact([2, 4, 5, 6, 7]) == [2, 4, 5, 6, 7]
    assert manager.page_allocator.calls == [1]


def test_exact_claim_can_use_preallocated_physical_backing():
    manager = make_manager()
    manager._get_cached_avail_physical_pages = lambda: 0
    manager.page_allocator.reserved_pages = 1
    assert manager.alloc_exact([4]) == [4]
    assert manager.page_allocator.calls == [1]


def test_exact_claim_cannot_spend_unrequested_reservations():
    manager = make_manager()
    manager.logical_num_blocks = 2
    manager._logical_allocated_blocks = 2
    manager.reserved_blocks = [2, 3]
    page_for(manager, 0, [])
    manager.available_size = lambda: 2
    assert manager.alloc_exact([4]) is None
    assert manager.page_allocator.calls == []
    assert manager.reserved_blocks == [2, 3]
    assert manager._logical_allocated_blocks == 2
    assert manager.alloc_exact([3, 2]) == [3, 2]
    assert manager.reserved_blocks == []
    assert manager._logical_allocated_blocks == 2


def test_exact_claim_charges_only_fresh_ids_and_enforces_remaining_limit():
    manager = make_manager()
    manager.logical_num_blocks = 4
    manager._logical_allocated_blocks = 2  # null block and reserved ID 2
    manager.reserved_blocks = [2]
    page_for(manager, 0, [3])
    assert manager.alloc_exact([2, 3, 4]) == [2, 3, 4]
    assert manager._logical_allocated_blocks == 4
    assert manager.alloc_exact([5]) is None
    assert manager._logical_allocated_blocks == 4


def test_exact_capacity_miss_does_not_charge_logical_blocks():
    manager = make_manager()
    manager.logical_num_blocks = 4
    manager._logical_allocated_blocks = 1
    manager.page_allocator.fail_page = 2
    assert manager.alloc_exact([4, 8]) is None
    assert manager._logical_allocated_blocks == 1
    assert manager.page_allocator.released == [1]


def test_exact_allocation_waits_before_acquiring_manager_lock():
    manager = make_manager()
    # Model initialization finishing under the same manager lock. A caller
    # waiting while holding it would prevent initialization from completing.
    started = threading.Event()
    initialized = threading.Event()
    def initialize():
        assert started.wait(2)
        with manager._lock:
            initialized.set()
    worker = threading.Thread(target=initialize, daemon=True)
    worker.start()
    def wait_for_init():
        started.set()
        assert initialized.wait(2), 'initialization blocked by allocation lock'
    manager._wait_post_init = wait_for_init
    try:
        assert manager.alloc_exact([4]) == [4]
    finally:
        worker.join(timeout=2)
    assert not worker.is_alive()
