# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Exercise the native resize watcher through real quota writes.

Worker map/unmap callbacks are stubbed; the watcher and page accounting are
real. An accelerator is required because the prealloc worker queries free
device memory.
"""

import os
import time
import uuid

import pytest

pytest.importorskip("torch")

from kvcached.utils import get_device_module, get_device_type  # noqa: E402

if not get_device_module().is_available():
    pytest.skip(f"no {get_device_type()} device available",
                allow_module_level=True)

from kvcached.cli.utils import update_kv_cache_limit  # noqa: E402
from kvcached.vmm_ops import PageAllocator  # noqa: E402

PAGE_SIZE = 2 * 1024 * 1024
STARTUP_PAGES = 8
NUM_LAYERS = 3


def wait_for_target(allocator, expected):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if allocator.get_resize_target() == expected:
            return
        time.sleep(0.01)
    assert allocator.get_resize_target() == expected


@pytest.fixture
def allocator_factory():
    allocators = []

    def create(num_kv_buffers=2, extra_bytes=0):
        ipc_name = f"kvcached-resize-{os.getpid()}-{uuid.uuid4().hex}"
        startup_size = STARTUP_PAGES * PAGE_SIZE + extra_bytes
        allocator = PageAllocator(
            num_layers=NUM_LAYERS,
            mem_size_per_layer=startup_size,
            page_size=PAGE_SIZE,
            world_size=1,
            pp_rank=0,
            async_sched=False,
            contiguous_layout=True,
            enable_page_prealloc=True,
            num_kv_buffers=num_kv_buffers,
            ipc_name=ipc_name,
        )
        allocators.append((allocator, ipc_name))
        allocator.set_use_worker_ipc(True)
        allocator.set_broadcast_map_callback(lambda world_size, offsets: None)
        allocator.set_broadcast_unmap_callback(lambda world_size, offsets: None)
        allocator.start_prealloc_thread()
        deadline = time.monotonic() + 10
        while allocator.get_num_reserved_pages() == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert allocator.get_num_reserved_pages() > 0

        def set_quota(per_pool_size):
            assert update_kv_cache_limit(
                ipc_name, per_pool_size * NUM_LAYERS * num_kv_buffers
            ) is not None

        return allocator, set_quota, startup_size

    yield create
    for allocator, ipc_name in allocators:
        allocator.stop_prealloc_thread()
        allocator.trim()
        # The native destructor also unlinks this segment.
        if os.path.exists(f"/dev/shm/{ipc_name}"):
            os.unlink(f"/dev/shm/{ipc_name}")


@pytest.mark.parametrize("num_kv_buffers", [1, 2])
@pytest.mark.parametrize("extra_bytes", [0, PAGE_SIZE // 2])
def test_quota_can_return_to_startup_size(allocator_factory, num_kv_buffers, extra_bytes):
    allocator, set_quota, startup_size = allocator_factory(num_kv_buffers, extra_bytes)

    for _ in range(3):
        shrink_size = 3 * PAGE_SIZE + extra_bytes
        set_quota(shrink_size)
        wait_for_target(allocator, shrink_size)
        assert allocator.resize(allocator.get_resize_target())
        assert allocator.get_num_total_pages() == 3

        set_quota(startup_size)
        wait_for_target(allocator, startup_size)
        assert allocator.resize(allocator.get_resize_target())
        assert allocator.get_num_total_pages() == STARTUP_PAGES
        # Keep the latest request available across subsequent watcher polls.
        time.sleep(0.3)
        assert allocator.get_resize_target() == startup_size

        pages = [allocator.alloc_page() for _ in range(STARTUP_PAGES)]
        page_ids = [page.page_id for page in pages]
        assert len(set(page_ids)) == STARTUP_PAGES
        assert allocator.get_num_free_pages() == 0
        allocator.free_pages(page_ids)


def test_failed_shrink_remains_pending_until_applied(allocator_factory):
    allocator, set_quota, _ = allocator_factory()
    pages = [allocator.alloc_page() for _ in range(4)]
    shrink_size = 2 * PAGE_SIZE
    set_quota(shrink_size)
    wait_for_target(allocator, shrink_size)
    assert not allocator.resize(shrink_size)
    assert allocator.get_num_total_pages() == STARTUP_PAGES
    time.sleep(0.3)  # Let the watcher poll again after the rejected resize.
    assert allocator.get_resize_target() == shrink_size

    allocator.free_pages([page.page_id for page in pages])
    assert allocator.resize(shrink_size)
    assert allocator.get_num_total_pages() == 2


def test_direct_resize_does_not_reapply_unchanged_quota(allocator_factory):
    allocator, _, _ = allocator_factory()
    # The revisioned memory-limit API also calls resize() directly, without
    # updating the legacy shared-memory quota. Do not undo its resize.
    assert allocator.resize(6 * PAGE_SIZE)
    time.sleep(0.3)
    assert allocator.get_resize_target() == -1
    assert allocator.get_num_total_pages() == 6


def test_cancel_pending_shrink_at_current_capacity(allocator_factory):
    allocator, set_quota, _ = allocator_factory()
    current_size = 6 * PAGE_SIZE
    set_quota(current_size)
    wait_for_target(allocator, current_size)
    assert allocator.resize(current_size)

    pages = [allocator.alloc_page() for _ in range(4)]
    set_quota(2 * PAGE_SIZE)
    wait_for_target(allocator, 2 * PAGE_SIZE)
    assert not allocator.resize(2 * PAGE_SIZE)

    # Although capacity is already six pages, Python needs this notification
    # to cancel its deferred shrink and leave in_shrink mode.
    set_quota(current_size)
    wait_for_target(allocator, current_size)
    assert allocator.resize(current_size)
    allocator.free_pages([page.page_id for page in pages])
