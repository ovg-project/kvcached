# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Exercise the native resize watcher through real quota writes.

Worker map/unmap callbacks are stubbed; the watcher and page accounting are
real. A GPU is required because the prealloc worker queries free GPU memory.
"""

import os
import subprocess
import sys
import time
import uuid

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("prealloc worker needs a GPU", allow_module_level=True)

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

    def create(num_kv_buffers=2, extra_bytes=0, enable_prealloc=True):
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
            enable_page_prealloc=enable_prealloc,
            num_kv_buffers=num_kv_buffers,
            ipc_name=ipc_name,
        )
        allocators.append((allocator, ipc_name))
        allocator.set_use_worker_ipc(True)
        allocator.set_broadcast_map_callback(lambda world_size, offsets: None)
        allocator.set_broadcast_unmap_callback(lambda world_size, offsets: None)
        allocator.start_prealloc_thread()
        if enable_prealloc:
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
        # Native close preserves the segment for kvctl delete.
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


def test_regrow_is_capped_at_startup_size(allocator_factory):
    allocator, _, _ = allocator_factory()
    assert allocator.resize(3 * PAGE_SIZE)
    # Growing past the startup size used to hand out reclaimed IDs twice.
    assert allocator.resize(20 * PAGE_SIZE)
    assert allocator.get_num_total_pages() == STARTUP_PAGES

    pages = [allocator.alloc_page() for _ in range(STARTUP_PAGES)]
    page_ids = [page.page_id for page in pages]
    assert sorted(page_ids) == list(range(STARTUP_PAGES))
    allocator.free_pages(page_ids)


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


def test_quota_watcher_without_preallocation_can_stop_and_restart(allocator_factory):
    allocator, set_quota, _ = allocator_factory(enable_prealloc=False)
    set_quota(3 * PAGE_SIZE)
    wait_for_target(allocator, 3 * PAGE_SIZE)
    assert allocator.get_num_reserved_pages() == 0

    allocator.stop_prealloc_thread()
    set_quota(2 * PAGE_SIZE)
    time.sleep(0.3)
    assert allocator.get_resize_target() == 3 * PAGE_SIZE

    allocator.start_prealloc_thread()
    wait_for_target(allocator, 2 * PAGE_SIZE)
    assert allocator.get_num_reserved_pages() == 0
    set_quota(0)
    wait_for_target(allocator, 0)
    assert allocator.resize(0)
    assert allocator.get_num_total_pages() == 0
    set_quota(3 * PAGE_SIZE)
    wait_for_target(allocator, 3 * PAGE_SIZE)
    assert allocator.resize(3 * PAGE_SIZE)
    assert allocator.get_num_total_pages() == 3
    assert allocator.get_num_reserved_pages() == 0


def test_quota_watcher_without_preallocation_stops_on_destruction():
    # A joinable native watcher must not outlive its allocator or abort Python.
    scenario = """
import gc
import os
import uuid
from kvcached.vmm_ops import PageAllocator

for i in range(5):
    name = f"kvcached-watcher-destroy-{os.getpid()}-{uuid.uuid4().hex}"
    path = f"/dev/shm/{name}"
    try:
        allocator = PageAllocator(
            num_layers=1, mem_size_per_layer=8 * 1024 * 1024,
            page_size=2 * 1024 * 1024, enable_page_prealloc=False,
            num_kv_buffers=1, ipc_name=name,
        )
        allocator.start_prealloc_thread()
        assert os.path.exists(path)
        del allocator
        gc.collect()
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)
print("watcher cleanup complete")
"""
    result = subprocess.run(
        [sys.executable, "-c", scenario], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "watcher cleanup complete" in result.stdout
