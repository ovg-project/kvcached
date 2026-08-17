# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA build of the native extension", allow_module_level=True)

from kvcached.vmm_ops import PageAllocator  # noqa: E402


def test_page_state_stays_coherent_during_resize():
    page_size = 2 * 1024 * 1024
    lower_pages = 16
    upper_pages = 32
    allocator = PageAllocator(
        num_layers=1,
        mem_size_per_layer=upper_pages * page_size,
        page_size=page_size,
        world_size=1,
        pp_rank=0,
        async_sched=False,
        contiguous_layout=True,
        enable_page_prealloc=False,
        num_kv_buffers=1,
        group_id=os.getpid(),
        ipc_name=f"/kvcached-page-state-{os.getpid()}",
    )

    stop_snapshot = threading.Event()
    resize_done = threading.Event()
    snapshot_done = threading.Event()
    failures = []

    def resize_repeatedly():
        try:
            for _ in range(1_000):
                assert allocator.resize(lower_pages * page_size)
                assert allocator.resize(upper_pages * page_size)
        except BaseException as exc:
            failures.append(exc)
        finally:
            resize_done.set()

    def snapshot_repeatedly():
        try:
            while not stop_snapshot.is_set():
                state = allocator.get_page_state()
                total_pages = state["total_pages"]
                free_pages = state["free_pages"]
                inuse_pages = state["inuse_pages"]
                assert total_pages in (lower_pages, upper_pages)
                assert 0 <= free_pages <= total_pages
                assert inuse_pages == total_pages - free_pages
                assert state["reserved_pages"] == 0
                time.sleep(0)
        except BaseException as exc:
            failures.append(exc)
        finally:
            snapshot_done.set()

    resize_thread = threading.Thread(target=resize_repeatedly, daemon=True)
    snapshot_thread = threading.Thread(target=snapshot_repeatedly, daemon=True)
    snapshot_thread.start()
    resize_thread.start()

    assert resize_done.wait(timeout=20), "resize deadlocked with page-state snapshots"
    stop_snapshot.set()
    assert snapshot_done.wait(timeout=10), "page-state snapshot did not finish"
    resize_thread.join(timeout=0)
    snapshot_thread.join(timeout=0)
    assert failures == []
