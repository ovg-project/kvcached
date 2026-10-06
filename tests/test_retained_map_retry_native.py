# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Real mappings retry at zero fresh headroom without losing batch ownership."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import test_vmm_failure_policy

fault_library = test_vmm_failure_policy.fault_library


@pytest.mark.parametrize("source", ["foreground", "background"])
@pytest.mark.parametrize("consumer", ["foreground", "background", "fatal", "gil"])
def test_retained_batch_retry(fault_library, source, consumer):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires native CUDA VMM")
    env = dict(os.environ, LD_PRELOAD=str(fault_library),
               ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0",
               KVCACHED_CONTIGUOUS_LAYOUT="false", KVCACHED_PAGE_SIZE_MB="2",
               KVCACHED_MIN_RESERVED_PAGES="2", KVCACHED_MAX_RESERVED_PAGES="0",
               KVCACHED_IPC_NAME=f"retained-{os.getpid()}-{source}-{consumer}")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), source, consumer],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def _case(source, consumer):
    import ctypes
    import threading
    import time

    import torch

    from kvcached import vmm_ops as vmm
    from kvcached.errors import MapRetainedError, RetainedResizeError, StateConsistencyError
    from kvcached.kv_cache_manager import KVCacheManager
    from kvcached.lifecycle import LifecyclePhase, LifecycleState

    page = 2 * 1024 * 1024
    torch.cuda.set_device(0)
    vmm.init_kvcached("cuda:0", page, False)
    tensors = vmm.create_kv_tensors(8 * page, 1, "cuda:0", 1, 1)
    allocator = vmm.PageAllocator(1, 8 * page, page, contiguous_layout=False,
                                  enable_page_prealloc=source == "background" or consumer == "background",
                                  num_kv_buffers=1, ipc_name=os.environ["KVCACHED_IPC_NAME"])
    allocator.set_use_worker_ipc(True)
    calls = []
    required = []
    formatting, release_format = threading.Event(), threading.Event()

    class SlowError(RuntimeError):
        def __str__(self):
            formatting.set()
            assert release_format.wait(10), "error formatting was not released"
            return "ordinary Python callback failure"

    driver = ctypes.CDLL(None)
    driver.kvcached_fault_zero_free_memory.argtypes = [ctypes.c_int]

    def map_pages(world_size, offsets):
        calls.append(tuple(offsets))
        if len(calls) > 1:
            assert calls[-1] == calls[0], "a retained batch was split or reordered"
        transaction = f"retry-{len(calls)}"
        prepared = vmm.prepare_map_to_kv_tensors(transaction, offsets, 0)
        assert prepared["success"]
        required.append(prepared["required_bytes"])
        assert vmm.commit_prepared_map(transaction, 0)["success"]
        if len(calls) == 1:
            driver.kvcached_fault_zero_free_memory(1)
            raise MapRetainedError("controlled peer commit failure after local mapping")
        if len(calls) == 2:
            if consumer == "fatal":
                raise StateConsistencyError("retry outcome is unknown")
            if consumer == "gil":
                raise SlowError()
            raise RuntimeError("peer still has no capacity")

    def unmap_pages(world_size, offsets):
        assert vmm.unmap_from_kv_tensors(offsets, 0)

    allocator.set_broadcast_map_callback(map_pages)
    allocator.set_broadcast_unmap_callback(unmap_pages)
    manager = object.__new__(KVCacheManager)
    manager.page_allocator = allocator
    manager.page_size = page
    manager.block_mem_size = page // 4
    manager.num_avail_blocks = 0
    manager.avail_pages = {}
    manager.full_pages = {}
    manager.reserved_blocks = []
    manager.null_block = None
    manager.in_shrink = False
    manager.target_num_blocks = None
    manager._lock = threading.RLock()
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()
    manager._lifecycle = LifecycleState("retained-retry")
    manager._lifecycle.mark_ready()

    def wait_for(predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            assert time.monotonic() < deadline, "native retry did not settle"
            time.sleep(0.01)

    try:
        if source == "background":
            allocator.start_prealloc_thread()
            wait_for(lambda: allocator.get_num_retryable_pages() == 2)
            allocator.stop_prealloc_thread()
            batch_pages = 2
        else:
            assert manager.alloc(1) is None
            batch_pages = 1
        assert allocator.get_num_retryable_pages() == batch_pages
        assert allocator.get_num_free_pages() == 8
        assert allocator.get_transaction_state()["retained_bytes_upper_bound"] == batch_pages * page
        allocator.trim()
        with pytest.raises(RetainedResizeError, match="retained"):
            allocator.resize(4 * page)
        assert allocator.get_num_retryable_pages() == batch_pages
        # The driver shim reports zero free memory to both the manager and
        # native preallocator. All actual retry mappings already exist locally.
        assert allocator.get_avail_physical_pages() == 0
        manager._avail_physical_pages_cache = None
        assert manager.available_size() == batch_pages * 4
        if consumer == "fatal":
            with pytest.raises(StateConsistencyError, match="unknown"):
                manager.alloc(1)
            assert manager.lifecycle_phase is LifecyclePhase.FAILED
            assert allocator.get_transaction_state()["state"] == "FAILED"
            assert allocator.get_num_retryable_pages() == 0
            with pytest.raises(StateConsistencyError):
                manager.alloc(1)
        else:
            if consumer == "background":
                allocator.start_prealloc_thread()
                wait_for(lambda: len(calls) >= 2 and allocator.get_num_retryable_pages() == batch_pages)
                allocator.stop_prealloc_thread()
                allocator.start_prealloc_thread()
                wait_for(lambda: allocator.get_num_reserved_pages() == batch_pages)
                allocator.stop_prealloc_thread()
            elif consumer == "gil":
                results = []
                errors = []

                def retry():
                    try:
                        results.append(manager.alloc(1))
                    except Exception as error:
                        errors.append(error)

                thread = threading.Thread(target=retry, daemon=True)
                thread.start()
                try:
                    assert formatting.wait(5)
                    # This getter holds the GIL while acquiring the native
                    # mutex. Formatting must not own that mutex while waiting.
                    assert allocator.get_num_retryable_pages() == 0
                finally:
                    release_format.set()
                    thread.join(5)
                assert not thread.is_alive() and not errors
                assert results == [None]
            else:
                assert manager.alloc(1) is None
                assert allocator.get_num_retryable_pages() == batch_pages
            blocks = manager.alloc(1)
            assert blocks == [0]
            assert allocator.get_num_retryable_pages() == 0
            assert allocator.get_num_free_pages() == 7
            tensors[0][:128].fill_(23)
            assert torch.all(tensors[0][:128] == 23).item()
            torch.cuda.synchronize()
            manager.free(blocks)
            allocator.trim()
            assert allocator.get_num_free_pages() == 8
            assert allocator.get_num_reserved_pages() == 0
            assert allocator.resize(4 * page)
        assert required[0] == batch_pages * page
        assert all(value == 0 for value in required[1:]), required
        print(f"PASS {source}/{consumer}: batch={batch_pages}, required={required}", flush=True)
    finally:
        allocator.stop_prealloc_thread()
        driver.kvcached_fault_zero_free_memory(0)
        vmm.shutdown_kvcached()


def test_clear_restores_page_zero_after_retained_map():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires native CUDA VMM")
    env = dict(os.environ, ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0",
               KVCACHED_CONTIGUOUS_LAYOUT="false", KVCACHED_PAGE_SIZE_MB="2",
               KVCACHED_PAGE_PREALLOC_ENABLED="false", KVCACHED_MIN_RESERVED_PAGES="0",
               KVCACHED_MAX_RESERVED_PAGES="0", KVCACHED_IPC_NAME=f"retry-clear-{os.getpid()}")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "clear"],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS clear" in result.stdout


def _clear_case():
    import torch

    from kvcached import vmm_ops as vmm
    from kvcached.errors import MapRetainedError
    from kvcached.kv_cache_manager import KVCacheManager

    page = 2 * 1024 * 1024
    vmm.init_kvcached("cuda:0", page, False)
    tensors = vmm.create_kv_tensors(8 * page, 1, "cuda:0", 1, 1)
    manager = KVCacheManager(32, 1, page // 4, 1, num_kv_buffers=1, reserve_null_block=True)
    calls = []

    def map_pages(world_size, offsets):
        calls.append(tuple(offsets))
        transaction = f"clear-{len(calls)}"
        assert vmm.prepare_map_to_kv_tensors(transaction, offsets, 0)["success"]
        assert vmm.commit_prepared_map(transaction, 0)["success"]
        if len(calls) == 1:
            raise MapRetainedError("retained nonzero page")

    def unmap_pages(world_size, offsets):
        assert vmm.unmap_from_kv_tensors(offsets, 0)

    try:
        manager.wait_ready(timeout=5)
        assert manager.null_block == [0]
        manager.page_allocator.set_use_worker_ipc(True)
        manager.page_allocator.set_broadcast_map_callback(map_pages)
        manager.page_allocator.set_broadcast_unmap_callback(unmap_pages)
        assert manager.alloc(4) is None
        assert calls == [(page,)]
        assert manager.page_allocator.get_num_retryable_pages() == 1
        manager.clear()
        manager.wait_ready(timeout=5)
        assert calls == [(page,), (page,), (0,)]
        assert manager.null_block == [0]
        assert manager.page_allocator.get_num_retryable_pages() == 0
        block = manager.alloc(1)
        assert block == [1]
        tensors[0][page // 4:page // 4 + 128].fill_(17)
        assert torch.all(tensors[0][page // 4:page // 4 + 128] == 17).item()
        torch.cuda.synchronize()
        manager.free(block)
        print("PASS clear restores page zero after retained batch recovery", flush=True)
    finally:
        manager.shutdown()
        vmm.shutdown_kvcached()


if __name__ == "__main__":
    if sys.argv[1] == "clear":
        _clear_case()
    else:
        _case(sys.argv[1], sys.argv[2])
