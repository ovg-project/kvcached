# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Native rollback-failure tests; each case runs in its own CUDA process."""

import ctypes
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def fault_library(tmp_path_factory):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        pytest.skip("CUDA headers required for the test-only driver shim")
    library = tmp_path_factory.mktemp("vmm-faults") / "faults.so"
    subprocess.run([
        "c++", "-std=c++17", "-shared", "-fPIC", "-O2",
        "-I", str(Path(CUDA_HOME) / "include"),
        str(Path(__file__).parent / "native" / "vmm_faults.cpp"),
        "-ldl", "-o", str(library),
    ], check=True)
    return library


@pytest.mark.parametrize("case", ["quarantine", "prealloc", "unmap", "callback", "fatal-callback", "manager", "serialize", "release", "commit-release", "rollback-release", "state-gil"])
def test_native_failure_policy(fault_library, case):
    env = dict(os.environ, LD_PRELOAD=str(fault_library),
               KVCACHED_MIN_RESERVED_PAGES="2", KVCACHED_MAX_RESERVED_PAGES="0",
               KVCACHED_IPC_NAME=f"f418-{os.getpid()}-{case}",
               ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), case],
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS {case}" in result.stdout


def _native_case(case):
    import torch

    from kvcached import vmm_ops
    from kvcached.errors import (
        MapQuarantinedError,
        QuarantinedResizeError,
        StateConsistencyError,
    )

    device = int(os.environ.get("KVCACHED_TEST_DEVICE", "0"))
    torch.cuda.set_device(device)
    page_size = 2 * 1024 * 1024
    vmm_ops.init_kvcached(f"cuda:{device}", page_size, False)
    tensors = vmm_ops.create_kv_tensors(8 * page_size, 2, f"cuda:{device}", 2, 2, 0, False)
    allocator = vmm_ops.PageAllocator(2, 4 * page_size, page_size,
                                     contiguous_layout=False, enable_page_prealloc=case in ("prealloc", "state-gil"),
                                     ipc_name=os.environ["KVCACHED_IPC_NAME"])
    driver = ctypes.CDLL(None)
    driver.kvcached_fault_arm.argtypes = [ctypes.c_int] * 3
    driver.kvcached_fault_hits.restype = ctypes.c_int
    driver.kvcached_fault_release.argtypes = [ctypes.c_int]
    if case == "state-gil":
        formatting = threading.Event()

        class CallbackError(RuntimeError):
            def __str__(self):
                formatting.set()
                # prealloc_worker holds its native mutex while formatting this.
                time.sleep(0.3)
                return "recoverable callback error"

        def fail(_world_size, _offsets):
            raise CallbackError()

        allocator.set_use_worker_ipc(True)
        allocator.set_broadcast_map_callback(fail)
        allocator.start_prealloc_thread()
        assert formatting.wait(5)
        assert allocator.get_transaction_state()["state"] == "HEALTHY"
        allocator.stop_prealloc_thread()
    elif case in ("callback", "fatal-callback"):
        error_type = MapQuarantinedError if case == "callback" else StateConsistencyError

        def fail(_world_size, _offsets):
            raise error_type("typed IPC failure")

        allocator.set_use_worker_ipc(True)
        allocator.set_broadcast_map_callback(fail)
        with pytest.raises(error_type):
            allocator.alloc_page()
        state = allocator.get_transaction_state()
        assert state["quarantined_page_ids"] == [0]
        if case == "fatal-callback":
            with pytest.raises(StateConsistencyError):
                allocator.get_num_free_pages()
            assert state["state"] == "FAILED"
        else:
            assert allocator.get_num_free_pages() == 3
            assert state["state"] == "DEGRADED"
            allocator.set_use_worker_ipc(False)
            assert allocator.alloc_page().page_id == 1
    elif case == "serialize":
        from concurrent.futures import ThreadPoolExecutor

        first = allocator.alloc_page()
        entered = threading.Event()
        release = threading.Event()
        unmap_entered = threading.Event()

        def map_callback(_size, offsets):
            torch.cuda.set_device(device)
            entered.set()
            assert release.wait(5)
            assert vmm_ops.map_to_kv_tensors(offsets, 0)

        def unmap_callback(_size, offsets):
            torch.cuda.set_device(device)
            unmap_entered.set()
            assert vmm_ops.prepare_unmap_from_kv_tensors(offsets, "serialize", 0)
            assert vmm_ops.commit_unmap_from_kv_tensors("serialize", 0)

        allocator.set_use_worker_ipc(True)
        allocator.set_broadcast_map_callback(map_callback)
        allocator.set_broadcast_unmap_callback(unmap_callback)
        with ThreadPoolExecutor(max_workers=2) as executor:
            mapping = executor.submit(allocator.alloc_page)
            assert entered.wait(5)
            unmapping = executor.submit(allocator.free_page, first.page_id)
            try:
                assert not unmap_entered.wait(0.2)
            finally:
                release.set()
            assert mapping.result(timeout=5).page_id == 1
            unmapping.result(timeout=5)
        assert unmap_entered.is_set()
        assert allocator.get_transaction_state()["state"] == "HEALTHY"
    elif case == "commit-release":
        page = allocator.alloc_page()
        assert vmm_ops.prepare_unmap_from_kv_tensors([page.page_id * page_size], "release", 0)
        driver.kvcached_fault_release(2)
        with pytest.raises(StateConsistencyError, match="physical page release"):
            vmm_ops.commit_unmap_from_kv_tensors("release", 0)
        assert driver.kvcached_fault_hits() == 1
        with pytest.raises(StateConsistencyError, match="cannot abort"):
            vmm_ops.abort_unmap_from_kv_tensors("release", 0)
        # Retry the same transaction: already released handles must not be released twice.
        assert vmm_ops.commit_unmap_from_kv_tensors("release", 0)
        assert vmm_ops.commit_unmap_from_kv_tensors("release", 0)
    elif case in ("unmap", "release"):
        page = allocator.alloc_page()
        # Fail zero-page restoration, then fail restoring the original mapping.
        if case == "unmap":
            driver.kvcached_fault_arm(0, 0, 1)
        else:
            driver.kvcached_fault_release(1)
        with pytest.raises(StateConsistencyError):
            allocator.free_page(page.page_id)
        assert driver.kvcached_fault_hits() == (2 if case == "unmap" else 1)
        driver.kvcached_fault_arm(0, 0, 0)
        assert allocator.get_transaction_state()["state"] == "FAILED"
        with pytest.raises(StateConsistencyError):
            allocator.alloc_page()
    elif case == "manager":
        from kvcached.kv_cache_manager import KVCacheManager

        # Real manager and native allocator, without startup's null-block reserve.
        manager = object.__new__(KVCacheManager)
        manager.page_allocator = allocator
        manager.page_size = page_size
        manager.block_mem_size = page_size // 4
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
        driver.kvcached_fault_arm(2, 3, 0)
        assert manager.alloc(1) is None
        assert driver.kvcached_fault_hits() == 2
        driver.kvcached_fault_arm(0, 0, 0)
        assert manager.available_size() == 12
        with pytest.raises(QuarantinedResizeError):
            manager.resize(5 * page_size)
        assert not manager.in_shrink
        assert manager.target_num_blocks is None
        assert manager.available_size() == 12
        for expected in ([4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]):
            assert manager.alloc(4) == expected
        manager.free(list(range(4, 16)))
        assert manager.available_size() == 12
        assert allocator.get_transaction_state()["quarantined_page_ids"] == [0]
        allocator.set_use_worker_ipc(True)

        def fatal(_size, _offsets):
            raise StateConsistencyError("unconfirmed peer")

        allocator.set_broadcast_map_callback(fatal)
        with pytest.raises(StateConsistencyError, match="unconfirmed peer"):
            manager.alloc(1)
        with pytest.raises(StateConsistencyError):
            manager.alloc(1)
    else:
        # K maps; V creation fails; rollback of K fails. Other calls are real VMM.
        driver.kvcached_fault_arm(2, 0 if case == "rollback-release" else 3, 0)
        if case == "rollback-release":
            driver.kvcached_fault_release(1)
        if case == "prealloc":
            allocator.start_prealloc_thread()
            deadline = time.monotonic() + 10
            while not allocator.get_transaction_state()["quarantined_pages"] and time.monotonic() < deadline:
                time.sleep(0.01)
            allocator.stop_prealloc_thread()
        else:
            with pytest.raises(MapQuarantinedError):
                allocator.alloc_page()
        assert driver.kvcached_fault_hits() == 2
        driver.kvcached_fault_arm(0, 0, 0)
        state = allocator.get_transaction_state()
        expected = [0, 1] if case == "prealloc" else [0]
        assert state["quarantined_page_ids"] == expected, state
        assert state["state"] == "DEGRADED"
        assert allocator.get_num_free_pages() == 4 - len(expected)
        assert state["retained_bytes_upper_bound"] == len(expected) * page_size * 4
        allocator.reset_free_page_order()
        with pytest.raises(QuarantinedResizeError):
            allocator.resize(3 * page_size)
        assert allocator.resize(4 * page_size)
        pages = [allocator.alloc_page() for _ in range(4 - len(expected))]
        assert [p.page_id for p in pages] == list(range(len(expected), 4))
        for page in pages:
            for tensor in tensors:
                data = tensor.view(-1)
                for base in (0, data.numel() // 2):
                    start = base + page.page_id * page_size // data.element_size()
                    data[start:start + 128].fill_(7)
                    assert bool((data[start:start + 128] == 7).all())
        torch.cuda.synchronize(device)
        allocator.free_pages([p.page_id for p in pages])
        assert allocator.get_num_free_pages() == 4 - len(expected)
    driver.kvcached_fault_arm(0, 0, 0)
    allocator.stop_prealloc_thread()
    torch.cuda.synchronize(device)
    vmm_ops.shutdown_kvcached()
    print(f"PASS {case}", flush=True)


if __name__ == "__main__":
    _native_case(sys.argv[1])
