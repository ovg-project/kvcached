# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import fcntl
import multiprocessing
import os
import time
import uuid
from multiprocessing.synchronize import Barrier

import pytest

from kvcached.cli.utils import (
    MemInfoStruct,
    RwLockedShm,
    get_ipc_path,
    get_kv_cache_limit,
    init_kv_cache_limit,
)
from kvcached.mem_info_tracker import MemInfoTracker
from kvcached.utils import DEFAULT_IPC_NAME

IPC_NAME = DEFAULT_IPC_NAME
IPC_PATH = get_ipc_path(IPC_NAME)
TOTAL_MEM = 10_000_000  # 10 MB
SHM_SIZE = MemInfoStruct.SHM_SIZE


@pytest.fixture(autouse=True)
def setup_shared_memory():
    # Create shared memory with known total limit and clean up afterward
    init_kv_cache_limit(IPC_NAME, TOTAL_MEM)
    yield
    if os.path.exists(IPC_PATH):
        os.remove(IPC_PATH)


def test_charge_behavior():
    # Initial state
    mem_info = get_kv_cache_limit(IPC_NAME)
    assert mem_info is not None
    assert mem_info.total_size == TOTAL_MEM
    initial_used_size = mem_info.used_size
    initial_prealloc_size = mem_info.prealloc_size

    # Simulate a charge of 100 bytes
    with RwLockedShm(IPC_NAME, SHM_SIZE, RwLockedShm.WLOCK) as mm:
        mem_info = MemInfoStruct.from_buffer(mm)
        mem_info.used_size += 100
        mem_info.write_to_buffer(mm)

    # Check updated usage
    updated = get_kv_cache_limit(IPC_NAME)
    assert updated is not None
    assert updated.used_size == initial_used_size + 100
    assert updated.prealloc_size == initial_prealloc_size
    assert updated.total_size == TOTAL_MEM


def test_uncharge_behavior():
    # Initial state
    mem_info = get_kv_cache_limit(IPC_NAME)
    assert mem_info is not None
    assert mem_info.total_size == TOTAL_MEM
    initial_used_size = mem_info.used_size
    initial_prealloc_size = mem_info.prealloc_size

    # First charge
    with RwLockedShm(IPC_NAME, SHM_SIZE, RwLockedShm.WLOCK) as mm:
        mem_info = MemInfoStruct.from_buffer(mm)
        mem_info.used_size += 200
        mem_info.write_to_buffer(mm)

    # Then uncharge
    with RwLockedShm(IPC_NAME, SHM_SIZE, RwLockedShm.WLOCK) as mm:
        mem_info = MemInfoStruct.from_buffer(mm)
        mem_info.used_size -= 200
        mem_info.write_to_buffer(mm)

    # Check it's back to 0
    updated = get_kv_cache_limit(IPC_NAME)
    assert updated is not None
    assert updated.used_size == initial_used_size
    assert updated.prealloc_size == initial_prealloc_size
    assert updated.total_size == TOTAL_MEM


def test_reserve_behavior():
    # Initial state
    mem_info = get_kv_cache_limit(IPC_NAME)
    assert mem_info is not None
    assert mem_info.total_size == TOTAL_MEM
    initial_used_size = mem_info.used_size
    initial_prealloc_size = mem_info.prealloc_size

    # Simulate a charge of 100 bytes
    with RwLockedShm(IPC_NAME, SHM_SIZE, RwLockedShm.WLOCK) as mm:
        mem_info = MemInfoStruct.from_buffer(mm)
        mem_info.prealloc_size += 300
        mem_info.write_to_buffer(mm)

    # Check updated usage
    updated = get_kv_cache_limit(IPC_NAME)
    assert updated is not None
    assert updated.used_size == initial_used_size
    assert updated.prealloc_size == initial_prealloc_size + 300
    assert updated.total_size == TOTAL_MEM


def test_unreserve_behavior():
    # Initial state
    mem_info = get_kv_cache_limit(IPC_NAME)
    assert mem_info is not None
    assert mem_info.total_size == TOTAL_MEM
    initial_used_size = mem_info.used_size
    initial_prealloc_size = mem_info.prealloc_size

    # First charge
    with RwLockedShm(IPC_NAME, SHM_SIZE, RwLockedShm.WLOCK) as mm:
        mem_info = MemInfoStruct.from_buffer(mm)
        mem_info.prealloc_size += 400
        mem_info.write_to_buffer(mm)

    # Then uncharge
    with RwLockedShm(IPC_NAME, SHM_SIZE, RwLockedShm.WLOCK) as mm:
        mem_info = MemInfoStruct.from_buffer(mm)
        mem_info.prealloc_size -= 400
        mem_info.write_to_buffer(mm)

    # Check it's back to 0
    updated = get_kv_cache_limit(IPC_NAME)
    assert updated is not None
    assert updated.used_size == initial_used_size
    assert updated.prealloc_size == initial_prealloc_size
    assert updated.total_size == TOTAL_MEM


def worker_charge_with_barrier(ipc_name, amount, barrier: Barrier):

    barrier.wait()  # Synchronize entry

    with RwLockedShm(ipc_name, MemInfoStruct.SHM_SIZE,
                     RwLockedShm.WLOCK) as mm:
        mem_info = MemInfoStruct.from_buffer(mm)
        time.sleep(0.1)
        mem_info.used_size += amount
        mem_info.write_to_buffer(mm)


def test_multi_process_concurrent_charge_variable():
    NUM_PROCESSES = 5
    CHARGE_PER_PROCESS = 500
    barrier = multiprocessing.Barrier(parties=NUM_PROCESSES)

    processes = []
    for _ in range(NUM_PROCESSES):
        p = multiprocessing.Process(target=worker_charge_with_barrier,
                                    args=(IPC_NAME, CHARGE_PER_PROCESS,
                                          barrier))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    updated = get_kv_cache_limit(IPC_NAME)
    assert updated is not None
    expected_total = NUM_PROCESSES * CHARGE_PER_PROCESS
    assert updated.used_size == expected_total, \
        f"Expected used_size={expected_total}, got {updated.used_size}"


def test_tracker_update_memory_usage():
    tracker = MemInfoTracker(total_mem_size=TOTAL_MEM)
    tracker.update_memory_usage(used_size=600, prealloc_size=600)

    mem_info = get_kv_cache_limit(IPC_NAME)
    assert mem_info is not None
    assert mem_info.used_size == 600
    assert mem_info.prealloc_size == 600
    assert mem_info.total_size == TOTAL_MEM


def test_check_and_get_resize_target_returns_new_value():
    tracker = MemInfoTracker(total_mem_size=TOTAL_MEM)

    num_layers = 10
    expected_mem_per_layer = TOTAL_MEM // num_layers // 2

    # Simulate a stale current value
    current_mem_size = expected_mem_per_layer - 1

    resize_target = tracker.check_and_get_resize_target(
        current_mem_size, num_layers)
    assert resize_target == expected_mem_per_layer


def test_check_and_get_resize_target_returns_none_if_same():
    tracker = MemInfoTracker(total_mem_size=TOTAL_MEM)

    num_layers = 10
    expected_mem_per_layer = TOTAL_MEM // num_layers // 2

    # Simulate a updated current value
    current_mem_size = expected_mem_per_layer

    resize_target = tracker.check_and_get_resize_target(
        current_mem_size, num_layers)
    assert resize_target is None


def test_reaper_python_tracker_and_tool_share_lifetime(tmp_path, monkeypatch):
    import kvcached.mem_info_tracker as tracker_module
    import kvcached_reaper as ipc

    name = f"kvcached-reaper-python-{uuid.uuid4().hex}"
    root = tmp_path / "reaper"
    monkeypatch.setenv("KVCACHED_IPC_CLEANUP", "reaper")
    monkeypatch.setenv("KVCACHED_REAPER_DIR", str(root))
    monkeypatch.setattr(tracker_module, "DEFAULT_IPC_NAME", name)
    tracker = MemInfoTracker(TOTAL_MEM)
    state = ipc.Reaper(str(root))
    assert tracker._lease is not None and tracker._lease.fd is not None
    state.add(get_ipc_path(name), tracker._lease.fd)
    try:
        tracker.update_memory_usage(600, 900)
        with RwLockedShm(name, SHM_SIZE, RwLockedShm.RLOCK) as mm:
            tracker._unlink_segment()
            state.tick()
            assert os.path.exists(get_ipc_path(name))
            assert MemInfoStruct.from_buffer(mm) == MemInfoStruct(TOTAL_MEM, 600, 900)
            with pytest.raises(BlockingIOError):
                ipc.remove(get_ipc_path(name))
        state.tick()
        assert not os.path.exists(get_ipc_path(name))
    finally:
        tracker._unlink_segment()
        tracker_module._active_trackers.remove(tracker)
        state.close()
        if os.path.exists(get_ipc_path(name)):
            os.unlink(get_ipc_path(name))


def test_reaper_tool_mapping_failure_releases_lease(tmp_path, monkeypatch):
    import kvcached.cli.utils as cli
    import kvcached_reaper as ipc

    name = f"kvcached-reaper-map-{uuid.uuid4().hex}"
    root = tmp_path / "reaper"
    monkeypatch.setenv("KVCACHED_IPC_CLEANUP", "reaper")
    monkeypatch.setenv("KVCACHED_REAPER_DIR", str(root))
    owner = ipc.Lease(get_ipc_path(name), register_owner=False)
    state = ipc.Reaper(str(root))
    assert owner.fd is not None
    state.add(owner.path, owner.fd)

    def fail(*args, **kwargs):
        raise OSError("injected mapping failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(cli.mmap, "mmap", fail)
            with pytest.raises(OSError, match="injected mapping failure"):
                with RwLockedShm(name, SHM_SIZE, RwLockedShm.RLOCK):
                    pass
        owner.close()
        state.tick()
        assert not os.path.exists(owner.path)
    finally:
        owner.close()
        state.close()
        if os.path.exists(owner.path):
            os.unlink(owner.path)


def test_reaper_signal_handler_keeps_lease_until_process_death(tmp_path, monkeypatch):
    import kvcached.mem_info_tracker as tracker_module
    import kvcached_reaper as ipc

    name = f"kvcached-reaper-signal-{uuid.uuid4().hex}"
    monkeypatch.setenv("KVCACHED_IPC_CLEANUP", "reaper")
    monkeypatch.setenv("KVCACHED_REAPER_DIR", str(tmp_path / "reaper"))
    monkeypatch.setattr(tracker_module, "DEFAULT_IPC_NAME", name)
    tracker = MemInfoTracker(TOTAL_MEM)
    probe = os.open(get_ipc_path(name), os.O_RDWR)
    delivered = []

    def deliver(pid, signum):
        assert pid == os.getpid() and signum == tracker_module.signal.SIGTERM
        with pytest.raises(BlockingIOError):
            ipc.usage_lock(probe, fcntl.F_WRLCK)
        delivered.append(signum)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(tracker_module, "_active_trackers", [tracker])
            patch.setattr(tracker_module.signal, "signal", lambda *args: None)
            patch.setattr(tracker_module.os, "kill", deliver)
            tracker_module._cleanup_all(tracker_module.signal.SIGTERM, None)
        assert delivered == [tracker_module.signal.SIGTERM]
    finally:
        tracker._unlink_segment()
        tracker_module._active_trackers.remove(tracker)
        os.close(probe)
        os.unlink(get_ipc_path(name))
