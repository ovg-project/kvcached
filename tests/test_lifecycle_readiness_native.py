# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Real-VMM lifetime and readiness agreement for a failed KV pool.

A native callback's hidden reference must not retain a FAILED pool, and
after an unmap failure the Python phase must agree with the native
transaction verdict (#478 review): the pool is FAILED, every entry stays
fail-closed per #418, and the pool is disposed of rather than cleared
back into service."""

import os
import subprocess
import sys

import pytest

SCENARIO = r"""
import gc
import threading
import time
import weakref

import torch
from kvcached import kv_cache_manager as kcm, tp_ipc_util as ipc
from kvcached import vmm_ops as native
from kvcached.errors import StateConsistencyError
from kvcached.lifecycle import LifecyclePhase

PAGE = 2 * 1024 * 1024
torch.cuda.set_device(0)
native.init_kvcached("cuda:0", PAGE, False)
kcm.broadcast_kv_tensors_created = (
    lambda world_size, pp_rank=0, group_id=0: native.kv_tensors_created(group_id))
inject = False
hits = 0

def map_pages(world_size, offsets, pp_rank=0, group_id=0):
    assert native.map_to_kv_tensors(offsets, group_id)

def unmap_pages(world_size, offsets, pp_rank=0, group_id=0):
    global hits
    if inject:
        hits += 1
        raise RuntimeError("injected native unmap failure")
    assert native.unmap_from_kv_tensors(offsets, group_id)

ipc.broadcast_map_to_kv_tensors = map_pages
ipc.broadcast_unmap_from_kv_tensors = unmap_pages

def expect_fail_closed(call):
    try:
        call()
    except StateConsistencyError:
        return
    raise AssertionError("a FAILED pool must stay fail-closed")

def run_case(group_id, fail, defer_release):
    global inject
    tensors = native.create_kv_tensors(PAGE * 4, 1, "cuda:0", 1, 1, group_id)
    # world_size=2 selects the production C++ -> Python broadcast callback.
    # Its body maps a single local device; this is not a distributed TP test.
    manager = kcm.KVCacheManager(
        16, 1, PAGE // 4, 1, world_size=2, num_kv_buffers=1,
        group_id=group_id, pool_name="native-lifetime",
        defer_physical_release=defer_release)
    manager.wait_ready(timeout=5)
    blocks = manager.alloc(1)
    assert blocks and len(blocks) == 1
    view = tensors[0].reshape(-1)[blocks[0] * (PAGE // 4):][:16]
    view.fill_(7)
    torch.cuda.synchronize()
    assert torch.all(view == 7).item()
    inject = fail
    try:
        manager.free(blocks)
        if defer_release:
            assert manager.lifecycle_phase is LifecyclePhase.READY
            manager.release_retired_pages_through(manager.capture_physical_release_marker())
        assert not fail, "fault did not reach the native callback"
    except StateConsistencyError as exc:
        assert fail and "injected native unmap failure" in str(exc)
        # The manager records the native verdict where it propagates: the
        # phase, the transaction state, the readiness gate, and the next
        # alloc all agree on FAILED now.
        assert manager.lifecycle_phase is LifecyclePhase.FAILED
        state = manager.page_allocator.get_transaction_state()
        assert state["state"] == "FAILED", state
        assert manager.lifecycle_error is not None
        assert manager.lifecycle_error.__traceback__ is not None
        snapshot = manager.observability_snapshot()
        assert snapshot.lifecycle_phase == "failed"
        assert snapshot.available_blocks == 0
        assert snapshot.available_bytes == 0
        assert snapshot.total_pages > 0
        expect_fail_closed(manager.available_size)
        expect_fail_closed(lambda: manager.wait_ready(timeout=5))
        expect_fail_closed(lambda: manager.alloc(1))
        assert manager.lifecycle_phase is LifecyclePhase.FAILED
    finally:
        inject = False
    if fail:
        # #418 stays fail-closed: a FAILED pool is disposed of, never
        # cleared back into service, and clear() on it re-raises the
        # verdict instead of reopening the readiness gate. Disposal is
        # the weakref collection checked by the caller.
        expect_fail_closed(manager.clear)
        assert manager.lifecycle_phase is LifecyclePhase.FAILED
    else:
        manager.clear()
        assert manager.lifecycle_phase is LifecyclePhase.READY
    return weakref.ref(manager), weakref.ref(manager._lifecycle), tensors

for fail, defer_release in ((False, False), (True, False),
                           (False, True), (True, True)):
    refs = []
    buffers = []
    for index in range(3):
        manager_ref, lifecycle_ref, tensors = run_case(
            100 + int(fail) * 10 + int(defer_release) * 20 + index, fail, defer_release)
        refs.append((manager_ref, lifecycle_ref))
        buffers.append(tensors)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        gc.collect()
        if all(ref() is None for pair in refs for ref in pair):
            break
        time.sleep(0.01)
    assert all(ref() is None for pair in refs for ref in pair), (
        "native callback retained the lifecycle/traceback/manager cycle", fail)
    del tensors, buffers
    gc.collect()
assert hits == 6, hits

def run_cancelled_clear(group_id, async_sched):
    tensors = native.create_kv_tensors(PAGE * 4, 1, "cuda:0", 1, 1, group_id)
    manager = kcm.KVCacheManager(
        16, 1, PAGE // 4, 1, world_size=2, num_kv_buffers=1,
        group_id=group_id, reserve_null_block=True, async_sched=async_sched,
        own_segment=True)
    manager.wait_ready(timeout=5)
    # Healthy clear still rebuilds the null page and makes it usable.
    manager.clear()
    manager.wait_ready(timeout=5)
    assert manager.page_allocator.get_num_inuse_pages() == 1
    view = tensors[0].reshape(-1)[:16]
    view.fill_(7)
    torch.cuda.synchronize()
    assert torch.all(view == 7).item()
    del view
    reserving = threading.Event()
    errors = []
    reserve = manager._reserve_null_block

    def reserve_without_capacity():
        # At this boundary clear has unmapped the old null page. Apply a
        # real native zero-capacity limit so reservation waits for shutdown.
        assert manager.page_allocator.get_num_inuse_pages() == 0
        assert manager.page_allocator.resize(0)
        assert manager.available_size() == 0
        reserving.set()
        reserve()

    def clear():
        try:
            manager.clear()
        except Exception as exc:
            errors.append(exc)

    manager._reserve_null_block = reserve_without_capacity
    clearer = threading.Thread(target=clear, daemon=True)
    clearer.start()
    try:
        assert reserving.wait(5)
        assert manager.lifecycle_phase is LifecyclePhase.INITIALIZING
        assert manager.shutdown()
    finally:
        manager._shutdown_requested.set()
        clearer.join(5)
    assert not clearer.is_alive() and not errors
    assert manager.lifecycle_phase is LifecyclePhase.FAILED
    assert manager.null_block is None
    assert manager.page_allocator.get_num_inuse_pages() == 0
    try:
        manager.wait_ready(timeout=0)
    except RuntimeError as exc:
        assert "clear() cancelled by shutdown" in str(exc)
    else:
        raise AssertionError("cancelled clear reopened the readiness gate")

for async_sched in (False, True):
    run_cancelled_clear(200 + int(async_sched), async_sched)
native.shutdown_kvcached()
print("native lifetime: healthy=6/6, failed=6/6, injections=6, cancelled-clear=2/2", flush=True)
"""


def test_native_unmap_error_does_not_retain_manager():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA VMM and the native PageAllocator")

    env = dict(os.environ)
    env.update(
        ENABLE_KVCACHED="false",
        KVCACHED_AUTOPATCH="0",
        KVCACHED_PAGE_SIZE_MB="2",
        KVCACHED_MIN_RESERVED_PAGES="0",
        KVCACHED_MAX_RESERVED_PAGES="0",
        KVCACHED_PAGE_PREALLOC_ENABLED="false",
        KVCACHED_CONTIGUOUS_LAYOUT="false",
        KVCACHED_IPC_NAME=f"lifecycle-native-{os.getpid()}",
    )
    result = subprocess.run(
        [sys.executable, "-c", SCENARIO], env=env,
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "healthy=6/6, failed=6/6, injections=6" in result.stdout
    assert "cancelled-clear=2/2" in result.stdout
