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


@pytest.mark.parametrize("source,consumer", [
    (source, consumer)
    for source in ("foreground", "background")
    for consumer in ("foreground", "background", "fatal", "gil", "resize",
                     "resize_fatal", "resize_quarantined")
] + [("background", "fresh"), ("background", "exact"), ("background", "mixed")])
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
    from types import SimpleNamespace

    import torch

    from kvcached import vmm_ops as vmm
    from kvcached.errors import (
        MapQuarantinedError,
        MapRetainedError,
        QuarantinedResizeError,
        StateConsistencyError,
    )
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
    calls: list[tuple[int, ...]] = []
    required = []
    unmapped = []
    formatting, release_format = threading.Event(), threading.Event()

    class SlowError(RuntimeError):
        def __str__(self):
            formatting.set()
            assert release_format.wait(10), "error formatting was not released"
            return "ordinary Python callback failure"

    driver = ctypes.CDLL(None)
    driver.kvcached_fault_zero_free_memory.argtypes = [ctypes.c_int]

    def map_pages(world_size, offsets):
        if calls and tuple(offsets) != calls[0]:
            if consumer not in ("fresh", "mixed"):
                raise RuntimeError("no headroom for a fresh page")
            assert offsets == [2 * page], "a retained batch was split or reused"
        calls.append(tuple(offsets))
        if len(calls) > 1 and consumer not in ("fresh", "mixed"):
            assert calls[-1] == calls[0], "a retained batch was split or reordered"
        transaction = f"retry-{len(calls)}"
        prepared = vmm.prepare_map_to_kv_tensors(transaction, offsets, 0)
        assert prepared["success"]
        required.append(prepared["required_bytes"])
        assert vmm.commit_prepared_map(transaction, 0)["success"]
        if len(calls) == 1:
            driver.kvcached_fault_zero_free_memory(1)
            raise MapRetainedError("controlled peer commit failure after local mapping")
        if len(calls) == 2 and consumer != "mixed":
            if consumer in ("fatal", "resize_fatal"):
                raise StateConsistencyError("retry outcome is unknown")
            if consumer == "resize_quarantined":
                raise MapQuarantinedError("unpublished mapping quarantined")
            if consumer == "gil":
                raise SlowError()
            raise RuntimeError("peer still has no capacity")

    def unmap_pages(world_size, offsets):
        assert vmm.unmap_from_kv_tensors(offsets, 0)
        unmapped.append(tuple(offsets))

    allocator.set_broadcast_map_callback(map_pages)
    allocator.set_broadcast_unmap_callback(unmap_pages)
    manager = object.__new__(KVCacheManager)
    manager.page_allocator = allocator
    manager.page_size = page
    manager.mem_size = 8 * page
    manager.block_mem_size = page // 4
    manager.num_avail_blocks = 0
    manager.avail_pages = {}
    manager.full_pages = {}
    manager.reserved_blocks = []
    manager.null_block = None
    manager.in_shrink = False
    manager.target_num_pages = None
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
        assert allocator.get_num_retryable_pages() == batch_pages
        # The driver shim reports zero free memory to both the manager and
        # native preallocator. All actual retry mappings already exist locally.
        assert allocator.get_avail_physical_pages() == 0
        manager._avail_physical_pages_cache = None
        assert manager.available_size() == batch_pages * 4
        # Non-divisible geometry must count retained page IDs, including gaps,
        # even with no fresh headroom; the block-ID bound still clips them.
        block_bytes = 3 * page // 4
        expected = sum((i + 1) * page // block_bytes
                       - (i * page + block_bytes - 1) // block_bytes
                       for i in range(batch_pages))
        assert allocator.get_allocatable_block_capacity(block_bytes, 0) == expected
        assert allocator.get_allocatable_block_capacity(block_bytes, 0, 1) == 1
        assert allocator.get_allocatable_block_capacity(block_bytes, 0, 0) == 0
        manager.block_mem_size = block_bytes
        assert manager.available_size() == expected
        manager.block_mem_size = page // 4
        if consumer == "mixed":
            # Force the native budget to two pages. Only the spare from the
            # retained batch can pay for the fresh ID, which is requested first.
            driver.kvcached_fault_zero_free_memory(0)
            assert vmm.set_instance_budget_limit(2 * page, 1, 0) == "applied"
            manager._avail_physical_pages_cache = None
            manager._virtual_num_blocks = 32
            manager.num_layers = manager.num_kv_buffers = 1
            manager.reserve_null_block = False
            manager._retired_pages = []

            forced = []

            def worker_quota(*, force=False):
                state = vmm.physical_budget_state()
                room = state["instance_limit_bytes"] - state["instance_charged_bytes"]
                if force:
                    forced.append(room)
                return [{"available_bytes": max(0, room), "signed_available_bytes": room, "page_bundle_bytes": page}]

            def retry_headroom():
                return worker_quota(force=True)[0]["signed_available_bytes"] // page

            setattr(manager, "_runtime_control", SimpleNamespace(
                worker_ipc=True, poll=worker_quota, exact_retry_headroom=retry_headroom, apply=lambda: None))
            assert manager._get_cached_avail_physical_pages() == 0
            assert manager.exact_recovery_quote([8, 4]) == (0, [], 0)
            assert manager.can_alloc_exact([8, 4])
            assert manager.alloc_exact([8, 4]) == [8, 4]
            assert calls == [(0, page), (0, page), (2 * page,)]
            assert unmapped == [(0,)]
            assert forced == [0]
            for block in [8, 4]:
                view = tensors[0][block * (page // 4):block * (page // 4) + 128]
                view.fill_(41)
                assert torch.all(view == 41).item()
            torch.cuda.synchronize()
            manager.free([8, 4])
            allocator.trim()
        elif consumer == "exact":
            # No local headroom. An unrelated ID must leave the batch alone,
            # and an exact retry of its second ID must replay both offsets.
            assert allocator.try_alloc_page_at(2) is None
            assert len(calls) == 1
            assert allocator.exact_recovery_quote([1]) == (0, [], 0)
            assert allocator.exact_recovery_quote([2]) == (page, [], 0)
            with pytest.raises(RuntimeError, match="peer still has no capacity"):
                allocator.try_alloc_page_at(1)
            assert allocator.get_num_retryable_pages() == 2
            recovered = allocator.try_alloc_page_at(1)
            assert recovered.page_id == 1
            assert calls == [(0, page)] * 3
            assert allocator.try_alloc_page_at(0).page_id == 0
            assert len(calls) == 3
            view = tensors[0][page:page + 128]
            view.fill_(37)
            assert torch.all(view == 37).item()
            torch.cuda.synchronize()
            allocator.free_pages([0, 1])
            allocator.trim()
        elif consumer == "fresh":
            driver.kvcached_fault_zero_free_memory(0)
            blocks = manager.alloc(1)
            assert blocks == [8]
            assert calls == [(0, page), (0, page), (2 * page,)]
            assert allocator.get_num_retryable_pages() == 2
            view = tensors[0][2 * page:2 * page + 128]
            view.fill_(29)
            assert torch.all(view == 29).item()
            torch.cuda.synchronize()
            manager.free(blocks)
            # A later allocation can still adopt the original, unsplit batch.
            blocks = manager.alloc(1)
            assert blocks == [0] and calls[-1] == calls[0]
            assert allocator.get_num_retryable_pages() == 0
            manager.free(blocks)
            allocator.trim()
            assert allocator.get_num_free_pages() == 8
            assert required == [2 * page, 0, page, 0]
        elif consumer == "resize":
            # The engine stays idle: only quota polling can drive recovery.
            # A still-failing peer must not complete the shrink or drop ownership.
            assert not manager._resize(0)
            assert manager.in_shrink and manager.target_num_pages == 0
            assert len(calls) == 2 and not unmapped
            assert allocator.get_num_retryable_pages() == batch_pages
            assert allocator.get_num_free_pages() == 8
            assert manager._resize(0)
            assert len(calls) == 3 and unmapped == [calls[0]]
            assert allocator.get_num_retryable_pages() == 0
            assert allocator.get_num_free_pages() == 0
            assert allocator.get_num_reserved_pages() == 0
        elif consumer == "resize_quarantined":
            with pytest.raises(QuarantinedResizeError, match="quarantined"):
                manager._resize(0)
            assert len(calls) == 2 and not unmapped
            assert allocator.get_num_retryable_pages() == 0
            with pytest.raises(QuarantinedResizeError):
                manager._resize(0)
            assert len(calls) == 2
        elif consumer in ("fatal", "resize_fatal"):
            with pytest.raises(StateConsistencyError, match="unknown"):
                if consumer == "resize_fatal":
                    manager._resize(0)
                else:
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
        if consumer not in ("fresh", "mixed"):
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


@pytest.mark.parametrize("mode", ["queued", "quota", "quota_null", "quota_busy", "quota_busy_null",
                                  "quota_cache", "quota_reserve", "quota_exact", "quota_impossible_exact", "null_init"])
def test_retained_recovery_boundaries(mode):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires native CUDA VMM")
    env = dict(os.environ, ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0",
               KVCACHED_CONTIGUOUS_LAYOUT="false", KVCACHED_PAGE_SIZE_MB="2",
               KVCACHED_PAGE_PREALLOC_ENABLED="false", KVCACHED_MIN_RESERVED_PAGES="2",
               KVCACHED_MAX_RESERVED_PAGES="1" if mode in ("quota_reserve", "quota_exact", "quota_impossible_exact") else "0",
               KVCACHED_IPC_NAME=f"retained-{mode}-{os.getpid()}")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), mode],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def _queued_case():
    """Real native batches with a constructed peer that cannot recover two pages."""
    import time

    import torch

    from kvcached import vmm_ops as vmm
    from kvcached.errors import MapRetainedError

    page = 2 * 1024 * 1024
    vmm.init_kvcached("cuda:0", page, False)
    tensors = vmm.create_kv_tensors(3 * page, 1, "cuda:0", 1, 1)
    allocator = vmm.PageAllocator(1, 3 * page, page, contiguous_layout=False,
                                  enable_page_prealloc=True, num_kv_buffers=1,
                                  ipc_name=os.environ["KVCACHED_IPC_NAME"])
    allocator.set_use_worker_ipc(True)
    calls: list[tuple[int, ...]] = []
    recover_large = False

    def map_pages(world_size, offsets):
        calls.append(tuple(offsets))
        if offsets == [0, page] and len(calls) > 1 and not recover_large:
            raise RuntimeError("peer cannot reserve two pages")
        transaction = f"queued-{len(calls)}"
        assert vmm.prepare_map_to_kv_tensors(transaction, offsets, 0)["success"]
        assert vmm.commit_prepared_map(transaction, 0)["success"]
        if calls.count(tuple(offsets)) == 1:
            raise MapRetainedError("controlled peer failure after local mapping")

    allocator.set_broadcast_map_callback(map_pages)
    allocator.set_broadcast_unmap_callback(
        lambda world_size, offsets: vmm.unmap_from_kv_tensors(offsets, 0))
    try:
        allocator.start_prealloc_thread()
        deadline = time.monotonic() + 10
        while allocator.get_num_retryable_pages() != 2:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        allocator.stop_prealloc_thread()
        with pytest.raises(MapRetainedError):
            allocator.alloc_page()
        assert allocator.get_num_retryable_pages() == 3
        # No fresh IDs remain. A failed two-page retry must rotate behind the
        # retained one-page batch, which can recover on the next attempt.
        with pytest.raises(RuntimeError, match="cannot reserve two pages"):
            allocator.alloc_page()
        recovered = allocator.alloc_page()
        assert recovered.page_id == 2
        assert calls[-1] == (2 * page,)
        assert allocator.get_num_retryable_pages() == 2
        view = tensors[0][2 * page:2 * page + 128]
        view.fill_(41)
        assert torch.all(view == 41).item()
        torch.cuda.synchronize()
        allocator.free_page(recovered.page_id)
        recover_large = True
        assert allocator.resize(0)
        assert allocator.get_num_retryable_pages() == 0
        assert allocator.get_num_free_pages() == 0
        print("PASS retained batches rotate without splitting payloads", flush=True)
    finally:
        allocator.stop_prealloc_thread()
        vmm.shutdown_kvcached()


def _quota_case(mode):
    """Real grants/control; the initial retained peer is simulated."""
    import torch

    from kvcached import vmm_ops as vmm
    from kvcached.errors import MapRetainedError, MapRetainedQuotaError
    from kvcached.kv_cache_manager import KVCacheManager
    from kvcached.runtime_control import RuntimeControl

    page = 2 * 1024 * 1024
    reserve_null = mode.endswith("_null")
    cached = mode == "quota_cache"
    transient = "busy" in mode or cached
    floor = page if reserve_null else 0
    vmm.init_kvcached("cuda:0", page, False)
    tensors = vmm.create_kv_tensors(8 * page, 1, "cuda:0", 1, 1)
    manager = KVCacheManager(32, 1, page // 4, 1, num_kv_buffers=1,
                             reserve_null_block=reserve_null, page_size=page)
    calls = []

    def map_pages(world_size, offsets):
        calls.append(tuple(offsets))
        if len(calls) == 1:
            raise MapRetainedError("a simulated peer mapped; this rank did not")
        transaction = f"quota-{len(calls)}"
        prepared = vmm.prepare_map_to_kv_tensors(transaction, offsets, 0)
        if not prepared["success"]:
            assert prepared["quota_rejections"] == 1
            assert prepared["quota_recovery_impossible"] == int(not transient)
            error = MapRetainedError if transient else MapRetainedQuotaError
            raise error("instance limit prevents peer recovery")
        assert vmm.commit_prepared_map(transaction, 0)["success"]

    def publish(size):
        result = manager.set_memory_limit(size)
        assert result["accepted"]
        runtime.poll(force=True)
        manager.apply_resize_target()
        runtime.poll(force=True)

    try:
        manager.wait_ready(timeout=5)
        runtime = RuntimeControl(manager, worker_ipc=False, can_shrink=True,
                                 expected_participants=1)
        setattr(manager, "_runtime_control", runtime)
        pinned = manager.alloc(3) if transient and reserve_null else []
        first = manager.alloc(4) if transient else []
        second = manager.alloc(4) if transient else []
        assert pinned is not None and first is not None and second is not None
        manager.page_allocator.set_use_worker_ipc(True)
        manager.page_allocator.set_broadcast_map_callback(map_pages)
        manager.page_allocator.set_broadcast_unmap_callback(
            lambda world_size, offsets: vmm.unmap_from_kv_tensors(offsets, 0))
        request = 4 if reserve_null and not transient else 1
        assert manager.alloc(request) is None
        assert manager.page_allocator.get_num_retryable_pages() == 1
        if transient:
            publish(floor + 2 * page)
            assert manager.memory_limit_state()["status"] == "pending"
            assert runtime.current[0]["failed"] == 0
            assert manager.page_allocator.get_num_retryable_pages() == 1
            if cached:
                class CacheOwner:
                    def __init__(self):
                        self.evictions = 0

                    def evict(self):
                        if manager.shrink_excess_pages() and self.evictions == 0:
                            self.evictions += 1
                            manager.free(first)

                owner = CacheOwner()
                manager._register_shrink_reclaimer(owner.evict)
                manager.reclaim_for_shrink()
                assert owner.evictions == 1
            else:
                manager.free(first)
                manager.trim()
            runtime.poll(force=True)
            manager.apply_resize_target()
            runtime.poll(force=True)
            assert manager.memory_limit_state()["status"] == "converged"
            assert manager.page_allocator.get_num_retryable_pages() == 0
            view = tensors[0][second[0] * (page // 4):second[0] * (page // 4) + 128]
            view.fill_(47)
            assert torch.all(view == 47).item()
            torch.cuda.synchronize()
            manager.free(second)
            manager.free(pinned)
            publish(floor)
            assert manager.memory_limit_state()["status"] == "converged"
            print("PASS occupied pages release without failing a recoverable quota", flush=True)
            return
        publish(floor)
        failed = manager.memory_limit_state()
        assert failed["status"] == "failed", failed
        assert runtime.current[0]["failure_reason"] == "retained_mapping_exceeds_limit"
        assert vmm.physical_budget_state()["instance_limit_bytes"] == floor
        assert vmm.physical_budget_state()["instance_charged_bytes"] == floor
        for _ in range(3):
            manager.apply_resize_target()
        assert len(calls) == 2
        assert manager.page_allocator.get_num_retryable_pages() == 1
        publish(8 * page)
        blocks = manager.alloc(request)
        assert blocks is not None and len(blocks) == request
        view = tensors[0][blocks[-1] * (page // 4):blocks[-1] * (page // 4) + 128]
        view.fill_(43)
        assert torch.all(view == 43).item()
        torch.cuda.synchronize()
        manager.free(blocks)
        publish(floor)
        assert manager.memory_limit_state()["status"] == "converged"
        assert manager.page_allocator.get_num_retryable_pages() == 0
        print("PASS impossible quota rejected and newer quota recovers", flush=True)
    finally:
        manager.shutdown()
        vmm.shutdown_kvcached()


def _reserved_quota_case(mode):
    """A native reserve funds recoverable retries and survives an impossible quota."""
    import torch

    from kvcached import vmm_ops as vmm
    from kvcached.errors import MapRetainedError, MapRetainedQuotaError

    page = 2 * 1024 * 1024
    vmm.init_kvcached("cuda:0", page, False)
    tensors = vmm.create_kv_tensors(8 * page, 1, "cuda:0", 1, 1)
    allocator = vmm.PageAllocator(1, 8 * page, page, contiguous_layout=False,
                                  enable_page_prealloc=False, num_kv_buffers=1,
                                  ipc_name=os.environ["KVCACHED_IPC_NAME"])
    first = allocator.alloc_page()
    live = allocator.alloc_page()
    calls = []

    def map_pages(world_size, offsets):
        calls.append(tuple(offsets))
        if len(calls) == 1:
            raise MapRetainedError("a simulated peer mapped the retained page")
        transaction = f"reserve-{len(calls)}"
        prepared = vmm.prepare_map_to_kv_tensors(transaction, offsets, 0)
        if not prepared["success"]:
            if prepared["quota_recovery_impossible"]:
                raise MapRetainedQuotaError("batch exceeds the peer quota")
            raise MapRetainedError("healthy reserve still holds the needed credit")
        assert vmm.commit_prepared_map(transaction, 0)["success"]

    allocator.set_use_worker_ipc(True)
    allocator.set_broadcast_map_callback(map_pages)
    allocator.set_broadcast_unmap_callback(
        lambda world_size, offsets: vmm.unmap_from_kv_tensors(offsets, 0))
    try:
        with pytest.raises(MapRetainedError):
            allocator.alloc_page()
        allocator.free_page(first.page_id)
        assert allocator.get_num_reserved_pages() == 1
        impossible = mode == "quota_impossible_exact"
        limit = 0 if impossible else 2 * page
        assert vmm.set_instance_budget_limit(limit, 1, 0) == "applied"
        if impossible:
            with pytest.raises(MapRetainedQuotaError):
                allocator.try_alloc_page_at(2)
            assert calls == [(2 * page,)] * 2
            assert allocator.get_num_reserved_pages() == 1
            idle = allocator.try_alloc_page_at(first.page_id)
            assert idle.page_id == first.page_id
            allocator.free_page(idle.page_id)
            limit = 3 * page
            assert vmm.set_instance_budget_limit(limit, 2, 0) == "applied"
        if mode != "quota_reserve":
            recovered = allocator.try_alloc_page_at(2)
            assert calls == [(2 * page,)] * 3
        else:
            assert allocator.resize(2 * page)
            recovered = allocator.alloc_page()
        assert allocator.get_num_retryable_pages() == 0
        assert vmm.physical_budget_state()["instance_charged_bytes"] == limit
        assert recovered.page_id == 2
        for pid in (live.page_id, recovered.page_id):
            view = tensors[0][pid * page:pid * page + 128]
            view.fill_(59)
            assert torch.all(view == 59).item()
        torch.cuda.synchronize()
        allocator.free_pages([live.page_id, recovered.page_id])
        allocator.trim()
        print("PASS native reservations release credit before retained recovery", flush=True)
    finally:
        vmm.shutdown_kvcached()


def _null_init_case():
    """Normal manager initialization with controlled peer outcomes and real CUDA."""
    import torch

    from kvcached import kv_cache_manager as manager_module, tp_ipc_util as ipc, vmm_ops as vmm
    from kvcached.errors import MapRetainedError

    page = 2 * 1024 * 1024
    vmm.init_kvcached("cuda:0", page, False)
    tensors = vmm.create_kv_tensors(8 * page, 1, "cuda:0", 1, 1)
    calls = []

    def map_pages(world_size, offsets, *args, **kwargs):
        calls.append(tuple(offsets))
        if len(calls) == 2:
            raise RuntimeError("peer is temporarily full")
        transaction = f"null-init-{len(calls)}"
        assert vmm.prepare_map_to_kv_tensors(transaction, offsets, 0)["success"]
        assert vmm.commit_prepared_map(transaction, 0)["success"]
        if len(calls) == 1:
            raise MapRetainedError("controlled peer commit failure")

    setattr(ipc, "broadcast_map_to_kv_tensors", map_pages)
    setattr(ipc, "broadcast_unmap_from_kv_tensors",
            lambda world_size, offsets, *args, **kwargs: vmm.unmap_from_kv_tensors(offsets, 0))
    manager_module.broadcast_kv_tensors_created = lambda *args, **kwargs: True
    manager = manager_module.KVCacheManager(
        32, 1, page // 4, 1, world_size=2, num_kv_buffers=1,
        reserve_null_block=True, page_size=page)
    try:
        manager.wait_ready(timeout=10)
        assert manager.null_block == [0]
        assert calls[:3] == [(0,), (0,), (0,)]
        blocks = manager.alloc(1)
        assert blocks == [1]
        tensors[0][page // 4:page // 4 + 128].fill_(53)
        assert torch.all(tensors[0][page // 4:page // 4 + 128] == 53).item()
        torch.cuda.synchronize()
        manager.free(blocks)
        print("PASS null initialization retries page zero without a fresh bypass", flush=True)
    finally:
        manager.shutdown()
        vmm.shutdown_kvcached()


if __name__ == "__main__":
    if sys.argv[1] == "clear":
        _clear_case()
    elif sys.argv[1] == "queued":
        _queued_case()
    elif sys.argv[1] == "null_init":
        _null_init_case()
    elif sys.argv[1] in ("quota_reserve", "quota_exact", "quota_impossible_exact"):
        _reserved_quota_case(sys.argv[1])
    elif sys.argv[1].startswith("quota"):
        _quota_case(sys.argv[1])
    else:
        _case(sys.argv[1], sys.argv[2])
