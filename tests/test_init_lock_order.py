# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Real constructor/operations with CPU stand-ins only for the VMM boundary.

Each child owns its threads and native-module substitutes. A failed regression
reports both stacks and exits without making a stuck pytest worker hang CI.
"""

from __future__ import annotations

import importlib
import math
import multiprocessing as mp
import sys
import threading
import time
import traceback
import types
from collections import defaultdict
from multiprocessing.connection import Connection
from typing import Any

import pytest


class CPUPage:
    def __init__(self, page_id: int, page_size: int):
        self.page_id = page_id
        self.page_size = page_size
        self.blocks: list[int] = []
        self.capacity = 0

    def init(self, block_mem_size: int) -> None:
        start, end = self.get_block_range(self.page_id, self.page_size, block_mem_size)
        self.blocks = list(range(start, end))
        self.capacity = end - start

    def alloc(self, count: int) -> list[int]:
        taken, self.blocks = self.blocks[:count], self.blocks[count:]
        return taken

    def num_free_blocks(self) -> int:
        return len(self.blocks)

    def full(self) -> bool:
        return not self.blocks

    def empty(self) -> bool:
        return len(self.blocks) == self.capacity

    def free_batch(self, indices: list[int]) -> None:
        self.blocks.extend(indices)

    @staticmethod
    def get_num_blocks(page_size: int, block_mem_size: int) -> int:
        return page_size // block_mem_size

    @staticmethod
    def get_block_range(page_id: int, page_size: int, block_mem_size: int) -> tuple[int, int]:
        return (math.ceil(page_id * page_size / block_mem_size),
                (page_id + 1) * page_size // block_mem_size)


class CPUAllocator:
    def __init__(self, num_layers: int, mem_size: int, page_size: int,
                 world_size: int, **kwargs: Any):
        self.page_size = page_size
        self.total_pages = math.ceil(mem_size / page_size)
        self.free_pages_queue = list(range(self.total_pages))
        self.prealloc_started = False
        self.resize_target = -1
        self.capacity_ready = threading.Event()
        self.capacity_ready.set()
        self.miss_seen = threading.Event()
        self.allocation_misses = 0
        self.fatal = False
        self.closed = False

    def set_use_worker_ipc(self, enabled: bool) -> None:
        pass

    def get_resize_target(self) -> int:
        return self.resize_target

    def get_num_free_pages(self) -> int:
        if self.closed or self.fatal:
            from kvcached.errors import StateConsistencyError
            raise StateConsistencyError("CPU backend failed closed")
        return len(self.free_pages_queue)

    def get_avail_physical_pages(self) -> int:
        return len(self.free_pages_queue)

    def get_num_reserved_pages(self) -> int:
        return 0

    def get_page_state(self) -> dict[str, int]:
        return {"total_pages": self.total_pages, "free_pages": len(self.free_pages_queue),
                "inuse_pages": self.total_pages - len(self.free_pages_queue), "reserved_pages": 0}

    def alloc_page(self) -> CPUPage:
        if not self.capacity_ready.is_set():
            self.allocation_misses += 1
            self.miss_seen.set()
            raise RuntimeError("physical backing temporarily unavailable")
        return CPUPage(self.free_pages_queue.pop(0), self.page_size)

    def free_pages(self, indices: list[int]) -> None:
        self.free_pages_queue.extend(indices)

    def group_indices_by_page(self, indices: list[int], block_mem_size: int) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            groups[index * block_mem_size // self.page_size].append(index)
        return groups

    def start_prealloc_thread(self) -> None:
        self.prealloc_started = True

    def stop_prealloc_thread(self) -> None:
        self.prealloc_started = False

    def trim(self) -> None:
        pass

    def resize(self, size: int) -> bool:
        self.resize_target = -1
        return True

    def reset_free_page_order(self) -> None:
        self.free_pages_queue.sort()

    def release_shared_segment(self) -> bool:
        self.closed = True
        return True

    def get_transaction_state(self) -> dict[str, str]:
        return {"state": "FAILED" if self.fatal or self.closed else "READY"}


def _stack(thread: threading.Thread) -> list[str]:
    frame = sys._current_frames().get(thread.ident) if thread.ident is not None else None
    return [] if frame is None else [item.name for item in traceback.extract_stack(frame)]


def _child(operation: str, mode: str, connection: Connection) -> None:
    tensors_ready = threading.Event()
    check_seen = threading.Event()
    initializer: list[threading.Thread] = []
    allocators: list[CPUAllocator] = []

    def tensors_created(group_id: int = 0) -> bool:
        if not initializer:
            initializer.append(threading.current_thread())
        check_seen.set()
        return tensors_ready.is_set()

    def allocator_factory(*args: Any, **kwargs: Any) -> CPUAllocator:
        allocator = CPUAllocator(*args, **kwargs)
        if mode == "pressure":
            allocator.capacity_ready.clear()
        if mode == "startup_resize":
            allocator.resize_target = int(args[1])
        allocators.append(allocator)
        return allocator

    native: Any = types.ModuleType("kvcached.vmm_ops")
    native.PageAllocator = allocator_factory
    native.InternalPage = CPUPage
    native.kv_tensors_created = tensors_created
    native.map_to_kv_tensors = lambda *args, **kwargs: None
    native.unmap_from_kv_tensors = lambda *args, **kwargs: None
    native.init_kvcached = lambda *args, **kwargs: None
    native.shutdown_kvcached = lambda *args, **kwargs: None
    native.create_kv_tensors = lambda *args, **kwargs: None
    sys.modules["kvcached.vmm_ops"] = native
    ipc: Any = types.ModuleType("kvcached.integration.vllm.interfaces")
    ipc.should_use_worker_ipc = lambda: False
    sys.modules["kvcached.integration.vllm.interfaces"] = ipc
    import kvcached
    import kvcached.kv_cache_manager as kcm
    kvcached.vmm_ops = native  # type: ignore[attr-defined]
    kcm.PageAllocator = allocator_factory
    kcm.InternalPage = CPUPage
    kcm.kv_tensors_created = tensors_created

    if mode in ("ready", "pressure", "startup_resize"):
        tensors_ready.set()
    if operation == "sglang_factory":
        interface = importlib.reload(importlib.import_module("kvcached.integration.sglang.interfaces"))
        interface.init_kvcached(device="cuda:0", async_sched=True)
        manager = interface.get_kv_cache_manager(16, 1, 4096, 1)
    else:
        manager = kcm.KVCacheManager(
            16, 1, 4096, 1, async_sched=(mode != "sync"),
            reserve_null_block=(mode != "no_null"), page_size=16384,
        )
    if not check_seen.wait(2):
        raise RuntimeError("Constructor did not start its initializer")
    allocator = allocators[0]
    if mode == "ready":
        manager.wait_ready(timeout=2)
    if mode == "pressure" and not allocator.miss_seen.wait(2):
        raise RuntimeError("Initializer did not reach the physical miss")

    outcomes: list[dict[str, Any]] = []

    def run(index: int) -> None:
        try:
            if operation in ("alloc", "sglang_factory", "many_allocs"):
                value = manager.alloc(1)
            elif operation == "alloc_packed":
                value = manager.alloc_packed(1)
            elif operation == "free":
                value = manager.free([])
            elif operation == "reserve":
                value = manager.try_to_reserve(1)
            elif operation == "resize":
                value = manager.resize(manager.mem_size)
            elif operation == "limits":
                value = manager.set_memory_limit(manager.mem_size * 2, revision=(1 if index == 0 else 3))
            elif operation == "invalid_limit":
                value = manager.set_memory_limit(-1, revision=1)
            elif operation == "wait_ready":
                value = manager.wait_ready(timeout=2)
            else:
                value = getattr(manager, operation)()
            outcomes.append({"index": index, "value": value})
        except Exception as error:
            outcomes.append({"index": index, "error_type": type(error).__name__, "error": str(error)})

    count = 4 if operation == "many_allocs" else 2 if operation == "limits" else 1
    threads = [threading.Thread(target=run, args=(index,), daemon=True) for index in range(count)]
    for thread in threads:
        thread.start()
    if mode not in ("ready", "startup_resize"):
        deadline = time.monotonic() + 2
        while any(thread.is_alive() and "_wait_post_init" not in _stack(thread)
                  and "wait_ready" not in _stack(thread) for thread in threads):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.001)
    if mode == "fatal":
        allocator.fatal = True
    if operation == "invalid_limit":
        threads[0].join(1)
        before_ready = not threads[0].is_alive()
    else:
        before_ready = False
    tensors_ready.set()
    allocator.capacity_ready.set()
    for thread in threads:
        thread.join(1)
    initializer[0].join(1)
    alive = [thread for thread in [*threads, initializer[0]] if thread.is_alive()]
    result: dict[str, Any] = {
        "blocked": bool(alive), "stacks": [_stack(thread) for thread in alive],
        "outcomes": outcomes, "null_block": manager.null_block,
        "ready": manager._post_init_done.is_set(),
        "misses": allocator.allocation_misses, "invalid_before_ready": before_ready,
    }
    if not alive:
        result["phase"] = manager.lifecycle_phase.value
        if mode == "fatal":
            try:
                manager.wait_ready(timeout=1)
            except Exception as error:
                result["readiness_error"] = type(error).__name__
        else:
            if operation == "limits":
                result["control_ready"] = getattr(manager, "_runtime_control", None) is not None
            for outcome in outcomes:
                if operation in ("alloc", "alloc_packed", "sglang_factory", "many_allocs"):
                    manager.free(outcome["value"])
            manager.free_reserved()
            result["reserved_after_cleanup"] = list(manager.reserved_blocks)
            result["counters"] = dict(manager._operation_counters)
    connection.send(result)
    connection.close()


def _observe(operation: str, mode: str = "waiting") -> dict[str, Any]:
    context: Any = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(operation, mode, child))
    process.start()
    child.close()
    try:
        assert parent.poll(20), "Child did not return its diagnostics"
        result = parent.recv()
        process.join(2)
        assert process.exitcode == 0, result
        return result
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join()
        parent.close()


@pytest.mark.parametrize("operation", [
    "alloc", "alloc_packed", "free", "reserve", "resize", "trim", "clear",
    "apply_resize_target", "reclaim_for_shrink",
])
def test_operation_waits_without_holding_init_lock(operation: str) -> None:
    result = _observe(operation)
    assert not result["blocked"], result
    assert result["ready"] and result["null_block"] == [0]
    assert all("error" not in outcome for outcome in result["outcomes"]), result
    assert result["reserved_after_cleanup"] == []


@pytest.mark.parametrize("mode", ["ready", "sync", "no_null"])
def test_normal_controls(mode: str) -> None:
    result = _observe("alloc", mode)
    assert not result["blocked"], result
    assert result["outcomes"][0]["value"] == ([0] if mode == "no_null" else [1])


@pytest.mark.parametrize("operation", ["alloc", "sglang_factory"])
def test_capacity_recovery_finishes_initialization(operation: str) -> None:
    result = _observe(operation, "pressure")
    assert result["misses"] >= 1
    assert not result["blocked"], result
    assert result["null_block"] == [0] and result["outcomes"][0]["value"] == [1]


def test_initializer_can_apply_startup_resize() -> None:
    result = _observe("alloc", "startup_resize")
    assert not result["blocked"], result
    assert result["null_block"] == [0] and result["outcomes"][0]["value"] == [1]


def test_multiple_waiters_preserve_private_slots_and_counters() -> None:
    result = _observe("many_allocs")
    assert not result["blocked"], result
    slots = [outcome["value"][0] for outcome in result["outcomes"]]
    assert len(set(slots)) == 4 and 0 not in slots
    assert result["counters"]["allocation_requests_total"] == 4
    assert result["counters"]["allocated_blocks_total"] == 4
    assert result["counters"]["freed_blocks_total"] == 4


def test_preinitialization_limits_are_explicitly_unavailable() -> None:
    result = _observe("limits")
    assert not result["blocked"], result
    assert not result["control_ready"]
    assert all(outcome["error_type"] == "RuntimeError" for outcome in result["outcomes"])


def test_invalid_limit_is_rejected_before_readiness() -> None:
    result = _observe("invalid_limit")
    assert not result["blocked"] and result["invalid_before_ready"], result
    assert result["outcomes"][0]["error_type"] == "ValueError"


def test_fatal_initialization_wakes_waiters_without_swallowing_error() -> None:
    result = _observe("alloc", "fatal")
    assert not result["blocked"], result
    assert result["phase"] == "failed"
    assert result["outcomes"][0]["error_type"] == "StateConsistencyError"
    assert result["readiness_error"] == "StateConsistencyError"


@pytest.mark.parametrize("mode", ["waiting", "pressure"])
def test_shutdown_cancels_real_initialization(mode: str) -> None:
    result = _observe("shutdown", mode)
    assert not result["blocked"], result
    assert result["ready"] and result["null_block"] is None
    assert result["phase"] == "failed" and result["outcomes"][0]["value"] is True
