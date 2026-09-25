# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import gc
import importlib
import sys
import threading
import types
import weakref
from typing import Any

import pytest


def _import_kv_cache_manager(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    vmm_ops: Any = types.ModuleType("kvcached.vmm_ops")
    vmm_ops.kv_tensors_created = lambda *args, **kwargs: False
    vmm_ops.map_to_kv_tensors = lambda *args, **kwargs: True
    vmm_ops.unmap_from_kv_tensors = lambda *args, **kwargs: True
    vmm_ops.PageAllocator = object
    vmm_ops.InternalPage = object
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", vmm_ops)

    interfaces: Any = types.ModuleType(
        "kvcached.integration.vllm.interfaces"
    )
    interfaces.should_use_worker_ipc = lambda: False
    monkeypatch.setitem(
        sys.modules, "kvcached.integration.vllm.interfaces", interfaces
    )

    from kvcached import kv_cache_manager

    return kv_cache_manager


def test_post_init_timeout_keeps_last_observed_error(monkeypatch):
    kv_cache_manager = _import_kv_cache_manager(monkeypatch)
    manager = kv_cache_manager.KVCacheManager.__new__(
        kv_cache_manager.KVCacheManager)
    manager.null_block = None
    manager.world_size = 1
    manager.pp_rank = 0
    manager.group_id = 0
    manager._post_init_done = threading.Event()
    manager._shutdown_requested = threading.Event()

    calls = 0

    def fake_kv_tensors_created(group_id=0):
        nonlocal calls
        assert group_id == 0
        calls += 1
        if calls == 1:
            raise RuntimeError("kv tensor map failed")
        return False

    monkeypatch.setattr(kv_cache_manager, "kv_tensors_created",
                        fake_kv_tensors_created)
    monkeypatch.setattr(kv_cache_manager, "KV_TENSOR_WAIT_TIMEOUT", 0.002)
    monkeypatch.setattr(kv_cache_manager.time, "sleep", lambda _: None)

    with pytest.raises(TimeoutError, match="last error: kv tensor map failed"):
        manager._post_init()

    assert manager._post_init_done.is_set()
    with pytest.raises(TimeoutError, match="last error: kv tensor map failed"):
        manager._wait_post_init()


def test_readiness_deadline_includes_time_spent_in_ipc(monkeypatch):
    module = _import_kv_cache_manager(monkeypatch)
    manager = module.KVCacheManager.__new__(module.KVCacheManager)
    manager.null_block = None
    manager.world_size = 2
    manager._shutdown_requested = threading.Event()
    manager.pp_rank = 0
    manager.group_id = 0
    manager._post_init_done = threading.Event()
    now = [0.0]
    calls = []

    def stalled(*args, timeout_s, **kwargs):
        calls.append(timeout_s)
        now[0] += timeout_s
        raise TimeoutError("worker did not reply")

    monkeypatch.setattr(module, "broadcast_kv_tensors_created", stalled)
    monkeypatch.setattr(module, "KV_TENSOR_WAIT_TIMEOUT", 10.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    with pytest.raises(TimeoutError, match="worker did not reply"):
        manager._post_init()
    assert calls == [10.0]


def test_broadcast_callbacks_preserve_runtime_group_context(monkeypatch):
    kv_cache_manager = _import_kv_cache_manager(monkeypatch)
    calls = []

    class FakePageAllocator:
        def __init__(self, *args, **kwargs):
            self.map_callback = None
            self.unmap_callback = None

        def set_use_worker_ipc(self, enabled):
            self.use_worker_ipc = enabled

        def set_broadcast_map_callback(self, callback):
            self.map_callback = callback

        def set_broadcast_unmap_callback(self, callback):
            self.unmap_callback = callback

        def start_prealloc_thread(self):
            pass

    monkeypatch.setattr(kv_cache_manager, "PageAllocator", FakePageAllocator)
    monkeypatch.setattr(kv_cache_manager, "broadcast_kv_tensors_created",
                        lambda *args, **kwargs: False)
    monkeypatch.setattr(kv_cache_manager, "KV_TENSOR_WAIT_TIMEOUT", 0.001)
    monkeypatch.setattr(kv_cache_manager.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        kv_cache_manager.threading,
        "Thread",
        lambda *args, **kwargs: types.SimpleNamespace(start=lambda: None),
    )

    tp_ipc_util = importlib.import_module("kvcached.tp_ipc_util")

    monkeypatch.setattr(
        tp_ipc_util,
        "broadcast_map_to_kv_tensors",
        lambda world_size, offsets, pp_rank=0, group_id=0, record_stats=None: calls.append(
            ("map", world_size, offsets, pp_rank, group_id)
        ),
    )
    monkeypatch.setattr(
        tp_ipc_util,
        "broadcast_unmap_from_kv_tensors",
        lambda world_size, offsets, pp_rank=0, group_id=0: calls.append(
            ("unmap", world_size, offsets, pp_rank, group_id)
        ),
    )

    manager = kv_cache_manager.KVCacheManager(
        num_blocks=4,
        block_size=1,
        cell_size=1,
        num_layers=1,
        world_size=4,
        pp_rank=2,
        async_sched=True,
        group_id=1007,
    )

    manager.page_allocator.map_callback(4, [1, 2, 3])
    manager.page_allocator.unmap_callback(4, [4, 5])

    assert calls == [
        ("map", 4, [1, 2, 3], 2, 1007),
        ("unmap", 4, [4, 5], 2, 1007),
    ]

    allocator = manager.page_allocator
    manager_ref = weakref.ref(manager)
    del manager
    gc.collect()
    assert manager_ref() is None
    assert allocator.map_callback is not None
    assert allocator.unmap_callback is not None
