# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict


def _load_manager_module(monkeypatch):
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
    torch.dtype = type("dtype", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)

    vmm_ops = types.ModuleType("kvcached.vmm_ops")
    vmm_ops.InternalPage = type("InternalPage", (), {})  # type: ignore[attr-defined]
    vmm_ops.PageAllocator = type("PageAllocator", (), {})  # type: ignore[attr-defined]
    vmm_ops.kv_tensors_created = lambda *args, **kwargs: False  # type: ignore[attr-defined]
    vmm_ops.map_to_kv_tensors = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    vmm_ops.unmap_from_kv_tensors = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", vmm_ops)

    interfaces = types.ModuleType("kvcached.integration.vllm.interfaces")
    interfaces.should_use_worker_ipc = lambda: False  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "kvcached.integration.vllm.interfaces",
        interfaces,
    )

    module_path = Path(__file__).parents[1] / "kvcached" / "kv_cache_manager.py"
    spec = importlib.util.spec_from_file_location(
        "kvcached._test_kv_cache_manager", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_control_only_manager_uses_worker_meminfo(monkeypatch):
    manager_module = _load_manager_module(monkeypatch)
    observed: Dict[str, Any] = {}

    class FakePageAllocator:
        def __init__(self, *args, **kwargs):
            observed["constructor"] = kwargs

        def set_mem_info(self, free_bytes, total_bytes):
            observed["mem_info"] = (free_bytes, total_bytes)

        def set_use_worker_ipc(self, enabled):
            observed["use_worker_ipc"] = enabled

        def set_broadcast_map_callback(self, callback):
            observed["map_callback"] = callback

        def set_broadcast_unmap_callback(self, callback):
            observed["unmap_callback"] = callback

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(manager_module, "PageAllocator", FakePageAllocator)
    monkeypatch.setattr(manager_module.threading, "Thread", FakeThread)

    import kvcached.meminfo_provider as meminfo_provider

    monkeypatch.setattr(
        meminfo_provider,
        "query_mem_info",
        lambda world_size, pp_rank: (123, 456),
    )

    manager = manager_module.KVCacheManager(
        num_blocks=4,
        block_size=1,
        cell_size=1,
        num_layers=1,
        world_size=2,
        cuda_control_plane=False,
    )
    manager._refresh_mem_info(force=True)

    assert observed["constructor"]["cuda_control_plane"] is False
    assert observed["constructor"]["enable_page_prealloc"] is False
    assert observed["use_worker_ipc"] is False
    assert observed["mem_info"] == (123, 456)


def test_meminfo_refresh_fails_closed_after_initial_snapshot(monkeypatch):
    manager_module = _load_manager_module(monkeypatch)
    updates = []
    manager = manager_module.KVCacheManager.__new__(manager_module.KVCacheManager)
    manager.cuda_control_plane = False
    manager.world_size = 1
    manager.pp_rank = 0
    manager._last_meminfo_refresh = 0.0
    manager._meminfo_initialized = True
    manager._meminfo_total_bytes = 456
    manager.page_allocator = types.SimpleNamespace(
        set_mem_info=lambda free, total: updates.append((free, total)))

    import kvcached.meminfo_provider as meminfo_provider

    def fail(*args, **kwargs):
        raise TimeoutError("worker unavailable")

    monkeypatch.setattr(meminfo_provider, "query_mem_info", fail)
    manager._refresh_mem_info(force=True)

    assert updates == [(0, 456)]


def test_meminfo_refresh_recovers_after_fail_closed_snapshot(monkeypatch):
    manager_module = _load_manager_module(monkeypatch)
    updates = []
    manager = manager_module.KVCacheManager.__new__(manager_module.KVCacheManager)
    manager.cuda_control_plane = False
    manager.world_size = 4
    manager.pp_rank = 0
    manager._last_meminfo_refresh = 0.0
    manager._meminfo_initialized = True
    manager._meminfo_total_bytes = 456
    manager.page_allocator = types.SimpleNamespace(
        set_mem_info=lambda free, total: updates.append((free, total)))

    import kvcached.meminfo_provider as meminfo_provider

    responses = [TimeoutError("worker unavailable"), (123, 456)]

    def query(*args, **kwargs):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(meminfo_provider, "query_mem_info", query)

    manager._refresh_mem_info(force=True)
    manager._refresh_mem_info(force=True)

    assert updates == [(0, 456), (123, 456)]


def test_control_only_initialization_does_not_touch_cuda(monkeypatch):
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
    torch.dtype = type("dtype", (), {})  # type: ignore[attr-defined]

    def unexpected_cuda_call():
        raise AssertionError("control-only initialization touched CUDA")

    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        current_device=unexpected_cuda_call)
    monkeypatch.setitem(sys.modules, "torch", torch)

    manager_module = types.ModuleType("kvcached.kv_cache_manager")
    manager_module.KVCacheManager = type(  # type: ignore[attr-defined]
        "KVCacheManager", (), {})
    monkeypatch.setitem(sys.modules, "kvcached.kv_cache_manager", manager_module)

    listeners = types.ModuleType("kvcached.tp_ipc_util")
    listeners.start_worker_listener_thread = unexpected_cuda_call  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kvcached.tp_ipc_util", listeners)

    calls = []
    vmm_ops = types.ModuleType("kvcached.vmm_ops")
    vmm_ops.create_kv_tensors = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    vmm_ops.init_kvcached = lambda *args, **kwargs: calls.append("init")  # type: ignore[attr-defined]
    vmm_ops.shutdown_kvcached = lambda *args, **kwargs: calls.append("shutdown")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", vmm_ops)

    module_path = (Path(__file__).parents[1] / "kvcached" / "integration" /
                   "vllm" / "interfaces.py")
    spec = importlib.util.spec_from_file_location(
        "kvcached.integration.vllm._test_interfaces", module_path)
    assert spec is not None and spec.loader is not None
    interfaces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interfaces)

    interfaces.init_kvcached(world_size=4, control_only=True)

    assert interfaces._kvcached_initialized is True
    assert interfaces._kvcached_gpu_initialized is False
    assert calls == []


def test_failed_gpu_initialization_does_not_mark_initialized(monkeypatch):
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
    torch.dtype = type("dtype", (), {})  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        current_device=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", torch)

    manager_module = types.ModuleType("kvcached.kv_cache_manager")
    manager_module.KVCacheManager = type(  # type: ignore[attr-defined]
        "KVCacheManager", (), {})
    monkeypatch.setitem(sys.modules, "kvcached.kv_cache_manager", manager_module)

    listeners = types.ModuleType("kvcached.tp_ipc_util")
    listeners.start_worker_listener_thread = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kvcached.tp_ipc_util", listeners)

    def fail_init(*args, **kwargs):
        raise RuntimeError("CUDA initialization failed")

    vmm_ops = types.ModuleType("kvcached.vmm_ops")
    vmm_ops.create_kv_tensors = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    vmm_ops.init_kvcached = fail_init  # type: ignore[attr-defined]
    vmm_ops.shutdown_kvcached = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", vmm_ops)

    module_path = (Path(__file__).parents[1] / "kvcached" / "integration" /
                   "vllm" / "interfaces.py")
    spec = importlib.util.spec_from_file_location(
        "kvcached.integration.vllm._test_failed_interfaces", module_path)
    assert spec is not None and spec.loader is not None
    interfaces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interfaces)

    try:
        interfaces.init_kvcached(device="cuda:0")
    except RuntimeError as exc:
        assert str(exc) == "CUDA initialization failed"
    else:
        raise AssertionError("GPU initialization failure was not propagated")

    assert interfaces._kvcached_initialized is False
    assert interfaces._kvcached_gpu_initialized is False


def test_worker_listener_uses_explicit_initialized_device(monkeypatch):
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
    torch.dtype = type("dtype", (), {})  # type: ignore[attr-defined]

    def unexpected_current_device():
        raise AssertionError("explicit device should not query current_device")

    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        current_device=unexpected_current_device)
    monkeypatch.setitem(sys.modules, "torch", torch)

    manager_module = types.ModuleType("kvcached.kv_cache_manager")
    manager_module.KVCacheManager = type(  # type: ignore[attr-defined]
        "KVCacheManager", (), {})
    monkeypatch.setitem(sys.modules, "kvcached.kv_cache_manager", manager_module)

    listener_calls = []
    listeners = types.ModuleType("kvcached.tp_ipc_util")
    listeners.start_worker_listener_thread = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: listener_calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "kvcached.tp_ipc_util", listeners)

    vmm_ops = types.ModuleType("kvcached.vmm_ops")
    vmm_ops.create_kv_tensors = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    vmm_ops.init_kvcached = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    vmm_ops.shutdown_kvcached = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", vmm_ops)

    module_path = (Path(__file__).parents[1] / "kvcached" / "integration" /
                   "vllm" / "interfaces.py")
    spec = importlib.util.spec_from_file_location(
        "kvcached.integration.vllm._test_device_interfaces", module_path)
    assert spec is not None and spec.loader is not None
    interfaces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interfaces)

    interfaces.init_kvcached(is_worker=True, device="cuda:2")

    assert listener_calls == [((0, 0), {"device_index": 2})]
