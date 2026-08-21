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
    vmm_ops.InternalPage = type(  # type: ignore[attr-defined]
        "InternalPage",
        (),
        {"get_num_blocks": staticmethod(lambda page_size, block_size: 4)},
    )
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


def test_control_only_manager_uses_logical_capacity(monkeypatch):
    manager_module = _load_manager_module(monkeypatch)
    observed: Dict[str, Any] = {}

    class FakePageAllocator:
        def __init__(self, *args, **kwargs):
            observed["constructor"] = kwargs

        def set_use_worker_ipc(self, enabled):
            observed["use_worker_ipc"] = enabled

        def set_broadcast_map_callback(self, callback):
            observed["map_callback"] = callback

        def set_broadcast_unmap_callback(self, callback):
            observed["unmap_callback"] = callback

        def get_num_free_pages(self):
            return 3

        def get_num_reserved_pages(self):
            raise AssertionError("control-only capacity queried CUDA state")

        def get_avail_physical_pages(self):
            raise AssertionError("control-only capacity queried CUDA state")

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(manager_module, "PageAllocator", FakePageAllocator)
    monkeypatch.setattr(manager_module.threading, "Thread", FakeThread)

    manager = manager_module.KVCacheManager(
        num_blocks=4,
        block_size=1,
        cell_size=1,
        num_layers=1,
        world_size=2,
        worker_physical_admission=True,
    )
    manager.num_avail_blocks = 2
    manager.reserved_blocks = [0]
    manager.in_shrink = False

    assert "worker_physical_admission" not in observed["constructor"]
    assert observed["constructor"]["enable_page_prealloc"] is False
    assert observed["use_worker_ipc"] is False
    assert manager.available_size() == 15


def test_default_manager_clamps_capacity_to_physical_pages(monkeypatch):
    manager_module = _load_manager_module(monkeypatch)
    manager = manager_module.KVCacheManager.__new__(manager_module.KVCacheManager)
    manager.worker_physical_admission = False
    manager._lock = manager_module.NoOpLock()
    manager.num_avail_blocks = 0
    manager.reserved_blocks = []
    manager.in_shrink = False
    manager.page_size = 16
    manager.block_mem_size = 4
    manager.page_allocator = types.SimpleNamespace(
        get_num_free_pages=lambda: 5,
        get_avail_physical_pages=lambda: 2,
        get_num_reserved_pages=lambda: 1,
    )

    assert manager.available_size() == 12


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


def test_worker_listener_restores_device_before_serving(monkeypatch):
    selected = []
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        set_device=lambda device: selected.append(device))
    monkeypatch.setitem(sys.modules, "torch", torch)

    utils = types.ModuleType("kvcached.utils")
    utils.DEFAULT_IPC_NAME = "test"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kvcached.utils", utils)

    vmm_ops = types.ModuleType("kvcached.vmm_ops")
    vmm_ops.kv_tensors_created = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    vmm_ops.map_to_kv_tensors = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    vmm_ops.unmap_from_kv_tensors = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", vmm_ops)

    module_path = Path(__file__).parents[1] / "kvcached" / "tp_ipc_util.py"
    spec = importlib.util.spec_from_file_location(
        "kvcached._test_tp_ipc_util", module_path)
    assert spec is not None and spec.loader is not None
    tp_ipc_util = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp_ipc_util)

    tp_ipc_util._set_listener_device(2)

    assert selected == [2]
