# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import contextlib
import functools
import importlib
import sys
import types
from typing import Any
from unittest import mock

import pytest


@pytest.fixture
def patches(monkeypatch):
    torch = mock.MagicMock()
    torch.__version__ = "2.6.0"
    inference_mode_enabled = False

    def inference_mode():
        def decorate(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                nonlocal inference_mode_enabled
                previous = inference_mode_enabled
                inference_mode_enabled = True
                try:
                    return func(*args, **kwargs)
                finally:
                    inference_mode_enabled = previous

            return wrapped

        return decorate

    torch.inference_mode = inference_mode
    torch.is_grad_enabled.side_effect = lambda: not inference_mode_enabled
    torch.is_inference_mode_enabled.side_effect = lambda: inference_mode_enabled
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", torch.cuda)
    monkeypatch.setitem(sys.modules, "torch.utils", torch.utils)
    monkeypatch.setitem(
        sys.modules, "torch.utils.cpp_extension", torch.utils.cpp_extension
    )
    monkeypatch.setitem(sys.modules, "posix_ipc", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())
    monkeypatch.delitem(
        sys.modules, "kvcached.integration.vllm.interfaces", raising=False
    )
    monkeypatch.delitem(
        sys.modules, "kvcached.integration.vllm.patches", raising=False
    )
    importlib.import_module("kvcached.integration.vllm.interfaces")
    return importlib.import_module("kvcached.integration.vllm.patches")


def _worker_config(*, utilization=0.9, explicit_budget=None):
    return types.SimpleNamespace(
        gpu_memory_utilization=utilization,
        kv_cache_memory_bytes=explicit_budget,
    )


def _patch_worker(patches, monkeypatch, worker_cls, *, enabled=True) -> Any:
    monkeypatch.setattr(patches, "enable_kvcached", lambda: enabled)
    module = types.ModuleType("mock_gpu_worker")
    setattr(module, "Worker", worker_cls)
    patch = patches.GPUWorkerPatch()
    assert patch.patch_worker_init_device(module)
    assert patch.patch_worker_determine_available_memory(module)
    return getattr(module, "Worker")


def _install_memory_profiling(
    monkeypatch, *, torch_peak_increase, before_torch_peak=0
):
    calls = []

    @contextlib.contextmanager
    def memory_profiling(init_snapshot, *, weights_memory):
        calls.append((init_snapshot, weights_memory))
        yield types.SimpleNamespace(
            weights_memory=weights_memory,
            torch_peak_increase=torch_peak_increase,
            non_torch_increase=10_000,
            before_profile=types.SimpleNamespace(
                torch_peak=before_torch_peak
            ),
        )

    module = types.ModuleType("vllm.utils.mem_utils")
    setattr(module, "memory_profiling", memory_profiling)
    monkeypatch.setitem(sys.modules, "vllm.utils.mem_utils", module)
    return calls


def test_virtual_capacity_uses_total_memory_and_utilization(patches):
    snapshot = types.SimpleNamespace(total_memory=1001)
    config = _worker_config(utilization=0.9)

    assert patches._get_virtual_kv_capacity_bytes(snapshot, config) == 901


def test_request_memory_uses_virtual_budget_and_warns(monkeypatch, patches):
    original_request_memory = mock.Mock(side_effect=AssertionError("must not run"))

    class Worker:
        def determine_available_memory(self):
            return 1

    module = types.ModuleType("mock_gpu_worker")
    setattr(module, "Worker", Worker)
    setattr(module, "request_memory", original_request_memory)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)

    patch = patches.GPUWorkerPatch()
    warning = mock.Mock()
    patch.logger.warning = warning
    assert patch.patch_worker_init_device(module)

    snapshot = types.SimpleNamespace(total_memory=1000, free_memory=100)
    config = _worker_config(utilization=0.9)
    assert getattr(module, "request_memory")(snapshot, config) == 900
    assert "whole-device startup memory guard" in warning.call_args.args[0]
    original_request_memory.assert_not_called()


def test_request_memory_keeps_original_behavior_when_disabled(monkeypatch, patches):
    original_request_memory = mock.Mock(return_value=123)

    class Worker:
        def determine_available_memory(self):
            return 1

    module = types.ModuleType("mock_gpu_worker")
    setattr(module, "Worker", Worker)
    setattr(module, "request_memory", original_request_memory)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: False)

    patch = patches.GPUWorkerPatch()
    assert patch.patch_worker_init_device(module)

    snapshot = types.SimpleNamespace(total_memory=1000, free_memory=100)
    config = _worker_config(utilization=0.9)
    assert getattr(module, "request_memory")(snapshot, config) == 123
    original_request_memory.assert_called_once_with(snapshot, config)


def test_init_device_does_not_hide_unrelated_value_error(monkeypatch, patches):
    class Worker:
        def init_device(self):
            raise ValueError("invalid model configuration")

        def determine_available_memory(self):
            return 1

    PatchedWorker = _patch_worker(patches, monkeypatch, Worker)

    with pytest.raises(ValueError, match="invalid model configuration"):
        PatchedWorker().init_device()


def test_startup_memory_guard_preserves_requested_memory(monkeypatch, patches):
    set_seed = mock.Mock()
    init_dist = mock.Mock()
    init_workspace = mock.Mock()
    report_usage = mock.Mock()

    class MemorySnapshot:
        def __init__(self, device):
            self.device = device
            self.total_memory = 1000
            self.free_memory = 100

    class GPUModelRunner:
        def __init__(self, config, device):
            self.config = config
            self.device = device

    modules: dict[str, dict[str, Any]] = {
        "vllm.utils.mem_utils": {"MemorySnapshot": MemorySnapshot},
        "vllm.utils.torch_utils": {"set_random_seed": set_seed},
        "vllm.v1.utils": {"report_usage_stats": report_usage},
        "vllm.v1.worker.gpu_model_runner": {"GPUModelRunner": GPUModelRunner},
        "vllm.v1.worker.gpu_worker": {
            "init_worker_distributed_environment": init_dist,
        },
        "vllm.v1.worker.workspace": {"init_workspace_manager": init_workspace},
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    class Worker:
        def __init__(self):
            self.device = "cuda:0"
            self.cache_config = _worker_config(utilization=0.9)
            self.vllm_config = types.SimpleNamespace(
                parallel_config=types.SimpleNamespace(enable_dbo=False)
            )
            self.model_config = types.SimpleNamespace(seed=7)
            self.rank = 0
            self.local_rank = 0
            self.distributed_init_method = "mock://"

        def init_device(self):
            self.init_snapshot = MemorySnapshot(device=self.device)
            self.requested_memory = 777
            raise ValueError(
                "Free memory on device cuda:0 (0.1/1.0 GiB) on startup "
                "is less than desired GPU memory utilization"
            )

        def determine_available_memory(self):
            return 1

    PatchedWorker = _patch_worker(patches, monkeypatch, Worker)
    worker = PatchedWorker()
    worker.init_device()

    assert getattr(worker, "requested_memory") == 777
    assert getattr(worker, "init_snapshot").free_memory == 100
    assert isinstance(getattr(worker, "model_runner"), GPUModelRunner)
    init_dist.assert_called_once()
    set_seed.assert_called_once_with(7)
    init_workspace.assert_called_once_with("cuda:0", 1)
    report_usage.assert_called_once_with(worker.vllm_config)


def test_legacy_init_device_persists_virtual_budget(monkeypatch, patches):
    torch = sys.modules["torch"]
    torch.cuda.get_device_properties.return_value = types.SimpleNamespace(
        total_memory=1000
    )

    class Worker:
        def __init__(self):
            self.device = "cuda:0"
            self.cache_config = _worker_config(utilization=0.75)

        def init_device(self):
            self.init_gpu_memory = 100

        def determine_available_memory(self):
            return 1

    PatchedWorker = _patch_worker(patches, monkeypatch, Worker)
    worker = PatchedWorker()
    worker.init_device()

    assert worker.requested_memory == 750
    assert not hasattr(worker, "init_snapshot")


def test_determine_available_memory_injects_automatic_virtual_budget(
    monkeypatch, patches
):
    torch = sys.modules["torch"]
    profile_modes = []

    def profile_run():
        profile_modes.append(
            (
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
            )
        )

    profile_run = mock.Mock(side_effect=profile_run)
    profile_calls = _install_memory_profiling(
        monkeypatch, torch_peak_increase=50
    )

    class Worker:
        def __init__(self):
            self.init_snapshot = types.SimpleNamespace(
                total_memory=4096, free_memory=200
            )
            self.cache_config = _worker_config(utilization=0.8)
            self.requested_memory = 800
            self.model_runner = types.SimpleNamespace(
                model_memory_usage=200, profile_run=profile_run
            )

        def init_device(self):
            return None

        def determine_available_memory(self):
            raise AssertionError("whole-device profiling must not run")

    PatchedWorker = _patch_worker(patches, monkeypatch, Worker)
    worker = PatchedWorker()
    capacity = mock.Mock(side_effect=AssertionError("must reuse requested_memory"))
    monkeypatch.setattr(patches, "_get_virtual_kv_capacity_bytes", capacity)

    assert worker.determine_available_memory() == 550
    assert profile_calls == [(worker.init_snapshot, 200)]
    profile_run.assert_called_once_with()
    assert worker.cache_config.kv_cache_memory_bytes is None
    assert getattr(worker, "available_kv_cache_memory_bytes") == 550
    assert worker.non_torch_memory == 0
    assert profile_modes == [(False, True)]
    capacity.assert_not_called()


def test_determine_available_memory_records_but_ignores_cudagraph_estimate(
    monkeypatch, patches
):
    torch = sys.modules["torch"]
    profile_modes = []

    def record_profile_mode(result=None):
        profile_modes.append(
            (
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
            )
        )
        return result

    profile_run = mock.Mock(side_effect=lambda: record_profile_mode())
    profile_cudagraph = mock.Mock(side_effect=lambda: record_profile_mode(30))
    _install_memory_profiling(
        monkeypatch,
        torch_peak_increase=999,
        before_torch_peak=10,
    )
    torch = sys.modules["torch"]
    torch.accelerator.memory_stats.return_value = {
        "allocated_bytes.all.peak": 80
    }
    vllm = types.ModuleType("vllm")
    setattr(
        vllm,
        "envs",
        types.SimpleNamespace(
            VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=True
        ),
    )
    platforms = types.ModuleType("vllm.platforms")
    setattr(
        platforms,
        "current_platform",
        types.SimpleNamespace(
            is_cuda=lambda: True,
            is_rocm=lambda: False,
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    class Worker:
        def __init__(self):
            self.device = "cuda:0"
            self.init_snapshot = types.SimpleNamespace(total_memory=1000)
            self.cache_config = _worker_config(utilization=0.8)
            self.requested_memory = 800
            self.model_config = types.SimpleNamespace(enforce_eager=False)
            self.vllm_config = types.SimpleNamespace(
                compilation_config=types.SimpleNamespace()
            )
            self.model_runner = types.SimpleNamespace(
                model_memory_usage=200,
                profile_run=profile_run,
                profile_cudagraph_memory=profile_cudagraph,
            )

        def init_device(self):
            return None

        def determine_available_memory(self):
            raise AssertionError("whole-device profiling must not run")

    PatchedWorker = _patch_worker(patches, monkeypatch, Worker)
    worker = PatchedWorker()

    assert worker.determine_available_memory() == 530
    assert worker.non_torch_memory == 0
    assert worker.peak_activation_memory == 70
    assert worker.cudagraph_memory_estimate == 30
    profile_run.assert_called_once_with()
    profile_cudagraph.assert_called_once_with()
    assert profile_modes == [(False, True), (False, True)]


def test_cudagraph_profile_respects_none_mode(patches):
    worker = types.SimpleNamespace(
        model_config=types.SimpleNamespace(enforce_eager=False),
        vllm_config=types.SimpleNamespace(
            compilation_config=types.SimpleNamespace(cudagraph_mode="NONE")
        ),
        model_runner=types.SimpleNamespace(
            profile_cudagraph_memory=mock.Mock()
        ),
    )

    assert patches._should_profile_cudagraph_memory(worker) is False


def test_determine_available_memory_preserves_explicit_user_budget(
    monkeypatch, patches
):
    calls = mock.Mock(return_value=321)

    class Worker:
        def __init__(self):
            self.init_snapshot = types.SimpleNamespace(total_memory=1000)
            self.cache_config = _worker_config(explicit_budget=321)

        def init_device(self):
            return None

        determine_available_memory = calls

    PatchedWorker = _patch_worker(patches, monkeypatch, Worker)
    worker = PatchedWorker()

    assert worker.determine_available_memory() == 321
    assert worker.cache_config.kv_cache_memory_bytes == 321
    calls.assert_called_once()


def test_determine_available_memory_propagates_profile_failure(
    monkeypatch, patches
):
    _install_memory_profiling(monkeypatch, torch_peak_increase=25)

    class Worker:
        def __init__(self):
            self.init_snapshot = types.SimpleNamespace(total_memory=1000)
            self.cache_config = _worker_config(utilization=0.5)
            self.requested_memory = 500
            self.model_runner = types.SimpleNamespace(
                model_memory_usage=100,
                profile_run=mock.Mock(side_effect=RuntimeError("profile run failed")),
            )

        def init_device(self):
            return None

        def determine_available_memory(self):
            raise AssertionError("whole-device profiling must not run")

    PatchedWorker = _patch_worker(patches, monkeypatch, Worker)
    worker = PatchedWorker()

    with pytest.raises(RuntimeError, match="profile run failed"):
        worker.determine_available_memory()
    assert worker.cache_config.kv_cache_memory_bytes is None


def test_legacy_determine_available_memory_runs_profile_without_device_delta(
    monkeypatch, patches
):
    profile_run = mock.Mock()
    torch = sys.modules["torch"]
    torch.cuda.memory_stats.return_value = {
        "allocated_bytes.all.peak": 125
    }

    class Worker:
        def __init__(self):
            self.cache_config = types.SimpleNamespace(gpu_memory_utilization=0.75)
            self.model_runner = types.SimpleNamespace(
                model_memory_usage=100, profile_run=profile_run
            )
            self.requested_memory = 750

        def init_device(self):
            return None

        def determine_available_memory(self):
            raise AssertionError("whole-device profiling must not run")

    PatchedWorker = _patch_worker(patches, monkeypatch, Worker)
    worker = PatchedWorker()

    assert worker.determine_available_memory() == 625
    assert worker.available_kv_cache_memory_bytes == 625
    torch.cuda.empty_cache.assert_called_once_with()
    torch.cuda.reset_peak_memory_stats.assert_called_once_with()
    profile_run.assert_called_once_with()


def test_null_block_reservation_waits_for_physical_capacity(monkeypatch, patches):
    manager_module = importlib.import_module("kvcached.kv_cache_manager")
    log = types.SimpleNamespace(warning=mock.Mock(), info=mock.Mock())
    monkeypatch.setattr(manager_module, "logger", log)
    manager = object.__new__(manager_module.KVCacheManager)
    manager.reserve_null_block = True
    manager.null_block = None
    manager.available_size = mock.Mock(side_effect=[0, 0, 1])
    manager._alloc = mock.Mock(return_value=[0])
    sleep = mock.Mock()
    monkeypatch.setattr(manager_module.time, "sleep", sleep)

    manager._reserve_null_block()

    assert manager.null_block == [0]
    assert sleep.call_count == 2
    manager._alloc.assert_called_once_with(1, _skip_wait=True)
    assert "reason=no_effective_capacity" in log.warning.call_args.args[0]
    assert "Reserved null block after waiting" in log.info.call_args.args[0]


def test_null_block_reservation_retries_allocator_race(monkeypatch, patches):
    manager_module = importlib.import_module("kvcached.kv_cache_manager")
    log = types.SimpleNamespace(warning=mock.Mock(), info=mock.Mock())
    monkeypatch.setattr(manager_module, "logger", log)
    manager = object.__new__(manager_module.KVCacheManager)
    manager.reserve_null_block = True
    manager.null_block = None
    manager.available_size = mock.Mock(return_value=1)
    manager._alloc = mock.Mock(side_effect=[None, [0]])
    sleep = mock.Mock()
    monkeypatch.setattr(manager_module.time, "sleep", sleep)

    manager._reserve_null_block()

    assert manager.null_block == [0]
    assert sleep.call_count == 1
    assert manager._alloc.call_count == 2
    manager._alloc.assert_called_with(1, _skip_wait=True)
    assert (
        "reason=alloc_returned_none_after_capacity_check"
        in log.warning.call_args.args[0]
    )


def test_null_block_wait_log_is_rate_limited(monkeypatch, patches):
    manager_module = importlib.import_module("kvcached.kv_cache_manager")
    log = types.SimpleNamespace(warning=mock.Mock(), info=mock.Mock())
    monkeypatch.setattr(manager_module, "logger", log)
    monkeypatch.setattr(
        manager_module.time,
        "monotonic",
        mock.Mock(side_effect=[100.0, 100.0, 105.0, 111.0, 112.0]),
    )
    monkeypatch.setattr(manager_module.time, "sleep", mock.Mock())

    manager = object.__new__(manager_module.KVCacheManager)
    manager.reserve_null_block = True
    manager.null_block = None
    manager.available_size = mock.Mock(side_effect=[0, 0, 0, 1])
    manager._alloc = mock.Mock(return_value=[0])

    manager._reserve_null_block()

    assert log.warning.call_count == 2
    assert "elapsed=0.0s" in log.warning.call_args_list[0].args[0]
    assert "elapsed=11.0s" in log.warning.call_args_list[1].args[0]


def test_null_block_reservation_keeps_wrong_id_fail_loud(monkeypatch, patches):
    manager_module = importlib.import_module("kvcached.kv_cache_manager")
    manager = object.__new__(manager_module.KVCacheManager)
    manager.reserve_null_block = True
    manager.null_block = None
    manager.available_size = mock.Mock(return_value=1)
    manager._alloc = mock.Mock(return_value=[1])

    with pytest.raises(RuntimeError, match="null block at index 0"):
        manager._reserve_null_block()


def test_kvcached_disabled_keeps_original_memory_paths(monkeypatch, patches):
    init_device = mock.Mock(return_value=None)
    determine = mock.Mock(return_value=123)

    class Worker:
        def __init__(self):
            self.cache_config = _worker_config()

        def init_device(self):
            return init_device()

        def determine_available_memory(self):
            return determine()

    PatchedWorker = _patch_worker(patches, monkeypatch, Worker, enabled=False)
    worker = PatchedWorker()

    assert worker.init_device() is None
    assert worker.determine_available_memory() == 123
    init_device.assert_called_once()
    determine.assert_called_once()
