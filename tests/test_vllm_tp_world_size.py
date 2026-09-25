# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from typing import Any
from unittest import mock

import pytest


@pytest.fixture
def vllm_modules(monkeypatch):
    torch = mock.MagicMock()
    torch.__version__ = "2.6.0"
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

    interfaces: Any = importlib.import_module(
        "kvcached.integration.vllm.interfaces"
    )
    patches: Any = importlib.import_module("kvcached.integration.vllm.patches")
    return interfaces, patches



@pytest.mark.parametrize("version", ["0.9.2", "0.10.2", "0.11.2", "0.12.0", "0.22.1", "0.28.0", None])
def test_hash_granularity_preserves_native_value(vllm_modules, version):
    _, patches = vllm_modules
    pool = types.SimpleNamespace(hash_block_size=4)
    assert patches._get_native_hash_block_size(pool, 16, version) == 4


@pytest.mark.parametrize("version", ["0.9.2", "0.10.2", "0.11.2"])
def test_legacy_pool_without_hash_granularity_uses_allocation_size(vllm_modules, version):
    _, patches = vllm_modules
    assert patches._get_native_hash_block_size(types.SimpleNamespace(), 16, version) == 16


@pytest.mark.parametrize("version", ["0.12.0", "0.22.1", "0.28.0", "unknown", None])
def test_modern_or_unknown_pool_must_expose_hash_granularity(vllm_modules, version):
    _, patches = vllm_modules
    with pytest.raises(RuntimeError, match="BlockPool.hash_block_size is missing"):
        patches._get_native_hash_block_size(types.SimpleNamespace(), 16, version)


@pytest.mark.parametrize("version", ["0.11.2", "0.28.0"])
@pytest.mark.parametrize("value", [None, 0, -4, "4"])
def test_invalid_native_hash_granularity_never_falls_back(vllm_modules, version, value):
    _, patches = vllm_modules
    with pytest.raises(RuntimeError, match="positive integer"):
        patches._get_native_hash_block_size(
            types.SimpleNamespace(hash_block_size=value), 16, version,
        )


@pytest.mark.parametrize("version", ["0.11.2", "0.28.0", None])
def test_missing_native_pool_never_uses_legacy_fallback(vllm_modules, version):
    _, patches = vllm_modules
    with pytest.raises(RuntimeError, match="missing its native BlockPool"):
        patches._get_native_hash_block_size(None, 16, version)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("version", ["0.11.2", "0.12.0", "0.22.1", "0.28.0", None])
def test_coordinator_missing_hash_field_fails_before_initialization(
    monkeypatch, vllm_modules, enabled, version,
):
    interfaces, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: enabled)
    monkeypatch.setattr(patches, "_validate_kv_cache_groups", lambda cfg: None)
    monkeypatch.setattr(patches, "_get_first_attention_group", lambda cfg: types.SimpleNamespace(
        kv_cache_spec=types.SimpleNamespace(block_size=16),
    ))
    monkeypatch.setattr(patches, "_infer_attention_type", lambda cfg: "MHA")
    monkeypatch.setattr(patches, "_get_kv_cache_params", lambda *args, **kwargs: (1024, 2))
    monkeypatch.setattr(patches, "_get_group_size", lambda cfg: 1)
    monkeypatch.setattr(interfaces, "get_world_size", lambda: 1)
    initialize = mock.Mock()
    monkeypatch.setattr(interfaces, "init_kvcached", initialize)
    constructor = mock.Mock(return_value=types.SimpleNamespace(null_block=object()))
    pool_module = types.ModuleType("vllm.v1.core.block_pool")
    setattr(pool_module, "ElasticBlockPool", constructor)
    monkeypatch.setitem(sys.modules, pool_module.__name__, pool_module)
    native_pool = types.SimpleNamespace()

    class Coordinator:
        def __init__(self):
            self.block_pool = native_pool
            # An unrelated coordinator field must not conceal a broken pool.
            self.hash_block_size = 16
            self.enable_caching = True
            self.kv_cache_config = types.SimpleNamespace(num_blocks=8)
            self.single_type_managers = []

    patch = patches.KVCacheCoordinatorPatch()
    patch.detected_version = version
    assert patch.patch_coordinator(types.SimpleNamespace(KVCacheCoordinator=Coordinator))
    if enabled and version != "0.11.2":
        with pytest.raises(RuntimeError, match="BlockPool.hash_block_size is missing"):
            Coordinator()
        initialize.assert_not_called()
        constructor.assert_not_called()
    elif enabled:
        Coordinator()
        assert constructor.call_args.kwargs["hash_block_size"] == 16
        initialize.assert_called_once()
    else:
        assert Coordinator().block_pool is native_pool
        initialize.assert_not_called()
        constructor.assert_not_called()


def test_get_world_size_returns_engine_core_recorded_value(
    monkeypatch, vllm_modules
):
    interfaces, _ = vllm_modules
    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    monkeypatch.setattr(interfaces, "_world_size", 4)

    assert interfaces.get_world_size() == 4


@pytest.mark.parametrize("device", [None, "cuda", "cpu:0"])
def test_model_runner_rejects_invalid_device_before_initializing(
    monkeypatch, vllm_modules, device
):
    interfaces, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    torch = sys.modules["torch"]
    torch.device.side_effect = lambda value: types.SimpleNamespace(
        type=value.split(":")[0],
        index=int(value.split(":")[1]) if ":" in value else None,
    )
    init = mock.Mock()
    monkeypatch.setattr(interfaces, "init_kvcached", init)

    class Runner:
        def __init__(self):
            if device is not None:
                self.device = device

    assert patches.GPUModelRunnerPatch().patch_model_runner_init(Runner)
    with pytest.raises((AttributeError, ValueError)):
        Runner()
    init.assert_not_called()
    torch.cuda.current_device.assert_not_called()


def test_model_runner_passes_initialized_device(monkeypatch, vllm_modules):
    interfaces, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(patches, "_should_enable_async_sched", lambda cfg: False)
    torch = sys.modules["torch"]
    torch.device.return_value = types.SimpleNamespace(type="cuda", index=3)
    init = mock.Mock()
    monkeypatch.setattr(interfaces, "init_kvcached", init)

    class Runner:
        def __init__(self):
            self.device = "cuda:3"
            self.vllm_config = object()

    assert patches.GPUModelRunnerPatch().patch_model_runner_init(Runner)
    Runner()
    init.assert_called_once()
    assert init.call_args.kwargs["device"] == "cuda:3"
    torch.cuda.current_device.assert_not_called()


def test_get_world_size_rejects_uninitialized_state(monkeypatch, vllm_modules):
    interfaces, _ = vllm_modules
    monkeypatch.setattr(interfaces, "_kvcached_initialized", False)

    with pytest.raises(RuntimeError, match="kvcached is not initialized"):
        interfaces.get_world_size()


def test_engine_core_records_tp_world_size_before_original_init(
    monkeypatch, vllm_modules
):
    interfaces, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(patches, "_should_enable_async_sched", lambda cfg: True)
    init_kvcached = mock.Mock()
    monkeypatch.setattr(interfaces, "init_kvcached", init_kvcached)

    def assert_kvcached_initialized_first(*args, **kwargs):
        init_kvcached.assert_called_once()

    original_init = mock.Mock(side_effect=assert_kvcached_initialized_first)
    engine_mod = types.ModuleType("mock_engine_mod")

    class FakeEngineCore:
        __init__ = original_init

    setattr(engine_mod, "EngineCore", FakeEngineCore)
    assert patches.EngineCorePatch().patch_engine_init(engine_mod)

    config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=4, pipeline_parallel_size=1
        )
    )
    engine_mod.EngineCore(config)

    init_kvcached.assert_called_once_with(
        tp_rank=0,
        world_size=4,
        pp_rank=0,
        is_worker=False,
        async_sched=True,
    )
    original_init.assert_called_once()


def test_engine_core_propagates_kvcached_initialization_failure(
    monkeypatch, vllm_modules
):
    interfaces, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(patches, "_should_enable_async_sched", lambda cfg: False)
    init_error = RuntimeError("failed to record TP state")
    monkeypatch.setattr(
        interfaces, "init_kvcached", mock.Mock(side_effect=init_error)
    )

    original_init = mock.Mock(return_value=None)
    engine_mod = types.ModuleType("mock_engine_mod")

    class FakeEngineCore:
        __init__ = original_init

    setattr(engine_mod, "EngineCore", FakeEngineCore)
    assert patches.EngineCorePatch().patch_engine_init(engine_mod)

    config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=4, pipeline_parallel_size=1
        )
    )
    with pytest.raises(RuntimeError, match="failed to record TP state"):
        engine_mod.EngineCore(config)

    original_init.assert_not_called()


class FakeElasticBlockPool:
    def __init__(self, *args, **kwargs):
        self.null_block = object()
        self.kwargs = kwargs


def test_coordinator_uses_recorded_world_size(monkeypatch, vllm_modules):
    interfaces, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(patches, "_validate_kv_cache_groups", lambda cfg: None)
    monkeypatch.setattr(
        patches,
        "_get_first_attention_group",
        lambda cfg: types.SimpleNamespace(
            kv_cache_spec=types.SimpleNamespace(block_size=64)
        ),
    )
    monkeypatch.setattr(patches, "_infer_attention_type", lambda cfg: "MHA")
    monkeypatch.setattr(
        patches, "_get_kv_cache_params", lambda *args, **kwargs: (1024, 2)
    )
    monkeypatch.setattr(patches, "_get_group_size", lambda cfg: 1)
    monkeypatch.setattr(patches, "_get_max_cached_blocks", lambda block_size: 0)
    monkeypatch.setattr(patches, "_should_enable_async_sched", lambda cfg: False)
    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    monkeypatch.setattr(interfaces, "get_world_size", lambda: 4)
    init_kvcached = mock.Mock()
    monkeypatch.setattr(interfaces, "init_kvcached", init_kvcached)

    fake_block_pool_mod = types.ModuleType("vllm.v1.core.block_pool")
    setattr(fake_block_pool_mod, "ElasticBlockPool", FakeElasticBlockPool)
    monkeypatch.setitem(
        sys.modules, "vllm.v1.core.block_pool", fake_block_pool_mod
    )

    kvcoord_mod = types.ModuleType("mock_kvcoord_mod")

    class FakeKVCacheCoordinator:
        def __init__(self, *args, **kwargs):
            self.enable_caching = False
            self.kv_cache_config = types.SimpleNamespace(num_blocks=8)
            self.single_type_managers = [types.SimpleNamespace()]
            self.block_pool = types.SimpleNamespace(hash_block_size=16)

    setattr(kvcoord_mod, "KVCacheCoordinator", FakeKVCacheCoordinator)

    assert patches.KVCacheCoordinatorPatch().patch_coordinator(kvcoord_mod)
    coordinator = kvcoord_mod.KVCacheCoordinator()

    assert init_kvcached.call_args.kwargs["world_size"] == 4
    assert isinstance(getattr(coordinator, "block_pool"), FakeElasticBlockPool)
    assert coordinator.block_pool.kwargs["hash_block_size"] == 16


def test_coordinator_propagates_uninitialized_world_size(
    monkeypatch, vllm_modules
):
    """An uninitialized kvcached must abort startup, not warn and continue.

    ``get_world_size()`` raises when ``init_kvcached()`` never ran, and the
    coordinator patch used to swallow that into a warning — leaving the engine
    running with a half-applied patch and no signal beyond one log line.
    """
    interfaces, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(patches, "_validate_kv_cache_groups", lambda cfg: None)
    monkeypatch.setattr(
        patches,
        "_get_first_attention_group",
        lambda cfg: types.SimpleNamespace(
            kv_cache_spec=types.SimpleNamespace(block_size=16)
        ),
    )
    monkeypatch.setattr(patches, "_infer_attention_type", lambda cfg: "MHA")
    monkeypatch.setattr(
        patches, "_get_kv_cache_params", lambda *args, **kwargs: (1024, 2)
    )
    monkeypatch.setattr(patches, "_get_group_size", lambda cfg: 1)
    monkeypatch.setattr(patches, "_get_max_cached_blocks", lambda block_size: 0)
    monkeypatch.setattr(patches, "_should_enable_async_sched", lambda cfg: False)
    # EngineCore never recorded a world size, so get_world_size() raises.
    monkeypatch.setattr(interfaces, "_kvcached_initialized", False)

    fake_block_pool_mod = types.ModuleType("vllm.v1.core.block_pool")
    setattr(fake_block_pool_mod, "ElasticBlockPool", FakeElasticBlockPool)
    monkeypatch.setitem(
        sys.modules, "vllm.v1.core.block_pool", fake_block_pool_mod
    )

    kvcoord_mod = types.ModuleType("mock_kvcoord_mod")

    class FakeKVCacheCoordinator:
        def __init__(self, *args, **kwargs):
            self.enable_caching = False
            self.kv_cache_config = types.SimpleNamespace(num_blocks=8)
            self.single_type_managers = [types.SimpleNamespace()]
            self.block_pool = types.SimpleNamespace(hash_block_size=16)

    setattr(kvcoord_mod, "KVCacheCoordinator", FakeKVCacheCoordinator)

    assert patches.KVCacheCoordinatorPatch().patch_coordinator(kvcoord_mod)

    with pytest.raises(RuntimeError, match="not initialized"):
        kvcoord_mod.KVCacheCoordinator()


@pytest.mark.parametrize(
    "version,enabled,use_v2,reject",
    [
        ("0.24.0", True, None, False),
        ("0.29.0", True, True, False),
        ("0.29.0", True, False, True),
        ("0.30.0", True, None, True),
        ("0.29.0", False, None, False),
        ("0.30.0", False, None, False),
    ],
)
def test_engine_core_rejects_partial_integration_before_initialization(
    monkeypatch, vllm_modules, version, enabled, use_v2, reject
):
    interfaces, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: enabled)
    monkeypatch.setattr(patches, "_should_enable_async_sched", lambda cfg: False)
    initialize = mock.Mock()
    monkeypatch.setattr(interfaces, "init_kvcached", initialize)
    original_init = mock.Mock(return_value=None)

    class EngineCore:
        __init__ = original_init

    patch = patches.EngineCorePatch()
    patch.detected_version = version
    assert patch.patch_engine_init(types.SimpleNamespace(EngineCore=EngineCore))
    config = types.SimpleNamespace(parallel_config=types.SimpleNamespace(
        tensor_parallel_size=1, pipeline_parallel_size=1,
    ))
    # Older/disabled routes must not read a property they do not need.
    if use_v2 is not None:
        config.use_v2_model_runner = use_v2
    if reject:
        with pytest.raises(patches.KVCachedConfigError):
            EngineCore(config)
        initialize.assert_not_called()
        original_init.assert_not_called()
    else:
        EngineCore(config)
        assert initialize.call_count == int(enabled)
        original_init.assert_called_once()
