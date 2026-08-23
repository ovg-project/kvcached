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


def test_get_world_size_returns_engine_core_recorded_value(
    monkeypatch, vllm_modules
):
    interfaces, _ = vllm_modules
    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    monkeypatch.setattr(interfaces, "_world_size", 4)

    assert interfaces.get_world_size() == 4


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
        parallel_config=types.SimpleNamespace(tensor_parallel_size=4)
    )
    engine_mod.EngineCore(config)

    init_kvcached.assert_called_once_with(
        tp_rank=0,
        world_size=4,
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
        parallel_config=types.SimpleNamespace(tensor_parallel_size=4)
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

    setattr(kvcoord_mod, "KVCacheCoordinator", FakeKVCacheCoordinator)

    assert patches.KVCacheCoordinatorPatch().patch_coordinator(kvcoord_mod)

    with pytest.raises(RuntimeError, match="not initialized"):
        kvcoord_mod.KVCacheCoordinator()
