# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from importlib.machinery import ModuleSpec
from unittest import mock

import pytest


def _load_patches(monkeypatch):
    torch_mock = mock.MagicMock()
    torch_mock.__version__ = "2.6.0"
    monkeypatch.setitem(sys.modules, "torch", torch_mock)
    monkeypatch.setitem(sys.modules, "torch.cuda", torch_mock.cuda)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_mock.utils)
    monkeypatch.setitem(
        sys.modules,
        "torch.utils.cpp_extension",
        torch_mock.utils.cpp_extension,
    )
    monkeypatch.setitem(sys.modules, "posix_ipc", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())

    interfaces_mod = types.ModuleType("kvcached.integration.vllm.interfaces")
    interfaces_mod.get_world_size = mock.Mock(return_value=2)
    interfaces_mod.init_kvcached = mock.Mock()
    monkeypatch.setitem(
        sys.modules,
        "kvcached.integration.vllm.interfaces",
        interfaces_mod,
    )

    parallel_state_mod = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state_mod.get_tensor_model_parallel_world_size = lambda: 1
    parallel_state_mod.__spec__ = ModuleSpec(
        "vllm.distributed.parallel_state",
        loader=None,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.parallel_state",
        parallel_state_mod,
    )

    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__spec__ = ModuleSpec("vllm", loader=None)
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)

    vllm_distributed_mod = types.ModuleType("vllm.distributed")
    vllm_distributed_mod.__spec__ = ModuleSpec("vllm.distributed", loader=None)
    monkeypatch.setitem(sys.modules, "vllm.distributed", vllm_distributed_mod)

    patches = importlib.import_module("kvcached.integration.vllm.patches")
    return importlib.reload(patches), interfaces_mod


class FakeElasticBlockPool:
    def __init__(self, *args, **kwargs):
        self.null_block = object()


def test_kv_cache_coordinator_reuses_enginecore_world_size(monkeypatch):
    patches, interfaces_mod = _load_patches(monkeypatch)

    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(patches, "_validate_kv_cache_groups", lambda cfg: None)
    monkeypatch.setattr(
        patches,
        "_get_first_attention_group",
        lambda cfg: types.SimpleNamespace(kv_cache_spec=types.SimpleNamespace(block_size=16)),
    )
    monkeypatch.setattr(patches, "_infer_attention_type", lambda cfg: "MHA")
    monkeypatch.setattr(patches, "_get_kv_cache_params", lambda *args, **kwargs: (1024, 2))
    monkeypatch.setattr(patches, "_get_group_size", lambda cfg: 1)
    monkeypatch.setattr(patches, "_get_max_cached_blocks", lambda block_size: 0)
    monkeypatch.setattr(patches, "_should_enable_async_sched", lambda cfg: False)

    fake_block_pool_mod = types.ModuleType("vllm.v1.core.block_pool")
    fake_block_pool_mod.ElasticBlockPool = FakeElasticBlockPool
    monkeypatch.setitem(sys.modules, "vllm.v1.core.block_pool", fake_block_pool_mod)

    kvcoord_mod = types.ModuleType("mock_kvcoord_mod")

    class FakeKVCacheCoordinator:
        def __init__(self, *args, **kwargs):
            self.enable_caching = False
            self.kv_cache_config = types.SimpleNamespace(num_blocks=8)
            self.single_type_managers = [types.SimpleNamespace()]

    kvcoord_mod.KVCacheCoordinator = FakeKVCacheCoordinator

    patch = patches.KVCacheCoordinatorPatch()
    assert patch.patch_coordinator(kvcoord_mod)

    coordinator = kvcoord_mod.KVCacheCoordinator()

    interfaces_mod.get_world_size.assert_called_once_with()
    interfaces_mod.init_kvcached.assert_called_once()
    assert interfaces_mod.init_kvcached.call_args.kwargs["world_size"] == 2
    assert isinstance(coordinator.block_pool, FakeElasticBlockPool)
