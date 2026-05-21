# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from importlib.machinery import ModuleSpec
from unittest import mock


_torch_mock = mock.MagicMock()
_torch_mock.__version__ = "2.6.0"
sys.modules.setdefault("torch", _torch_mock)
sys.modules.setdefault("torch.cuda", _torch_mock.cuda)
sys.modules.setdefault("torch.utils", _torch_mock.utils)
sys.modules.setdefault("torch.utils.cpp_extension", _torch_mock.utils.cpp_extension)
sys.modules.setdefault("posix_ipc", mock.MagicMock())
sys.modules.setdefault("kvcached.vmm_ops", mock.MagicMock())

_interfaces_mod = types.ModuleType("kvcached.integration.vllm.interfaces")
_interfaces_mod._world_size = 2
_interfaces_mod.init_kvcached = mock.Mock()
sys.modules["kvcached.integration.vllm.interfaces"] = _interfaces_mod

_parallel_state_mod = types.ModuleType("vllm.distributed.parallel_state")
_parallel_state_mod.get_tensor_model_parallel_world_size = lambda: 1
_vllm_mod = sys.modules.setdefault("vllm", types.ModuleType("vllm"))
_vllm_mod.__spec__ = ModuleSpec("vllm", loader=None)
_vllm_distributed_mod = sys.modules.setdefault(
    "vllm.distributed", types.ModuleType("vllm.distributed")
)
_vllm_distributed_mod.__spec__ = ModuleSpec("vllm.distributed", loader=None)
_parallel_state_mod.__spec__ = ModuleSpec("vllm.distributed.parallel_state", loader=None)
sys.modules["vllm.distributed.parallel_state"] = _parallel_state_mod

import pytest

from kvcached.integration.vllm import patches


class FakeElasticBlockPool:
    def __init__(self, *args, **kwargs):
        self.null_block = object()


def test_kv_cache_coordinator_reuses_enginecore_world_size(monkeypatch):
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
    sys.modules["vllm.v1.core.block_pool"] = fake_block_pool_mod

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

    _interfaces_mod.init_kvcached.assert_called_once()
    assert _interfaces_mod.init_kvcached.call_args.kwargs["world_size"] == 2
    assert isinstance(coordinator.block_pool, FakeElasticBlockPool)
