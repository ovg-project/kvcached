# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.machinery
import sys
import types
from typing import Any
from unittest import mock


def _load_patches(monkeypatch):
    torch = mock.MagicMock()
    torch.__version__ = "2.6.0"
    torch.float8_e4m3fn = types.SimpleNamespace(
        itemsize=1, name="float8_e4m3fn"
    )
    torch.uint64 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())

    for name in ("sglang", "sglang.srt"):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        monkeypatch.setitem(sys.modules, name, module)

    distributed = types.ModuleType("sglang.srt.distributed")
    setattr(distributed, "get_tensor_model_parallel_rank", lambda: 0)
    setattr(distributed, "get_tensor_model_parallel_world_size", lambda: 1)
    setattr(distributed, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed)

    from kvcached.integration.sglang import patches

    importlib.reload(patches)
    return patches, torch


def _install_interfaces(monkeypatch, calls):
    interfaces: Any = types.ModuleType("kvcached.integration.sglang.interfaces")
    interfaces.init_kvcached = lambda **kwargs: calls.setdefault(
        "init_kvcached", kwargs
    )

    def alloc_kv_cache(**kwargs):
        calls["alloc_kv_cache"] = kwargs
        buffers = [mock.Mock(data_ptr=lambda: 1), mock.Mock(data_ptr=lambda: 2)]
        if kwargs["attention_type"] == "MLA":
            return buffers
        return buffers, [mock.Mock(data_ptr=lambda: 3), mock.Mock(data_ptr=lambda: 4)]

    interfaces.alloc_kv_cache = alloc_kv_cache

    def get_kv_cache_manager(*args, **kwargs):
        calls["get_kv_cache_manager"] = (args, kwargs)
        return mock.Mock()

    interfaces.get_kv_cache_manager = get_kv_cache_manager
    monkeypatch.setitem(
        sys.modules, "kvcached.integration.sglang.interfaces", interfaces
    )


def _manager_cell_size(calls):
    args, kwargs = calls["get_kv_cache_manager"]
    if "cell_size" in kwargs:
        return kwargs["cell_size"]
    return args[2]


def test_mha_pool_uses_storage_dtype(monkeypatch):
    patches, _torch = _load_patches(monkeypatch)
    logical_dtype = types.SimpleNamespace(itemsize=2)
    storage_dtype = types.SimpleNamespace(itemsize=1)
    calls: dict[str, Any] = {}
    _install_interfaces(monkeypatch, calls)

    class MHATokenToKVPool:
        def __init__(
            self,
            size,
            page_size,
            dtype,
            head_num,
            head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer=None,
            end_layer=None,
        ):
            self.size = size
            self.page_size = page_size
            self.dtype = dtype
            self.store_dtype = storage_dtype
            self.head_num = head_num
            self.head_dim = head_dim
            self.layer_num = layer_num
            self.device = device
            getattr(self, "_create_buffers")()

        def get_kv_size_bytes(self):
            return 0, 0

    mem_pool_mod: Any = types.SimpleNamespace(MHATokenToKVPool=MHATokenToKVPool)
    assert patches.ElasticMemoryPoolPatch().inject_elastic_mem_pool(mem_pool_mod)

    pool = mem_pool_mod.ElasticMHATokenToKVPool(
        128, 16, logical_dtype, 8, 64, 2, "cuda:0", False
    )

    assert calls["alloc_kv_cache"]["dtype"] is storage_dtype
    assert pool.cell_size == 8 * 64 * storage_dtype.itemsize
    assert pool.get_kv_size_bytes_phy() == (2 * 144 * 8 * 64, 2 * 144 * 8 * 64)
    assert _manager_cell_size(calls) == pool.cell_size


def test_mla_pool_uses_storage_dtype(monkeypatch):
    patches, torch = _load_patches(monkeypatch)
    logical_dtype = types.SimpleNamespace(itemsize=2)
    storage_dtype = torch.float8_e4m3fn
    calls: dict[str, Any] = {}
    _install_interfaces(monkeypatch, calls)

    class KVCache:
        def __init__(
            self,
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        ):
            self.size = size
            self.page_size = page_size
            self.dtype = dtype
            self.store_dtype = storage_dtype
            self.layer_num = layer_num
            self.device = device

    class MLATokenToKVPool:
        def get_kv_size_bytes(self):
            return 0

    mem_pool_mod: Any = types.SimpleNamespace(
        KVCache=KVCache,
        MLATokenToKVPool=MLATokenToKVPool,
    )
    assert patches.ElasticMLAMemoryPoolPatch().inject_elastic_mla_mem_pool(
        mem_pool_mod
    )

    pool = mem_pool_mod.ElasticMLATokenToKVPool(
        128,
        16,
        logical_dtype,
        32,
        16,
        2,
        "cuda:0",
        False,
        use_nsa=True,
        override_kv_cache_dim=64,
    )

    assert calls["alloc_kv_cache"]["dtype"] is storage_dtype
    assert calls["alloc_kv_cache"]["kvcache_shape"][-1] == 64
    assert pool.cell_size == 64 * storage_dtype.itemsize
    assert pool.get_kv_size_bytes_phy() == 2 * 144 * 64
    assert _manager_cell_size(calls) == pool.cell_size
    assert torch.tensor.called


def test_mha_pool_falls_back_to_logical_dtype(monkeypatch):
    patches, _torch = _load_patches(monkeypatch)
    logical_dtype = types.SimpleNamespace(itemsize=2)
    calls: dict[str, Any] = {}
    _install_interfaces(monkeypatch, calls)

    class MHATokenToKVPool:
        def __init__(
            self,
            size,
            page_size,
            dtype,
            head_num,
            head_dim,
            layer_num,
            device,
            enable_memory_saver,
            **kwargs,
        ):
            self.size = size
            self.page_size = page_size
            self.dtype = dtype
            self.head_num = head_num
            self.head_dim = head_dim
            self.layer_num = layer_num
            self.device = device
            getattr(self, "_create_buffers")()

        def get_kv_size_bytes(self):
            return 0, 0

    mem_pool_mod: Any = types.SimpleNamespace(MHATokenToKVPool=MHATokenToKVPool)
    assert patches.ElasticMemoryPoolPatch().inject_elastic_mem_pool(mem_pool_mod)

    pool = mem_pool_mod.ElasticMHATokenToKVPool(
        128, 16, logical_dtype, 8, 64, 2, "cuda:0", False
    )

    assert calls["alloc_kv_cache"]["dtype"] is logical_dtype
    assert pool.cell_size == 8 * 64 * logical_dtype.itemsize


def test_mla_pool_falls_back_to_logical_dtype(monkeypatch):
    patches, _torch = _load_patches(monkeypatch)
    logical_dtype = types.SimpleNamespace(itemsize=2)
    calls: dict[str, Any] = {}
    _install_interfaces(monkeypatch, calls)

    class KVCache:
        def __init__(
            self,
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        ):
            self.size = size
            self.page_size = page_size
            self.dtype = dtype
            self.layer_num = layer_num
            self.device = device

    class MLATokenToKVPool:
        def get_kv_size_bytes(self):
            return 0

    mem_pool_mod: Any = types.SimpleNamespace(
        KVCache=KVCache,
        MLATokenToKVPool=MLATokenToKVPool,
    )
    assert patches.ElasticMLAMemoryPoolPatch().inject_elastic_mla_mem_pool(
        mem_pool_mod
    )

    pool = mem_pool_mod.ElasticMLATokenToKVPool(
        128, 16, logical_dtype, 32, 16, 2, "cuda:0", False
    )

    assert calls["alloc_kv_cache"]["dtype"] is logical_dtype
    assert pool.cell_size == 48 * logical_dtype.itemsize
