# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.machinery
import sys
import types
from typing import Any
from unittest import mock

import pytest


def _load_patches(monkeypatch):
    torch = mock.MagicMock()
    torch.__version__ = "2.6.0"
    torch.float8_e4m3fn = types.SimpleNamespace(
        itemsize=1, name="float8_e4m3fn"
    )
    torch.float8_e5m2 = types.SimpleNamespace(itemsize=1, name="float8_e5m2")
    torch.uint8 = types.SimpleNamespace(itemsize=1, name="uint8")
    torch.bfloat16 = types.SimpleNamespace(itemsize=2, name="bfloat16")
    torch.uint64 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)

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
    # Dotted imports can reuse the parent's reference after another test imports it.
    package = importlib.import_module("kvcached.integration.sglang")
    monkeypatch.setattr(package, "interfaces", interfaces, raising=False)


def _manager_cell_size(calls):
    args, kwargs = calls["get_kv_cache_manager"]
    if "cell_size" in kwargs:
        return kwargs["cell_size"]
    return args[2]


def test_interfaces_fixture_replaces_cached_package_attribute(monkeypatch):
    package = importlib.import_module("kvcached.integration.sglang")
    previous = types.ModuleType("kvcached.integration.sglang.interfaces")
    monkeypatch.setattr(package, "interfaces", previous, raising=False)
    monkeypatch.setitem(sys.modules, previous.__name__, previous)

    with monkeypatch.context() as isolated:
        calls: dict[str, Any] = {}
        _install_interfaces(isolated, calls)
        import kvcached.integration.sglang.interfaces as kvi

        assert kvi is sys.modules[previous.__name__]
        kvi.init_kvcached(world_size=1)
        assert calls["init_kvcached"] == {"world_size": 1}

    assert getattr(package, "interfaces") is previous
    assert sys.modules[previous.__name__] is previous


@pytest.mark.parametrize("logical_name", ["float8_e4m3fn", "float8_e5m2", "bfloat16"])
def test_mha_pool_uses_storage_dtype(monkeypatch, logical_name):
    patches, torch = _load_patches(monkeypatch)
    logical_dtype = getattr(torch, logical_name)
    storage_dtype = torch.uint8 if logical_name.startswith("float8") else logical_dtype
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
    assert pool.dtype is logical_dtype
    assert pool.cell_size == 8 * 64 * storage_dtype.itemsize
    physical_bytes = 2 * 144 * 8 * 64 * storage_dtype.itemsize
    assert pool.get_kv_size_bytes_phy() == (physical_bytes, physical_bytes)
    assert _manager_cell_size(calls) == pool.cell_size


@pytest.mark.parametrize(
    "logical_name, storage_name, use_nsa, expected_dim",
    [
        ("float8_e4m3fn", "uint8", False, 48),
        ("float8_e5m2", "uint8", False, 48),
        ("bfloat16", "bfloat16", False, 48),
        ("float8_e4m3fn", "uint8", True, 64),
        # A physical FP8 dtype alone must not select the legacy NSA layout.
        ("bfloat16", "float8_e4m3fn", True, 48),
    ],
)
def test_mla_pool_uses_storage_dtype(
    monkeypatch, logical_name, storage_name, use_nsa, expected_dim
):
    patches, torch = _load_patches(monkeypatch)
    logical_dtype = getattr(torch, logical_name)
    storage_dtype = getattr(torch, storage_name)
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
        use_nsa=use_nsa,
        override_kv_cache_dim=64,
    )

    assert calls["alloc_kv_cache"]["dtype"] is storage_dtype
    assert pool.dtype is logical_dtype
    assert calls["alloc_kv_cache"]["kvcache_shape"][-1] == expected_dim
    assert pool.kv_cache_dim == expected_dim
    assert pool.cell_size == expected_dim * storage_dtype.itemsize
    assert pool.get_kv_size_bytes_phy() == 2 * 144 * expected_dim * storage_dtype.itemsize
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
