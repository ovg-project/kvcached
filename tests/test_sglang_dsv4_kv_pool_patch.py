# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from typing import Any
from unittest import mock

import pytest


class FakeTensor:
    def __init__(self, nbytes: int = 0, contiguous: bool = True):
        self.nbytes = nbytes
        self.contiguous = contiguous
        self.view_shape: Any = None

    def __getitem__(self, _key):
        return self

    def view(self, *shape):
        self.view_shape = shape
        return self

    def is_contiguous(self):
        return self.contiguous


def _load_modules(monkeypatch):
    torch_mock = mock.MagicMock()
    torch_mock.__version__ = "2.6.0"
    torch_mock.uint8.itemsize = 1
    torch_mock.cuda.is_available.return_value = True
    torch_mock.cuda.current_device.return_value = 0
    monkeypatch.setitem(sys.modules, "torch", torch_mock)
    monkeypatch.setitem(sys.modules, "torch.cuda", torch_mock.cuda)
    monkeypatch.setitem(sys.modules, "posix_ipc", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())

    from kvcached.integration.sglang import interfaces, patches

    importlib.reload(interfaces)
    importlib.reload(patches)
    return interfaces, patches, torch_mock


def _make_dsv4_module():
    class DeepSeekV4SingleKVPool:
        def __init__(
            self,
            size,
            page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
        ):
            self.size = size
            self.page_size = page_size
            self.dtype = dtype
            self.qk_nope_head_dim = qk_nope_head_dim
            self.qk_rope_head_dim = qk_rope_head_dim
            self.layer_num = layer_num
            self.device = device
            self.enable_memory_saver = enable_memory_saver
            self._create_buffers()

        def get_bytes_per_token(self):
            return 36

        def _create_buffers(self):
            self.kv_buffer = [FakeTensor(101) for _ in range(self.layer_num)]

    class DeepSeekV4TokenToKVPool:
        def __init__(self, *, unified=False):
            self.device = "cuda:0"
            self.qk_nope_head_dim = 16
            self.qk_rope_head_dim = 8
            if unified:
                self.unified_kv_pool: Any = types.SimpleNamespace(
                    kv_buffer=[FakeTensor(303)]
                )
                self.swa_kv_pool = None
                self.c4_kv_pool = None
                return

            self.unified_kv_pool = None
            self.swa_kv_pool = self._make_kv_pool(
                size=1024,
                page_size=16,
                dtype=types.SimpleNamespace(itemsize=1),
                layer_num=2,
                device="cuda:0",
                enable_memory_saver=False,
                global_page_size=16,
            )
            self.c4_kv_pool = self._make_kv_pool(
                size=256,
                page_size=4,
                dtype=types.SimpleNamespace(itemsize=1),
                layer_num=1,
                device="cuda:0",
                enable_memory_saver=False,
                global_page_size=16,
            )

        def _make_kv_pool(
            self,
            *,
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            global_page_size,
            cls=DeepSeekV4SingleKVPool,
        ):
            del global_page_size
            return cls(
                size,
                page_size,
                dtype,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                layer_num,
                device,
                enable_memory_saver,
            )

    return types.SimpleNamespace(
        DeepSeekV4SingleKVPool=DeepSeekV4SingleKVPool,
        DeepSeekV4TokenToKVPool=DeepSeekV4TokenToKVPool,
    )


def test_dsv4_pool_patch_takes_over_only_swa_during_construction(monkeypatch):
    interfaces, patches, _torch_mock = _load_modules(monkeypatch)
    dsv4_module = _make_dsv4_module()
    calls: dict[str, Any] = {}

    monkeypatch.setattr(
        patches.DeepSeekV4KVPoolPatch,
        "initialize_version_info",
        lambda self: True,
    )
    monkeypatch.setattr(patches, "CONTIGUOUS_LAYOUT", False)
    monkeypatch.setattr(patches, "_resolve_sglang_parallel_context", lambda: (0, 4, 1))
    monkeypatch.setattr(
        interfaces,
        "init_kvcached",
        lambda **kwargs: calls.setdefault("init", kwargs),
    )

    managed_buffers = [FakeTensor(201), FakeTensor(203)]
    raw_tensors = [FakeTensor(4096), FakeTensor(4096)]

    def alloc_dsv4_swa_cache(**kwargs):
        calls["alloc"] = kwargs
        return managed_buffers, raw_tensors

    monkeypatch.setattr(interfaces, "alloc_dsv4_swa_cache", alloc_dsv4_swa_cache)
    manager = object()
    monkeypatch.setattr(
        patches,
        "_new_tp_scoped_kvcached_allocator",
        lambda _kvi, **kwargs: calls.setdefault("manager", (manager, kwargs))[0],
    )

    patch = patches.DeepSeekV4KVPoolPatch()
    assert patch.inject_dsv4_swa_pool(dsv4_module)

    pool = dsv4_module.DeepSeekV4TokenToKVPool()
    assert pool.swa_kv_pool.kv_buffer is managed_buffers
    assert pool.swa_kv_pool.kvcached_allocator is manager
    assert pool.swa_kv_pool._kvcached_managed is True
    assert pool.c4_kv_pool.kv_buffer[0].nbytes == 101
    assert not hasattr(pool.c4_kv_pool, "_kvcached_managed")
    assert calls["init"] == {
        "tp_rank": 0,
        "world_size": 4,
        "pp_rank": 1,
        "device": "cuda:0",
        "async_sched": True,
    }
    assert calls["alloc"] == {
        "num_pages": 65,
        "bytes_per_page": 576,
        "num_layers": 2,
        "device": "cuda:0",
        "group_id": 20_000,
    }
    assert calls["manager"][1]["block_size"] == 1
    assert calls["manager"][1]["cell_size"] == 576
    assert calls["manager"][1]["reserve_null_block"] is True


def test_dsv4_pool_patch_leaves_unified_layout_runtime_owned(monkeypatch):
    _interfaces, patches, _torch_mock = _load_modules(monkeypatch)
    dsv4_module = _make_dsv4_module()
    monkeypatch.setattr(
        patches.DeepSeekV4KVPoolPatch,
        "initialize_version_info",
        lambda self: True,
    )

    assert patches.DeepSeekV4KVPoolPatch().inject_dsv4_swa_pool(dsv4_module)
    pool = dsv4_module.DeepSeekV4TokenToKVPool(unified=True)

    assert pool.swa_kv_pool is None
    assert pool.unified_kv_pool.kv_buffer[0].nbytes == 303
    assert not hasattr(pool.unified_kv_pool, "_kvcached_managed")


def test_dsv4_pool_patch_leaves_swa_native_for_compound_layout(monkeypatch):
    _interfaces, patches, _torch_mock = _load_modules(monkeypatch)
    dsv4_module = _make_dsv4_module()
    monkeypatch.setattr(
        patches.DeepSeekV4KVPoolPatch,
        "initialize_version_info",
        lambda self: True,
    )
    monkeypatch.setattr(patches, "CONTIGUOUS_LAYOUT", True)

    assert patches.DeepSeekV4KVPoolPatch().inject_dsv4_swa_pool(dsv4_module)
    pool = dsv4_module.DeepSeekV4TokenToKVPool()

    assert pool.swa_kv_pool.kv_buffer[0].nbytes == 101
    assert not hasattr(pool.swa_kv_pool, "_kvcached_managed")


def test_swa_allocator_dispatches_by_pool_ownership(monkeypatch):
    _interfaces, patches, _torch_mock = _load_modules(monkeypatch)
    allocator_module: Any = types.ModuleType("sglang.srt.mem_cache.allocator")

    class NativeTokenAllocator:
        def __init__(self, *args, **kwargs):
            self.kind = "native-token"

    class NativePagedAllocator:
        def __init__(self, *args, **kwargs):
            self.kind = "native-paged"

    class ElasticTokenAllocator:
        def __init__(self, *args, **kwargs):
            self.kind = "elastic-token"

    class ElasticPagedAllocator:
        def __init__(self, *args, **kwargs):
            self.kind = "elastic-paged"

    allocator_module.ElasticTokenToKVPoolAllocator = ElasticTokenAllocator
    allocator_module.ElasticPagedTokenToKVPoolAllocator = ElasticPagedAllocator
    mem_cache_module: Any = types.ModuleType("sglang.srt.mem_cache")
    mem_cache_module.allocator = allocator_module
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache", mem_cache_module)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.mem_cache.allocator", allocator_module
    )

    swa_module: Any = types.SimpleNamespace(
        TokenToKVPoolAllocator=NativeTokenAllocator,
        PagedTokenToKVPoolAllocator=NativePagedAllocator,
    )

    class SWATokenToKVPoolAllocator:
        def __init__(
            self,
            size,
            size_swa,
            page_size,
            dtype,
            device,
            kvcache,
            need_sort,
        ):
            del size_swa, need_sort
            self.full = swa_module.PagedTokenToKVPoolAllocator(
                size, page_size, dtype, device, kvcache.full_kv_pool
            )
            self.swa = swa_module.PagedTokenToKVPoolAllocator(
                size, page_size, dtype, device, kvcache.swa_kv_pool
            )

    swa_module.SWATokenToKVPoolAllocator = SWATokenToKVPoolAllocator
    monkeypatch.setattr(
        patches.DeepSeekV4SWAAllocatorPatch,
        "initialize_version_info",
        lambda self: True,
    )

    patch = patches.DeepSeekV4SWAAllocatorPatch()
    assert patch.patch_swa_allocator(swa_module)

    managed_cache = types.SimpleNamespace(
        swa_kv_pool=types.SimpleNamespace(_kvcached_managed=True)
    )
    managed = swa_module.SWATokenToKVPoolAllocator(
        64, 32, 4, object(), "cuda:0", managed_cache, False
    )
    assert managed.full.kind == "elastic-paged"
    assert managed.swa.kind == "elastic-paged"
    assert managed_cache.full_kv_pool.kvcached_allocator.available_size() == 16

    native_cache = types.SimpleNamespace(
        full_kv_pool=object(), swa_kv_pool=object()
    )
    native = swa_module.SWATokenToKVPoolAllocator(
        64, 32, 4, object(), "cuda:0", native_cache, False
    )
    assert native.full.kind == "native-paged"
    assert native.swa.kind == "native-paged"


def test_alloc_dsv4_swa_cache_builds_per_layer_contiguous_views(monkeypatch):
    interfaces, _patches, _torch_mock = _load_modules(monkeypatch)
    raw_tensors = [FakeTensor(), FakeTensor()]
    captured: dict[str, Any] = {}

    def create_kv_tensors(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return raw_tensors

    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    monkeypatch.setattr(interfaces, "_contiguous_layout", False)
    monkeypatch.setattr(interfaces, "create_kv_tensors", create_kv_tensors)

    buffers, returned_raw = interfaces.alloc_dsv4_swa_cache(
        num_pages=3,
        bytes_per_page=1000,
        num_layers=2,
        device="cuda:0",
        group_id=23,
    )

    assert returned_raw is raw_tensors
    assert buffers == raw_tensors
    assert captured["args"][0] == interfaces.PAGE_SIZE
    assert captured["kwargs"] == {
        "num_kv_buffers": 1,
        "group_id": 23,
        "unified_pool": True,
    }
    assert all(tensor.view_shape == (3, 1000) for tensor in raw_tensors)


def test_alloc_dsv4_swa_cache_rejects_compound_layout(monkeypatch):
    interfaces, _patches, _torch_mock = _load_modules(monkeypatch)
    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    monkeypatch.setattr(interfaces, "_contiguous_layout", True)

    with pytest.raises(RuntimeError, match="KVCACHED_CONTIGUOUS_LAYOUT=false"):
        interfaces.alloc_dsv4_swa_cache(
            num_pages=3,
            bytes_per_page=1000,
            num_layers=2,
            device="cuda:0",
            group_id=23,
        )
