# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from typing import Any
from unittest import mock


def _load_patches(monkeypatch):
    torch_mock = mock.MagicMock()
    torch_mock.__version__ = "2.6.0"
    monkeypatch.setitem(sys.modules, "torch", torch_mock)
    monkeypatch.setitem(sys.modules, "torch.cuda", torch_mock.cuda)
    monkeypatch.setitem(sys.modules, "posix_ipc", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())

    from kvcached.integration.sglang import patches

    importlib.reload(patches)
    return patches


def _make_pool_class():
    class DeepSeekV4TokenToKVPool:
        def __init__(self):
            self.device = "cuda:0"
            self.swa_kv_pool = types.SimpleNamespace(
                size=1024,
                page_size=16,
                layer_num=2,
                device="cuda:0",
                bytes_per_page_padded=576,
                kv_buffer=["native-swa-0", "native-swa-1"],
            )
            self.c4_kv_pool = types.SimpleNamespace(
                kv_buffer=[mock.Mock(nbytes=40)]
            )
            self.c128_kv_pool = types.SimpleNamespace(
                kv_buffer=[mock.Mock(nbytes=80)]
            )
            self.c4_indexer_kv_pool = types.SimpleNamespace(
                index_k_with_scale_buffer=[]
            )
            self.compress_state_pools = []
            self.indexer_compress_state_pools = []

    return DeepSeekV4TokenToKVPool


def _make_allocator_class():
    class FakeSWATokenToKVPoolAllocator:
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
            self.args = (
                size,
                size_swa,
                page_size,
                dtype,
                device,
                kvcache,
                need_sort,
            )
            self.full_proxy = kvcache.full_kv_pool

    return FakeSWATokenToKVPoolAllocator


def _install_allocator_module(monkeypatch):
    allocator_module = types.ModuleType("sglang.srt.mem_cache.allocator")
    setattr(allocator_module, "TokenToKVPoolAllocator", object())
    setattr(allocator_module, "PagedTokenToKVPoolAllocator", object())
    mem_cache_module = types.ModuleType("sglang.srt.mem_cache")
    setattr(mem_cache_module, "allocator", allocator_module)
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache", mem_cache_module)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.mem_cache.allocator", allocator_module
    )
    return allocator_module


def test_dsv4_bridge_replaces_real_swa_buffers_and_keeps_side_pools(monkeypatch):
    patches = _load_patches(monkeypatch)
    elastic_allocators = _install_allocator_module(monkeypatch)
    managed_allocator = object()
    managed_buffers = ["managed-swa-0", "managed-swa-1"]
    proxy = types.SimpleNamespace(
        kv_buffer=managed_buffers,
        kvcached_allocator=managed_allocator,
        raw_tensors=["raw-0", "raw-1"],
        raw_allocator="manager",
        group_id=20000,
    )
    monkeypatch.setattr(
        patches, "_new_dsv4_swa_kvcached_proxy", lambda _pool: proxy
    )

    pool_class = _make_pool_class()
    allocator_class = _make_allocator_class()
    pool_module = types.SimpleNamespace(
        DeepSeekV4TokenToKVPool=pool_class
    )
    allocator_module = types.SimpleNamespace(
        SWATokenToKVPoolAllocator=allocator_class
    )
    patch = patches.DeepSeekV4KVPoolBridgePatch()
    assert patch.patch_dsv4_pool_init(pool_module)
    assert patch.patch_swa_allocator(allocator_module)
    assert (
        allocator_module.PagedTokenToKVPoolAllocator
        is elastic_allocators.PagedTokenToKVPoolAllocator
    )

    pool = pool_module.DeepSeekV4TokenToKVPool()
    assert pool.swa_kv_pool.kv_buffer is managed_buffers
    assert pool.swa_kv_pool.kvcached_allocator is managed_allocator
    assert pool.swa_kv_pool._kvcached_dsv4_managed is True
    assert pool.c4_kv_pool.kv_buffer[0].nbytes == 40
    assert pool.c128_kv_pool.kv_buffer[0].nbytes == 80

    allocator = allocator_module.SWATokenToKVPoolAllocator(
        4096,
        1024,
        16,
        "float16",
        "cuda:0",
        pool,
        False,
    )
    logical = allocator.full_proxy.kvcached_allocator
    assert logical.available_size() == 256
    assert logical.alloc(2) == [1, 2]


def test_dsv4_reservations_exclude_only_managed_swa_pool(monkeypatch):
    patches = _load_patches(monkeypatch)
    pool = _make_pool_class()()
    pool.swa_kv_pool.kv_buffer = [mock.Mock(nbytes=120)]

    native = patches._collect_dsv4_runtime_reservations(pool)
    assert native["dsv4.swa_kv_pool"] == 120
    assert native["dsv4.c4_kv_pool"] == 40
    assert native["dsv4.c128_kv_pool"] == 80

    pool.swa_kv_pool._kvcached_dsv4_managed = True
    managed = patches._collect_dsv4_runtime_reservations(pool)
    assert managed["dsv4.swa_kv_pool"] == 0
    assert managed["dsv4.c4_kv_pool"] == 40
    assert managed["dsv4.c128_kv_pool"] == 80


def test_dsv4_bridge_is_opt_in(monkeypatch):
    patches = _load_patches(monkeypatch)
    pool_class = _make_pool_class()
    pool_module = types.SimpleNamespace(
        DeepSeekV4TokenToKVPool=pool_class
    )
    original_init = pool_class.__init__

    monkeypatch.delenv("KVCACHED_SGLANG_DSV4_KV_POOL_BRIDGE", raising=False)
    patch = patches.DeepSeekV4KVPoolBridgePatch()
    assert patch.apply(pool_module)
    assert pool_module.DeepSeekV4TokenToKVPool.__init__ is original_init


def test_dsv4_bridge_requires_swa_pool(monkeypatch):
    patches = _load_patches(monkeypatch)
    with mock.patch.object(
        patches, "_new_dsv4_swa_kvcached_proxy"
    ) as create_proxy:
        try:
            patches._attach_dsv4_allocator_bridge(types.SimpleNamespace())
        except ValueError as exc:
            assert "swa_kv_pool" in str(exc)
        else:
            raise AssertionError("missing swa_kv_pool must fail")
        create_proxy.assert_not_called()


def test_dsv4_swa_proxy_maps_real_pool_shape(monkeypatch):
    patches = _load_patches(monkeypatch)
    calls: dict[str, Any] = {}
    buffers = [mock.Mock(), mock.Mock()]
    raw_tensors = [mock.Mock(), mock.Mock()]

    interfaces: Any = types.ModuleType("kvcached.integration.sglang.interfaces")
    interfaces.init_kvcached = lambda **kwargs: calls.setdefault("init", kwargs)

    def alloc_dsv4_swa_cache(**kwargs):
        calls["alloc"] = kwargs
        return buffers, raw_tensors

    interfaces.alloc_dsv4_swa_cache = alloc_dsv4_swa_cache

    def get_manager(**kwargs):
        calls["manager"] = kwargs
        return types.SimpleNamespace(available_size=lambda: kwargs["num_blocks"])

    interfaces.get_kv_cache_manager = get_manager
    monkeypatch.setitem(
        sys.modules, "kvcached.integration.sglang.interfaces", interfaces
    )
    import kvcached.integration.sglang as sglang_integration

    monkeypatch.setattr(sglang_integration, "interfaces", interfaces, raising=False)
    monkeypatch.setattr(patches, "_distributed_ranks", lambda: (0, 4, 0))

    pool = types.SimpleNamespace(
        size=1024,
        page_size=16,
        layer_num=2,
        device="cuda:0",
        bytes_per_page_padded=576,
    )
    proxy = patches._new_dsv4_swa_kvcached_proxy(pool)

    assert proxy.kv_buffer is buffers
    assert proxy.raw_tensors is raw_tensors
    assert proxy.kvcached_allocator.available_size() == 64
    assert calls["init"] == {
        "tp_rank": 0,
        "world_size": 4,
        "pp_rank": 0,
        "device": "cuda:0",
        "async_sched": True,
    }
    assert calls["alloc"]["bytes_per_page"] == 576
    assert calls["alloc"]["num_layers"] == 2
    assert calls["manager"]["block_size"] == 16
    assert calls["manager"]["cell_size"] == 36
    assert calls["manager"]["num_layers"] == 2


def test_dsv4_bridge_patch_is_idempotent(monkeypatch):
    patches = _load_patches(monkeypatch)
    _install_allocator_module(monkeypatch)
    pool_class = _make_pool_class()
    allocator_class = _make_allocator_class()
    pool_module = types.SimpleNamespace(
        DeepSeekV4TokenToKVPool=pool_class
    )
    allocator_module = types.SimpleNamespace(
        SWATokenToKVPoolAllocator=allocator_class
    )
    patch = patches.DeepSeekV4KVPoolBridgePatch()

    assert patch.patch_dsv4_pool_init(pool_module)
    first_pool_init = pool_module.DeepSeekV4TokenToKVPool.__init__
    assert patch.patch_dsv4_pool_init(pool_module)
    assert pool_module.DeepSeekV4TokenToKVPool.__init__ is first_pool_init

    assert patch.patch_swa_allocator(allocator_module)
    first_allocator_init = allocator_module.SWATokenToKVPoolAllocator.__init__
    assert patch.patch_swa_allocator(allocator_module)
    assert (
        allocator_module.SWATokenToKVPoolAllocator.__init__
        is first_allocator_init
    )
