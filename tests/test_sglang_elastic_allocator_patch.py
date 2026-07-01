# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.machinery
import sys
import types
from typing import Any, cast
from unittest import mock


class FakeTensor(list):
    def __getitem__(self, index):
        value = super().__getitem__(index)
        if isinstance(index, slice):
            return FakeTensor(value)
        return value

    def numel(self):
        return len(self)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self)

    def split(self, size):
        assert size == 1
        return [FakeTensor([value]) for value in self]


class FakeKVCachedAllocator:
    def __init__(self):
        self.alloc_calls = []

    def alloc(self, need_size):
        self.alloc_calls.append(need_size)
        return list(range(need_size))

    def available_size(self):
        return 16


def _load_sglang_patches(monkeypatch):
    torch_mock = mock.MagicMock()
    torch_mock.__version__ = "2.6.0"
    torch_mock.int64 = "int64"
    torch_mock.empty.side_effect = lambda shape, **_kwargs: FakeTensor(
        [0] * (shape if isinstance(shape, int) else shape[0])
    )
    torch_mock.arange.side_effect = lambda start, end=None, **_kwargs: FakeTensor(
        range(start if end is not None else 0, end if end is not None else start)
    )
    torch_mock.tensor.side_effect = lambda values, **_kwargs: FakeTensor(values)
    torch_mock.cat.side_effect = lambda tensors, **_kwargs: FakeTensor(
        [value for tensor in tensors for value in tensor]
    )
    monkeypatch.setitem(sys.modules, "torch", torch_mock)

    utils_mod: Any = types.ModuleType("sglang.srt.utils")
    utils_mod.next_power_of_2 = lambda value: 1 << (int(value) - 1).bit_length()

    def get_num_new_pages(**_kwargs):
        raise AssertionError("get_num_new_pages should not be called")

    utils_mod.get_num_new_pages = get_num_new_pages

    for name in ("sglang", "sglang.srt"):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_mod)

    from kvcached.integration.sglang import patches

    importlib.reload(patches)
    return patches


def _install_sglang_interfaces(monkeypatch, interfaces):
    import kvcached.integration.sglang as sglang_integration

    monkeypatch.setitem(
        sys.modules, "kvcached.integration.sglang.interfaces", interfaces
    )
    monkeypatch.setattr(sglang_integration, "interfaces", interfaces)


def test_elastic_paged_alloc_extend_accepts_precomputed_num_new_pages(monkeypatch):
    patches = _load_sglang_patches(monkeypatch)

    class BaseTokenToKVPoolAllocator:
        def __init__(self, size, page_size, _dtype, device, _kvcache):
            self.size = size
            self.page_size = page_size
            self.device = device
            self.free_group = []
            self.is_not_in_free_group = True

    class Kernel:
        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def run(*args):
                self.calls.append((grid, args))

            return run

    alloc_extend_kernel = Kernel()
    alloc_mod: Any = types.SimpleNamespace(
        BaseTokenToKVPoolAllocator=BaseTokenToKVPoolAllocator,
        alloc_extend_kernel=alloc_extend_kernel,
        alloc_decode_kernel=Kernel(),
    )

    patch = patches.ElasticAllocatorPatch()
    assert patch.inject_elastic_paged_allocator(alloc_mod)

    allocator = FakeKVCachedAllocator()
    kvcache = types.SimpleNamespace(kvcached_allocator=allocator)
    paged = alloc_mod.ElasticPagedTokenToKVPoolAllocator(
        128,
        16,
        object(),
        "cuda:0",
        kvcache,
    )

    out = paged.alloc_extend(
        prefix_lens=[0, 0],
        prefix_lens_cpu=[0, 0],
        seq_lens=[8, 24],
        seq_lens_cpu=[8, 24],
        last_loc=FakeTensor([0, 0]),
        extend_num_tokens=32,
        num_new_pages=2,
    )

    assert len(out) == 32
    assert allocator.alloc_calls == [2]
    assert len(alloc_extend_kernel.calls) == 1


def test_elastic_paged_allocator_supports_split_sglang_allocator_module(monkeypatch):
    patches = _load_sglang_patches(monkeypatch)

    class BaseTokenToKVPoolAllocator:
        def __init__(self, size, page_size, _dtype, device, _kvcache):
            self.size = size
            self.page_size = page_size
            self.device = device
            self.free_group = []
            self.is_not_in_free_group = True

    class Kernel:
        def __init__(self):
            self.calls = []

            def fn(
                prefix_lens,
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                bs_upper,
                page_size,
            ):
                return None

            self.fn = fn

        def __getitem__(self, grid):
            def run(*args):
                self.calls.append((grid, args))

            return run

    alloc_extend_kernel = Kernel()
    paged_mod: Any = types.ModuleType("sglang.srt.mem_cache.allocator.paged")
    paged_mod.alloc_extend_kernel = alloc_extend_kernel
    paged_mod.alloc_decode_kernel = Kernel()
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache", types.ModuleType("sglang.srt.mem_cache"))
    allocator_pkg: Any = types.ModuleType("sglang.srt.mem_cache.allocator")
    allocator_pkg.paged = paged_mod
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.allocator", allocator_pkg)
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.allocator.paged", paged_mod)

    alloc_mod: Any = types.SimpleNamespace(
        BaseTokenToKVPoolAllocator=BaseTokenToKVPoolAllocator
    )

    patch = patches.ElasticAllocatorPatch()
    assert patch.inject_elastic_paged_allocator(alloc_mod)
    assert patch.alias_paged_allocator_to_elastic(alloc_mod)
    assert paged_mod.PagedTokenToKVPoolAllocator is alloc_mod.ElasticPagedTokenToKVPoolAllocator

    allocator = FakeKVCachedAllocator()
    kvcache = types.SimpleNamespace(kvcached_allocator=allocator)
    paged = alloc_mod.PagedTokenToKVPoolAllocator(
        128,
        16,
        object(),
        "cuda:0",
        kvcache,
    )

    out = paged.alloc_extend(
        prefix_lens=[0, 0],
        prefix_lens_cpu=[0, 0],
        seq_lens=[8, 24],
        seq_lens_cpu=[8, 24],
        last_loc=FakeTensor([0, 0]),
        extend_num_tokens=32,
        num_new_pages=2,
    )

    assert len(out) == 32
    assert allocator.alloc_calls == [2]
    assert len(alloc_extend_kernel.calls) == 1
    assert len(alloc_extend_kernel.calls[0][1]) == 7


def test_elastic_paged_allocator_falls_back_for_native_pools(monkeypatch):
    patches = _load_sglang_patches(monkeypatch)

    class BaseTokenToKVPoolAllocator:
        def __init__(self, size, page_size, _dtype, device, _kvcache):
            self.size = size
            self.page_size = page_size
            self.device = device

    class NativePagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
        pass

    class Kernel:
        def __getitem__(self, grid):
            def run(*_args):
                return None

            return run

    alloc_mod: Any = types.SimpleNamespace(
        BaseTokenToKVPoolAllocator=BaseTokenToKVPoolAllocator,
        PagedTokenToKVPoolAllocator=NativePagedTokenToKVPoolAllocator,
        alloc_extend_kernel=Kernel(),
        alloc_decode_kernel=Kernel(),
    )

    patch = patches.ElasticAllocatorPatch()
    assert patch.inject_elastic_paged_allocator(alloc_mod)
    assert patch.alias_paged_allocator_to_elastic(alloc_mod)

    native = alloc_mod.PagedTokenToKVPoolAllocator(
        128,
        16,
        object(),
        "cuda:0",
        types.SimpleNamespace(),
    )

    assert isinstance(native, NativePagedTokenToKVPoolAllocator)


def test_elastic_allocator_falls_back_for_native_pools(monkeypatch):
    patches = _load_sglang_patches(monkeypatch)

    class BaseTokenToKVPoolAllocator:
        def __init__(self, size, page_size, _dtype, device, _kvcache):
            self.size = size
            self.page_size = page_size
            self.device = device

    class NativeTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
        def __init__(self, size, _dtype, device, _kvcache):
            super().__init__(size, 1, _dtype, device, _kvcache)

    alloc_mod: Any = types.SimpleNamespace(
        BaseTokenToKVPoolAllocator=BaseTokenToKVPoolAllocator,
        TokenToKVPoolAllocator=NativeTokenToKVPoolAllocator,
    )

    patch = patches.ElasticAllocatorPatch()
    assert patch.inject_elastic_allocator(alloc_mod)
    assert patch.alias_allocator_to_elastic(alloc_mod)

    native = alloc_mod.TokenToKVPoolAllocator(
        128,
        object(),
        "cuda:0",
        types.SimpleNamespace(),
    )

    assert isinstance(native, NativeTokenToKVPoolAllocator)


def test_elastic_mha_pool_uses_storage_dtype(monkeypatch):
    patches = _load_sglang_patches(monkeypatch)

    logical_dtype = types.SimpleNamespace(itemsize=1, name="float8_e4m3fn")
    storage_dtype = types.SimpleNamespace(itemsize=1, name="uint8")
    calls: dict[str, Any] = {}

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
            self.enable_memory_saver = enable_memory_saver
            self.start_layer = start_layer
            self.end_layer = end_layer
            self._create_buffers()

        def get_kv_size_bytes(self):
            return (self.layer_num * self.size, self.layer_num * self.size)

        def _create_buffers(self):
            return None

    def fake_alloc_kv_cache(**kwargs):
        calls["alloc_kv_cache"] = kwargs
        return ([object()], [object()])

    interfaces: Any = types.ModuleType("kvcached.integration.sglang.interfaces")
    interfaces.init_kvcached = lambda **kwargs: calls.setdefault("init_kvcached", kwargs)
    interfaces.get_kv_cache_manager = (
        lambda *args, **kwargs: types.SimpleNamespace(args=args, kwargs=kwargs)
    )
    interfaces.alloc_kv_cache = fake_alloc_kv_cache
    _install_sglang_interfaces(monkeypatch, interfaces)

    mem_pool_mod: Any = types.SimpleNamespace(MHATokenToKVPool=MHATokenToKVPool)
    patch = patches.ElasticMemoryPoolPatch()
    assert patch.inject_elastic_mem_pool(mem_pool_mod)

    pool = mem_pool_mod.ElasticMHATokenToKVPool(
        size=128,
        page_size=16,
        dtype=logical_dtype,
        head_num=8,
        head_dim=64,
        layer_num=2,
        device="cuda",
        enable_memory_saver=False,
    )

    assert pool.cell_size == 8 * 64 * storage_dtype.itemsize
    assert calls["init_kvcached"]["device"] == "cuda:0"
    assert calls["alloc_kv_cache"]["dtype"] is storage_dtype
    assert calls["alloc_kv_cache"]["device"] == "cuda:0"
    assert pool.device == "cuda:0"


def test_elastic_mha_nonzero_tp_rank_uses_logical_allocator(monkeypatch):
    patches = _load_sglang_patches(monkeypatch)

    dist_mod: Any = types.ModuleType("sglang.srt.distributed")
    dist_mod.get_tensor_model_parallel_rank = lambda: 2
    dist_mod.get_tensor_model_parallel_world_size = lambda: 4
    dist_mod.get_pipeline_model_parallel_rank = lambda: 0
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", dist_mod)

    calls: dict[str, Any] = {}

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
            self.store_dtype = dtype
            self.head_num = head_num
            self.head_dim = head_dim
            self.layer_num = layer_num
            self.device = device
            self.enable_memory_saver = enable_memory_saver
            self.start_layer = start_layer
            self.end_layer = end_layer
            self._create_buffers()

        def get_kv_size_bytes(self):
            return (self.layer_num * self.size, self.layer_num * self.size)

        def _create_buffers(self):
            return None

    interfaces: Any = types.ModuleType("kvcached.integration.sglang.interfaces")
    interfaces.init_kvcached = lambda **kwargs: calls.setdefault("init_kvcached", kwargs)

    def fake_alloc_kv_cache(**kwargs):
        calls["alloc_kv_cache"] = kwargs
        return [object()], [object()]

    interfaces.alloc_kv_cache = fake_alloc_kv_cache

    def fail_get_manager(*_args, **_kwargs):
        raise AssertionError("nonzero TP rank should not drive physical KV manager")

    interfaces.get_kv_cache_manager = fail_get_manager
    _install_sglang_interfaces(monkeypatch, interfaces)

    mem_pool_mod: Any = types.SimpleNamespace(MHATokenToKVPool=MHATokenToKVPool)
    patch = patches.ElasticMemoryPoolPatch()
    assert patch.inject_elastic_mem_pool(mem_pool_mod)

    pool = mem_pool_mod.ElasticMHATokenToKVPool(
        size=128,
        page_size=16,
        dtype=types.SimpleNamespace(itemsize=1),
        head_num=8,
        head_dim=64,
        layer_num=2,
        device="cuda",
        enable_memory_saver=False,
    )

    assert pool.device == "cuda:2"
    assert calls["init_kvcached"]["device"] == "cuda:2"
    assert calls["alloc_kv_cache"]["device"] == "cuda:2"
    assert pool.kvcached_allocator.alloc(2) == [1, 2]


def test_resolve_tp_device_keeps_explicit_device(monkeypatch):
    patches = _load_sglang_patches(monkeypatch)

    assert patches._resolve_tp_device("cuda", 2) == "cuda:2"
    assert patches._resolve_tp_device("hip", 3) == "hip:3"
    assert patches._resolve_tp_device("cuda:1", 2) == "cuda:1"


def test_capped_allocator_reports_logical_available_but_checks_physical(monkeypatch):
    patches = _load_sglang_patches(monkeypatch)

    class Manager:
        def __init__(self):
            self.available = 2
            self.alloc_calls = []

        def available_size(self):
            return self.available

        def alloc(self, need_size):
            self.alloc_calls.append(need_size)
            return list(range(1, need_size + 1))

        def free(self, _block_ids):
            return None

    manager = Manager()
    allocator = patches._CappedKVCachedAllocator(manager, capacity_blocks=5)

    assert allocator.available_size() == 5
    assert allocator.alloc(3) is None
    assert manager.alloc_calls == []

    manager.available = 4
    assert allocator.alloc(3) == [1, 2, 3]
    assert allocator.available_size() == 2


def test_capped_allocator_only_frees_allocated_ids(monkeypatch):
    patches = _load_sglang_patches(monkeypatch)

    class Manager:
        def __init__(self):
            self.freed = []

        def available_size(self):
            return 8

        def alloc(self, need_size):
            return list(range(1, need_size + 1))

        def free(self, block_ids):
            self.freed.append(list(block_ids))

    manager = Manager()
    allocator = patches._CappedKVCachedAllocator(manager, capacity_blocks=4)

    assert allocator.alloc(2) == [1, 2]
    allocator.free([0, 1, 99])

    assert manager.freed == [[1]]
    assert allocator.available_size() == 3


def test_logical_allocator_only_frees_allocated_ids(monkeypatch):
    patches = _load_sglang_patches(monkeypatch)

    allocator = patches._LogicalKVCachedAllocator(capacity_blocks=4)

    assert allocator.alloc(2) == [1, 2]
    allocator.free([0, 1, 99])

    assert sorted(allocator.alloc(2)) == [1, 3]


def test_elastic_mamba_pool_uses_rank_local_device_for_spec_buffers(monkeypatch):
    patches = _load_sglang_patches(monkeypatch)

    torch_mod = cast(Any, sys.modules["torch"])
    torch_mod.zeros.side_effect = lambda size, **kwargs: types.SimpleNamespace(
        size=size,
        kwargs=kwargs,
        mem_usage_bytes=lambda: 0,
    )

    calls: dict[str, Any] = {"zeros": []}

    def fake_zeros(*args, **kwargs):
        calls["zeros"].append(kwargs)
        return types.SimpleNamespace(
            args=args,
            kwargs=kwargs,
            mem_usage_bytes=lambda: 0,
        )

    torch_mod.zeros.side_effect = fake_zeros

    dist_mod: Any = types.ModuleType("sglang.srt.distributed")
    dist_mod.get_tensor_model_parallel_rank = lambda: 2
    dist_mod.get_tensor_model_parallel_world_size = lambda: 4
    dist_mod.get_pipeline_model_parallel_rank = lambda: 0
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", dist_mod)

    interfaces: Any = types.ModuleType("kvcached.integration.sglang.interfaces")
    interfaces.init_kvcached = lambda **kwargs: calls.setdefault("init_kvcached", kwargs)
    interfaces.alloc_mamba_states = lambda **kwargs: (
        [types.SimpleNamespace(mem_usage_bytes=lambda: 0)],
        types.SimpleNamespace(mem_usage_bytes=lambda: 0),
        {"is_contiguous": True, "cell_size": 1},
    )
    interfaces.get_kv_cache_manager = lambda *args, **kwargs: types.SimpleNamespace(
        args=args,
        kwargs=kwargs,
        available_size=lambda: 8,
    )
    _install_sglang_interfaces(monkeypatch, interfaces)

    class CacheParams:
        layers = [0]
        shape = types.SimpleNamespace(conv=[(2, 2)], temporal=(2, 2, 2))
        dtype = types.SimpleNamespace(conv="conv-dtype", temporal="temporal-dtype")

    class MambaPool:
        class State:
            def __init__(self, *, conv, temporal):
                self.conv = conv
                self.temporal = temporal

            def mem_usage_bytes(self):
                return 0

        class SpeculativeState(State):
            def __init__(self, *, conv, temporal, intermediate_ssm, intermediate_conv_window):
                super().__init__(conv=conv, temporal=temporal)
                self.intermediate_ssm = intermediate_ssm
                self.intermediate_conv_window = intermediate_conv_window

    mem_pool_mod: Any = types.SimpleNamespace(MambaPool=MambaPool)
    patch = patches.ElasticMambaPoolPatch()
    assert patch.inject_elastic_mamba_pool(mem_pool_mod)

    pool = mem_pool_mod.ElasticMambaPool(
        size=16,
        spec_state_size=1,
        cache_params=CacheParams(),
        device="cuda",
        speculative_num_draft_tokens=2,
    )

    assert pool.device == "cuda:2"
    assert calls["init_kvcached"]["device"] == "cuda:2"
    assert calls["zeros"]
    assert {call["device"] for call in calls["zeros"]} == {"cuda:2"}
