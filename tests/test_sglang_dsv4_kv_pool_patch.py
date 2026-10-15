# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from contextvars import ContextVar
from typing import Any
from unittest import mock

import pytest


class FakeTensor:
    def __init__(self, nbytes: int = 0, contiguous: bool = True):
        self.nbytes = nbytes
        self.contiguous = contiguous
        self.view_shape: Any = None
        self.view_dtype: Any = None

    def __getitem__(self, _key):
        return self

    def view(self, *shape, **kwargs):
        if "dtype" in kwargs:
            self.view_dtype = kwargs["dtype"]
        else:
            self.view_shape = shape
        return self

    def is_contiguous(self):
        return self.contiguous


def _load_modules(monkeypatch):
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
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
    distributed = types.ModuleType("sglang.srt.distributed")
    setattr(distributed, "get_tensor_model_parallel_world_size", lambda: 1)
    setattr(distributed, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    monkeypatch.setattr(patches, "CONTIGUOUS_LAYOUT", False)
    # These direct constructor tests explicitly model a standalone target.
    monkeypatch.setattr(patches, "_dsv4_pool_context", ContextVar(
        "test_dsv4_pool_context", default=(False, None)
    ))
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
    monkeypatch.setattr(patches, "_resolve_sglang_parallel_context", lambda: (0, 1, 1))
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
        "world_size": 1,
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
    physical_bytes = interfaces.get_page_size_for_block(576, interfaces.PAGE_SIZE)
    assert calls["manager"][1]["cell_size"] == physical_bytes
    assert calls["manager"][1]["num_blocks"] == (65 * 576 + physical_bytes - 1) // physical_bytes
    assert pool.swa_kv_pool._kvcached_dsv4_backing.manager is manager
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


@pytest.mark.parametrize("wrapped", [False, True])
def test_shared_draft_uses_native_swa_and_retains_reservation(monkeypatch, wrapped):
    _interfaces, patches, _torch_mock = _load_modules(monkeypatch)
    module = _make_dsv4_module()
    target = types.SimpleNamespace(swa_kv_pool=types.SimpleNamespace(size=1024, page_size=16))
    shared = types.SimpleNamespace(get_kvcache=lambda: target)
    if wrapped:
        shared = types.SimpleNamespace(logical_attn_allocator=shared)

    class Runner:
        is_draft_worker = True
        token_to_kv_pool_allocator = shared

        def _init_pools(self):
            self.pool = module.DeepSeekV4TokenToKVPool()

    assert patches.DeepSeekV4KVPoolPatch().inject_dsv4_swa_pool(module)
    runner_mod: Any = types.SimpleNamespace(ModelRunner=Runner)
    patch = patches.DeepSeekV4RunnerOwnershipPatch()
    assert patch.patch_pool_initialization(runner_mod)
    assert patch.patch_pool_initialization(runner_mod)
    runner = Runner()
    runner._init_pools()
    assert runner.token_to_kv_pool_allocator is shared
    assert not hasattr(runner.pool.swa_kv_pool, "kvcached_allocator")
    assert patches._collect_dsv4_runtime_reservations(runner.pool)["dsv4.swa_kv_pool"] == 202
    assert patches._dsv4_pool_context.get() == (False, None)


@pytest.mark.parametrize("size,page_size", [(2048, 16), (1024, 32)])
def test_shared_draft_rejects_incompatible_target_before_allocation(monkeypatch, size, page_size):
    _interfaces, patches, _torch_mock = _load_modules(monkeypatch)
    module = _make_dsv4_module()
    target = types.SimpleNamespace(swa_kv_pool=types.SimpleNamespace(size=size, page_size=page_size))
    shared = types.SimpleNamespace(get_kvcache=lambda: target)
    native_create = mock.Mock(side_effect=AssertionError("must validate before allocation"))
    monkeypatch.setattr(module.DeepSeekV4SingleKVPool, "_create_buffers", native_create)
    assert patches.DeepSeekV4KVPoolPatch().inject_dsv4_swa_pool(module)
    token = patches._dsv4_pool_context.set((True, shared))
    try:
        with pytest.raises(patches.KVCachedConfigError, match="cover the shared target"):
            module.DeepSeekV4TokenToKVPool()
    finally:
        patches._dsv4_pool_context.reset(token)
    native_create.assert_not_called()


@pytest.mark.parametrize("entry", ["_init_pools", "alloc_memory_pool"])
def test_runner_ownership_context_restores_nested_calls_and_exceptions(monkeypatch, entry):
    _interfaces, patches, _torch_mock = _load_modules(monkeypatch)
    shared = object()
    seen = []

    class Runner:
        def __init__(self, draft, allocator):
            self.is_draft_worker = draft
            self.token_to_kv_pool_allocator = allocator

        def _init_pools(self):
            seen.append(patches._dsv4_pool_context.get())
            if self.is_draft_worker:
                getattr(Runner(False, None), entry)()
                seen.append(patches._dsv4_pool_context.get())
                raise RuntimeError("construction failed")

    if entry != "_init_pools":
        setattr(Runner, "alloc_memory_pool", Runner._init_pools)
        del Runner._init_pools
    runner_mod: Any = types.SimpleNamespace(ModelRunner=Runner)
    assert patches.DeepSeekV4RunnerOwnershipPatch().patch_pool_initialization(runner_mod)
    with pytest.raises(RuntimeError, match="construction failed"):
        getattr(Runner(True, shared), entry)()
    assert seen == [(True, shared), (False, None), (True, shared)]
    assert patches._dsv4_pool_context.get() == (False, None)


@pytest.mark.parametrize("version", ["0.5.15", "0.5.19", "0.5.21"])
def test_unaudited_version_does_not_take_over_dsv4_storage(monkeypatch, version):
    _interfaces, patches, _torch_mock = _load_modules(monkeypatch)
    patch = patches.DeepSeekV4KVPoolPatch()
    monkeypatch.setattr(patch.version_manager, "detect_version", lambda _: version)
    module = _make_dsv4_module()
    original = module.DeepSeekV4TokenToKVPool
    assert patch.apply(module) is False
    assert module.DeepSeekV4TokenToKVPool is original


def test_dsv4_requires_runner_identity_before_elastic_allocation(monkeypatch):
    _interfaces, patches, _torch_mock = _load_modules(monkeypatch)
    module = _make_dsv4_module()
    assert patches.DeepSeekV4KVPoolPatch().inject_dsv4_swa_pool(module)
    token = patches._dsv4_pool_context.set(None)
    try:
        with pytest.raises(patches.KVCachedConfigError, match="ownership context"):
            module.DeepSeekV4TokenToKVPool()
    finally:
        patches._dsv4_pool_context.reset(token)


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
        _native_allocator_cls = NativeTokenAllocator

        def __init__(self, *args, **kwargs):
            self.kind = "elastic-token"

    class ElasticPagedAllocator:
        _native_allocator_cls = NativePagedAllocator

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
                size, page_size, dtype, device, getattr(kvcache, "full_kv_pool", None)
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

    # Exercise the registered order, rather than capturing pristine aliases.
    assert patches.ElasticSWAAllocatorPatch().alias_swa_sub_allocators(swa_module)
    patch = patches.DeepSeekV4SWAAllocatorPatch()
    assert patch.patch_swa_allocator(swa_module)

    from kvcached.integration.sglang import dsv4_backing

    bind = mock.Mock()
    monkeypatch.setattr(dsv4_backing, "bind_native_swa", bind)
    managed_cache = types.SimpleNamespace(
        swa_kv_pool=types.SimpleNamespace(
            _kvcached_managed=True, kvcached_allocator=object(),
            _kvcached_dsv4_backing=object()
        )
    )
    managed = swa_module.SWATokenToKVPoolAllocator(
        64, 32, 4, object(), "cuda:0", managed_cache, False
    )
    assert managed.full.kind == "native-paged"
    assert managed.swa.kind == "native-paged"
    assert not hasattr(managed_cache, "full_kv_pool")
    bind.assert_called_once_with(managed, managed_cache.swa_kv_pool._kvcached_dsv4_backing, None)

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
    monkeypatch.setattr(interfaces, "PAGE_SIZE", 2048)
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
    assert captured["args"][0] == 2 * interfaces.PAGE_SIZE
    assert captured["kwargs"] == {
        "num_kv_buffers": 1,
        "group_id": 23,
        "unified_pool": True,
        "page_size": 2048,
    }
    assert all(tensor.view_shape == (3, 1000) for tensor in raw_tensors)
    assert all(tensor.view_dtype is _torch_mock.uint8 for tensor in raw_tensors)


@pytest.mark.parametrize(
    ("bytes_per_page", "physical_page_size"),
    [(4 * 1024 * 1024, 4 * 1024 * 1024),
     (5 * 1024 * 1024 // 2, 6 * 1024 * 1024)],
)
def test_alloc_dsv4_swa_cache_resolves_backing_geometry(
    monkeypatch, bytes_per_page, physical_page_size
):
    interfaces, _patches, _torch_mock = _load_modules(monkeypatch)
    monkeypatch.delenv("KVCACHED_PAGE_SIZE_MB", raising=False)
    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    monkeypatch.setattr(interfaces, "_contiguous_layout", False)
    monkeypatch.setattr(interfaces, "PAGE_SIZE", 2 * 1024 * 1024)
    create = mock.Mock(return_value=[FakeTensor()])
    monkeypatch.setattr(interfaces, "create_kv_tensors", create)

    buffers, _ = interfaces.alloc_dsv4_swa_cache(
        num_pages=5, bytes_per_page=bytes_per_page, num_layers=1,
        device="cuda:0", group_id=23,
    )

    assert create.call_args.kwargs["page_size"] == physical_page_size
    expected_bytes = ((5 * bytes_per_page + physical_page_size - 1)
                      // physical_page_size * physical_page_size)
    assert create.call_args.args[0] == expected_bytes
    assert buffers[0].view_shape == (5, bytes_per_page)


@pytest.mark.parametrize("bytes_per_page", [4 * 1024 * 1024, 3 * 1024 * 1024 // 2])
def test_alloc_dsv4_swa_cache_validates_explicit_geometry_before_backing(
    monkeypatch, bytes_per_page
):
    interfaces, _patches, _torch_mock = _load_modules(monkeypatch)
    from kvcached.utils import KVCachedConfigError

    monkeypatch.setenv("KVCACHED_PAGE_SIZE_MB", "2")
    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    monkeypatch.setattr(interfaces, "_contiguous_layout", False)
    monkeypatch.setattr(interfaces, "PAGE_SIZE", 2 * 1024 * 1024)
    create = mock.Mock()
    monkeypatch.setattr(interfaces, "create_kv_tensors", create)

    with pytest.raises(KVCachedConfigError, match="cannot manage this KV geometry"):
        interfaces.alloc_dsv4_swa_cache(
            num_pages=5, bytes_per_page=bytes_per_page, num_layers=1,
            device="cuda:0", group_id=23,
        )
    create.assert_not_called()


def test_alloc_dsv4_swa_cache_rejects_pool_without_a_physical_page(monkeypatch):
    interfaces, _patches, _torch_mock = _load_modules(monkeypatch)
    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    monkeypatch.setattr(interfaces, "_contiguous_layout", False)
    monkeypatch.setattr(interfaces, "PAGE_SIZE", 4096)
    create = mock.Mock()
    monkeypatch.setattr(interfaces, "create_kv_tensors", create)

    with pytest.raises(ValueError, match="at least one physical page"):
        interfaces.alloc_dsv4_swa_cache(
            num_pages=3, bytes_per_page=1000, num_layers=2,
            device="cuda:0", group_id=23,
        )
    create.assert_not_called()


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


@pytest.mark.parametrize("size,layers,managed", [(4096, 1, True), (64, 1, False), (4096, 0, False)])
@pytest.mark.parametrize("ratio", [4, 128])
def test_compressed_factory_selects_only_useful_supported_storage(monkeypatch, size, layers, managed, ratio):
    interfaces, patches, _ = _load_modules(monkeypatch)
    module = _make_dsv4_module()
    monkeypatch.setattr(module.DeepSeekV4SingleKVPool, "get_bytes_per_token", lambda self: 584)
    monkeypatch.setattr(interfaces, "init_kvcached", lambda **_: None)
    monkeypatch.setattr(patches, "_resolve_sglang_parallel_context", lambda: (0, 1, 0))
    monkeypatch.setattr(interfaces, "alloc_dsv4_swa_cache", lambda **_: ([FakeTensor(200)], []))
    managers = []

    def manager(_kvi, **kwargs):
        managers.append(kwargs)
        return object()

    monkeypatch.setattr(patches, "_new_tp_scoped_kvcached_allocator", manager)
    assert patches.DeepSeekV4KVPoolPatch().inject_dsv4_swa_pool(module)
    pool = module.DeepSeekV4TokenToKVPool()
    pool.qk_nope_head_dim, pool.qk_rope_head_dim = 448, 64
    kwargs = dict(size=size, page_size=2, dtype=types.SimpleNamespace(itemsize=1),
                  layer_num=layers, device="cuda:0", enable_memory_saver=False,
                  global_page_size=2 * ratio)
    pool.c128_kv_pool = pool._make_kv_pool(**kwargs)
    assert bool(getattr(pool.c128_kv_pool, "_kvcached_managed", False)) is managed
    breakdown = patches._collect_dsv4_runtime_reservations(pool)
    assert breakdown["dsv4.c128_kv_pool"] == (0 if managed else 101 * layers)
    assert breakdown["dsv4.c4_kv_pool"] == 101
    if managed:
        assert managers[-1]["pool_name"] == f"dsv4_c{ratio}_kv"
        assert managers[0]["group_id"] != managers[1]["group_id"]
        assert pool.c128_kv_pool.bytes_per_page_padded == 1728
    token = patches._dsv4_pool_context.set((True, types.SimpleNamespace(
        page_size=2 * ratio, size_full=4096 * ratio)))
    try:
        kwargs.update(size=4096, layer_num=1)
        draft_pool = pool._make_kv_pool(**kwargs)
        assert not hasattr(draft_pool, "_kvcached_managed")
        assert patches._pool_buffer_nbytes(draft_pool, "kv_buffer") == 101
        kwargs["size"] = 4094
        with pytest.raises(patches.KVCachedConfigError, match="shared target full IDs"):
            pool._make_kv_pool(**kwargs)
    finally:
        patches._dsv4_pool_context.reset(token)


@pytest.mark.parametrize("tp_size,pp_size", [(2, 1), (4, 1), (1, 2), (1, 4), (2, 2)])
def test_dsv4_parallel_retains_native_storage_before_any_backing(monkeypatch, tp_size, pp_size):
    interfaces, patches, _ = _load_modules(monkeypatch)
    module = _make_dsv4_module()
    distributed = sys.modules["sglang.srt.distributed"]
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(distributed, "get_pipeline_model_parallel_world_size", lambda: pp_size)
    create = mock.Mock(side_effect=AssertionError("Distributed pools must retain native backing"))
    monkeypatch.setattr(interfaces, "alloc_dsv4_swa_cache", create)
    assert patches.DeepSeekV4KVPoolPatch().inject_dsv4_swa_pool(module)
    pool = module.DeepSeekV4TokenToKVPool()
    assert not hasattr(pool.swa_kv_pool, "_kvcached_managed")
    assert not hasattr(pool.c4_kv_pool, "_kvcached_managed")
    assert patches._collect_dsv4_runtime_reservations(pool)["dsv4.swa_kv_pool"] == 202
    create.assert_not_called()


@pytest.mark.parametrize("error", [AssertionError, RuntimeError, AttributeError, ValueError])
def test_dsv4_unknown_topology_retains_native_storage(monkeypatch, error):
    interfaces, patches, _ = _load_modules(monkeypatch)
    distributed = sys.modules["sglang.srt.distributed"]
    monkeypatch.setattr(distributed, "get_pipeline_model_parallel_world_size",
                        mock.Mock(side_effect=error))
    module = _make_dsv4_module()
    create = mock.Mock(side_effect=AssertionError("Unknown topology must stay native"))
    monkeypatch.setattr(interfaces, "alloc_dsv4_swa_cache", create)
    assert patches.DeepSeekV4KVPoolPatch().inject_dsv4_swa_pool(module)
    pool = module.DeepSeekV4TokenToKVPool()
    assert not hasattr(pool.swa_kv_pool, "_kvcached_managed")
    assert patches._collect_dsv4_runtime_reservations(pool)["dsv4.swa_kv_pool"] == 202
    create.assert_not_called()


@pytest.mark.parametrize("layout,managed", [("fp8", True), ("fp4", True), ("aiter", False),
                                          ("small", False), ("hisparse", False)])
def test_c4_indexer_preserves_native_layout_and_ownership(monkeypatch, layout, managed):
    interfaces, patches, _ = _load_modules(monkeypatch)
    module = _make_dsv4_module()

    class Indexer:
        def __init__(self, size, page_size, dtype, index_head_dim, layer_num,
                     device, enable_memory_saver):
            self.size, self.page_size = size, page_size
            self.index_head_dim, self.layer_num = index_head_dim, layer_num
            self.device = device
            self.uses_aiter_fp4_layout = layout == "aiter"
            self._create_buffer()

        def get_bytes_per_token(self):
            return 68 if layout in ("fp4", "aiter") else 132

        def _create_buffer(self):
            self.index_k_with_scale_buffer = [FakeTensor(103)]

    def make_indexer(self, size, page_size, dtype, index_head_dim, layer_num,
                     device, enable_memory_saver):
        return Indexer(size, page_size, dtype, index_head_dim, layer_num,
                       device, enable_memory_saver)

    module.DeepSeekV4IndexerPool = Indexer
    module.DeepSeekV4TokenToKVPool._make_indexer_pool = make_indexer
    monkeypatch.setattr(module.DeepSeekV4SingleKVPool, "get_bytes_per_token", lambda self: 584)
    monkeypatch.setattr(interfaces, "init_kvcached", lambda **_: None)
    monkeypatch.setattr(patches, "_resolve_sglang_parallel_context", lambda: (0, 1, 0))
    allocations, managers = [], []

    def allocate(**kwargs):
        allocations.append(kwargs)
        return [FakeTensor(200)], []

    def manager(_kvi, **kwargs):
        managers.append(kwargs)
        return object()

    monkeypatch.setattr(interfaces, "alloc_dsv4_swa_cache", allocate)
    monkeypatch.setattr(patches, "_new_tp_scoped_kvcached_allocator", manager)
    assert patches.DeepSeekV4KVPoolPatch().inject_dsv4_swa_pool(module)
    pool = module.DeepSeekV4TokenToKVPool()
    pool.qk_nope_head_dim, pool.qk_rope_head_dim = 448, 64

    original_single = module.DeepSeekV4SingleKVPool

    HiSparse = type("HiSparse", (original_single,), {})

    c4 = pool._make_kv_pool(size=32768, page_size=64, dtype=object(), layer_num=1,
                           device="cuda:0", enable_memory_saver=False, global_page_size=256,
                           cls=HiSparse if layout == "hisparse" else module.DeepSeekV4SingleKVPool)
    pool.c4_kv_pool = None  # Native factory creates aliases after indexer construction.
    pool.kv_pools = {4: c4}
    indexer = pool._make_indexer_pool(64 if layout == "small" else 32768,
                                    64, object(), 128, 1, "cuda:0", False)
    pool.c4_kv_pool, pool.c4_indexer_kv_pool = c4, indexer
    assert bool(getattr(indexer, "_kvcached_managed", False)) is managed
    breakdown = patches._collect_dsv4_runtime_reservations(pool)
    assert breakdown["dsv4.c4_indexer_kv_pool"] == (0 if managed else 103)
    if managed:
        assert allocations[-1]["bytes_per_page"] == 64 * (68 if layout == "fp4" else 132)
        assert managers[-1]["pool_name"] == "dsv4_c4_indexer_kv"
        assert len({m["group_id"] for m in managers}) == len(managers)
    if layout == "hisparse":
        assert type(c4) is HiSparse
        assert breakdown["dsv4.c4_kv_pool"] == 101
    token = patches._dsv4_pool_context.set((True, object()))
    try:
        draft = pool._make_indexer_pool(32768, 64, object(), 128, 1, "cuda:0", False)
        assert not hasattr(draft, "_kvcached_managed")
    finally:
        patches._dsv4_pool_context.reset(token)


@pytest.mark.parametrize("mode", ["managed", "small", "c128", "unified", "native_swa", "saver"])
def test_c4_state_factory_groups_layers_and_preserves_native_constructor(monkeypatch, mode):
    interfaces, _patches, _ = _load_modules(monkeypatch)
    from test_sglang_dsv4_native_backing import Manager

    from kvcached.integration.sglang.dsv4_state import install_c4_state_backing

    native = types.ModuleType("test_native_dsv4_state")
    managers: list[Any] = []
    allocations: list[Any] = []

    class KVAndScore:
        def __init__(self, tensor):
            self.kv_score = tensor

        def __getitem__(self, index):
            assert index == -1
            return self

        def clear(self):
            # This represents the unchanged native constructor's sentinel write.
            assert managers and 4 in managers[-1].live

    class State:
        kv_score_buffer: Any

        def __init__(self, owner, ratio, head_dim, enable_memory_saver):
            self.ratio, self.online, self.page_size = ratio, False, 1
            self._size = 8 if mode == "small" else 40
            self.ring_size, self.last_dim = 8, head_dim * 4
            self._alloc_kv_score_buffer(dtype=owner.dtype, device="cuda:0",
                                        enable_memory_saver=enable_memory_saver)
            if getattr(self, "_kvcached_managed", False):
                self.kv_score_buffer[-1].clear()

        def _alloc_kv_score_buffer(self, **kwargs):
            self.native_allocation = True

    State.__module__ = native.__name__
    setattr(native, "CompressStatePool", State)
    setattr(native, "KVAndScore", KVAndScore)
    setattr(native, "TorchMemorySaverAdapter", types.SimpleNamespace(create=lambda **_: object()))
    monkeypatch.setitem(sys.modules, native.__name__, native)

    class Top:
        compression_ratios = [4, 4]
        _stage_start, _stage_end = 0, 2
        dtype = object()
        _unified_kv = mode == "unified"
        swa_kv_pool = (None if mode == "native_swa" else types.SimpleNamespace(
            _kvcached_dsv4_backing=types.SimpleNamespace(native_pages=5)))

        def _make_compress_state_pool(self, ratio, *, head_dim, enable_memory_saver):
            return State(self, ratio, head_dim, enable_memory_saver)

    # Dtypes are hashable in the runtime.
    Top.dtype = type("Dtype", (), {"itemsize": 2})()

    def allocate(**kwargs):
        allocations.append(kwargs)
        return [FakeTensor() for _ in range(kwargs["num_layers"])], []

    def manager(**kwargs):
        result = Manager()
        managers.append(result)
        return result

    monkeypatch.setattr(interfaces, "get_page_size_for_block", lambda *_: 256)
    monkeypatch.setattr(interfaces, "alloc_dsv4_swa_cache", allocate)
    monkeypatch.setattr(interfaces, "get_kv_cache_manager", manager)
    install_c4_state_backing(native, Top, lambda: 25000)
    owner = Top()
    pools = [owner._make_compress_state_pool(128 if mode == "c128" else 4,
                                            head_dim=4, enable_memory_saver=mode == "saver")
             for _ in range(2)]
    if mode == "managed":
        assert len(managers) == len(allocations) == 1
        assert allocations[0]["num_layers"] == 2
        assert pools[0]._kvcached_dsv4_backing is pools[1]._kvcached_dsv4_backing
        assert all(p.kv_score_buffer.kv_score.view_shape == (40, 16) for p in pools)
        assert managers[0].live == {4}
        with pytest.raises(RuntimeError, match="layer count"):
            owner._make_compress_state_pool(4, head_dim=4, enable_memory_saver=False)
    else:
        assert all(p.native_allocation for p in pools)
        assert not allocations and not managers

@pytest.mark.parametrize("setting", [None, "false", "0"])
def test_dsv4_disabled_autopatch_retains_native_storage(monkeypatch, setting):
    interfaces, patches, _ = _load_modules(monkeypatch)
    monkeypatch.setenv("KVCACHED_AUTOPATCH", "1")
    if setting is None:
        monkeypatch.delenv("ENABLE_KVCACHED", raising=False)
    else:
        monkeypatch.setenv("ENABLE_KVCACHED", setting)
    monkeypatch.setattr(patches, "_dsv4_pool_context", ContextVar("disabled_dsv4", default=None))
    create = mock.Mock(side_effect=AssertionError("Disabled storage must not initialize a manager"))
    monkeypatch.setattr(interfaces, "init_kvcached", create)
    monkeypatch.setattr(patches, "_new_tp_scoped_kvcached_allocator", create)
    module = _make_dsv4_module()
    assert patches.DeepSeekV4KVPoolPatch().inject_dsv4_swa_pool(module)
    pool = module.DeepSeekV4TokenToKVPool()
    assert pool.swa_kv_pool.kv_buffer[0].nbytes == 101
    assert not hasattr(pool.swa_kv_pool, "_kvcached_managed")
    assert not hasattr(pool.c4_kv_pool, "_kvcached_managed")
    pool.qk_nope_head_dim, pool.qk_rope_head_dim = 448, 64
    compressed = pool._make_kv_pool(
        size=4096, page_size=2, dtype=object(), layer_num=1,
        device="cuda:0", enable_memory_saver=False, global_page_size=256,
    )
    assert not hasattr(compressed, "_kvcached_managed")
    create.assert_not_called()
