# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for zero-attention-layer SGLang pools (pure Mamba2 models).

SGLang 0.5.16+ builds the full-attention sub-pool of ``HybridLinearKVPool``
with ``layer_num=0`` for models without full-attention layers: a pure Mamba2
model (e.g. Mamba-Codestral-7B-v0.1) puts every layer in the Mamba pool, so
``full_attention_layer_ids`` is empty. Native SGLang allocates no buffers for
such a pool (its per-layer loops run zero times) and serves the model. The
elastic path instead reached ``alloc_kv_cache``, whose per-layer sizing
divides the GPU budget by ``num_layers`` and raised ``ZeroDivisionError``
(found by the v0.1.6 release validation, issue #509). These tests pin the
fix: the elastic pool mirrors native for zero layers (empty buffers, no
kvcached manager), the elastic token allocators hand the pool to the native
allocator classes, and ``alloc_kv_cache`` rejects ``num_layers <= 0`` with a
clear error instead of dividing by it.
"""

import importlib.util
import sys
import types
from abc import ABC
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from kvcached.integration.sglang.patches import (
    ElasticAllocatorPatch,
    ElasticHybridLinearKVPoolPatch,
    ElasticMemoryPoolPatch,
)

# Shapes from the failing validation run: Mamba-Codestral-7B-v0.1 on an
# L40S (46068 MiB), sglang 0.5.20 defaults (page_size=1, bf16 KV,
# max_total_num_tokens=2048), zero full-attention layers.
L40S_TOTAL_MEMORY = 46068 * 1024 * 1024
CODESTRAL_SHAPE = (2048 + 1, 8, 128)


def _make_elastic_env(monkeypatch) -> Dict[str, Any]:
    """Stub the kvcached interfaces module and allow cpu-device pools."""
    calls: Dict[str, Any] = {}
    stub: Any = types.ModuleType("kvcached.integration.sglang.interfaces")

    def init_kvcached(**kwargs):
        calls["init_kvcached"] = kwargs

    def alloc_kv_cache(**kwargs):
        calls["alloc_kv_cache"] = kwargs
        num_layers = kwargs["num_layers"]
        shape = tuple(kwargs["kvcache_shape"])
        k = [torch.zeros(shape, dtype=kwargs["dtype"]) for _ in range(num_layers)]
        v = [torch.zeros(shape, dtype=kwargs["dtype"]) for _ in range(num_layers)]
        return k, v

    def get_kv_cache_manager(*args, **kwargs):
        calls["get_kv_cache_manager"] = (args, kwargs)
        return object()

    stub.init_kvcached = init_kvcached
    stub.alloc_kv_cache = alloc_kv_cache
    stub.get_kv_cache_manager = get_kv_cache_manager

    import kvcached.integration.sglang as sglang_integration_pkg
    monkeypatch.setitem(
        sys.modules, "kvcached.integration.sglang.interfaces", stub
    )
    monkeypatch.setattr(
        sglang_integration_pkg, "interfaces", stub, raising=False
    )

    from kvcached.integration.sglang import patches
    monkeypatch.setattr(
        patches, "_is_supported_gpu_device", lambda device: True
    )
    return calls


class SeamMHATokenToKVPool:
    """Mirrors the sglang 0.5.16..0.5.20 buffer-creation dispatch,
    including the native zero-layer guard in ``_init_kv_copy_and_warmup``."""

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
        enable_kv_cache_copy=False,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.store_dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.device = device
        self.kv_cache_layout = "nhd"
        self.use_hnd = False
        self.quant_method = None
        self.native_normal_calls = 0
        self._create_buffers()
        if enable_kv_cache_copy and not self.use_hnd:
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

    @property
    def is_quantized_kv_cache(self):
        return False

    def _create_buffers(self):
        self.k_scale_buffer = None
        self.v_scale_buffer = None
        self.dq_k_buffer = None
        self.dq_v_buffer = None
        self._create_buffers_normal()
        self._kv_buffer_descs = self._build_kv_buffer_descs()
        self._init_data_ptrs_and_strides()

    def _buffer_shape(self):
        return (self.size + self.page_size, self.head_num, self.head_dim)

    def _create_buffers_normal(self):
        self.native_normal_calls += 1
        self.k_buffer = [
            torch.zeros(self._buffer_shape(), dtype=self.dtype)
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            torch.zeros(self._buffer_shape(), dtype=self.dtype)
            for _ in range(self.layer_num)
        ]

    def _build_kv_buffer_descs(self):
        return [tuple(t.shape) for t in (*self.k_buffer, *self.v_buffer)]

    def _init_data_ptrs_and_strides(self):
        buffers = [*self.k_buffer, *self.v_buffer]
        self.data_ptrs = [t.data_ptr() for t in buffers]
        self.data_strides = [t[0].numel() * t.element_size() for t in buffers]

    def _init_kv_copy_and_warmup(self):
        # Native v0.5.16..v0.5.20 guard: zero-layer pools have no buffers.
        if self.layer_num == 0:
            self._kv_copy_config = None
            return
        self._kv_copy_config = {"stride": int(self.data_strides[0])}

    def get_kv_size_bytes(self):
        k = sum(t.numel() * t.element_size() for t in self.k_buffer)
        v = sum(t.numel() * t.element_size() for t in self.v_buffer)
        return k, v


def _inject_mha(pool_cls):
    module: Any = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    module.MHATokenToKVPool = pool_cls
    assert ElasticMemoryPoolPatch().inject_elastic_mem_pool(module)
    return module


# ---------------------------------------------------------------------------
# interfaces.alloc_kv_cache: reject zero layers instead of dividing by them
# ---------------------------------------------------------------------------


class _FakeProps:
    def __init__(self, total_memory):
        self.total_memory = total_memory


def _load_interfaces_under_stubs(monkeypatch):
    """Load the real interfaces.py with its process-level deps stubbed,
    the test_observability.py pattern, so no compiled extension or GPU is
    needed to reach the sizing arithmetic."""
    fake_torch = types.ModuleType("torch")
    setattr(fake_torch, "dtype", object)
    setattr(fake_torch, "Tensor", object)
    setattr(
        fake_torch, "cuda",
        types.SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda dev=None: _FakeProps(
                L40S_TOTAL_MEMORY),
        ))

    manager_module = types.ModuleType("kvcached.kv_cache_manager")
    setattr(manager_module, "KVCacheManager", object)

    tp_ipc_module = types.ModuleType("kvcached.tp_ipc_util")
    setattr(tp_ipc_module, "resolve_gpu_device_index", lambda device: 0)
    setattr(tp_ipc_module, "start_worker_listener_thread", lambda *args: None)
    setattr(tp_ipc_module, "stop_worker_listener_threads", lambda: True)

    utils_module = types.ModuleType("kvcached.utils")
    setattr(utils_module, "CONTIGUOUS_LAYOUT", False)
    setattr(utils_module, "PAGE_SIZE", 2 * 1024 * 1024)
    setattr(utils_module, "get_page_size_for_block", lambda block, page: page)
    setattr(utils_module, "get_kvcached_logger",
            lambda: types.SimpleNamespace())
    setattr(utils_module, "normalize_gpu_device", lambda device: device)

    vmm_ops_module = types.ModuleType("kvcached.vmm_ops")
    setattr(
        vmm_ops_module, "create_kv_tensors",
        lambda *args, **kwargs: pytest.fail(
            "must not allocate ftensors for zero layers"))
    setattr(vmm_ops_module, "init_kvcached", lambda *args, **kwargs: None)
    setattr(vmm_ops_module, "shutdown_kvcached", lambda: None)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "kvcached.kv_cache_manager",
                        manager_module)
    monkeypatch.setitem(sys.modules, "kvcached.tp_ipc_util", tp_ipc_module)
    monkeypatch.setitem(sys.modules, "kvcached.utils", utils_module)
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", vmm_ops_module)

    module_path = (
        Path(__file__).parents[1]
        / "kvcached"
        / "integration"
        / "sglang"
        / "interfaces.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_sglang_interfaces_zero_layers", module_path)
    assert spec is not None and spec.loader is not None
    interfaces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interfaces)
    setattr(interfaces, "_kvcached_initialized", True)
    return interfaces


def test_alloc_kv_cache_rejects_zero_layers(monkeypatch):
    """The validation crash shape: num_layers=0 reached
    ``gpu_mem_bytes // num_layers`` and raised ZeroDivisionError. The
    contract error must name the condition instead."""
    interfaces = _load_interfaces_under_stubs(monkeypatch)

    with pytest.raises(ValueError, match="num_layers must be positive"):
        interfaces.alloc_kv_cache(
            CODESTRAL_SHAPE,
            types.SimpleNamespace(itemsize=2),  # bf16 storage width
            "cuda:0",
            num_layers=0,
            page_size=1,
            attention_type="MHA",
        )


# ---------------------------------------------------------------------------
# ElasticMHATokenToKVPool: zero layers mirror native (no elastic alloc)
# ---------------------------------------------------------------------------


def test_zero_layer_elastic_pool_allocates_nothing(monkeypatch):
    calls = _make_elastic_env(monkeypatch)
    module = _inject_mha(SeamMHATokenToKVPool)

    pool = module.ElasticMHATokenToKVPool(
        2048, 1, torch.bfloat16, 8, 128, 0, "cpu", False)

    # Same empty buffers native leaves, and the native tail ran over them.
    assert pool.k_buffer == []
    assert pool.v_buffer == []
    assert pool._kv_buffer_descs == []
    assert pool.data_ptrs == []
    # Nothing elastic was engaged for the empty attention group.
    assert "alloc_kv_cache" not in calls
    assert "get_kv_cache_manager" not in calls
    assert pool.kvcached_allocator is None
    assert pool.mem_usage == 0.0


def test_zero_layer_pool_skips_copy_warmup(monkeypatch):
    _make_elastic_env(monkeypatch)
    module = _inject_mha(SeamMHATokenToKVPool)

    pool = module.ElasticMHATokenToKVPool(
        2048, 1, torch.bfloat16, 8, 128, 0, "cpu", False,
        enable_kv_cache_copy=True)

    assert pool._kv_copy_config is None


def test_nonzero_layer_elastic_pool_still_engages(monkeypatch):
    """Regression guard: the zero-layer skip must not widen."""
    calls = _make_elastic_env(monkeypatch)
    module = _inject_mha(SeamMHATokenToKVPool)

    pool = module.ElasticMHATokenToKVPool(
        2048, 1, torch.bfloat16, 8, 128, 2, "cpu", False)

    assert calls["alloc_kv_cache"]["num_layers"] == 2
    assert "get_kv_cache_manager" in calls
    assert pool.kvcached_allocator is not None
    assert len(pool.k_buffer) == 2


# ---------------------------------------------------------------------------
# HybridLinearKVPool: the delegating property reports the absent manager
# ---------------------------------------------------------------------------


class FakeHybridLinearKVPool:
    """Stands in for sglang's HybridLinearKVPool: builds the full-attention
    sub-pool from the module-level MHA class, zero layers for pure Mamba2."""

    def __init__(self, full_kv_pool):
        self.full_kv_pool = full_kv_pool


def test_hybrid_pool_property_reports_missing_manager(monkeypatch):
    _make_elastic_env(monkeypatch)
    mem_module = _inject_mha(SeamMHATokenToKVPool)
    mem_module.HybridLinearKVPool = FakeHybridLinearKVPool
    assert ElasticHybridLinearKVPoolPatch().inject_elastic_hybrid_linear_pool(
        mem_module)

    full = mem_module.ElasticMHATokenToKVPool(
        2048, 1, torch.bfloat16, 8, 128, 0, "cpu", False)
    hybrid = mem_module.ElasticHybridLinearKVPool(full)

    # hasattr stays True (the allocator dispatch reads the None, it must
    # not see an AttributeError) and the delegated value is the sentinel.
    assert hasattr(hybrid, "kvcached_allocator")
    assert hybrid.kvcached_allocator is None


# ---------------------------------------------------------------------------
# Elastic allocators: zero-layer pools get the native allocator classes
# ---------------------------------------------------------------------------


class FakeBaseTokenToKVPoolAllocator(ABC):
    def __init__(self, size, page_size, dtype, device, kvcache, *args, **kwargs):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.kvcache = kvcache
        self.is_not_in_free_group = True
        self.free_group = []


class FakeNativeTokenToKVPoolAllocator(FakeBaseTokenToKVPoolAllocator):
    def __init__(self, size, dtype, device, kvcache, *args, **kwargs):
        super().__init__(size, 1, dtype, device, kvcache, *args, **kwargs)


class FakeNativePagedTokenToKVPoolAllocator(FakeBaseTokenToKVPoolAllocator):
    pass


def _make_allocator_module(monkeypatch, with_native=True):
    sglang: Any = types.ModuleType("sglang")
    srt: Any = types.ModuleType("sglang.srt")
    utils: Any = types.ModuleType("sglang.srt.utils")
    utils.get_num_new_pages = lambda **kwargs: 0
    utils.next_power_of_2 = lambda value: 1 << (max(value, 1) - 1).bit_length()
    sglang.srt = srt
    srt.utils = utils
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils)

    alloc_mod: Any = types.ModuleType("sglang.srt.mem_cache.allocator")
    alloc_mod.BaseTokenToKVPoolAllocator = FakeBaseTokenToKVPoolAllocator
    if with_native:
        alloc_mod.TokenToKVPoolAllocator = FakeNativeTokenToKVPoolAllocator
        alloc_mod.PagedTokenToKVPoolAllocator = (
            FakeNativePagedTokenToKVPoolAllocator)

    def _kernel(*args, **kwargs):
        pass

    alloc_mod.alloc_extend_kernel = _kernel
    alloc_mod.alloc_decode_kernel = _kernel
    return alloc_mod


class FakeZeroLayerPool:
    """A hybrid pool over a zero-layer attention sub-pool, as the fix
    leaves it: the manager slot exists and is None."""

    def __init__(self):
        self.full_kv_pool = types.SimpleNamespace(layer_num=0)
        self.kvcached_allocator = None


class FakeElasticPool:
    def __init__(self):
        self.full_kv_pool = types.SimpleNamespace(layer_num=24)
        self.kvcached_allocator = object()


def test_token_allocator_dispatches_native_for_zero_layer_pool(monkeypatch):
    alloc_mod = _make_allocator_module(monkeypatch)
    patch = ElasticAllocatorPatch()
    assert patch.inject_elastic_allocator(alloc_mod)
    assert patch.alias_allocator_to_elastic(alloc_mod)

    allocator = alloc_mod.TokenToKVPoolAllocator(
        2048, "bf16", "cuda:0", FakeZeroLayerPool())

    assert type(allocator) is FakeNativeTokenToKVPoolAllocator
    assert isinstance(allocator, alloc_mod.ElasticTokenToKVPoolAllocator)
    assert allocator.size == 2048


def test_paged_allocator_dispatches_native_for_zero_layer_pool(monkeypatch):
    alloc_mod = _make_allocator_module(monkeypatch)
    patch = ElasticAllocatorPatch()
    assert patch.inject_elastic_paged_allocator(alloc_mod)
    assert patch.alias_paged_allocator_to_elastic(alloc_mod)

    allocator = alloc_mod.PagedTokenToKVPoolAllocator(
        2048, 16, "bf16", "cuda:0", FakeZeroLayerPool())

    assert type(allocator) is FakeNativePagedTokenToKVPoolAllocator
    assert isinstance(
        allocator, alloc_mod.ElasticPagedTokenToKVPoolAllocator)
    assert allocator.page_size == 16


def test_token_allocator_stays_elastic_for_attention_pools(monkeypatch):
    """Regression guard: the native dispatch must not widen beyond
    zero-layer pools; silent fallback was exactly the #509 finding-2
    failure mode."""
    alloc_mod = _make_allocator_module(monkeypatch)
    patch = ElasticAllocatorPatch()
    assert patch.inject_elastic_allocator(alloc_mod)
    assert patch.alias_allocator_to_elastic(alloc_mod)

    allocator = alloc_mod.TokenToKVPoolAllocator(
        2048, "bf16", "cuda:0", FakeElasticPool())

    assert isinstance(allocator, alloc_mod.ElasticTokenToKVPoolAllocator)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("import_before_patch", [False, True])
def test_zero_layer_allocator_passes_cache_constructor(
    monkeypatch, paged, import_before_patch
):
    alloc_mod = _make_allocator_module(monkeypatch)
    token_cls = alloc_mod.TokenToKVPoolAllocator
    paged_cls = alloc_mod.PagedTokenToKVPoolAllocator
    patch = ElasticAllocatorPatch()
    monkeypatch.setattr(patch.version_manager, "detect_version", lambda _: "0.5.15")
    assert patch.apply(alloc_mod)
    # Re-entry must retain the native classes captured before aliasing.
    assert patch.apply(alloc_mod)
    if not import_before_patch:
        token_cls = alloc_mod.TokenToKVPoolAllocator
        paged_cls = alloc_mod.PagedTokenToKVPoolAllocator

    class MambaCache:
        def __init__(self, allocator):
            # MambaRadixCache checks the classes it imported from allocator.
            assert isinstance(allocator, token_cls) or isinstance(allocator, paged_cls)
            self.allocator = allocator

    if paged:
        allocator = alloc_mod.PagedTokenToKVPoolAllocator(
            2048, 16, "bf16", "cuda:0", FakeZeroLayerPool())
        assert type(allocator) is FakeNativePagedTokenToKVPoolAllocator
        assert not isinstance(allocator, alloc_mod.ElasticTokenToKVPoolAllocator)
    else:
        allocator = alloc_mod.TokenToKVPoolAllocator(
            2048, "bf16", "cuda:0", FakeZeroLayerPool())
        assert type(allocator) is FakeNativeTokenToKVPoolAllocator
        assert not isinstance(allocator, alloc_mod.ElasticPagedTokenToKVPoolAllocator)
    assert MambaCache(allocator).allocator is allocator


@pytest.mark.parametrize("paged", [False, True])
def test_nonzero_native_allocator_is_not_treated_as_elastic(monkeypatch, paged):
    alloc_mod = _make_allocator_module(monkeypatch)
    patch = ElasticAllocatorPatch()
    monkeypatch.setattr(patch.version_manager, "detect_version", lambda _: "0.5.15")
    assert patch.apply(alloc_mod)
    allocator: FakeBaseTokenToKVPoolAllocator
    if paged:
        allocator = FakeNativePagedTokenToKVPoolAllocator(
            2048, 16, "bf16", "cuda:0", FakeElasticPool())
    else:
        allocator = FakeNativeTokenToKVPoolAllocator(
            2048, "bf16", "cuda:0", FakeElasticPool())
    assert not isinstance(allocator, alloc_mod.ElasticTokenToKVPoolAllocator)
    assert not isinstance(allocator, alloc_mod.ElasticPagedTokenToKVPoolAllocator)
