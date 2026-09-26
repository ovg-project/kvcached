# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the elastic pool overrides on the SGLang 0.5.16+ layout.

SGLang 0.5.16 split ``MHATokenToKVPool._create_buffers()`` into
``_create_buffers_normal()`` plus a tail that derives ``_kv_buffer_descs``
(read by PD transfer, prefill-decode disaggregation) and the
``data_ptrs``/``data_strides`` tensors (read by the speculative-decode kv
copy). The elastic pool must override only the inner stage there so the
native tail still runs. SGLang 0.5.13 renamed the MLA sparse-attention
attributes from ``use_nsa``/``nsa_kv_cache_store_fp8`` to
``use_dsa``/``dsa_kv_cache_store_fp8``, and inherited write paths read the
new spelling on every call.
"""

import sys
import types
from typing import Any, Dict

import pytest
import torch

from kvcached.integration.sglang.patches import (
    ElasticMemoryPoolPatch,
    ElasticMLAMemoryPoolPatch,
)


@pytest.fixture
def elastic_env(monkeypatch):
    """Stub the kvcached interfaces module and allow cpu-device pools."""
    calls: Dict[str, Any] = {}
    stub: Any = types.ModuleType("kvcached.integration.sglang.interfaces")

    def init_kvcached(**kwargs):
        calls["init_kvcached"] = kwargs

    def alloc_kv_cache(**kwargs):
        calls["alloc_kv_cache"] = kwargs

        def make_buffer():
            return torch.zeros(tuple(kwargs["kvcache_shape"]),
                               dtype=kwargs["dtype"])

        k_buffers = [make_buffer() for _ in range(kwargs["num_layers"])]
        if kwargs["attention_type"] == "MLA":
            calls["mla_buffers"] = k_buffers
            return k_buffers
        v_buffers = [make_buffer() for _ in range(kwargs["num_layers"])]
        calls["mha_buffers"] = (k_buffers, v_buffers)
        return k_buffers, v_buffers

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
    """Mirrors the sglang 0.5.16..0.5.20 buffer-creation dispatch."""

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
        kv_cache_layout=None,
        quant_method=None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.store_dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = head_dim
        self.layer_num = layer_num
        self.device = device
        self.kv_cache_layout = kv_cache_layout or "nhd"
        self.use_hnd = self.kv_cache_layout == "hnd"
        self.quant_method = quant_method
        self.native_normal_calls = 0
        self.native_quantized_calls = 0
        self._create_buffers()

    @property
    def is_quantized_kv_cache(self):
        return self.quant_method is not None

    def _create_buffers(self):
        # Same dispatch as sglang v0.5.16..v0.5.20 memory_pool.py.
        if self.is_quantized_kv_cache:
            self._create_quantized_buffers()
        else:
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

    def _create_quantized_buffers(self):
        self.native_quantized_calls += 1
        self.k_buffer = []
        self.v_buffer = []

    def _build_kv_buffer_descs(self):
        return [tuple(t.shape) for t in (*self.k_buffer, *self.v_buffer)]

    def _init_data_ptrs_and_strides(self):
        buffers = [*self.k_buffer, *self.v_buffer]
        self.data_ptrs = [t.data_ptr() for t in buffers]
        self.data_strides = [
            t[0].numel() * t.element_size() for t in buffers
        ]

    def get_kv_size_bytes(self):
        return 0, 0


class PreSeamMHATokenToKVPool:
    """Mirrors the pre-0.5.16 single-stage buffer creation."""

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
        kv_cache_layout=None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.device = device
        self.kv_cache_layout = kv_cache_layout or "nhd"
        self.use_hnd = self.kv_cache_layout == "hnd"
        self.native_create_calls = 0
        self._create_buffers()

    def _create_buffers(self):
        self.native_create_calls += 1
        self.k_buffer = []
        self.v_buffer = []

    def get_kv_size_bytes(self):
        return 0, 0


def _inject_mha(pool_cls):
    module: Any = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    module.MHATokenToKVPool = pool_cls
    assert ElasticMemoryPoolPatch().inject_elastic_mem_pool(module)
    return module


def _make_mha_pool(module, **kwargs):
    return module.ElasticMHATokenToKVPool(
        8, 4, torch.float16, 2, 4, 2, "cpu", False, **kwargs
    )


def test_seam_layout_keeps_native_descriptor_tail(elastic_env):
    module = _inject_mha(SeamMHATokenToKVPool)

    pool = _make_mha_pool(module)

    elastic_k, elastic_v = elastic_env["mha_buffers"]
    assert pool.k_buffer is elastic_k
    assert pool.v_buffer is elastic_v
    # The native torch.zeros stage must not run in addition.
    assert pool.native_normal_calls == 0
    # The native tail ran over the elastic buffers.
    assert pool._kv_buffer_descs == [
        tuple(t.shape) for t in (*elastic_k, *elastic_v)
    ]
    assert pool.data_ptrs == [
        t.data_ptr() for t in (*elastic_k, *elastic_v)
    ]
    assert len(pool.data_strides) == 4


def test_seam_layout_pins_quant_adjacent_attributes(elastic_env):
    module = _inject_mha(SeamMHATokenToKVPool)

    pool = _make_mha_pool(module)

    assert pool.k_scale_buffer is None
    assert pool.v_scale_buffer is None
    assert pool.dq_k_buffer is None
    assert pool.dq_v_buffer is None


@pytest.mark.parametrize("layout", ["hnd", "vectorized_5d"])
def test_seam_layout_rejects_non_nhd_layouts(elastic_env, layout):
    module = _inject_mha(SeamMHATokenToKVPool)

    with pytest.raises(NotImplementedError, match="NHD"):
        _make_mha_pool(module, kv_cache_layout=layout)


def test_seam_layout_rejects_quantized_recipes(elastic_env):
    module = _inject_mha(SeamMHATokenToKVPool)

    with pytest.raises(NotImplementedError, match="quantized KV cache"):
        _make_mha_pool(module, quant_method=object())


def test_pre_seam_layout_still_replaces_create_buffers(elastic_env):
    module = _inject_mha(PreSeamMHATokenToKVPool)

    pool = _make_mha_pool(module)

    elastic_k, elastic_v = elastic_env["mha_buffers"]
    assert pool.k_buffer is elastic_k
    assert pool.v_buffer is elastic_v
    assert pool.native_create_calls == 0
    assert elastic_env["alloc_kv_cache"]["kv_layout"] == "NHD"


def test_pre_seam_layout_rejects_non_nhd_layouts(elastic_env):
    module = _inject_mha(PreSeamMHATokenToKVPool)

    with pytest.raises(NotImplementedError, match="NHD"):
        _make_mha_pool(module, kv_cache_layout="hnd")


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
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1


class MLATokenToKVPool(KVCache):
    # Set by the elastic subclass constructor; declared for mypy.
    use_dsa: bool
    dsa_kv_cache_store_fp8: bool

    def native_write_gate(self):
        """Mirrors the sglang 0.5.13+ set_kv_buffer entry assertion."""
        assert not self.dsa_kv_cache_store_fp8
        return self.use_dsa

    def get_kv_size_bytes(self):
        return 0


def _inject_mla():
    module: Any = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    module.KVCache = KVCache
    module.MLATokenToKVPool = MLATokenToKVPool
    assert ElasticMLAMemoryPoolPatch().inject_elastic_mla_mem_pool(module)
    return module


def _make_mla_pool(module, dtype=torch.float16, **kwargs):
    return module.ElasticMLATokenToKVPool(
        8, 4, dtype, 4, 2, 2, "cpu", False, **kwargs
    )


def test_mla_pool_sets_renamed_dsa_attributes(elastic_env):
    module = _inject_mla()

    pool = _make_mla_pool(module)

    assert pool.use_dsa is False
    assert pool.dsa_kv_cache_store_fp8 is False
    # Compatibility spellings for sglang older than 0.5.13.
    assert pool.use_nsa is False
    assert pool.nsa_kv_cache_store_fp8 is False
    assert pool.kv_cache_dim == 6
    assert pool.native_write_gate() is False
    assert elastic_env["alloc_kv_cache"]["kvcache_shape"] == (12, 1, 6)


@pytest.mark.parametrize("spelling", ["use_dsa", "use_nsa"])
def test_mla_pool_accepts_either_dsa_kwarg_spelling(elastic_env, spelling):
    module = _inject_mla()

    pool = _make_mla_pool(
        module,
        dtype=torch.float8_e4m3fn,
        override_kv_cache_dim=16,
        **{spelling: True},
    )

    assert pool.use_dsa is True
    assert pool.use_nsa is True
    assert pool.dsa_kv_cache_store_fp8 is True
    assert pool.nsa_kv_cache_store_fp8 is True
    assert pool.kv_cache_dim == 16
    assert elastic_env["alloc_kv_cache"]["kvcache_shape"] == (12, 1, 16)


def test_mla_pool_fp8_flag_requires_override_dim(elastic_env):
    module = _inject_mla()

    pool = _make_mla_pool(module, dtype=torch.float8_e4m3fn, use_dsa=True)

    # Native derivation on every version since 0.5.9: without
    # override_kv_cache_dim the fp8 store flag stays off and the pool
    # keeps the kv_lora_rank + qk_rope_head_dim layout.
    assert pool.dsa_kv_cache_store_fp8 is False
    assert pool.kv_cache_dim == 6
    assert pool.native_write_gate() is True
