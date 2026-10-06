# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the elastic pool overrides on the SGLang 0.5.16+ layout.

SGLang 0.5.16 split ``MHATokenToKVPool._create_buffers()`` into
``_create_buffers_normal()`` plus a tail that derives ``_kv_buffer_descs``
(read by PD transfer, prefill-decode disaggregation) and the
``data_ptrs``/``data_strides`` tensors (read by the speculative-decode kv
copy). The elastic pool overrides the inner stage there so the native tail
still runs, but the pointer tables and the accessors built on them assume
independently contiguous K/V rows: the copy kernel
(``kernels/ops/kvcache/cache_move.py``) uses one scalar per buffer as both
the token-address pitch and the bytes to copy, and PD transfer walks pages
at ``ptr + page * item_len``. The default elastic layout interleaves all
layers and K/V within each token row, so those paths are gated on actual
view contiguity. SGLang 0.5.13 renamed the MLA sparse-attention attributes
from ``use_nsa``/``nsa_kv_cache_store_fp8`` to
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


def _storage_bytes(t: torch.Tensor) -> torch.Tensor:
    """1-D uint8 view of the whole allocation backing ``t``."""
    u8 = torch.empty(0, dtype=torch.uint8)
    u8.set_(t.untyped_storage())
    return u8


def _emulate_copy_all_layer_kv_cache(buffers, data_ptrs, data_strides,
                                     tgt_loc, src_loc):
    """CPU mirror of the sglang copy-kernel address math.

    ``copy_all_layer_kv_cache_tiled`` (v0.5.16..v0.5.20,
    kernels/ops/kvcache/cache_move.py) loads one scalar per buffer and
    uses it both as the token-address pitch (``base + loc * stride``) and
    as the byte count (``byte_off < stride``). All loads happen before
    stores per buffer, as in the kernel.
    """
    for buf, base_ptr, stride in zip(buffers, data_ptrs, data_strides):
        mem = _storage_bytes(buf)
        base = base_ptr - buf.untyped_storage().data_ptr()
        rows = [
            mem[base + int(s) * stride:base + int(s) * stride +
                stride].clone() for s in src_loc.tolist()
        ]
        for d, row in zip(tgt_loc.tolist(), rows):
            mem[base + int(d) * stride:base + int(d) * stride +
                stride] = row


def _reference_move(k_buffers, v_buffers, tgt_loc, src_loc):
    """Token-slot move on cloned views: the intended copy semantics."""
    k_out = [t.clone() for t in k_buffers]
    v_out = [t.clone() for t in v_buffers]
    for k, v in zip(k_out, v_out):
        k[tgt_loc] = k[src_loc]
        v[tgt_loc] = v[src_loc]
    return k_out, v_out


def _make_elastic_env(monkeypatch, interleaved):
    """Stub the kvcached interfaces module and allow cpu-device pools."""
    calls: Dict[str, Any] = {}
    stub: Any = types.ModuleType("kvcached.integration.sglang.interfaces")

    def init_kvcached(**kwargs):
        calls["init_kvcached"] = kwargs

    def alloc_kv_cache(**kwargs):
        calls["alloc_kv_cache"] = kwargs
        shape = tuple(kwargs["kvcache_shape"])
        num_layers = kwargs["num_layers"]

        def make_buffer():
            return torch.zeros(shape, dtype=kwargs["dtype"])

        if kwargs["attention_type"] == "MLA":
            calls["mla_buffers"] = [make_buffer() for _ in range(num_layers)]
            return calls["mla_buffers"]
        if interleaved:
            # Mirror the interfaces.py contiguous layout: one
            # (tokens, layers, 2, *rest) buffer, per-layer K/V views
            # strided across tokens.
            big = torch.zeros((shape[0], num_layers, 2) + shape[1:],
                              dtype=kwargs["dtype"])
            calls["mha_backing"] = big
            k_buffers = [big[:, i, 0] for i in range(num_layers)]
            v_buffers = [big[:, i, 1] for i in range(num_layers)]
        else:
            k_buffers = [make_buffer() for _ in range(num_layers)]
            v_buffers = [make_buffer() for _ in range(num_layers)]
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


@pytest.fixture
def elastic_env(monkeypatch):
    """Per-layer buffers, each independently contiguous."""
    return _make_elastic_env(monkeypatch, interleaved=False)


@pytest.fixture
def interleaved_elastic_env(monkeypatch):
    """The default contiguous layout: K/V views interleaved per token."""
    return _make_elastic_env(monkeypatch, interleaved=True)


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
        enable_kv_cache_copy=False,
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
        self.use_native_move_kv_cache = False
        self.quant_method = quant_method
        self.native_normal_calls = 0
        self.native_quantized_calls = 0
        self.native_warmup_calls = 0
        self.native_kernel_moves = 0
        self._create_buffers()
        # Same gate as the native __init__: spec-decode call sites pass
        # enable_kv_cache_copy=True.
        if enable_kv_cache_copy and not self.use_hnd:
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

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
        # Same formula as sglang v0.5.16..v0.5.20: one scalar per buffer,
        # prod(shape[1:]) * itemsize.
        buffers = [*self.k_buffer, *self.v_buffer]
        self.k_data_ptrs = [t.data_ptr() for t in self.k_buffer]
        self.v_data_ptrs = [t.data_ptr() for t in self.v_buffer]
        self.data_ptrs = [t.data_ptr() for t in buffers]
        self.data_strides = [
            t[0].numel() * t.element_size() for t in buffers
        ]

    def _init_kv_copy_and_warmup(self):
        # Mirrors the native tile-config math and the warmup launch of the
        # copy kernel with loc 0 -> 0.
        if self.layer_num == 0:
            self._kv_copy_config = None
            return
        stride_bytes = int(self.data_strides[0])
        if stride_bytes >= 8192:
            bytes_per_tile = 512
        elif stride_bytes >= 4096:
            bytes_per_tile = 256
        else:
            bytes_per_tile = 128
        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1)
            // bytes_per_tile,
            "num_locs_upper": 128 if bytes_per_tile >= 512 else 256,
        }
        self.native_warmup_calls += 1
        dummy = torch.zeros(1, dtype=torch.int64)
        _emulate_copy_all_layer_kv_cache(
            [*self.k_buffer, *self.v_buffer], self.data_ptrs,
            self.data_strides, dummy, dummy)

    def move_kv_cache(self, tgt_loc, src_loc):
        # Base flow minus the OOB checks and the HND arm (elastic pools
        # refuse HND before buffers exist).
        if self.layer_num == 0:
            return
        self._move_kv_cache_impl(tgt_loc, src_loc)

    def _move_kv_cache_impl(self, tgt_loc, src_loc):
        if self.use_native_move_kv_cache:
            for k_cache, v_cache in zip(self.k_buffer, self.v_buffer):
                k_cache[tgt_loc] = k_cache[src_loc]
                v_cache[tgt_loc] = v_cache[src_loc]
            return
        if tgt_loc.numel() == 0:
            return
        assert self._kv_copy_config is not None, (
            "KV copy not initialized. Set enable_kv_cache_copy=True "
            "in __init__")
        self.native_kernel_moves += 1
        _emulate_copy_all_layer_kv_cache(
            [*self.k_buffer, *self.v_buffer], self.data_ptrs,
            self.data_strides, tgt_loc, src_loc)

    def get_contiguous_buf_infos(self):
        # Mirrors the native (ptrs, lens, item_lens) math for the NHD
        # slot-row layout (tokens_per_row == 1).
        assert not self.use_hnd
        buffers = [*self.k_buffer, *self.v_buffer]
        row_bytes = [t[0].numel() * t.element_size() for t in buffers]
        rows = self.size + self.page_size
        ptrs = [t.data_ptr() for t in buffers]
        lens = [rows * rb for rb in row_bytes]
        item_lens = [self.page_size * rb for rb in row_bytes]
        return ptrs, lens, item_lens

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
    # For independently contiguous rows the copy-kernel contract holds:
    # the recorded scalar equals both the token pitch and the row bytes.
    assert pool.data_strides == [
        t.stride(0) * t.element_size() for t in (*elastic_k, *elastic_v)
    ]


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


def test_native_stride_formula_corrupts_interleaved_copy():
    """CPU mirror of the T4 failure: 2 layers, 2 heads, head_dim 8, fp16.

    The native formula records prod(shape[1:]) * itemsize = 32 bytes while
    the interleaved views' token pitch is 128 bytes. Walking the kernel's
    address math with the recorded scalar leaves every intended
    destination row unchanged and rewrites 64 unrelated elements; the same
    walk over independently contiguous buffers matches the reference.
    """
    tokens, layers, heads, dim = 8, 2, 2, 8
    big = torch.arange(
        tokens * layers * 2 * heads * dim, dtype=torch.float16
    ).reshape(tokens, layers, 2, heads, dim)
    k_views = [big[:, i, 0] for i in range(layers)]
    v_views = [big[:, i, 1] for i in range(layers)]
    buffers = [*k_views, *v_views]

    native_strides = [t[0].numel() * t.element_size() for t in buffers]
    true_pitches = [t.stride(0) * t.element_size() for t in buffers]
    assert native_strides == [32, 32, 32, 32]
    assert true_pitches == [128, 128, 128, 128]

    src = torch.tensor([2], dtype=torch.int64)
    tgt = torch.tensor([6], dtype=torch.int64)
    before = big.clone()
    before_views = [before[:, i, 0] for i in range(layers)]
    before_views += [before[:, i, 1] for i in range(layers)]
    expected_k, expected_v = _reference_move(k_views, v_views, tgt, src)

    _emulate_copy_all_layer_kv_cache(
        buffers, [t.data_ptr() for t in buffers], native_strides, tgt, src)

    for view, old, expected in zip(buffers, before_views,
                                   [*expected_k, *expected_v]):
        # Every intended destination row kept its old value instead of
        # receiving the source row.
        assert torch.equal(view[6], old[6])
        assert not torch.equal(view[6], expected[6])
    changed = int((big != before).sum().item())
    assert changed == 64

    # Control: independently contiguous buffers, same emulated walk.
    contiguous = [t.clone() for t in buffers]
    exp_k, exp_v = _reference_move(contiguous[:layers], contiguous[layers:],
                                   tgt, src)
    _emulate_copy_all_layer_kv_cache(
        contiguous, [t.data_ptr() for t in contiguous],
        [t[0].numel() * t.element_size() for t in contiguous], tgt, src)
    for view, expected in zip(contiguous, [*exp_k, *exp_v]):
        assert torch.equal(view, expected)


def test_seam_interleaved_pool_publishes_no_pointer_tables(
        interleaved_elastic_env):
    module = _inject_mha(SeamMHATokenToKVPool)

    pool = _make_mha_pool(module, enable_kv_cache_copy=True)

    elastic_k, elastic_v = interleaved_elastic_env["mha_buffers"]
    assert pool.k_buffer is elastic_k
    assert pool.v_buffer is elastic_v
    # The tables encode pitch == payload, which interleaved views break,
    # so the pool must not publish them; consumers fail loudly instead.
    for attr in ("data_ptrs", "data_strides", "k_data_ptrs", "v_data_ptrs"):
        assert not hasattr(pool, attr)
    # No copy config and no kernel warmup over the bad tables.
    assert pool._kv_copy_config is None
    assert pool.native_warmup_calls == 0
    # The layout-independent part of the tail still ran.
    assert pool._kv_buffer_descs == [
        tuple(t.shape) for t in (*elastic_k, *elastic_v)
    ]


def test_seam_interleaved_move_kv_cache_matches_reference(
        interleaved_elastic_env):
    module = _inject_mha(SeamMHATokenToKVPool)

    pool = _make_mha_pool(module, enable_kv_cache_copy=True)

    big = interleaved_elastic_env["mha_backing"]
    big.copy_(torch.arange(big.numel(), dtype=big.dtype).reshape(big.shape))
    before = big.clone()
    src = torch.tensor([2, 4], dtype=torch.int64)
    tgt = torch.tensor([6, 1], dtype=torch.int64)
    expected_k, expected_v = _reference_move(pool.k_buffer, pool.v_buffer,
                                             tgt, src)

    pool.move_kv_cache(tgt, src)

    for view, expected in zip((*pool.k_buffer, *pool.v_buffer),
                              (*expected_k, *expected_v)):
        assert torch.equal(view, expected)
    # Untouched token rows stayed intact (no neighbouring-byte writes).
    untouched = [t for t in range(big.shape[0]) if t not in (1, 6)]
    assert torch.equal(big[untouched], before[untouched])
    # The kernel path over the pointer tables must not have run.
    assert pool.native_kernel_moves == 0


def test_seam_contiguous_move_keeps_native_kernel_path(elastic_env):
    module = _inject_mha(SeamMHATokenToKVPool)

    pool = _make_mha_pool(module, enable_kv_cache_copy=True)

    assert pool._kv_copy_config is not None
    assert pool.native_warmup_calls == 1
    for buf in (*pool.k_buffer, *pool.v_buffer):
        buf.copy_(torch.arange(buf.numel(),
                               dtype=buf.dtype).reshape(buf.shape))
    src = torch.tensor([2, 4], dtype=torch.int64)
    tgt = torch.tensor([6, 1], dtype=torch.int64)
    expected_k, expected_v = _reference_move(pool.k_buffer, pool.v_buffer,
                                             tgt, src)

    pool.move_kv_cache(tgt, src)

    assert pool.native_kernel_moves == 1
    for view, expected in zip((*pool.k_buffer, *pool.v_buffer),
                              (*expected_k, *expected_v)):
        assert torch.equal(view, expected)


def test_seam_interleaved_buf_infos_refused(interleaved_elastic_env):
    module = _inject_mha(SeamMHATokenToKVPool)

    pool = _make_mha_pool(module)

    with pytest.raises(NotImplementedError,
                       match="per-layer contiguous regions"):
        pool.get_contiguous_buf_infos()


def test_seam_contiguous_buf_infos_stay_native(elastic_env):
    module = _inject_mha(SeamMHATokenToKVPool)

    pool = _make_mha_pool(module)

    ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
    buffers = (*pool.k_buffer, *pool.v_buffer)
    assert ptrs == [t.data_ptr() for t in buffers]
    assert lens == [
        (pool.size + pool.page_size) * t[0].numel() * t.element_size()
        for t in buffers
    ]
    assert item_lens == [
        pool.page_size * t[0].numel() * t.element_size() for t in buffers
    ]


def test_pre_seam_interleaved_pool_stays_untouched(interleaved_elastic_env):
    module = _inject_mha(PreSeamMHATokenToKVPool)

    pool = _make_mha_pool(module)

    elastic_k, elastic_v = interleaved_elastic_env["mha_buffers"]
    assert pool.k_buffer is elastic_k
    assert pool.v_buffer is elastic_v
    # Pre-0.5.16 never derived pointer tables for elastic pools; the
    # seam-only overrides must not change that.
    for attr in ("data_ptrs", "data_strides", "k_data_ptrs", "v_data_ptrs"):
        assert not hasattr(pool, attr)


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
