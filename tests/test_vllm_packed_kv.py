# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Packed KV views and their physical-buffer contract for vLLM 0.28."""

import importlib
import math
from types import SimpleNamespace

import pytest
import torch

BLOCK_SIZE = 16
LAYERS = 3


@pytest.fixture
def interfaces(monkeypatch):
    mod = importlib.import_module("kvcached.integration.vllm.interfaces")
    monkeypatch.setattr(mod, "_kvcached_initialized", True)
    monkeypatch.setattr(mod, "PAGE_SIZE", 256)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(total_memory=24 * 1024))

    def create(size, itemsize, device, num_layers, **kwargs):
        assert kwargs["unified_pool"] is True
        assert kwargs["num_kv_buffers"] == 1
        if mod._contiguous_layout:
            return [torch.zeros(size * num_layers, dtype=torch.uint8)]
        return [torch.zeros(size, dtype=torch.uint8) for _ in range(num_layers)]

    monkeypatch.setattr(mod, "create_kv_tensors", create)
    return mod


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("ratio", [1, 2, 4])
@pytest.mark.parametrize("attention_type", ["MHA", "HYBRID_LINEAR"])
def test_packed_views_keep_every_block_and_layer_distinct(
    interfaces, monkeypatch, contiguous, layout, ratio, attention_type,
):
    mod = interfaces
    monkeypatch.setattr(mod, "_contiguous_layout", contiguous)
    shape = (2, 2, BLOCK_SIZE, 16)
    kernel_size = BLOCK_SIZE // ratio
    result = mod.alloc_kv_cache(
        shape, BLOCK_SIZE, torch.float16, "cuda:0", LAYERS,
        attention_type=attention_type, kv_layout=layout,
        kernel_block_size=kernel_size, return_meta=True,
    )
    views, meta = result[0], result[-1]
    count = meta["num_blocks_per_layer"]
    elements = shape[1] * kernel_size * shape[3]
    stride = elements * (LAYERS if contiguous else 1)
    for layer, view in enumerate(views):
        assert tuple(view.shape) == (count * ratio, 2, kernel_size, 16)
        assert view.stride(0) == stride
        assert view.stride(3) == 1
        assert view.stride(1) == (16 if layout == "NHD" else kernel_size * 16)
        assert view.stride(2) == (32 if layout == "NHD" else 16)
        for block in range(count * ratio):
            for head in range(2):
                view[block, head, :, :8].fill_(layer * 100 + block * 4 + head)
                view[block, head, :, 8:].fill_(layer * 100 + block * 4 + head + 2)

    # Check after all writes, so cross-layer and cross-block aliasing is visible.
    for layer, view in enumerate(views):
        raw = meta["raw_kv_tensors"][0 if contiguous else layer].view(torch.float16)
        for block in range(count * ratio):
            base = block * stride + (layer * elements if contiguous else 0)
            logical_block = block // ratio
            assert base >= logical_block * ratio * stride
            assert base + elements <= (logical_block + 1) * ratio * stride
            native_shape = (kernel_size, 2, 16) if layout == "NHD" else (2, kernel_size, 16)
            native = raw[base:base + elements].view(native_shape)
            if layout == "NHD":
                native = native.permute(1, 0, 2)
            assert torch.equal(native, view[block])
            for head in range(2):
                assert (native[head, :, :8] == layer * 100 + block * 4 + head).all()
                assert (native[head, :, 8:] == layer * 100 + block * 4 + head + 2).all()

    rebuilt, page_bytes = mod.build_kv_views(
        meta["raw_kv_tensors"], shape, BLOCK_SIZE, torch.float16, attention_type,
        count, meta["gpu_mem_bytes_per_layer_k_or_v"], LAYERS,
        kernel_block_size=kernel_size, kv_layout=layout,
    )
    assert page_bytes == math.prod(shape[1:]) * 2
    for original, copy in zip(views, rebuilt):
        assert original.data_ptr() == copy.data_ptr()
        assert original.stride() == copy.stride()
        assert torch.equal(original, copy)


@pytest.mark.parametrize("shape", [(2, 2, 8, 16), (2, 2, 16, 15), (2, 0, 16, 16)])
def test_invalid_packed_shape_fails_before_native_allocation(interfaces, monkeypatch, shape):
    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid geometry reached the native allocator")

    monkeypatch.setattr(interfaces, "create_kv_tensors", unexpected)
    with pytest.raises(ValueError, match="Unsupported packed KV cache shape"):
        interfaces.alloc_kv_cache(shape, BLOCK_SIZE, torch.float16, "cuda:0", LAYERS)


def test_invalid_packed_stride_layout_is_rejected(interfaces):
    with pytest.raises(ValueError, match="KV layout"):
        interfaces.alloc_kv_cache((2, 2, 16, 16), BLOCK_SIZE, torch.float16,
                                 "cuda:0", LAYERS, kv_layout="unknown")


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_native_packed_scales_preserve_layer_offsets(interfaces, monkeypatch, contiguous, layout):
    backend = pytest.importorskip("vllm.v1.attention.backends.triton_attn")
    from kvcached.integration.vllm.patches import _uses_packed_attention_kv
    if not _uses_packed_attention_kv():
        pytest.skip("Native packed scale handling requires vLLM 0.28")
    monkeypatch.setattr(interfaces, "_contiguous_layout", contiguous)
    views = interfaces.alloc_kv_cache((2, 2, 16, 24), BLOCK_SIZE, torch.uint8,
                                     "cuda:0", LAYERS, kv_layout=layout)
    runners = []
    for layer, view in enumerate(views):
        runner = SimpleNamespace(_k_scale_cache=None, _v_scale_cache=None)
        backend.TritonAttentionImpl._ensure_scale_caches(runner, view)
        runner._k_scale_cache.fill_(layer + 1)
        runner._v_scale_cache.fill_(layer + 11)
        runners.append(runner)
    for layer, (runner, view) in enumerate(zip(runners, views)):
        assert (runner._k_scale_cache == layer + 1).all()
        assert (runner._v_scale_cache == layer + 11).all()
        assert not view[..., :8].any()
        assert not view[..., 12:20].any()
