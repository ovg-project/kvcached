# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Real CPU tensor views for the V1 Mamba binding contract in vLLM 0.28."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from packaging.version import Version

from kvcached.integration.vllm import patches


def _pool(contiguous):
    blocks, pools, page_bytes = 4, 3, 64
    # Nonzero storage offset catches accidental rebasing in as_strided.
    buffers = [torch.zeros(blocks * page_bytes * (pools if contiguous else 1)
                           + 16, dtype=torch.int8)[16:]
               for _ in range(1 if contiguous else pools)]
    return dict(buffers=buffers, num_blocks=blocks, num_pools=pools,
                page_size_bytes=page_bytes, is_contiguous=contiguous,
                block_stride_bytes=page_bytes * (pools if contiguous else 1))


@pytest.mark.parametrize("contiguous", [False, True])
def test_page_views_alias_each_block_and_layer(contiguous):
    info = _pool(contiguous)
    spec = SimpleNamespace(page_size_bytes=64)
    views = [patches._reshape_mamba_page_tensor(info, spec, i) for i in range(3)]
    for layer, view in enumerate(views):
        assert view.shape == (4, 1, 1, 64)
        assert view.dtype == torch.int8
        assert view.stride(0) == info["block_stride_bytes"]
        pages = view.squeeze(dim=(1, 2))
        # Mirror native binding with mixed state dtypes and page padding.
        conv = pages[:, :16].view(torch.float16).view(4, 2, 4)
        ssm = pages[:, 16:48].view(torch.float32).view(4, 2, 4)
        for block in range(4):
            conv[block].fill_(layer * 10 + block + 1)
            ssm[block].fill_(layer * 10 + block + 101)
    for layer in range(3):
        raw = info["buffers"][0 if contiguous else layer]
        for block in range(4):
            start = ((block * 3 + layer) if contiguous else block) * 64
            cell = raw[start:start + 64]
            assert (cell[:16].view(torch.float16) == layer * 10 + block + 1).all()
            assert (cell[16:48].view(torch.float32) == layer * 10 + block + 101).all()
            assert (cell[48:] == 0).all()


@pytest.mark.parametrize("size", [0, 32, 128])
def test_page_geometry_mismatch_is_rejected(size):
    with pytest.raises(ValueError, match="page size"):
        patches._reshape_mamba_page_tensor(
            _pool(False), SimpleNamespace(page_size_bytes=size), 0,
        )


@pytest.mark.parametrize("buffer", [torch.zeros(256), torch.zeros((4, 64), dtype=torch.int8),
                                    torch.zeros(512, dtype=torch.int8)[::2]])
def test_non_byte_backing_is_rejected(buffer):
    info = _pool(False)
    info["buffers"][0] = buffer
    with pytest.raises(ValueError, match="flat int8"):
        patches._reshape_mamba_page_tensor(info, SimpleNamespace(page_size_bytes=64), 0)


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("version", ["0.22.1", "0.27.0", "0.28.0", "0.28.0+cu129"])
def test_v1_reshape_uses_versioned_binding_contract(monkeypatch, contiguous, version):
    class Runner(SimpleNamespace):
        pass

    spec = SimpleNamespace(page_size_bytes=64,
                           shapes=((2, 4), (2, 4)),
                           dtypes=(torch.float16, torch.float32))
    config = SimpleNamespace(kv_cache_groups=[
        SimpleNamespace(kv_cache_spec=spec, layer_names=["m0", "m1", "profile"]),
        SimpleNamespace(kv_cache_spec=object(), layer_names=["attn"]),
    ])
    monkeypatch.setattr(patches, "_is_mamba_spec", lambda value: value is spec)
    torch_utils = ModuleType("vllm.utils.torch_utils")
    setattr(torch_utils, "get_dtype_size", lambda dtype: dtype.itemsize)
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)
    patch = patches.GPUModelRunnerPatch()
    patch.detected_version = version
    assert patch.add_reshape_methods(Runner)
    runner = Runner()
    runner._kvcached_mamba_raw_info = _pool(contiguous)
    runner.runner_only_attn_layers = {"profile"}
    attention_view = torch.zeros(1)
    caches = runner._reshape_kv_cache_tensors_from_kvcached(config, [attention_view])
    assert set(caches) == {"m0", "m1", "attn"}
    assert caches["attn"] is attention_view
    for name in ("m0", "m1"):
        if Version(version) >= Version("0.28.0"):
            assert isinstance(caches[name], torch.Tensor)
            assert caches[name].shape == (4, 1, 1, 64)
        else:
            assert isinstance(caches[name], list)
            assert [state.dtype for state in caches[name]] == list(spec.dtypes)
            assert [tuple(state.shape) for state in caches[name]] == [(4, 2, 4)] * 2
