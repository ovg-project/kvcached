# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for the contiguous KV layout of hybrid linear-attention
(HYBRID_LINEAR) models in the vLLM integration.

Background
----------
Hybrid linear-attention models (Qwen3-Next / Qwen3.5/3.6 GDN, Jamba, Bamba,
NemotronH, Zamba2, Plamo2) mix full-attention layers with linear-attention
(mamba/SSM) layers. In kvcached's vLLM integration both kinds of layer share one
*unified* physical pool per layer: K and V are interleaved into a single
``page_size_bytes`` cell per block so a mamba layer can reinterpret the same
cell as recurrent state.

For a long time only the *non-contiguous* layout was supported for these models.
This test suite pins the two invariants that make the *contiguous* layout work:

1. ``num_kv_buffers`` reconciliation: the C++ contiguous allocator sizes its
   compound page as ``kPageSize * num_layers * num_kv_buffers``. For the unified
   (hybrid) pool the manager/PageAllocator use ``num_kv_buffers=1``, so
   ``create_kv_tensors`` must be called with ``1`` -- not ``2`` -- or the
   compound page is 2x too large and ``FTensor::map``'s offset-alignment assert
   fails on every odd page id.  (test_num_kv_buffers_*)
2. The restored ``_reshape_mamba_contiguous`` view aliases exactly the same
   physical cell as the contiguous attention view: block N of pool L lives at
   byte ``(N*num_pools + L) * page_size_bytes`` in the single shared base
   buffer.  (test_contiguous_mamba_aliases_attention)

Plus a guard test for the one unsupported combination (contiguous + ratio>1).

These run CPU-only: ``create_kv_tensors`` is intercepted and
``torch.cuda.get_device_properties`` is stubbed, so no GPU / VMM is exercised.
The end-to-end token-parity check against vanilla vLLM still requires a GPU and
a real hybrid model.
"""
import importlib

import pytest
import torch

BLOCK_SIZE = 16
DTYPE = torch.float16
# HYBRID_LINEAR uses the MHA-style FlashAttn shape:
# (2, num_blocks, block_size, head_num, head_dim)
HYBRID_SHAPE = (2, 8, BLOCK_SIZE, 8, 128)
MHA_SHAPE = (2, 8, BLOCK_SIZE, 8, 128)
MLA_SHAPE = (8, BLOCK_SIZE, 576)


class _FakeProps:
    def __init__(self, total_memory):
        self.total_memory = total_memory


class _CapturedCall(Exception):
    """Raised by the create_kv_tensors stub to surface its call arguments."""

    def __init__(self, args, kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs


@pytest.fixture()
def ifc(monkeypatch):
    """The vLLM interfaces module with a fake GPU + intercepted allocator."""
    mod = importlib.import_module("kvcached.integration.vllm.interfaces")
    monkeypatch.setattr(mod, "_kvcached_initialized", True, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda dev=None: _FakeProps(80 * (1024 ** 3)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _stub(*args, **kwargs):
        raise _CapturedCall(args, kwargs)

    monkeypatch.setattr(mod, "create_kv_tensors", _stub)
    return mod, monkeypatch


def _capture_alloc(mod, attention_type, shape=HYBRID_SHAPE, kernel_block_size=None):
    with pytest.raises(_CapturedCall) as ei:
        mod.alloc_kv_cache(shape, BLOCK_SIZE, DTYPE, "cuda:0", 24,
                           attention_type=attention_type,
                           kernel_block_size=kernel_block_size)
    return ei.value


@pytest.mark.parametrize("contiguous", [True, False])
def test_num_kv_buffers_hybrid_is_one(ifc, contiguous):
    """HYBRID_LINEAR must pass num_kv_buffers=1 (unified K+V cell) regardless of
    layout -- the crux fix that keeps the contiguous compound page consistent
    with the manager's num_kv_buffers=1."""
    mod, monkeypatch = ifc
    monkeypatch.setattr(mod, "_contiguous_layout", contiguous)
    call = _capture_alloc(mod, "HYBRID_LINEAR")
    assert call.kwargs["num_kv_buffers"] == 1
    assert call.kwargs["unified_pool"] is True


def test_num_kv_buffers_mha_unchanged(ifc):
    """MHA/GQA still pass num_kv_buffers=2 (separate K and V), unaffected by the
    unified_pool gating."""
    mod, monkeypatch = ifc
    monkeypatch.setattr(mod, "_contiguous_layout", True)
    call = _capture_alloc(mod, "MHA", shape=MHA_SHAPE)
    assert call.kwargs["num_kv_buffers"] == 2
    assert call.kwargs["unified_pool"] is False


def test_num_kv_buffers_mla_unchanged(ifc):
    """MLA still passes num_kv_buffers=1 (combined KV), unaffected."""
    mod, monkeypatch = ifc
    monkeypatch.setattr(mod, "_contiguous_layout", True)
    call = _capture_alloc(mod, "MLA", shape=MLA_SHAPE)
    assert call.kwargs["num_kv_buffers"] == 1
    assert call.kwargs["unified_pool"] is False


def test_hybrid_contiguous_no_longer_raises(ifc):
    """The old ValueError guard is gone: HYBRID_LINEAR + contiguous now proceeds
    all the way to create_kv_tensors instead of failing at startup."""
    mod, monkeypatch = ifc
    monkeypatch.setattr(mod, "_contiguous_layout", True)
    # Reaching _CapturedCall means we passed the (now-removed) guard.
    call = _capture_alloc(mod, "HYBRID_LINEAR")
    assert call.kwargs["unified_pool"] is True


def test_hybrid_contiguous_ratio_gt_1_fails_loud(ifc):
    """contiguous + ratio>1 (kernel_block_size != block_size) is unexpressible
    for the contiguous attention view and must raise NotImplementedError."""
    mod, monkeypatch = ifc
    monkeypatch.setattr(mod, "_contiguous_layout", True)
    with pytest.raises(NotImplementedError, match="contiguous KV layout"):
        mod.alloc_kv_cache(HYBRID_SHAPE, BLOCK_SIZE, DTYPE, "cuda:0", 24,
                           attention_type="HYBRID_LINEAR",
                           kernel_block_size=BLOCK_SIZE // 2)


def test_hybrid_noncontiguous_ratio_gt_1_ok(ifc):
    """The ratio>1 guard is contiguous-only: non-contiguous hybrid with ratio>1
    still reaches create_kv_tensors (non-contiguous handles kernel blocks)."""
    mod, monkeypatch = ifc
    monkeypatch.setattr(mod, "_contiguous_layout", False)
    # Should NOT raise NotImplementedError; should reach the allocator stub.
    call = _capture_alloc(mod, "HYBRID_LINEAR",
                          kernel_block_size=BLOCK_SIZE // 2)
    assert call.kwargs["num_kv_buffers"] == 1


def test_contiguous_mamba_aliases_attention():
    """The restored contiguous mamba view and the contiguous attention view
    address the SAME physical cell: block N of pool L at
    (N*num_pools + L)*page_size_bytes. Written through the attention view, read
    back through the mamba view -> identical value.
    """
    from kvcached.integration.vllm.patches import _reshape_mamba_contiguous

    num_pools = 3          # layers
    num_blocks = 4
    head, dim = 1, 8
    block_size = 1
    itemsize = torch.finfo(DTYPE).bits // 8  # 2 for fp16
    hidden = block_size * head * dim         # 8 elems
    page_size_bytes = 2 * hidden * itemsize  # K+V cell = 32 bytes

    # Single shared base int8 buffer (interleaved across all pools & blocks).
    base = torch.zeros(num_blocks * num_pools * page_size_bytes,
                       dtype=torch.int8)

    # --- Contiguous ATTENTION view (mirrors interfaces.py contiguous branch) ---
    # contiguous_shape = [num_blocks, num_layers, 2, block_size, head, dim]
    contiguous_shape = [num_blocks, num_pools, 2, block_size, head, dim]
    n_eles = 1
    for s in contiguous_shape:
        n_eles *= s
    contiguous_tensor = base.view(DTYPE)[:n_eles].view(contiguous_shape)
    # attention view for layer L: contiguous_tensor[:, L].permute([1,0,2,3,4])
    attn_views = [contiguous_tensor[:, L].permute(1, 0, 2, 3, 4)
                  for L in range(num_pools)]

    # --- Contiguous MAMBA view via the restored function ---
    class _FakeSpec:
        # Two state components summing to <= page_size_bytes (K half here).
        shapes = [(hidden,)]          # 8 fp16 = 16 bytes, fits in the K region
        dtypes = [DTYPE]

    def _get_dtype_size(dt):
        return torch.finfo(dt).bits // 8

    mamba_info = {
        "buffers": [base],
        "num_blocks": num_blocks,
        "page_size_bytes": page_size_bytes,
        "block_stride_bytes": num_pools * page_size_bytes,
    }

    for L in range(num_pools):
        mamba_view = _reshape_mamba_contiguous(
            mamba_info, _FakeSpec(), L, _get_dtype_size)[0]
        for N in range(num_blocks):
            # Write a unique sentinel through the ATTENTION view at the K slot
            # (kv=0), block N, first element.
            sentinel = float(100 + N * 10 + L)
            attn_views[L][0, N, 0, 0, 0] = sentinel
            # The mamba view's block N element 0 aliases the same K byte.
            assert float(mamba_view[N, 0]) == sentinel, (
                f"aliasing mismatch at pool={L} block={N}: "
                f"mamba={float(mamba_view[N, 0])} != attn={sentinel}")
            # And the physical byte offset of block N must be
            # (N*num_pools+L)*page_size_bytes (component 0 starts at the cell
            # base, inner_offset=0).
            expected_byte = (N * num_pools + L) * page_size_bytes
            got_byte = mamba_view[N].storage_offset() * itemsize
            assert got_byte == expected_byte, (
                f"pool={L} block={N}: storage offset {got_byte} != "
                f"{expected_byte}")


def test_contiguous_mamba_blocks_are_interleaved_by_num_pools():
    """Consecutive pools for the same block are exactly page_size_bytes apart,
    and consecutive blocks of one pool are num_pools*page_size_bytes apart --
    i.e. the pools are interleaved, which is what distinguishes the contiguous
    layout from the non-contiguous (per-pool compact) one."""
    from kvcached.integration.vllm.patches import _reshape_mamba_contiguous

    num_pools, num_blocks = 4, 5
    page_size_bytes = 64
    itemsize = 2

    base = torch.zeros(num_blocks * num_pools * page_size_bytes,
                       dtype=torch.int8)

    class _FakeSpec:
        shapes = [(8,)]
        dtypes = [DTYPE]

    mamba_info = {
        "buffers": [base],
        "num_blocks": num_blocks,
        "page_size_bytes": page_size_bytes,
        "block_stride_bytes": num_pools * page_size_bytes,
    }

    views = [_reshape_mamba_contiguous(mamba_info, _FakeSpec(), L,
                                       lambda dt: itemsize)[0]
             for L in range(num_pools)]

    # Same block N, adjacent pools -> page_size_bytes apart.
    for N in range(num_blocks):
        for L in range(num_pools - 1):
            off_l = views[L][N].storage_offset() * itemsize
            off_l1 = views[L + 1][N].storage_offset() * itemsize
            assert off_l1 - off_l == page_size_bytes

    # Same pool L, adjacent blocks -> num_pools*page_size_bytes apart.
    for L in range(num_pools):
        for N in range(num_blocks - 1):
            off_n = views[L][N].storage_offset() * itemsize
            off_n1 = views[L][N + 1].storage_offset() * itemsize
            assert off_n1 - off_n == num_pools * page_size_bytes


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
