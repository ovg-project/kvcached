# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for heterogeneous attention KV groups under BOTH KV layouts.

Background
----------
Gemma 3/4 interleave sliding-window and full-attention layers whose per-token
KV geometry differs -- sliding layers use ``head_dim=256`` over 8 KV heads,
full-attention layers use ``head_dim=512`` over 1 KV head -- while vLLM equalizes
the *physical* bytes per block (65536 B in both cases). kvcached therefore
allocates one uniform pool and gives each group its own ``as_strided`` view over
it (``build_kv_views``).

Until this test landed, the vLLM patch refused to start when that shape met the
*contiguous* layout, on the assumption that only the non-contiguous branch could
express per-group geometry. That assumption was never measured. It is wrong: the
shared ``block_mem_size`` is exactly what makes both branches work, because block
N occupies the same bytes whichever group's view addresses it.

These tests pin that property directly -- they are what makes it safe to run
either layout -- and they run CPU-only over plain tensors, no GPU or VMM.
The end-to-end token-parity check against vanilla vLLM still needs a GPU.
"""
import importlib

import pytest
import torch

DTYPE = torch.float16
NUM_LAYERS = 4
NUM_BLOCKS = 8

# Gemma-shaped pair: different (block_size, num_kv_heads, head_size), identical
# bytes per block. FlashAttn MHA shape is (2, num_blocks, block_size, H, D).
SLIDING = {"block_size": 16, "num_kv_heads": 8, "head_size": 256}
FULL = {"block_size": 64, "num_kv_heads": 1, "head_size": 512}

BLOCK_MEM_BYTES = (SLIDING["block_size"] * SLIDING["num_kv_heads"] *
                   SLIDING["head_size"] * DTYPE.itemsize)
BYTES_PER_LAYER_K_OR_V = NUM_BLOCKS * BLOCK_MEM_BYTES


def _shape(group):
    return (2, NUM_BLOCKS, group["block_size"], group["num_kv_heads"],
            group["head_size"])


@pytest.fixture()
def ifc():
    return importlib.import_module("kvcached.integration.vllm.interfaces")


def _raw_pools(contiguous):
    """Allocate the raw pool(s) the way alloc_kv_cache would."""
    eles_per_layer = BYTES_PER_LAYER_K_OR_V * 2 // DTYPE.itemsize
    if contiguous:
        # One compound buffer covering every layer.
        return [torch.zeros(eles_per_layer * NUM_LAYERS, dtype=DTYPE)]
    return [torch.zeros(eles_per_layer, dtype=DTYPE) for _ in range(NUM_LAYERS)]


def _views(ifc, monkeypatch, group, raw, contiguous):
    monkeypatch.setattr(ifc, "_contiguous_layout", contiguous)
    views, _ = ifc.build_kv_views(
        raw, _shape(group), group["block_size"], DTYPE, "MHA",
        NUM_BLOCKS, BYTES_PER_LAYER_K_OR_V, NUM_LAYERS,
    )
    return views


def test_the_two_groups_really_are_heterogeneous():
    """Guard the premise: different geometry, identical bytes per block.

    If a future shape change breaks this, the other tests would still pass while
    testing nothing interesting.
    """
    assert (SLIDING["block_size"], SLIDING["num_kv_heads"],
            SLIDING["head_size"]) != (FULL["block_size"],
                                      FULL["num_kv_heads"], FULL["head_size"])
    full_block_bytes = (FULL["block_size"] * FULL["num_kv_heads"] *
                        FULL["head_size"] * DTYPE.itemsize)
    assert full_block_bytes == BLOCK_MEM_BYTES


@pytest.mark.parametrize("contiguous", [True, False])
def test_hetero_groups_alias_the_same_block_bytes(ifc, monkeypatch, contiguous):
    """Block N of layer L must land on the same bytes in both groups' views.

    This is the whole premise of the shared pool, and it is what the removed
    contiguous-layout guard was hedging against.
    """
    raw = _raw_pools(contiguous)
    sliding = _views(ifc, monkeypatch, SLIDING, raw, contiguous)
    full = _views(ifc, monkeypatch, FULL, raw, contiguous)

    assert len(sliding) == len(full) == NUM_LAYERS
    for layer in range(NUM_LAYERS):
        for k_or_v in (0, 1):
            for block in range(NUM_BLOCKS):
                a = sliding[layer][k_or_v][block]
                b = full[layer][k_or_v][block]
                assert a.data_ptr() == b.data_ptr(), (
                    f"layer={layer} kv={k_or_v} block={block} not aliased "
                    f"(contiguous={contiguous})")
                assert a.numel() == b.numel()


@pytest.mark.parametrize("contiguous", [True, False])
def test_writes_through_one_group_are_visible_through_the_other(
        ifc, monkeypatch, contiguous):
    """Aliasing is real memory sharing, not just equal addresses."""
    raw = _raw_pools(contiguous)
    sliding = _views(ifc, monkeypatch, SLIDING, raw, contiguous)
    full = _views(ifc, monkeypatch, FULL, raw, contiguous)

    sliding[2][0][5].fill_(0.0)
    sliding[2][0][5].view(-1)[7] = 1.5
    assert full[2][0][5].reshape(-1)[7].item() == pytest.approx(1.5)


@pytest.mark.parametrize("contiguous", [True, False])
def test_both_groups_agree_on_the_block_stride(ifc, monkeypatch, contiguous):
    """Both groups must derive the same inter-block stride.

    The value differs by layout -- non-contiguous packs one layer's blocks back
    to back, while the contiguous compound page interleaves every layer's K and
    V, so one layer's blocks are ``num_layers * num_kv_buffers`` further apart.
    What matters is that the two geometries agree, since that is what lets one
    physical pool serve both.
    """
    expected = BLOCK_MEM_BYTES * (NUM_LAYERS * 2 if contiguous else 1)
    raw = _raw_pools(contiguous)
    for group in (SLIDING, FULL):
        views = _views(ifc, monkeypatch, group, raw, contiguous)
        v = views[1]
        step = v[0][3].data_ptr() - v[0][2].data_ptr()
        assert step == expected, (
            f"group={group} contiguous={contiguous} stride={step}")


@pytest.mark.parametrize("contiguous", [True, False])
def test_distinct_layers_do_not_overlap(ifc, monkeypatch, contiguous):
    """Different layers must address disjoint memory for the same block id."""
    raw = _raw_pools(contiguous)
    views = _views(ifc, monkeypatch, SLIDING, raw, contiguous)
    ptrs = {views[layer][0][0].data_ptr() for layer in range(NUM_LAYERS)}
    assert len(ptrs) == NUM_LAYERS
