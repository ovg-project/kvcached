# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Regression test for MLA ftensor alignment in ``alloc_kv_cache``.

The C++ allocator's ``get_v_base_offset()`` splits each per-layer ftensor in
half for the K/V regions and asserts the ftensor byte size is a multiple of
``2 * PAGE_SIZE``.  The ftensor byte size equals
``gpu_mem_bytes_per_layer_k_or_v * num_k_or_v``:

* MLA  (num_k_or_v == 1): the value must be aligned to ``2 * PAGE_SIZE``
  *directly* -- ``PAGE_SIZE`` alignment alone can leave an odd multiple of
  ``PAGE_SIZE`` and trip the assert (the bug fixed for vLLM in #349 and for
  SGLang here).
* MHA/GQA (num_k_or_v == 2): the ``* 2`` makes ``PAGE_SIZE`` alignment
  automatically ``2 * PAGE_SIZE``-aligned.

This pins the invariant for both the vLLM and SGLang integrations without a
GPU: it stubs ``torch.cuda.get_device_properties`` and intercepts the ``size``
argument passed to ``create_kv_tensors`` (== ftensor_bytes_per_layer).
"""
import importlib

import pytest
import torch

from kvcached.utils import PAGE_SIZE

ALIGN = 2 * PAGE_SIZE

# (gpu_gb, num_layers).  (20, 48) and (80, 81) are witnesses where the old
# PAGE_SIZE-only alignment leaves an odd multiple of PAGE_SIZE for MLA.
CONFIGS = [(8, 24), (8, 48), (20, 48), (24, 32), (40, 61), (80, 81), (16, 17)]
INTEGRATIONS = ["vllm", "sglang"]
ATTENTION_TYPES = ["MLA", "MHA"]
BLOCK_SIZE = 16
DTYPE = torch.float16


class _CapturedSize(Exception):
    """Raised by the create_kv_tensors stub to surface its ``size`` argument."""

    def __init__(self, size):
        super().__init__(size)
        self.size = size


class _FakeProps:
    def __init__(self, total_memory):
        self.total_memory = total_memory


def _kvcache_shape(integration, attention_type):
    """A shape valid enough to reach the create_kv_tensors call."""
    if attention_type == "MLA":
        # (num_blocks, block_size, head_size)
        return (8, BLOCK_SIZE, 576)
    if integration == "vllm":
        # FlashAttn MHA: (2, num_blocks, block_size, head_num, head_dim)
        return (2, 8, BLOCK_SIZE, 8, 128)
    # SGLang MHA, NHD layout: (num_tokens, head_num, head_dim)
    return (1024, 8, 128)


def _call_alloc(mod, integration, attention_type, shape, num_layers):
    if integration == "vllm":
        return mod.alloc_kv_cache(shape, BLOCK_SIZE, DTYPE, "cuda:0", num_layers,
                                  attention_type=attention_type)
    return mod.alloc_kv_cache(shape, DTYPE, "cuda:0", num_layers,
                              page_size=BLOCK_SIZE, attention_type=attention_type)


@pytest.mark.parametrize("integration", INTEGRATIONS)
@pytest.mark.parametrize("attention_type", ATTENTION_TYPES)
@pytest.mark.parametrize("gpu_gb,num_layers", CONFIGS)
def test_ftensor_bytes_aligned_to_2x_page_size(monkeypatch, integration,
                                               attention_type, gpu_gb, num_layers):
    mod = importlib.import_module(f"kvcached.integration.{integration}.interfaces")
    gpu_mem_bytes = gpu_gb * (1024 ** 3)

    monkeypatch.setattr(mod, "_kvcached_initialized", True, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda dev=None: _FakeProps(gpu_mem_bytes))

    def _fake_create_kv_tensors(size, *args, **kwargs):
        raise _CapturedSize(size)

    monkeypatch.setattr(mod, "create_kv_tensors", _fake_create_kv_tensors)

    shape = _kvcache_shape(integration, attention_type)
    with pytest.raises(_CapturedSize) as excinfo:
        _call_alloc(mod, integration, attention_type, shape, num_layers)

    ftensor_bytes_per_layer = excinfo.value.size
    assert ftensor_bytes_per_layer % ALIGN == 0, (
        f"{integration}/{attention_type} gpu={gpu_gb}GB layers={num_layers}: "
        f"ftensor_bytes_per_layer={ftensor_bytes_per_layer} is not a multiple of "
        f"2*PAGE_SIZE={ALIGN}")


def test_sweep_includes_pre_fix_failure_witness():
    """Guard against a vacuous invariant: at least one MLA config in the sweep
    must be one where the old PAGE_SIZE-only alignment would have failed."""
    def old_mla_ftensor_bytes(gpu_bytes, num_layers):
        per_layer = gpu_bytes // num_layers // 1  # num_k_or_v == 1 for MLA
        return (per_layer // PAGE_SIZE) * PAGE_SIZE

    witnesses = [
        (gpu_gb, num_layers) for (gpu_gb, num_layers) in CONFIGS
        if old_mla_ftensor_bytes(gpu_gb * (1024 ** 3), num_layers) % ALIGN != 0
    ]
    assert witnesses, (
        "No config exercises the pre-fix failure mode; the alignment test would "
        "pass even without the fix. Add a (gpu_gb, num_layers) that yields an odd "
        "multiple of PAGE_SIZE for MLA.")
