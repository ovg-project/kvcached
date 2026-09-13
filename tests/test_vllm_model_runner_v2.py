# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""CPU tensors with actual vLLM 0.29 view helpers; no GPU or native VMM needed."""

import importlib.metadata
import logging
import sys
import types

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
if not importlib.metadata.version("vllm").startswith("0.29."):
    pytest.skip("requires the vLLM 0.29 view contract", allow_module_level=True)

import torch
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from kvcached.integration.vllm import model_runner_v2 as adapter


def uniform_config(contiguous):
    spec = FullAttentionSpec(block_size=2, num_kv_heads=2, head_size=4, dtype=torch.bfloat16)
    assert spec.page_size_bytes == 64
    groups = [KVCacheGroupSpec(["a", "b"], spec), KVCacheGroupSpec(["alias_a"], spec)]
    tensors = [
        KVCacheTensor(
            size=384, layers=group.layer_names,
            layer_stride=64 if contiguous else 192,
            block_stride=128 if contiguous else 64,
        )
        for group in groups
    ]
    return KVCacheConfig(num_blocks=3, kv_cache_tensors=tensors, kv_cache_groups=groups)


def mock_native_allocator(monkeypatch, contiguous):
    monkeypatch.setattr(adapter, "CONTIGUOUS_LAYOUT", contiguous)
    monkeypatch.setattr(adapter, "PAGE_SIZE", 64)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: types.SimpleNamespace(total_memory=1024))
    captured = []

    def create(size, dtype_size, device, num_layers, **kwargs):
        captured.append((size, dtype_size, device, num_layers, kwargs))
        if contiguous:
            return [torch.zeros(size * num_layers, dtype=torch.int8)]
        return [torch.zeros(size, dtype=torch.int8) for _ in range(num_layers)]

    interfaces = types.ModuleType("kvcached.integration.vllm.interfaces")
    setattr(interfaces, "_kvcached_initialized", True)
    setattr(interfaces, "create_kv_tensors", create)
    setattr(interfaces, "logger", logging.getLogger(__name__))
    monkeypatch.setitem(sys.modules, interfaces.__name__, interfaces)
    import kvcached.integration.vllm as package

    monkeypatch.setattr(package, "interfaces", interfaces, raising=False)
    return captured


@pytest.mark.parametrize("layout_name", ["BLNHC", "BLHNC", "LBNHC", "LBHNC"])
def test_native_views_preserve_block_layer_and_group_aliases(monkeypatch, layout_name):
    contiguous = layout_name.startswith("B")
    captured = mock_native_allocator(monkeypatch, contiguous)
    config = uniform_config(contiguous)
    geometry = adapter.cache_geometry(config)
    caches = adapter.allocate_kv_cache(config, torch.device("cpu"), KVCacheLayout[layout_name], [2, 2])
    assert geometry.block_size * geometry.cell_size == 64
    assert geometry.num_pools == 2
    assert captured == [(512, 1, "cpu", 2, {"num_kv_buffers": 1, "unified_pool": True})]
    for layer_index, name in enumerate(("a", "b")):
        assert caches[name].shape == (3, 2, 2, 8)
        for block in range(3):
            caches[name][block, :, :, :4].fill_(10 * layer_index + block + 1)
            caches[name][block, :, :, 4:].fill_(100 + 10 * layer_index + block)
    for layer_index, name in enumerate(("a", "b")):
        for block in range(3):
            assert torch.all(caches[name][block, :, :, :4] == 10 * layer_index + block + 1)
            assert torch.all(caches[name][block, :, :, 4:] == 100 + 10 * layer_index + block)
    assert torch.equal(caches["alias_a"], caches["a"])
    caches["alias_a"][1].fill_(55)
    assert torch.all(caches["a"][1] == 55)
    assert torch.all(caches["b"][1, :, :, :4] == 12)


@pytest.mark.parametrize("layout_name", ["BLNHC", "LBNHC"])
def test_mla_native_views_preserve_latent_vectors_and_neighboring_pages(monkeypatch, layout_name):
    contiguous = layout_name.startswith("B")
    captured = mock_native_allocator(monkeypatch, contiguous)
    monkeypatch.setattr(adapter, "PAGE_SIZE", 2 * 1024**2)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: types.SimpleNamespace(total_memory=4 * 1024**2))
    # DeepSeek-V2 stores a 512-element latent vector and 64 RoPE elements,
    # without a separate V buffer.
    spec = MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[KVCacheTensor(
            size=442368, layers=["a", "b"],
            layer_stride=73728 if contiguous else 221184,
            block_stride=147456 if contiguous else 73728,
        )],
        kv_cache_groups=[KVCacheGroupSpec(["a", "b"], spec)],
    )
    assert adapter.cache_geometry(config) == adapter.CacheGeometry(64, 73728, 2)
    caches = adapter.allocate_kv_cache(config, torch.device("cpu"), KVCacheLayout[layout_name], [64])
    assert captured == [(2 * 1024**2, 1, "cpu", 2, {"num_kv_buffers": 1, "unified_pool": True})]
    for layer, name in enumerate(("a", "b")):
        assert caches[name].shape == (3, 1, 64, 576)
        for block in range(3):
            caches[name][block, ..., :512].fill_(10 * layer + block + 1)
            caches[name][block, ..., 512:].fill_(100 + 10 * layer + block)
    # Rewriting one latent page must not overwrite RoPE or a live neighbor.
    caches["a"][1, ..., :512].fill_(55)
    for layer, name in enumerate(("a", "b")):
        for block in range(3):
            latent = 55 if (name, block) == ("a", 1) else 10 * layer + block + 1
            assert torch.all(caches[name][block, ..., :512] == latent)
            assert torch.all(caches[name][block, ..., 512:] == 100 + 10 * layer + block)


def mixed_config(layout_name):
    small = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=4, dtype=torch.bfloat16)
    large = FullAttentionSpec(block_size=2, num_kv_heads=3, head_size=4, dtype=torch.bfloat16)
    sliding = SlidingWindowSpec(
        block_size=2, num_kv_heads=2, head_size=4, dtype=torch.bfloat16, sliding_window=8,
    )
    groups = [
        KVCacheGroupSpec(
            ["a", "b"], UniformTypeKVCacheSpecs(block_size=2, kv_cache_specs={"a": small, "b": large}),
        ),
        KVCacheGroupSpec(["c"], sliding),
    ]
    config = types.SimpleNamespace(cache_config=types.SimpleNamespace(
        get_resolved_kv_cache_layout=lambda: KVCacheLayout[layout_name],
        num_gpu_blocks_override=3,
        prefix_cache_retention_interval=None,
    ))
    return get_kv_cache_config_from_groups(config, groups, available_memory=384)


@pytest.mark.parametrize("layout_name", ["BLNHC", "BLHNC"])
def test_mixed_native_packing_survives_scheduler_collapse_and_group_reuse(monkeypatch, layout_name):
    captured = mock_native_allocator(monkeypatch, True)
    config = mixed_config(layout_name)
    scheduler_config = generate_scheduler_kv_cache_config([config])
    assert isinstance(config.kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs)
    assert isinstance(scheduler_config.kv_cache_groups[0].kv_cache_spec, FullAttentionSpec)
    geometry = adapter.cache_geometry(config)
    assert adapter.cache_geometry(scheduler_config) == geometry
    # The shared block holds 32 + 96 bytes, not two copies of the first layer.
    assert geometry.page_bytes * geometry.num_pools == 128
    caches = adapter.allocate_kv_cache(config, torch.device("cpu"), KVCacheLayout[layout_name], [2, 2])
    assert captured == [(512, 1, "cpu", 2, {"num_kv_buffers": 1, "unified_pool": True})]
    assert caches["b"].data_ptr() - caches["a"].data_ptr() == 32
    assert all(cache.stride(0) * cache.element_size() == 128 for cache in caches.values())

    # The full-attention group owns blocks 0/2, while the sliding group owns 1.
    for block in (0, 2):
        caches["a"][block].fill_(10 + block)
        caches["b"][block].fill_(20 + block)
    caches["c"][1].fill_(30)
    for block in (0, 2):
        assert torch.all(caches["a"][block] == 10 + block)
        assert torch.all(caches["b"][block] == 20 + block)
    assert torch.all(caches["c"][1] == 30)

    # After releasing block 0, the other group reuses its overlaid bytes.
    caches["c"][0].fill_(40)
    assert torch.all(caches["a"][0] == 40)
    assert torch.count_nonzero(caches["b"][0] == 40) == 16
    assert torch.count_nonzero(caches["b"][0] == 20) == 32
    assert torch.all(caches["b"][2] == 22)
    assert torch.all(caches["c"][1] == 30)


def test_mixed_packing_rejects_incompatible_physical_placement(monkeypatch):
    config = mixed_config("BLNHC")
    monkeypatch.setattr(adapter, "CONTIGUOUS_LAYOUT", False)
    with pytest.raises(adapter.KVCachedConfigError, match="require uniform pages"):
        adapter.cache_geometry(config)
    monkeypatch.setattr(adapter, "CONTIGUOUS_LAYOUT", True)
    # Moving the large layer would overwrite the next group's block ID.
    config.kv_cache_tensors[1].offset += 32
    with pytest.raises(adapter.KVCachedConfigError, match="exceeds its packed block"):
        adapter.cache_geometry(config)


def test_mixed_packing_retains_native_kernel_split_rejection(monkeypatch):
    mock_native_allocator(monkeypatch, True)
    with pytest.raises(ValueError, match="cannot be split"):
        adapter.allocate_kv_cache(
            mixed_config("BLNHC"), torch.device("cpu"), KVCacheLayout.BLNHC, [1, 2],
        )


def test_profile_and_failed_initialization_do_not_leak_persistent_scope(monkeypatch):
    calls = []
    attn_utils = types.ModuleType("vllm.v1.worker.gpu.attn_utils")
    setattr(attn_utils, "allocate_kv_cache", lambda config: calls.append(("vanilla", config)))
    monkeypatch.setitem(sys.modules, attn_utils.__name__, attn_utils)
    import vllm.v1.worker.gpu as package

    monkeypatch.setattr(package, "attn_utils", attn_utils, raising=False)
    monkeypatch.setattr(adapter, "allocate_kv_cache", lambda config: calls.append(("native", config)))
    monkeypatch.setenv("ENABLE_KVCACHED", "true")

    class Runner:
        def initialize_kv_cache(self, config, is_profiling=False, kv_cache_allocation_context=None):
            attn_utils.allocate_kv_cache(config)
            if config == "fail":
                raise RuntimeError("injected binding failure")
            calls.append(("bound", config))

    runner = Runner()
    assert adapter.ModelRunnerV2Patch().apply(types.SimpleNamespace(GPUModelRunner=Runner))
    runner.initialize_kv_cache("profile", is_profiling=True)
    runner.initialize_kv_cache("persistent")
    with pytest.raises(RuntimeError, match="binding failure"):
        runner.initialize_kv_cache("fail")
    attn_utils.allocate_kv_cache("outside")
    monkeypatch.setenv("ENABLE_KVCACHED", "false")
    runner.initialize_kv_cache("disabled")
    assert calls == [
        ("vanilla", "profile"), ("bound", "profile"),
        ("native", "persistent"), ("bound", "persistent"),
        ("native", "fail"), ("vanilla", "outside"),
        ("vanilla", "disabled"), ("bound", "disabled"),
    ]


@pytest.mark.parametrize("contiguous", [False, True])
def test_layout_negotiation_respects_backend_support(monkeypatch, contiguous):
    monkeypatch.setattr(adapter, "CONTIGUOUS_LAYOUT", contiguous)
    monkeypatch.setenv("ENABLE_KVCACHED", "true")

    class Worker:
        def get_supported_kv_cache_layouts(self):
            return ["LBNHC", "BLHNC", "BLNHC"]

    assert adapter.KVLayoutV2Patch().apply(types.SimpleNamespace(Worker=Worker))
    expected = ["BLHNC", "BLNHC"] if contiguous else ["LBNHC"]
    assert Worker().get_supported_kv_cache_layouts() == expected
    monkeypatch.setenv("ENABLE_KVCACHED", "false")
    assert Worker().get_supported_kv_cache_layouts() == ["LBNHC", "BLHNC", "BLNHC"]
