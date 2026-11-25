# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""CPU tensors with actual vLLM 0.29/0.30 view helpers; no GPU or native VMM needed."""

import copy
import dataclasses
import importlib.metadata
import logging
import sys
import types

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
if not importlib.metadata.version("vllm").startswith(("0.29.", "0.30.")):
    pytest.skip("requires the vLLM 0.29/0.30 view contract", allow_module_level=True)

import torch
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
)
from vllm.v1.kv_cache_interface import (
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import add_kv_sharing_layers_to_kv_cache_groups

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
    assert captured == [(512, 1, "cpu", 2, {"num_kv_buffers": 1, "unified_pool": True, "page_size": 64})]
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
def test_automatic_page_preserves_native_views(monkeypatch, layout_name):
    contiguous = layout_name.startswith("B")
    captured = mock_native_allocator(monkeypatch, contiguous)
    mib = 1024**2
    monkeypatch.delenv("KVCACHED_PAGE_SIZE_MB", raising=False)
    monkeypatch.setattr(adapter, "PAGE_SIZE", 2 * mib)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: types.SimpleNamespace(total_memory=64 * mib))
    spec = FullAttentionSpec(block_size=2048, num_kv_heads=5,
                             head_size=128, dtype=torch.uint8)
    block_bytes = spec.page_size_bytes
    assert block_bytes == 5 * mib // 2
    config = KVCacheConfig(
        num_blocks=5,
        kv_cache_tensors=[KVCacheTensor(
            size=10 * block_bytes, layers=["a", "b"],
            layer_stride=block_bytes if contiguous else 5 * block_bytes,
            block_stride=2 * block_bytes if contiguous else block_bytes,
        )],
        kv_cache_groups=[KVCacheGroupSpec(["a", "b"], spec)],
    )
    scheduler_geometry = adapter.cache_geometry(generate_scheduler_kv_cache_config([config]))
    assert scheduler_geometry == adapter.cache_geometry(config)
    caches = adapter.allocate_kv_cache(
        config, torch.device("cpu"), KVCacheLayout[layout_name], [2048])
    assert captured == [(30 * mib, 1, "cpu", 2, {"num_kv_buffers": 1, "unified_pool": True, "page_size": 6 * mib})]
    # Verify native view offsets; CUDA tests separately allocate usable block IDs.
    for layer, name in enumerate(("a", "b")):
        for block in range(5):
            caches[name][block].fill_(10 * layer + block + 1)
    for layer, name in enumerate(("a", "b")):
        for block in range(5):
            assert torch.all(caches[name][block] == 10 * layer + block + 1)


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
    assert adapter.cache_geometry(config) == adapter.CacheGeometry(64, 73728, 2, 2 * 1024**2)
    caches = adapter.allocate_kv_cache(config, torch.device("cpu"), KVCacheLayout[layout_name], [64])
    assert captured == [(2 * 1024**2, 1, "cpu", 2, {"num_kv_buffers": 1, "unified_pool": True, "page_size": 2 * 1024**2})]
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
    ), attention_config=types.SimpleNamespace(hisparse_config=None))  # read by 0.30
    return get_kv_cache_config_from_groups(config, groups, available_memory=384)


@pytest.mark.parametrize("layout_name", ["BLNHC", "BLHNC", "LBNHC", "LBHNC"])
@pytest.mark.parametrize("borrower_count", [1, 4])
def test_native_borrowers_preserve_owner_geometry(monkeypatch, layout_name, borrower_count):
    config = uniform_config(layout_name.startswith("B"))
    # Use native allocation descriptors: discovery must not mutate their layers.
    native_config = types.SimpleNamespace(cache_config=types.SimpleNamespace(
        get_resolved_kv_cache_layout=lambda: KVCacheLayout[layout_name],
        num_gpu_blocks_override=3,
        prefix_cache_retention_interval=None,
    ), attention_config=types.SimpleNamespace(hisparse_config=None))  # read by 0.30
    config = get_kv_cache_config_from_groups(
        native_config, config.kv_cache_groups, available_memory=384,
    )
    _check_borrower_geometry(monkeypatch, config, layout_name, borrower_count)


@pytest.mark.parametrize("layout_name", ["BLNHC", "BLHNC"])
def test_native_borrowers_preserve_mixed_owner_geometry(monkeypatch, layout_name):
    config = mixed_config(layout_name)
    assert len(config.kv_cache_tensors) == 3
    _check_borrower_geometry(monkeypatch, config, layout_name, 1)


def _check_borrower_geometry(monkeypatch, config, layout_name, borrower_count):
    from vllm.v1.worker.gpu import attn_utils
    from vllm.v1.worker.utils import allocate_kv_cache as native_allocate

    captured = mock_native_allocator(monkeypatch, layout_name.startswith("B"))
    scheduler = generate_scheduler_kv_cache_config([config])
    backing = copy.deepcopy(config.kv_cache_tensors)
    expected = adapter.CacheGeometry(2, 64, 2, 64)
    assert adapter.cache_geometry(scheduler) == expected
    sharing = {f"borrower_{index}": "b" for index in range(borrower_count)}
    add_kv_sharing_layers_to_kv_cache_groups(sharing, config.kv_cache_groups)
    assert config.kv_cache_tensors == backing
    assert adapter.cache_geometry(config) == expected
    assert adapter.cache_geometry(generate_scheduler_kv_cache_config([config])) == expected

    layout = KVCacheLayout[layout_name]
    native = native_allocate(config, torch.device("cpu"), layout, [2, 2])
    owners = adapter.allocate_kv_cache(config, torch.device("cpu"), layout, [2, 2])
    assert set(owners) == {layer for tensor in backing for layer in tensor.layers}
    assert captured == [(512, 1, "cpu", 2, {"num_kv_buffers": 1, "unified_pool": True, "page_size": 64})]
    for name, cache in owners.items():
        assert cache.shape == native[name].shape
        assert cache.stride() == native[name].stride()
    assert owners["a"].data_ptr() != owners["b"].data_ptr()
    owners["a"].fill_(11)
    owners["b"].fill_(22)
    assert torch.all(owners["a"] == 11)

    # Exercise native post-allocation aliasing, capturing the binding boundary
    # without requiring model Attention objects for these CPU tensor contracts.
    bound = []
    monkeypatch.setattr(attn_utils, "allocate_kv_cache", lambda *args: owners)
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda _: sharing)
    engine_config = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(get_resolved_kv_cache_layout=lambda: layout),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type="test")),
        attention_config=types.SimpleNamespace(hisparse_config=None),
    )
    if hasattr(attn_utils, "bind_kv_cache"):  # 0.29
        monkeypatch.setattr(attn_utils, "bind_kv_cache", lambda caches, *args: bound.append(caches))
        caches = attn_utils.init_kv_cache([], {}, config, torch.device("cpu"), [2, 2], engine_config)
        assert bound == [caches]
    else:  # 0.30 binds only the layers present in the forward context
        monkeypatch.setattr(attn_utils, "bind_kv_cache_to_layers",
                            lambda caches, *args, **kwargs: bound.append(caches))
        caches = attn_utils.init_kv_cache({}, config, torch.device("cpu"), [2, 2], engine_config)
        assert bound == [{}]
    for borrower in sharing:
        assert caches[borrower] is caches["b"]
        assert caches[borrower].data_ptr() == caches["b"].data_ptr()
    caches["borrower_0"][1].fill_(33)
    assert torch.all(caches["b"][1] == 33)
    assert torch.all(caches["b"][0] == 22)
    assert torch.all(caches["a"] == 11)
    assert config.kv_cache_tensors == backing


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
    assert captured == [(512, 1, "cpu", 2, {"num_kv_buffers": 1, "unified_pool": True, "page_size": 64})]
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


def test_mixed_packing_rejects_kernel_split_before_allocating(monkeypatch):
    captured = mock_native_allocator(monkeypatch, True)
    with pytest.raises(adapter.KVCachedConfigError, match="KVCACHED_CONTIGUOUS_LAYOUT=false"):
        adapter.allocate_kv_cache(
            mixed_config("BLNHC"), torch.device("cpu"), KVCacheLayout.BLNHC, [1, 2],
        )
    assert captured == []


@pytest.mark.parametrize("layout_name", ["BLNHC", "BLHNC", "LBNHC", "LBHNC"])
@pytest.mark.parametrize("padded,kernel_size", [(None, 1), (128, 1), (128, 2)])
def test_single_layer_kernel_split_uses_dense_page_size(monkeypatch, layout_name, padded, kernel_size):
    contiguous = layout_name.startswith("B")
    captured = mock_native_allocator(monkeypatch, contiguous)
    monkeypatch.setattr(adapter, "PAGE_SIZE", 128)
    spec = FullAttentionSpec(
        block_size=2, num_kv_heads=2, head_size=4,
        dtype=torch.bfloat16, page_size_padded=padded,
    )
    page_bytes = spec.page_size_bytes
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[KVCacheTensor(
            size=3 * page_bytes, layers=["a"], block_stride=page_bytes,
            layer_stride=page_bytes if contiguous else 3 * page_bytes,
        )],
        kv_cache_groups=[KVCacheGroupSpec(["a"], spec)],
    )
    if padded is not None and kernel_size == 1:
        before = copy.deepcopy(config)
        with pytest.raises(adapter.KVCachedConfigError, match="dense page is 64 bytes") as error:
            adapter.allocate_kv_cache(
                config, torch.device("cpu"), KVCacheLayout[layout_name], [kernel_size],
            )
        assert "Changing the KV layout does not remove" in str(error.value)
        assert "KVCACHED_CONTIGUOUS_LAYOUT=false" not in str(error.value)
        assert captured == []
        assert config == before
    else:
        caches = adapter.allocate_kv_cache(
            config, torch.device("cpu"), KVCacheLayout[layout_name], [kernel_size],
        )
        assert len(captured) == 1
        assert caches["a"].shape[0] == 3 * (spec.block_size // kernel_size)
        for block in range(caches["a"].shape[0]):
            caches["a"][block].fill_(block + 1)
        for block in range(caches["a"].shape[0]):
            assert torch.all(caches["a"][block] == block + 1)


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
    assert adapter.ModelRunnerV2Patch().apply(types.SimpleNamespace(
        GPUModelRunner=Runner, init_attn_backend=lambda *args, **kwargs: None,
    ))
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


@pytest.mark.parametrize("spec_type", [CrossAttentionSpec, SinkFullAttentionSpec])
def test_rejects_unqualified_attention_types_before_native_allocation(monkeypatch, spec_type):
    captured = mock_native_allocator(monkeypatch, True)
    config = uniform_config(True)
    spec = spec_type(block_size=2, num_kv_heads=2, head_size=4, dtype=torch.bfloat16)
    # Exercise expansion as well as subclasses of supported attention types.
    config.kv_cache_groups[0].kv_cache_spec = UniformTypeKVCacheSpecs(
        block_size=2, kv_cache_specs={"a": spec, "b": spec},
    )
    with pytest.raises(adapter.KVCachedConfigError):
        adapter.allocate_kv_cache(config, torch.device("cpu"), KVCacheLayout.BLNHC, [2, 2])
    assert not captured


_HAS_030_FIELDS = (
    "host_resident" in KVCacheTensor.__dataclass_fields__
    and "storage_block_size" in MLAAttentionSpec.__dataclass_fields__
)


@pytest.mark.skipif(not _HAS_030_FIELDS, reason="vLLM 0.30 KV tensor and MLA fields")
def test_host_resident_tensor_is_rejected(monkeypatch):
    mock_native_allocator(monkeypatch, True)
    config = uniform_config(True)
    config.kv_cache_tensors[0] = dataclasses.replace(config.kv_cache_tensors[0], host_resident=True)
    with pytest.raises(adapter.KVCachedConfigError, match="host-resident"):
        adapter.allocate_kv_cache(config, torch.device("cpu"), KVCacheLayout.BLNHC, [2, 2])


@pytest.mark.skipif(not _HAS_030_FIELDS, reason="vLLM 0.30 KV tensor and MLA fields")
def test_mla_storage_block_size_overrides_kernel_block(monkeypatch):
    # Mirrors vLLM 0.30 worker/utils.allocate_kv_cache for storage-sized MLA views.
    mock_native_allocator(monkeypatch, False)
    monkeypatch.setattr(adapter, "PAGE_SIZE", 2 * 1024**2)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: types.SimpleNamespace(total_memory=4 * 1024**2))
    import vllm.v1.kv_cache_interface as kv_interface

    seen = []
    native_views = kv_interface.create_kv_cache_views

    def record(*args, **kwargs):
        seen.append(args[5] if len(args) > 5 else kwargs.get("kernel_block_size"))
        return native_views(*args, **kwargs)

    monkeypatch.setattr(kv_interface, "create_kv_cache_views", record)
    spec = MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=576, dtype=torch.bfloat16,
                            storage_block_size=32)
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[KVCacheTensor(size=442368, layers=["a", "b"],
                                        layer_stride=221184, block_stride=73728)],
        kv_cache_groups=[KVCacheGroupSpec(["a", "b"], spec)],
    )
    caches = adapter.allocate_kv_cache(config, torch.device("cpu"), KVCacheLayout.LBNHC, [64])
    assert seen == [32, 32]
    assert caches["a"].shape[0] == 6


def make_qsa_plan(layout, tokens, draft_tokens, num_blocks):
    from vllm.models.qwen4_exp.common.qsa_cache import (
        QSACompressedKeyCache,
        QSAKeyStateCache,
    )
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
    engine = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(block_size=tokens, num_gpu_blocks_override=num_blocks,
                                           prefix_cache_retention_interval=None,
                                           get_resolved_kv_cache_layout=lambda: layout),
        model_config=types.SimpleNamespace(max_model_len=1, original_max_model_len=1),
        device_config=types.SimpleNamespace(device_type="cuda"),
        parallel_config=types.SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        scheduler_config=types.SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        compilation_config=types.SimpleNamespace(static_forward_context={}),
        attention_config=types.SimpleNamespace(hisparse_config=None),
        speculative_config=None,
        num_speculative_tokens=draft_tokens,
    )
    kwargs = dict(head_size=128, dtype=torch.bfloat16, compress_ratio=4,
                  cache_config=engine.cache_config, vllm_config=engine)
    ring = QSAKeyStateCache(prefix="ring", cache_rope_positions=True, **kwargs)
    index = QSACompressedKeyCache(prefix="index", **kwargs)
    specs = {
        "ring": ring.get_kv_cache_spec(engine),
        "main": FullAttentionSpec(block_size=tokens, num_kv_heads=2, head_size=256,
                                  dtype=torch.bfloat16),
        "index": index.get_kv_cache_spec(engine),
    }
    groups = get_kv_cache_groups(engine, specs)
    config = get_kv_cache_config_from_groups(engine, groups, available_memory=16 * 1024**2)
    return engine, specs, config, ring, index


@pytest.mark.parametrize("layout", ["BLNHC", "BLHNC"])
@pytest.mark.parametrize("tokens,draft_tokens,ring_slots,page_mib", [
    (64, 0, 4, 2), (64, 2, 8, 2), (1024, 0, 4, 4), (1104, 6, 12, 4),
])
def test_qsa_ring_and_compressed_index(monkeypatch, layout, tokens, draft_tokens, ring_slots, page_mib):
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
    from vllm.v1.worker.utils import allocate_kv_cache as allocate_native

    layout = KVCacheLayout[layout]
    allocated = mock_native_allocator(monkeypatch, True)
    monkeypatch.delenv("KVCACHED_PAGE_SIZE_MB", raising=False)
    monkeypatch.setattr(adapter, "PAGE_SIZE", 2 * 1024**2)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: types.SimpleNamespace(total_memory=32 * 1024**2))
    num_blocks = 64 if tokens == 64 else 8
    engine, specs, config, ring, index = make_qsa_plan(layout, tokens, draft_tokens, num_blocks)
    adapter.validate_qsa_cache_plan(engine, [config])
    scheduler = generate_scheduler_kv_cache_config([config])
    geometry = adapter.cache_geometry(scheduler)
    assert adapter.cache_geometry(config) == geometry
    assert geometry.block_size == tokens
    assert geometry.physical_page_size == page_mib * 1024**2
    other_specs = {**specs, "main": dataclasses.replace(specs["main"], num_kv_heads=1)}
    other_groups = get_kv_cache_groups(engine, other_specs)
    other = get_kv_cache_config_from_groups(engine, other_groups, available_memory=16 * 1024**2)
    with pytest.raises(adapter.KVCachedConfigError, match="different block-to-page mappings"):
        adapter.validate_qsa_cache_plan(engine, [config, other])

    native = allocate_native(config, torch.device("cpu"), layout)
    caches = adapter.allocate_kv_cache(config, torch.device("cpu"), layout)
    assert allocated[0][-1]["page_size"] == geometry.physical_page_size
    for name, expected in native.items():
        assert caches[name].shape == expected.shape
        assert caches[name].stride() == expected.stride()
        assert caches[name].storage_offset() == expected.storage_offset()

    ring.bind_kv_cache(caches["ring"])
    index.bind_kv_cache(caches["index"])
    assert ring.key_cache.shape == (num_blocks, ring_slots, 1, 128)
    assert index.kv_cache.shape == (num_blocks, tokens // 4, 1, 128)
    # Main KV and its compressed index share a logical block. The request's
    # private ring gets a different block ID from vLLM's CircularBufferManager.
    caches["main"][1].fill_(1)
    index.kv_cache[1].fill_(2)
    ring.key_cache[2].fill_(3)
    ring.rope_position_cache[2].fill_(2**40 + 1)
    assert torch.all(ring.key_cache[2] == 3)
    assert torch.all(ring.rope_position_cache[2] == 2**40 + 1)
    assert torch.all(caches["main"][1] == 1)
    assert torch.all(index.kv_cache[1] == 2)


def test_qsa_admission_and_auto_fit_use_whole_blocks(monkeypatch):
    monkeypatch.setattr(adapter, "CONTIGUOUS_LAYOUT", True)
    monkeypatch.setattr(adapter, "PAGE_SIZE", 2 * 1024**2)
    monkeypatch.delenv("KVCACHED_PAGE_SIZE_MB", raising=False)
    from vllm.v1.core import kv_cache_utils

    from kvcached.integration.vllm import patches

    engine, specs, config, _, _ = make_qsa_plan(KVCacheLayout.BLNHC, 1024, 0, 128)
    engine.model_config.hf_text_config = types.SimpleNamespace(indexer_compress_ratio=4)
    hooks = types.SimpleNamespace(
        EngineCore=type("EngineCore", (), {}),
        get_kv_cache_configs=kv_cache_utils.get_kv_cache_configs,
        update_kv_cache_capacity=kv_cache_utils.update_kv_cache_capacity,
    )
    patch = patches.EngineCorePatch()
    patch.detected_version = importlib.metadata.version("vllm")
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patch.patch_engine_init(hooks)
    geometry = adapter.cache_geometry(config)
    assert geometry.page_bytes == 1_081_344
    assert geometry.usable_blocks(128) == 96

    # One null and one request-private ring leave 94 token blocks.
    limit = 94 * 1024
    engine.model_config.original_max_model_len = engine.model_config.max_model_len = limit + 1
    args = (engine, [specs], [128 * geometry.num_pools * geometry.page_bytes])
    assert kv_cache_utils.get_kv_cache_configs(*args)[0].num_blocks == 128
    with pytest.raises(ValueError, match="max|KV cache"):
        hooks.get_kv_cache_configs(*args)
    engine.model_config.original_max_model_len = -1
    configs = hooks.get_kv_cache_configs(*args)
    assert engine.model_config.max_model_len == limit
    assert configs[0].num_blocks == 128
    scheduler = generate_scheduler_kv_cache_config(configs)
    hooks.update_kv_cache_capacity(engine, scheduler)
    assert scheduler.num_blocks == 128
    expected = dataclasses.replace(scheduler, num_blocks=95)
    assert engine.cache_config.kv_cache_size_tokens == kv_cache_utils.get_kv_cache_capacity(engine, expected)[0]
    engine.model_config.max_model_len = 1024
    hooks.update_kv_cache_capacity(engine, scheduler)
    assert engine.cache_config.kv_cache_max_concurrency == 47.5
    engine.model_config.max_model_len = limit
    engine.model_config.original_max_model_len = limit
    assert hooks.get_kv_cache_configs(*args)[0].num_blocks == 128
