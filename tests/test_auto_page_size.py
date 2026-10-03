# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Automatic pages must be usable and agree across scheduler/worker paths."""
# ruff: noqa: F811
import math
import sys
import types
from unittest import mock

import pytest
from test_vllm_ftensor_capacity_guard import iface  # noqa: F401

from kvcached.kv_geometry import MIB, recommend_page_geometry, select_page_size
from kvcached.utils import get_page_size_for_block


@pytest.mark.parametrize("block,expected_mb", [
    (32768, 2), (2 * MIB, 2), (2162688, 6), (3211264, 8),
    (4 * MIB, 4), (6 * MIB, 6),
    (5 * MIB // 2, 10), (17 * MIB // 8, 34),
    (64 * MIB, 64), (65 * MIB, 130),
])
def test_select_page_size(block, expected_mb):
    assert select_page_size(block) == expected_mb * MIB


def test_tiling_preference_and_safe_fallback_by_enumerating_block_boundaries():
    def capacities(block, page):
        return [((i + 1) * page) // block - (i * page + block - 1) // block
                for i in range(block // math.gcd(block, page))]

    for block in range(65536, 129 * 65536, 65536):
        chosen = select_page_size(block)
        assert min(capacities(block, chosen)) > 0
        tiling_pages = [page for page in range(2 * MIB, 65 * MIB, 2 * MIB)
                        if page % block == 0]
        if tiling_pages:
            assert chosen == min(tiling_pages)
        else:
            for smaller in range(2 * MIB, chosen, 2 * MIB):
                assert min(capacities(block, smaller)) <= 0


@pytest.mark.parametrize("block", [0, -1])
def test_nonpositive_block_is_rejected(block):
    with pytest.raises(ValueError, match="positive"):
        select_page_size(block)


def test_recommended_block_keeps_its_tiling_page_at_allocation():
    for cell in (640, 768, 1280, 1536, 2048, 2560, 3072, 4096, 5120, 6144, 8192):
        for tokens in range(16, 4097, 16):
            if tokens * cell <= 2 * MIB:
                continue
            recommendation = recommend_page_geometry(tokens, cell)
            if recommendation is None:
                continue
            page_mb, aligned = recommendation
            page = select_page_size(aligned * cell)
            assert page == page_mb * MIB, (tokens, cell, recommendation)
            assert page % (aligned * cell) == 0


def test_auto_selection_is_local_and_logged(monkeypatch):
    from kvcached import utils

    monkeypatch.delenv("KVCACHED_PAGE_SIZE_MB", raising=False)
    logger = mock.Mock()
    monkeypatch.setattr(utils, "get_kvcached_logger", lambda: logger)
    default = utils.PAGE_SIZE
    assert get_page_size_for_block(4 * MIB, 2 * MIB) == 4 * MIB
    assert utils.PAGE_SIZE == default
    assert get_page_size_for_block(32768, 2 * MIB) == 2 * MIB
    assert logger.info.call_count == 1
    assert logger.info.call_args.args[1:] == (2 * MIB, 4 * MIB, 4 * MIB)


@pytest.mark.parametrize("configured_mb", [2, 4, 6, 8, 10])
def test_explicit_page_is_preserved(monkeypatch, configured_mb):
    monkeypatch.setenv("KVCACHED_PAGE_SIZE_MB", str(configured_mb))
    assert get_page_size_for_block(4 * MIB, configured_mb * MIB) == configured_mb * MIB
    assert get_page_size_for_block(5 * MIB // 2, configured_mb * MIB) == configured_mb * MIB


@pytest.mark.parametrize("tokens,head_dim,attention_type,expected_mb", [
    (2048, 256, "HYBRID_LINEAR", 4),
    (1056, 256, "HYBRID_LINEAR", 6),  # fixed block / releases without the align hook
    (2048, 160, "HYBRID_LINEAR", 10),
    (2048, 256, "MHA", 2),          # K and V are separate allocation units
    (4096, 160, "MHA", 10),
    (1024, 160, "MHA", 2),          # do not grow an already usable default page
])
@pytest.mark.parametrize("contiguous", [False, True])
def test_worker_and_scheduler_resolve_identical_pages(
        iface, monkeypatch, tokens, head_dim, attention_type, expected_mb, contiguous):
    monkeypatch.delenv("KVCACHED_PAGE_SIZE_MB", raising=False)
    monkeypatch.setattr(iface, "PAGE_SIZE", 2 * MIB)
    monkeypatch.setattr(iface, "_kvcached_initialized", True)
    monkeypatch.setattr(iface, "_contiguous_layout", contiguous)
    iface.torch.cuda.get_device_properties.return_value.total_memory = 24 * 1024**3
    capture = mock.Mock(side_effect=RuntimeError("captured tensor creation"))
    monkeypatch.setattr(iface, "create_kv_tensors", capture)
    with pytest.raises(RuntimeError, match="captured tensor creation"):
        iface.alloc_kv_cache(
            (2, 32, tokens, 4, head_dim), tokens, types.SimpleNamespace(itemsize=1),
            "cuda:0", 8, attention_type=attention_type)
    page = capture.call_args.kwargs["page_size"]
    assert page == expected_mb * MIB
    assert capture.call_args.args[0] % (2 * page) == 0

    manager = mock.Mock()
    monkeypatch.setattr(iface, "KVCacheManager", manager)
    monkeypatch.setattr(iface, "register_kv_cache_pool", mock.Mock())
    buffers = 1 if attention_type == "HYBRID_LINEAR" else 2
    iface.get_kv_cache_manager(32, tokens, 8 * head_dim // buffers, 8, num_kv_buffers=buffers)
    assert manager.call_args.kwargs["page_size"] == page


def test_tensor_page_mismatch_is_rejected_before_manager_creation(iface, monkeypatch):
    monkeypatch.delenv("KVCACHED_PAGE_SIZE_MB", raising=False)
    monkeypatch.setattr(iface, "PAGE_SIZE", 2 * MIB)
    record = {"num_layers": 1, "num_kv_buffers": 1, "page_size": 2 * MIB}
    with pytest.raises(ValueError, match="Manager page size"):
        iface._validate_manager_against_created_tensors(record, 8, 4 * MIB, 1, 1, 0)


def test_manager_first_checks_the_page_size_already_in_use(iface, monkeypatch):
    manager = types.SimpleNamespace(group_id=0, num_blocks=8, block_mem_size=32768,
                                    num_layers=1, num_kv_buffers=2, page_size=4 * MIB)
    monkeypatch.setattr(iface, "get_registered_kv_cache_pools", lambda **kw: [(manager, {})])
    record = {"num_layers": 1, "num_kv_buffers": 2, "page_size": 2 * MIB}
    with pytest.raises(ValueError, match="Manager page size"):
        iface._validate_registered_managers(0, record)


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("tokens,cell_bytes,explicit,expected_mb", [
    (2048, 2048, None, 4), (1056, 2048, None, 6), (2048, 2048, "2", None),
    (2048, 1280, None, 10),
])
def test_mrv2_resolves_page_before_geometry_check_and_native_allocation(
        iface, monkeypatch, contiguous, tokens, cell_bytes, explicit, expected_mb):
    from kvcached.integration.vllm import model_runner_v2 as adapter

    # Exercise the adapter without requiring vLLM 0.29 or CUDA. The existing
    # 0.29 integration tests separately check native view construction.
    specs = types.ModuleType("vllm.v1.kv_cache_interface")
    attention = type("AttentionSpec", (types.SimpleNamespace,), {})
    setattr(specs, "AttentionSpec", attention)
    for name in ("FullAttentionSpec", "SlidingWindowSpec", "MLAAttentionSpec",
                 "MambaSpec", "UniformTypeKVCacheSpecs"):
        setattr(specs, name, type(name, (attention,), {}))
    block_bytes = tokens * cell_bytes
    spec = getattr(specs, "FullAttentionSpec")(block_size=tokens, page_size_bytes=block_bytes)
    setattr(specs, "compute_layer_kv_cache_shape_bytes", lambda *a: (1, tokens, cell_bytes))
    setattr(specs, "create_kv_cache_views", mock.Mock())
    monkeypatch.setitem(sys.modules, specs.__name__, specs)
    monkeypatch.setattr(adapter, "CONTIGUOUS_LAYOUT", contiguous)
    monkeypatch.setattr(adapter, "PAGE_SIZE", 2 * MIB)
    if explicit is None:
        monkeypatch.delenv("KVCACHED_PAGE_SIZE_MB", raising=False)
    else:
        monkeypatch.setenv("KVCACHED_PAGE_SIZE_MB", explicit)
    config = types.SimpleNamespace(
        num_blocks=3,
        kv_cache_groups=[types.SimpleNamespace(layer_names=["attn"], kv_cache_spec=spec)],
        kv_cache_tensors=[types.SimpleNamespace(
            size=3 * block_bytes, layers=["attn"], offset=0,
            block_stride=block_bytes,
            layer_stride=block_bytes if contiguous else 3 * block_bytes)],
    )
    monkeypatch.setattr(iface, "_kvcached_initialized", True)
    iface.torch.cuda.get_device_properties.return_value.total_memory = 25 * MIB
    capture = mock.Mock(side_effect=RuntimeError("captured tensor creation"))
    monkeypatch.setattr(iface, "create_kv_tensors", capture)
    layout = types.SimpleNamespace(name="BLNHC" if contiguous else "LBNHC")
    if expected_mb is None:
        with pytest.raises(adapter.KVCachedConfigError, match="cannot manage"):
            adapter.allocate_kv_cache(config, "cuda:0", layout)
        capture.assert_not_called()
    else:
        assert adapter.cache_geometry(config).page_bytes == block_bytes
        with pytest.raises(RuntimeError, match="captured tensor creation"):
            adapter.allocate_kv_cache(config, "cuda:0", layout)
        assert capture.call_args.kwargs["page_size"] == expected_mb * MIB
        assert capture.call_args.args[0] == (25 // expected_mb) * expected_mb * MIB
