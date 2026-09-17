# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_get_kv_cache_shape_compat`` (issue #424).

GPU-free: the helpers are module-level functions in
``kvcached.integration.vllm.patches`` and need neither an installed vLLM nor
a GPU. They guard the per-token-head quantization fix: ``cache_dtype_str``
must be forwarded to ``get_kv_cache_shape`` on vLLM versions that accept it
(the widened head_size inlines per-head scales into the KV page), and must be
omitted on older versions whose signature does not declare it.
"""
import ast
import pathlib
from types import SimpleNamespace

import pytest

import kvcached.integration.vllm.patches as patches
from kvcached.integration.vllm.patches import (
    _cache_dtype_str,
    _get_kv_cache_shape_compat,
)

# Widening applied by per-token-head modes in the fake backend below.
_SCALE_ELEMS = 4


class ModernBackend:
    """Mimics vLLM >= 0.19 FlashAttention: accepts cache_dtype_str."""

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                           cache_dtype_str="auto"):
        if cache_dtype_str and cache_dtype_str.endswith("per_token_head"):
            head_size += _SCALE_ELEMS
        return (2, num_blocks, block_size, num_kv_heads, head_size)


class LegacyBackend:
    """Mimics older vLLM: no cache_dtype_str parameter at all."""

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
        return (2, num_blocks, block_size, num_kv_heads, head_size)


def test_forwards_dtype_to_modern_backend():
    shape = _get_kv_cache_shape_compat(ModernBackend, 8, 16, 4, 128,
                                       "fp8_per_token_head")
    assert shape == (2, 8, 16, 4, 128 + _SCALE_ELEMS)


def test_auto_dtype_leaves_shape_unwidened():
    shape = _get_kv_cache_shape_compat(ModernBackend, 8, 16, 4, 128, "auto")
    assert shape == (2, 8, 16, 4, 128)


def test_none_dtype_is_not_forwarded():
    shape = _get_kv_cache_shape_compat(ModernBackend, 8, 16, 4, 128, None)
    assert shape == (2, 8, 16, 4, 128)


def test_legacy_backend_does_not_receive_kwarg():
    """Must not raise TypeError on signatures without cache_dtype_str."""
    shape = _get_kv_cache_shape_compat(LegacyBackend, 8, 16, 4, 128,
                                       "fp8_per_token_head")
    assert shape == (2, 8, 16, 4, 128)


def test_cache_dtype_str_from_cache_config():
    runner = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="fp8_per_token_head"))
    assert _cache_dtype_str(runner) == "fp8_per_token_head"


def test_cache_dtype_str_via_vllm_config():
    runner = SimpleNamespace(
        cache_config=None,
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(cache_dtype="fp8")),
    )
    assert _cache_dtype_str(runner) == "fp8"


def test_cache_dtype_str_absent_returns_none():
    assert _cache_dtype_str(SimpleNamespace()) is None


def test_no_direct_get_kv_cache_shape_calls():
    """Every call site must route through the compat helper, so the #424 fix
    cannot be reverted at one of them while the helper stays behind."""
    tree = ast.parse(pathlib.Path(patches.__file__).read_text())
    # The helper itself is the one place allowed to call the backend directly.
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_get_kv_cache_shape_compat"):
            node.body = []
    direct = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "get_kv_cache_shape"
    ]
    assert not direct, (
        "get_kv_cache_shape must be called via _get_kv_cache_shape_compat so "
        f"cache_dtype_str is forwarded (#424); direct calls at lines {direct}")


@pytest.mark.parametrize("version,buffers", [("0.22.1", 2), ("0.27.0", 2),
                                            ("0.28.0", 1), ("0.28.0+cu129", 1)])
def test_engine_packed_geometry_matches_worker_version(monkeypatch, version, buffers):
    manager = patches.VersionManager.get_instance()
    monkeypatch.setattr(manager, "detect_version", lambda _: version)
    monkeypatch.setattr(patches, "_is_mla_kv_cache_spec", lambda _: False)
    spec = SimpleNamespace(page_size_bytes=4096)
    cell, actual_buffers = patches._get_kv_cache_params(spec, 16, "MHA")
    assert actual_buffers == buffers
    assert cell * 16 * actual_buffers == 4096


@pytest.mark.parametrize("order,layout", [((0, 2, 1, 3), "NHD"),
                                         ((0, 1, 2, 3), "HND")])
def test_packed_layout_comes_from_backend_stride_order(order, layout):
    backend = SimpleNamespace(get_kv_cache_stride_order=lambda: order)
    assert patches._get_packed_kv_layout(backend) == layout


def test_unrecognized_packed_stride_order_fails_closed():
    backend = SimpleNamespace(get_kv_cache_stride_order=lambda: (0, 1, 3, 2))
    with pytest.raises(NotImplementedError, match="stride order"):
        patches._get_packed_kv_layout(backend)


def test_packed_backend_keeps_native_scale_views(monkeypatch):
    monkeypatch.setattr(patches, "_uses_packed_attention_kv", lambda: True)

    class Impl:
        def _ensure_scale_caches(self, kv_cache):
            return kv_cache

    original = Impl._ensure_scale_caches
    module = SimpleNamespace(TritonAttentionImpl=Impl)
    assert patches.TritonAttentionPatch().patch_ensure_scale_caches(module)
    assert Impl._ensure_scale_caches is original
