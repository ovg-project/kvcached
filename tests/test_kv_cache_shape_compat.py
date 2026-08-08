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
from types import SimpleNamespace

from kvcached.integration.vllm.patches import (
    _cache_dtype_str,
    _get_kv_cache_shape_compat,
    _kv_cache_uses_inline_scales,
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


def test_inline_scales_detection():
    assert _kv_cache_uses_inline_scales("fp8_per_token_head")
    assert _kv_cache_uses_inline_scales("int8_per_token_head")
    assert _kv_cache_uses_inline_scales("nvfp4")
    assert not _kv_cache_uses_inline_scales("auto")
    assert not _kv_cache_uses_inline_scales("fp8")
    assert not _kv_cache_uses_inline_scales(None)
    assert not _kv_cache_uses_inline_scales("")
