# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_get_max_cached_blocks`` (prefix-cache block budget).

GPU-free: ``_get_max_cached_blocks`` is a module-level function in
``kvcached.integration.vllm.patches`` and needs neither the ``vmm_ops``
extension, a GPU, nor an installed vLLM. It guards the low-boundary fix: a
positive ``KVCACHED_MAX_CACHED_TOKENS`` smaller than ``block_size`` used to
truncate to 0, which silently disabled prefix caching (0 means "evict on
free").
"""
import pytest

import kvcached.utils
from kvcached.integration.vllm.patches import _get_max_cached_blocks


def _set_max_cached_tokens(monkeypatch, value: int) -> None:
    # The function re-imports MAX_CACHED_TOKENS from kvcached.utils on each
    # call, so patching the module attribute is sufficient.
    monkeypatch.setattr(kvcached.utils, "MAX_CACHED_TOKENS", value)


def test_negative_means_unlimited(monkeypatch):
    _set_max_cached_tokens(monkeypatch, -1)
    assert _get_max_cached_blocks(block_size=16) == -1


def test_zero_means_disabled(monkeypatch):
    _set_max_cached_tokens(monkeypatch, 0)
    assert _get_max_cached_blocks(block_size=16) == 0


def test_exact_multiple(monkeypatch):
    _set_max_cached_tokens(monkeypatch, 16000)
    assert _get_max_cached_blocks(block_size=16) == 1000


def test_rounds_down_to_whole_blocks(monkeypatch):
    _set_max_cached_tokens(monkeypatch, 100)
    assert _get_max_cached_blocks(block_size=16) == 6


@pytest.mark.parametrize("tokens", [1, 8, 15])
def test_small_positive_budget_keeps_caching_enabled(monkeypatch, tokens):
    """Regression: tokens < block_size must not truncate to 0 (disabled)."""
    _set_max_cached_tokens(monkeypatch, tokens)
    assert _get_max_cached_blocks(block_size=16) == 1
