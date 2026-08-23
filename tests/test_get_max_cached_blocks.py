# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_get_max_cached_blocks`` (MAX_CACHED_TOKENS → block budget).

GPU-free: ``_get_max_cached_blocks`` is a module-level pure function in
``kvcached.integration.vllm.patches`` and needs neither the ``vmm_ops``
extension, a GPU, nor an installed vLLM.

It guards the sub-block-size floor fix: a *positive* ``KVCACHED_MAX_CACHED_TOKENS``
smaller than ``block_size`` used to integer-divide to ``0`` — the same value the
``== 0`` "disabled" sentinel produces — which silently turned prefix caching off
(issue #343). The positive branch now floors at one block.
"""
import logging

import pytest

from kvcached.integration.vllm import patches
from kvcached.integration.vllm.patches import _get_max_cached_blocks


@pytest.fixture
def set_max_cached_tokens(monkeypatch):
    """Set the module-level MAX_CACHED_TOKENS the function reads at call time."""

    def _set(value: int) -> None:
        monkeypatch.setattr("kvcached.utils.MAX_CACHED_TOKENS", value)

    return _set


@pytest.fixture
def captured_warnings():
    """Collect records from the module logger.

    ``get_kvcached_logger`` sets ``propagate = False``, so pytest's root-attached
    ``caplog`` never sees these records; attach a handler to the logger directly.
    """
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    old_level = patches.logger.level
    patches.logger.addHandler(handler)
    patches.logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        patches.logger.removeHandler(handler)
        patches.logger.setLevel(old_level)


def test_negative_is_unlimited(set_max_cached_tokens):
    set_max_cached_tokens(-1)
    assert _get_max_cached_blocks(16) == -1
    set_max_cached_tokens(-9999)
    assert _get_max_cached_blocks(16) == -1


def test_zero_is_disabled(set_max_cached_tokens):
    set_max_cached_tokens(0)
    assert _get_max_cached_blocks(16) == 0


def test_small_positive_floors_to_one_block(set_max_cached_tokens):
    """Regression for #343: 0 < tokens < block_size must not collapse to 0."""
    set_max_cached_tokens(8)
    assert _get_max_cached_blocks(16) == 1


@pytest.mark.parametrize("tokens", [1, 8, 15])
def test_below_block_size_never_disables(set_max_cached_tokens, tokens):
    """Any positive budget below one block stays enabled (>= 1 block)."""
    set_max_cached_tokens(tokens)
    assert _get_max_cached_blocks(16) == 1


def test_exact_multiple(set_max_cached_tokens):
    set_max_cached_tokens(32)
    assert _get_max_cached_blocks(16) == 2


def test_non_multiple_rounds_down_but_stays_positive(set_max_cached_tokens):
    set_max_cached_tokens(40)
    assert _get_max_cached_blocks(16) == 2  # floor(40 / 16) == 2


def test_large_value(set_max_cached_tokens):
    set_max_cached_tokens(16000)
    assert _get_max_cached_blocks(16) == 1000


def test_positive_result_distinct_from_disabled(set_max_cached_tokens):
    """A positive budget must never return the 0 'disabled' sentinel."""
    for tokens in range(1, 17):
        set_max_cached_tokens(tokens)
        assert _get_max_cached_blocks(16) >= 1


def test_clamp_emits_warning(set_max_cached_tokens, captured_warnings):
    """The floor must not be silent — a warning explains the effective budget."""
    set_max_cached_tokens(8)
    assert _get_max_cached_blocks(16) == 1
    assert any("flooring max cached blocks to 1" in r.getMessage()
               for r in captured_warnings)


def test_no_warning_when_budget_is_a_full_block(
        set_max_cached_tokens, captured_warnings):
    set_max_cached_tokens(16)
    assert _get_max_cached_blocks(16) == 1
    assert not captured_warnings
