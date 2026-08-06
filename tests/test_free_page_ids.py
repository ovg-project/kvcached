# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_free_page_ids`` (SGLang paged-allocator free path).

GPU-free: ``_free_page_ids`` is a module-level function in
``kvcached.integration.sglang.patches`` and needs neither an installed SGLang
nor a GPU. It guards the CPU-conversion fix: unique page ids must be derived
from CPU data so allocator free bookkeeping does not couple to CUDA device
execution and synchronization (issue #382).
"""
import pytest
import torch

from kvcached.integration.sglang.patches import _free_page_ids


def test_single_page():
    assert _free_page_ids(torch.tensor([0, 1, 2, 3]), page_size=4) == [0]


def test_multiple_pages_deduplicated_and_sorted():
    free_index = torch.tensor([9, 8, 0, 1, 17, 16, 3])
    assert _free_page_ids(free_index, page_size=4) == [0, 2, 4]


def test_partial_page_indices():
    # Freeing only some tokens of a page still yields that page id once.
    assert _free_page_ids(torch.tensor([5, 6], dtype=torch.int64), page_size=4) == [1]


def test_empty_tensor():
    assert _free_page_ids(torch.empty((0,), dtype=torch.int64), page_size=4) == []


def test_returns_python_ints():
    """Bookkeeping needs plain ints, not tensor scalars."""
    result = _free_page_ids(torch.tensor([4, 5]), page_size=4)
    assert all(type(page_id) is int for page_id in result)


@pytest.mark.parametrize("page_size", [1, 2, 16, 64])
def test_page_size_division(page_size):
    free_index = torch.arange(3 * page_size)
    assert _free_page_ids(free_index, page_size) == [0, 1, 2]


def test_result_derived_from_cpu_tensor():
    """The unique/int conversion must happen on a CPU copy of the input."""
    calls = []

    class SpyTensor(torch.Tensor):

        def cpu(self):
            calls.append("cpu")
            return torch.Tensor.cpu(self)

    spy = torch.tensor([0, 5, 9]).as_subclass(SpyTensor)
    assert _free_page_ids(spy, page_size=4) == [0, 1, 2]
    assert calls == ["cpu"]
