# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Padding blocks must never become SGLang Mamba slot IDs."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires the GPU native extension", allow_module_level=True)

from kvcached.kv_cache_manager import _page_capacity  # noqa: E402
from kvcached.vmm_ops import InternalPage  # noqa: E402


def test_padded_page_returns_only_logical_mamba_slot_ids():
    mib = 1024 * 1024
    page = InternalPage(0, 12 * mib)
    page.init(3 * mib, 2)
    assert page.alloc(1) == [0]  # null slot
    assert page.alloc(1) == [1]
    page.free(1)
    assert page.alloc(1) == [1]
    assert page.num_free_blocks() == 0


def test_padding_ids_do_not_count_as_allocated_capacity():
    mib = 1024 * 1024
    assert _page_capacity(0, 12 * mib, 3 * mib,
                          max_block_id=2) == 2
