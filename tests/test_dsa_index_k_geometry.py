# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""CPU-only page geometry for SGLang DSA index-K storage.

SGLang's ``DSATokenToKVPool`` allocates one uint8 index-K tensor per layer.
At the current geometry one token is 132 B and a block is 64 tokens, so a
block is 8,448 B. This pins the geometry before the elastic adapter changes
the tensor allocation path; no SGLang install or GPU is required.
"""

import pytest

from kvcached.kv_geometry import (
    MIB,
    backing_blocks_for_capacity,
    check_page_geometry,
    has_zero_capacity_pages,
    select_page_size,
)

BLOCK_TOKENS = 64
BYTES_PER_TOKEN = 132
BLOCK_BYTES = BLOCK_TOKENS * BYTES_PER_TOKEN
PAGE_BYTES = 2 * MIB


def _block_range(page_id: int) -> tuple[int, int]:
    """InternalPage.get_block_range for the fixed page/block geometry."""
    return (
        (page_id * PAGE_BYTES + BLOCK_BYTES - 1) // BLOCK_BYTES,
        ((page_id + 1) * PAGE_BYTES) // BLOCK_BYTES,
    )


def _page_capacity(page_id: int) -> int:
    start, end = _block_range(page_id)
    return max(0, end - start)


def test_index_k_block_geometry_is_pinned():
    assert BLOCK_BYTES == 8_448
    assert PAGE_BYTES // BLOCK_BYTES == 248
    assert PAGE_BYTES % BLOCK_BYTES == 2_048


def test_index_k_pages_hold_whole_blocks_but_straddle_boundaries():
    # Page 0 contains blocks 0..247. Page 1 starts at block 249 because block
    # 248 crosses the page boundary and belongs to neither page.
    assert _block_range(0) == (0, 248)
    assert _block_range(1) == (249, 496)
    assert [_page_capacity(page) for page in range(3)] == [248, 247, 247]
    assert sum(_page_capacity(page) for page in range(2)) == 495
    assert not has_zero_capacity_pages(BLOCK_BYTES, PAGE_BYTES)
    assert check_page_geometry(BLOCK_BYTES, PAGE_BYTES, BLOCK_TOKENS) is None
    assert select_page_size(BLOCK_BYTES) == PAGE_BYTES


def test_index_k_backing_extent_covers_logical_blocks():
    # 256 logical blocks fit in two pages (495 usable blocks), but the
    # allocator's block extent is 497 because it rounds the backing byte range
    # up to whole blocks.
    assert backing_blocks_for_capacity(256, BLOCK_BYTES, PAGE_BYTES) == 497
    assert sum(_page_capacity(page) for page in range(2)) >= 256
    assert sum(_page_capacity(page) for page in range(1)) < 256


@pytest.mark.parametrize("logical,expected", [(0, 0), (1, 249), (248, 249), (249, 497), (495, 497)])
def test_index_k_backing_extent_boundaries(logical: int, expected: int):
    assert backing_blocks_for_capacity(logical, BLOCK_BYTES, PAGE_BYTES) == expected


def test_skipped_index_k_layer_must_request_zero_blocks():
    # A skipped layer must not allocate a zero-row tensor that still rounds
    # up to a page. This is the contract the adapter should preserve.
    assert backing_blocks_for_capacity(0, BLOCK_BYTES, PAGE_BYTES) == 0
