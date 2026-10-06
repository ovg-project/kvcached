# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Physical-page geometry of fixed-size KV blocks.

vLLM places block ``k`` of a pool at ``k * block_mem_size``, and kvcached maps
physical memory in ``page_size`` pages. A block that straddles a page boundary
belongs to no page (``InternalPage.get_block_range``), so when blocks do not
tile the page some block ids are never usable, and when a block is larger than
half a page some pages hold no whole block at all: such a page would be mapped
and could never be released. Pure functions only, so this is testable without
the compiled extension.
"""

from math import gcd
from typing import Optional

MIB = 1024 * 1024
# KVCACHED_PAGE_SIZE_MB must be a multiple of this.
PAGE_GRANULARITY = 2 * MIB
# Do not grow a block by more than this factor to make it tile a page.
MAX_BLOCK_GROWTH = 2


def has_zero_capacity_pages(block_mem_size: int, page_size: int) -> bool:
    """Whether some page would hold no whole block.

    Page starts visit multiples of ``gcd(page_size, block_mem_size)`` modulo
    the block size. The largest gap to the next block start is therefore
    ``block_mem_size - gcd(...)``. Every page holds a block exactly when it
    can fit that worst-case gap plus one complete block.
    """
    return page_size < 2 * block_mem_size - gcd(page_size, block_mem_size)


def select_page_size(block_mem_size: int, block_size: Optional[int] = None) -> int:
    """Smallest 2 MiB multiple with a whole block in every page.

    ``block_size`` enables the existing hybrid block alignment at each page
    candidate. Stop as soon as either the original or aligned block is safe.
    """
    if block_mem_size <= 0:
        raise ValueError("KV block size must be positive")
    page_size = ((block_mem_size + PAGE_GRANULARITY - 1)
                 // PAGE_GRANULARITY * PAGE_GRANULARITY)
    while has_zero_capacity_pages(block_mem_size, page_size):
        if (block_size and block_mem_size % block_size == 0
                and aligned_block_size(block_size, block_mem_size // block_size,
                                       page_size) is not None):
            break
        page_size += PAGE_GRANULARITY
    return page_size


def aligned_block_size(block_size: int, bytes_per_token: int,
                       page_size: int) -> Optional[int]:
    """Smallest block size whose block tiles ``page_size`` exactly.

    Candidates are ``block_size`` and larger multiples of its lowest
    power-of-two factor (the kernel alignment vLLM already applied), up to
    ``MAX_BLOCK_GROWTH`` times ``block_size`` and at most one page per block.
    Returns ``None`` when no candidate fits.
    """
    if block_size <= 0 or bytes_per_token <= 0:
        return None
    step = block_size & -block_size
    candidate = block_size
    while (candidate <= MAX_BLOCK_GROWTH * block_size
           and candidate * bytes_per_token <= page_size):
        if page_size % (candidate * bytes_per_token) == 0:
            return candidate
        candidate += step
    return None


def recommend_page_geometry(block_size: int,
                            bytes_per_token: int,
                            max_page_mb: int = 64) -> Optional[tuple[int, int]]:
    """Smallest ``(KVCACHED_PAGE_SIZE_MB, block_size)`` whose blocks tile the page."""
    for page_mb in range(PAGE_GRANULARITY // MIB, max_page_mb + 1,
                         PAGE_GRANULARITY // MIB):
        candidate = aligned_block_size(block_size, bytes_per_token,
                                       page_mb * MIB)
        if candidate is not None:
            return page_mb, candidate
    return None


def check_page_geometry(block_mem_size: int,
                        page_size: int,
                        block_size: Optional[int] = None) -> Optional[str]:
    """Return an error message when the block cannot be managed in this page.

    Unusable geometries: a block larger than a page (no page holds one), and a
    block that leaves some page with no whole block (that page would stay
    mapped forever). ``block_size`` (tokens per block) enables a concrete
    ``--block-size`` suggestion.
    """
    if block_mem_size <= page_size and not has_zero_capacity_pages(
            block_mem_size, page_size):
        return None
    page_mb = page_size // MIB
    problem = (
        f"the KV block ({block_mem_size} bytes, {block_mem_size / MIB:.2f} MiB) "
        f"is larger than the page ({page_mb} MiB)"
        if block_mem_size > page_size else
        f"the KV block ({block_mem_size} bytes, {block_mem_size / MIB:.2f} MiB) "
        f"does not tile the {page_mb} MiB page, so some pages would hold no "
        f"whole block and stay mapped")
    advice = []
    if block_size and block_mem_size % block_size == 0:
        bytes_per_token = block_mem_size // block_size
        here = aligned_block_size(block_size, bytes_per_token, page_size)
        if here is not None:
            advice.append(f"--block-size {here} with the current page")
        best = recommend_page_geometry(block_size, bytes_per_token)
        if best is not None and best[0] != page_mb:
            advice.append(
                f"KVCACHED_PAGE_SIZE_MB={best[0]} with --block-size {best[1]}")
    if not advice:
        tiling_mb = next(
            (mb for mb in range(2, 1025, 2)
             if (mb * MIB) % block_mem_size == 0 or 2 * block_mem_size <= mb * MIB),
            None)
        if tiling_mb is not None:
            advice.append(f"KVCACHED_PAGE_SIZE_MB={tiling_mb}")
    return (f"kvcached cannot manage this KV geometry: {problem}. "
            f"Re-launch with {' or '.join(advice) if advice else 'a larger KVCACHED_PAGE_SIZE_MB'}.")
