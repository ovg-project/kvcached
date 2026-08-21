# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Regression test for boundary-aware block counting in
``KVCacheManager._get_num_alloced_blocks``.

When ``block_mem_size`` does not evenly divide ``page_size`` (HYBRID_LINEAR /
Mamba GDN per-block state), ``InternalPage.get_block_range`` — which
``get_page_occupancy`` already uses — drops blocks straddling the page
boundary. Some page ids therefore yield *zero* usable blocks while
``InternalPage.get_num_blocks`` reports one or more. The 0-block pages are
parked in ``full_pages`` (kv_cache_manager.py:335) but were still counted by
the old ``len(self.full_pages) * get_num_blocks(...)`` in
``_get_num_alloced_blocks``, inflating the lazy-shrink completion gate
``_get_num_alloced_blocks() <= target_num_blocks`` and stalling the
operator's ``kvctl limit`` shrink.

GPU-free: ``kv_cache_manager`` is imported via a stubbed ``kvcached.vmm_ops``
(only the import-time entrypoints are needed); ``InternalPage`` is force-bound
to ``RichInternalPage`` by an autouse fixture regardless of which other cpu
test installed ``vmm_ops`` first.

Red on master: ``_get_num_alloced_blocks`` sizes every page with
``get_num_blocks`` (theoretical ``page_size // block_mem_size``), so a parked
0-block page is counted as 1. Green on branch: it sizes each page with
``_page_capacity`` (= ``get_block_range`` end - start), so a parked 0-block
page contributes 0.
"""
import sys
import types

import pytest


class RichInternalPage:
    """Mirror of the REAL InternalPage static methods (csrc/page_allocator.cpp:90-103)
    so ``_page_capacity`` resolves to boundary-aware math without the compiled
    extension or a GPU."""

    @staticmethod
    def get_block_range(page_id, page_size, block_mem_size):
        start_block = (page_id * page_size +
                       block_mem_size - 1) // block_mem_size
        end_block = ((page_id + 1) * page_size) // block_mem_size
        return (start_block, end_block)

    @staticmethod
    def get_num_blocks(page_size, block_mem_size):
        return page_size // block_mem_size


def _install_fake_vmm_ops():
    """Register a minimal fake ``kvcached.vmm_ops`` so ``kv_cache_manager``'s
    hard import succeeds without the compiled CUDA/HIP extension. Only the
    PageAllocator / kv_tensors entrypoints are needed for import; the
    per-page static methods are supplied by ``RichInternalPage`` via the autouse
    fixture below, so this stub's InternalPage is only a fallback for the
    very first import (before any other cpu test has stubbed it)."""

    class FakePageAllocator:
        pass

    fake = types.ModuleType("kvcached.vmm_ops")
    fake.PageAllocator = FakePageAllocator  # type: ignore[attr-defined]
    fake.InternalPage = RichInternalPage  # type: ignore[attr-defined]
    fake.kv_tensors_created = lambda *a, **kw: True  # type: ignore[attr-defined]
    fake.map_to_kv_tensors = lambda *a, **kw: None  # type: ignore[attr-defined]
    fake.unmap_from_kv_tensors = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["kvcached.vmm_ops"] = fake


try:
    import kvcached.vmm_ops  # noqa: F401
except Exception:  # noqa: BLE001 - any import failure means no GPU build
    _install_fake_vmm_ops()

import kvcached.kv_cache_manager as _kcm  # noqa: E402
from kvcached.kv_cache_manager import KVCacheManager  # noqa: E402
from kvcached.locks import NoOpLock  # noqa: E402


@pytest.fixture(autouse=True)
def _bind_rich_internal_page():
    """Force ``kv_cache_manager.InternalPage`` to ``RichInternalPage`` for every
    test in this module, regardless of which cpu test installed ``vmm_ops``
    first (others install an empty FakeInternalPage and may have already bound
    the cached module global before this module collected). Restored after each
    test so the swap does not leak into sibling test modules."""
    saved = _kcm.InternalPage
    _kcm.InternalPage = RichInternalPage
    try:
        yield
    finally:
        _kcm.InternalPage = saved


def _make_manager(page_size, block_mem_size, full_pages=None,
                  avail_pages=None, num_avail_blocks=0):
    """Build a KVCacheManager without running __init__ (which spawns a
    background thread and talks to the real page allocator); set exactly
    the attributes _get_num_alloced_blocks reads."""
    manager = object.__new__(KVCacheManager)
    manager._lock = NoOpLock()
    manager.page_size = page_size
    manager.block_mem_size = block_mem_size
    manager.full_pages = full_pages or {}
    manager.avail_pages = avail_pages or {}
    manager.num_avail_blocks = num_avail_blocks
    return manager


def test_page_capacity_mirrors_get_block_range_not_get_num_blocks():
    """_page_capacity must drop blocks straddling the page boundary.

    page_size=4, block_mem_size=3: page_id=1's block range is [2,2) — zero
    usable blocks — while get_num_blocks reports 1 for every page id. This
    is the 0-block page that _alloc parks in full_pages.
    """
    # Imported lazily so the bug-reproduction tests below collect and run on
    # master (where _page_capacity does not exist yet) instead of failing at
    # import time.
    from kvcached.kv_cache_manager import _page_capacity

    assert _page_capacity(0, 4, 3) == 1  # range [0, 1)
    assert _page_capacity(1, 4, 3) == 0  # range [2, 2) — parked 0-block page
    assert _page_capacity(2, 4, 3) == 1  # range [3, 4)
    # The theoretical get_num_blocks reports 1 for every page id, so counting
    # page_id=1 with it inflates capacity by 1.
    assert RichInternalPage.get_num_blocks(4, 3) == 1


def test_get_num_alloced_blocks_counts_parked_zero_block_page_as_zero():
    """A parked 0-block page (page_id=1) must contribute 0, not 1.

    Master: len(full_pages) * get_num_blocks(4,3) = 1 * 1 = 1 (inflated).
    Branch: _page_capacity(1, 4, 3) = 0 (boundary-aware).
    """
    manager = _make_manager(
        page_size=4,
        block_mem_size=3,
        full_pages={1: object()},  # page_id=1, 0 usable blocks
        avail_pages={},
        num_avail_blocks=0,
    )

    assert manager._get_num_alloced_blocks() == 0


def test_get_num_alloced_blocks_sums_per_page_capacity():
    """Mixed full_pages: page_id=1 (0 blocks) + page_id=2 (1 block) = 1.

    Master: 2 * get_num_blocks(4,3) = 2 (inflated by the parked page).
    Branch: 0 + 1 = 1.
    """
    manager = _make_manager(
        page_size=4,
        block_mem_size=3,
        full_pages={1: object(), 2: object()},
        avail_pages={},
        num_avail_blocks=0,
    )

    assert manager._get_num_alloced_blocks() == 1


def test_get_num_alloced_blocks_avail_pages_uses_boundary_capacity():
    """page_size=7, block_mem_size=3: page_id=1's range is [3,4) — 1 usable
    block — while get_num_blocks reports 2. A partially-allocated page_id=1
    with 0 free blocks must count as 1 allocated, not 2.

    Master: 1 * get_num_blocks(7,3) - 0 = 2 (inflated).
    Branch: _page_capacity(1, 7, 3) - 0 = 1.
    """
    manager = _make_manager(
        page_size=7,
        block_mem_size=3,
        full_pages={},
        avail_pages={1: object()},
        num_avail_blocks=0,
    )

    assert manager._get_num_alloced_blocks() == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
