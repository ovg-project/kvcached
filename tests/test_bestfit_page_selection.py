# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Allocation picks the page an allocation fits into best (issue #359).

kvcached returns physical memory one page at a time, and only once every block
on that page is free. So it matters a great deal *which* page a request's
blocks come from: a request smeared over a dozen partly-filled pages pins all
twelve for as long as any one of its blocks survives in the prefix cache.

`avail_pages.popitem()` hands back the most recently touched page, which walks
whatever small holes sit at the tail of the dict. These tests pin down the
replacement rule: take the smallest page that still holds the whole remaining
run, or the emptiest page when none does.

The compiled kvcached.vmm_ops extension is stubbed out (as in
test_resize_reserved_order.py) so this runs on CPU-only hardware.
"""

import sys
import threading
import types

import pytest


def _install_fake_vmm_ops():
    """Stand in for the compiled extension so this runs without a GPU build.

    Installed only when the real one cannot be imported: it is a partial stub,
    and leaving it in sys.modules would break the tests that need the genuine
    module (test_kvcache_manager, test_paged_allocator_aliasing) whenever this
    file is collected before them.
    """

    class FakeInternalPage:
        pass

    class FakePageAllocator:

        def __init__(self, *args, **kwargs):
            pass

    fake = types.ModuleType("kvcached.vmm_ops")
    fake.PageAllocator = FakePageAllocator  # type: ignore[attr-defined]
    fake.InternalPage = FakeInternalPage  # type: ignore[attr-defined]
    fake.kv_tensors_created = lambda *a, **kw: True  # type: ignore[attr-defined]
    fake.map_to_kv_tensors = lambda *a, **kw: None  # type: ignore[attr-defined]
    fake.unmap_from_kv_tensors = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["kvcached.vmm_ops"] = fake


try:
    import kvcached.vmm_ops  # noqa: F401
except Exception:  # noqa: BLE001 - any import failure means no GPU build
    _install_fake_vmm_ops()

from kvcached.kv_cache_manager import KVCacheManager  # noqa: E402
from kvcached.locks import NoOpLock  # noqa: E402

BLOCKS_PER_PAGE = 64


class FakePage:
    """A page that hands out its own block ids, so victims trace back to it."""

    def __init__(self, page_id: int, num_free: int):
        self.page_id = page_id
        base = page_id * BLOCKS_PER_PAGE
        # The free slots sit at the end of the page, as they would after the
        # earlier blocks were handed out.
        self.free_list = list(range(base + BLOCKS_PER_PAGE - num_free,
                                    base + BLOCKS_PER_PAGE))

    def num_free_blocks(self) -> int:
        return len(self.free_list)

    def alloc(self, n: int):
        got, self.free_list = self.free_list[:n], self.free_list[n:]
        return got

    def full(self) -> bool:
        return not self.free_list


def _make_manager(pages):
    """A manager with only what _alloc / _pick_avail_page read.

    __init__ is skipped: it starts a background thread and talks to the real
    page allocator.
    """
    mgr = object.__new__(KVCacheManager)
    mgr._lock = NoOpLock()
    mgr.block_mem_size = 32 * 1024
    mgr.reserved_blocks = []
    mgr.avail_pages = {p.page_id: p for p in pages}
    mgr.full_pages = {}
    mgr.num_avail_blocks = sum(p.num_free_blocks() for p in pages)
    mgr._post_init_done = threading.Event()
    mgr._post_init_done.set()

    class StubPageAllocator:

        def get_resize_target(self):
            return -1

        def alloc_page(self):  # pragma: no cover - the tests never run dry
            raise AssertionError("test should not need a fresh page")

    mgr.page_allocator = StubPageAllocator()
    mgr.available_size = lambda: mgr.num_avail_blocks
    return mgr


def _pages_touched(block_ids):
    return {bid // BLOCKS_PER_PAGE for bid in block_ids}


class TestPickAvailPage:

    def test_takes_the_smallest_page_that_fits_the_whole_run(self):
        mgr = _make_manager([FakePage(10, 5), FakePage(11, 20),
                             FakePage(12, 40)])
        assert mgr._pick_avail_page(10).page_id == 11, (
            "20 free is the tightest fit for 10 blocks; 40 wastes a big hole")

    def test_falls_back_to_the_emptiest_page_when_none_fits(self):
        mgr = _make_manager([FakePage(10, 5), FakePage(11, 20),
                             FakePage(12, 40)])
        assert mgr._pick_avail_page(50).page_id == 12, (
            "no page holds 50, so take the biggest bite available")

    def test_exact_fit_wins(self):
        mgr = _make_manager([FakePage(10, 16), FakePage(11, 17)])
        assert mgr._pick_avail_page(16).page_id == 10

    def test_removes_the_page_it_returns(self):
        mgr = _make_manager([FakePage(10, 5), FakePage(11, 20)])
        page = mgr._pick_avail_page(4)
        assert page.page_id not in mgr.avail_pages, (
            "the caller re-inserts the page itself, as with popitem()")


class TestAllocationKeepsARunTogether:

    def test_a_run_lands_on_one_page_instead_of_walking_the_small_holes(self):
        """The regression this exists for.

        The 40-free page is inserted first, so it sits at the head of the dict
        and the small holes sit at the tail -- exactly where `popitem()` looks.
        Draining those first spreads a 40-block request over six pages, and
        every one of them stays pinned while any of those blocks is cached.
        """
        pages = [FakePage(12, 40)] + [FakePage(pid, n) for pid, n in
                                      ((13, 2), (14, 3), (15, 2), (16, 1),
                                       (17, 2))]
        mgr = _make_manager(pages)

        block_ids = mgr._alloc(40)

        assert block_ids is not None and len(block_ids) == 40
        assert _pages_touched(block_ids) == {12}, (
            "a run that fits in one page must not be smeared across the holes")

    def test_a_run_too_big_for_any_page_still_allocates_everything(self):
        """Falling back must stay correct, not just tidy."""
        mgr = _make_manager([FakePage(20, 10), FakePage(21, 30),
                             FakePage(22, 20)])

        block_ids = mgr._alloc(55)

        assert block_ids is not None
        assert len(block_ids) == 55
        assert len(set(block_ids)) == 55, "no block may be handed out twice"
        # Biggest bite first, so the emptiest page is drained before the rest.
        assert _pages_touched(block_ids) == {20, 21, 22}

    def test_single_block_allocations_are_unaffected_in_count(self):
        """Decode asks for one block at a time; it must still get exactly one."""
        mgr = _make_manager([FakePage(30, 4), FakePage(31, 64)])
        for _ in range(4):
            got = mgr._alloc(1)
            assert got is not None and len(got) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
