# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for page-aware eviction in ElasticBlockPool (issue #359).

kvcached only unmaps a page once every block on it is free, so evicting blocks
in pure LRU order can return no memory at all when the retained blocks stay
scattered. These tests mock KVCacheManager's page bookkeeping to assert that
eviction prefers victims that empty whole pages.

No GPU or vLLM dependency: torch and the C extension are mocked, as in
test_prefix_cache.py.
"""

import sys
import types
from unittest import mock

_torch_mock = mock.MagicMock()
_torch_mock.__version__ = "2.6.0"
_torch_mock.cuda.mem_get_info.return_value = (8 * 1024**3, 16 * 1024**3)
sys.modules.setdefault("torch", _torch_mock)
sys.modules.setdefault("torch.cuda", _torch_mock.cuda)
sys.modules.setdefault("torch.utils", _torch_mock.utils)
sys.modules.setdefault("torch.utils.cpp_extension", _torch_mock.utils.cpp_extension)
sys.modules.setdefault("posix_ipc", mock.MagicMock())
sys.modules.setdefault("kvcached.vmm_ops", mock.MagicMock())
sys.modules.setdefault("kvcached.integration.vllm.interfaces", mock.MagicMock())

import pytest  # noqa: E402

BLOCKS_PER_PAGE = 4


class MockBlockPool:
    pass


class MockKVCacheBlock:

    def __init__(self, block_id: int, ref_cnt: int = 0):
        self.block_id = block_id
        self.ref_cnt = ref_cnt
        self.is_null = False


class MockRequest:

    def __init__(self, block_hashes: list):
        self.block_hashes = block_hashes


class MockPageAllocator:
    """Groups block ids by page the same way the C++ allocator does."""

    def group_indices_by_page(self, indices, block_mem_size):
        by_page: dict = {}
        for idx in indices:
            by_page.setdefault(idx // BLOCKS_PER_PAGE, []).append(idx)
        return by_page


class MockPagedKVCacheManager:
    """Allocator that tracks which blocks are live on each page.

    Mirrors the real manager closely enough for eviction: alloc/free move ids,
    and get_page_occupancy reports how many blocks a page still holds.
    """

    def __init__(self, num_blocks: int):
        self._free: list[int] = list(range(num_blocks))
        self._allocated: set[int] = set()
        self.page_allocator = MockPageAllocator()
        self.block_mem_size = 16 * 1024
        self.page_size = BLOCKS_PER_PAGE * self.block_mem_size

    def alloc(self, n: int):
        if len(self._free) < n:
            return None
        ids = self._free[:n]
        self._free = self._free[n:]
        self._allocated.update(ids)
        return ids

    def free(self, ids: list):
        for i in ids:
            self._allocated.discard(i)
            self._free.append(i)

    def available_size(self) -> int:
        return len(self._free)

    def get_page_occupancy(self, page_ids: list) -> dict:
        occupancy = {pid: 0 for pid in page_ids}
        for bid in self._allocated:
            pid = bid // BLOCKS_PER_PAGE
            if pid in occupancy:
                occupancy[pid] += 1
        return occupancy

    def pages_pinned(self) -> int:
        """Pages holding at least one live block -- what stays mapped."""
        return len({bid // BLOCKS_PER_PAGE for bid in self._allocated})


@pytest.fixture
def pool_factory():

    def _make(num_blocks: int = 64):
        manager = MockPagedKVCacheManager(num_blocks)

        mock_mod = types.ModuleType("mock_block_pool")
        mock_mod.BlockPool = MockBlockPool  # type: ignore[attr-defined]
        mock_mod.KVCacheBlock = MockKVCacheBlock  # type: ignore[attr-defined]

        with mock.patch(
                "kvcached.integration.vllm.interfaces.get_kv_cache_manager",
                return_value=manager,
        ):
            from kvcached.integration.vllm.patches import ElasticBlockPoolPatch

            ElasticBlockPoolPatch().inject_elastic_block_pool(mock_mod)
            ElasticBlockPool = mock_mod.ElasticBlockPool  # type: ignore[attr-defined]
            pool = ElasticBlockPool(
                num_gpu_blocks=num_blocks,
                block_size=16,
                cell_size=1024,
                num_layers=1,
                enable_caching=True,
            )
        return pool, manager

    return _make


def _cache_n(pool, n: int) -> list:
    """Allocate, cache and release `n` blocks so they become evictable.

    Ids come from the pool itself, so the null block the constructor holds is
    accounted for rather than overwritten.
    """
    blocks = pool.get_new_blocks(n)
    req = MockRequest([f"h{b.block_id}" for b in blocks])
    pool.cache_full_blocks(req, blocks, 0, n, 16, 0)
    pool.free_blocks(blocks)
    return blocks


class TestPageAwareEviction:

    def test_evicting_a_pages_worth_frees_a_page(self, pool_factory):
        """Evicting BLOCKS_PER_PAGE blocks should empty one page outright.

        The null block the pool allocates at construction occupies page 0, so
        caching 8 more blocks spans pages 0..2. Evicting 4 must concentrate on
        one page rather than take blocks from each and free nothing.
        """
        pool, mgr = pool_factory(64)
        _cache_n(pool, 8)

        before = mgr.pages_pinned()
        pool._evict_blocks_from_pool(BLOCKS_PER_PAGE)
        assert mgr.pages_pinned() == before - 1, (
            "evicting one page's worth of blocks should free exactly one page")

    def test_skips_pages_pinned_by_active_blocks(self, pool_factory):
        """A page holding a running request's block cannot be emptied.

        One block is left active (ref_cnt>0) so its page can never be freed by
        eviction; the pages that are fully evictable should be chosen instead.
        """
        pool, mgr = pool_factory(64)
        blocks = _cache_n(pool, 8)

        # Re-activate one block: it leaves the evictable pool but keeps its page.
        pool.touch([blocks[0]])
        pinned_page = blocks[0].block_id // BLOCKS_PER_PAGE

        pool._evict_blocks_from_pool(BLOCKS_PER_PAGE)
        assert blocks[0].block_id in mgr._allocated
        assert pinned_page in {b // BLOCKS_PER_PAGE for b in mgr._allocated}

    def test_evicts_requested_count(self, pool_factory):
        """Page preference must not change how many blocks get evicted."""
        pool, mgr = pool_factory(64)
        _cache_n(pool, 8)
        assert len(pool._evictable_blocks) == 8

        evicted = pool._evict_blocks_from_pool(6)
        assert evicted == 6
        assert len(pool._evictable_blocks) == 2

    def test_falls_back_to_lru_without_page_allocator(self, pool_factory):
        """A manager without page bookkeeping still evicts in LRU order."""
        pool, mgr = pool_factory(64)
        blocks = _cache_n(pool, 8)
        del mgr.page_allocator

        evicted = pool._evict_blocks_from_pool(3)
        assert evicted == 3
        # The three oldest cached blocks go first.
        for block in blocks[:3]:
            assert pool.get_cached_block(f"h{block.block_id}", [0]) is None
        assert pool.get_cached_block(f"h{blocks[3].block_id}", [0]) is not None

    def test_lru_eviction_ignores_page_layout(self, pool_factory):
        """page_aware=False evicts the LRU victim even off a fuller page.

        The allocation-shortage path reuses the freed slot immediately, so no
        page is unmapped and page-aware selection would only swap a newer
        prefix for an older one. Lay out the oldest cached block on a full page
        and the newest alone on a nearly-empty page: page-aware would take the
        newest to empty its page, so LRU must instead take the oldest.
        """
        pool, mgr = pool_factory(64)
        blocks = _cache_n(pool, 8)
        oldest, newest = blocks[0], blocks[-1]
        # The lone newest block sits on a page by itself; the oldest shares a
        # full page. Page-aware selection would prefer the newest here.
        assert (sum(1 for b in mgr._allocated
                    if b // BLOCKS_PER_PAGE
                    == newest.block_id // BLOCKS_PER_PAGE) == 1)

        pool._evict_blocks_from_pool(1, page_aware=False)

        # LRU dropped the oldest; the newest prefix survives.
        assert pool.get_cached_block(f"h{oldest.block_id}", [0]) is None
        assert pool.get_cached_block(f"h{newest.block_id}", [0]) is not None
