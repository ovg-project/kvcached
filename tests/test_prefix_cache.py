# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ElasticBlockPool prefix cache (step 2a: lazy eviction).

These tests mock kvcached's KVCacheManager and vLLM's BlockPool/KVCacheBlock
to test the prefix cache logic in isolation without GPU or vLLM dependency.
"""

import sys
import types
from typing import Optional
from unittest import mock

# ---------------------------------------------------------------------------
# Pre-mock heavy dependencies so kvcached can be imported without torch/CUDA.
# This MUST happen before any kvcached import.
# ---------------------------------------------------------------------------
_torch_mock = mock.MagicMock()
_torch_mock.__version__ = "2.6.0"
_torch_mock.cuda.mem_get_info.return_value = (8 * 1024**3, 16 * 1024**3)
sys.modules.setdefault("torch", _torch_mock)
sys.modules.setdefault("torch.cuda", _torch_mock.cuda)
sys.modules.setdefault("torch.utils", _torch_mock.utils)
sys.modules.setdefault("torch.utils.cpp_extension", _torch_mock.utils.cpp_extension)
sys.modules.setdefault("posix_ipc", mock.MagicMock())

# Mock the C extension module and heavy submodules
sys.modules.setdefault("kvcached.vmm_ops", mock.MagicMock())

# Pre-mock the interfaces module so mock.patch can resolve its attributes.
# This avoids importing torch / C extensions transitively via interfaces.py.
sys.modules.setdefault("kvcached.integration.vllm.interfaces",
                       mock.MagicMock())
import kvcached.integration.vllm as _vllm_pkg  # noqa: E402

# Binding the stub into sys.modules is not enough: mock.patch() resolves a
# dotted target with getattr() on the parent package, and a hand-installed
# sys.modules entry never sets that attribute. Point it at whatever is
# actually in sys.modules -- another test module may have stubbed this first,
# and patching a different object than the code imports silently does nothing.
_vllm_pkg.interfaces = sys.modules["kvcached.integration.vllm.interfaces"]

import pytest  # noqa: E402

# ---------------------------------------------------------------------------
# Mock classes
# ---------------------------------------------------------------------------

class MockBlockPool:
    """Minimal stand-in for vLLM's BlockPool base class."""
    pass


class MockKVCacheBlock:
    """Minimal stand-in for vLLM's KVCacheBlock."""

    def __init__(self, block_id: int, ref_cnt: int = 0):
        self.block_id = block_id
        self.ref_cnt = ref_cnt
        self.is_null = False
        self._block_hash: Optional[object] = None
        self._block_hash_num_tokens: Optional[int] = None

    @property
    def block_hash(self):
        return self._block_hash

    def set_block_hash(self, value, num_tokens=None):
        assert self._block_hash is None
        self._block_hash = value
        self._block_hash_num_tokens = num_tokens

    def reset_hash(self):
        self._block_hash = None
        self._block_hash_num_tokens = None


class MockKVCacheManager:
    """Simple allocator that tracks block IDs.

    alloc(n) pops from a free list; free(ids) pushes back.
    This is enough to test ElasticBlockPool's cache logic.
    """

    def __init__(self, num_blocks: int):
        # Mirror KVCacheManager(reserve_null_block=True): block 0 is reserved
        # as the null block during manager init and never enters circulation,
        # so the free list starts at 1 (same id sequence the old code produced
        # by allocating block 0 for the null block).
        self._free: list[int] = list(range(1, num_blocks))
        self._allocated: set[int] = set()
        self.null_block = [0]

    def alloc(self, n: int):
        if len(self._free) < n:
            return None
        ids = self._free[:n]
        self._free = self._free[n:]
        self._allocated.update(ids)
        return ids

    def free(self, ids: list[int]):
        for i in ids:
            self._allocated.discard(i)
            self._free.append(i)

    def available_size(self) -> int:
        return len(self._free)


class MockRequest:
    """Minimal stand-in for vLLM's Request."""

    def __init__(self, block_hashes: list):
        self.block_hashes = block_hashes


class MockBlockHashListWithBlockSize:
    def __init__(self, block_hashes, hash_block_size, target_block_size):
        assert target_block_size % hash_block_size == 0
        self.block_hashes = block_hashes
        self.scale_factor = target_block_size // hash_block_size

    def __len__(self):
        return len(self.block_hashes) // self.scale_factor

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        return self.block_hashes[(index + 1) * self.scale_factor - 1]


def test_set_block_hash_supports_legacy_writable_property():
    """vLLM before 0.24 uses a writable block_hash property."""
    from kvcached.integration.vllm.patches import _set_block_hash

    block = types.SimpleNamespace(block_hash=None)
    key = b"legacy-key"

    _set_block_hash(block, key)

    assert block.block_hash == key


# ---------------------------------------------------------------------------
# Fixture: create an ElasticBlockPool via the real patch injection
# ---------------------------------------------------------------------------

@pytest.fixture
def pool_factory(monkeypatch):
    """Factory that builds an ElasticBlockPool with a given number of blocks.

    Returns (pool, manager) so tests can inspect both.
    """

    def _make(num_blocks: int = 100, enable_caching: bool = True, manager=None):
        if manager is None:
            manager = MockKVCacheManager(num_blocks)

        kv_cache_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")
        setattr(
            kv_cache_utils,
            "BlockHashListWithBlockSize",
            MockBlockHashListWithBlockSize,
        )
        monkeypatch.setitem(
            sys.modules, "vllm.v1.core.kv_cache_utils", kv_cache_utils
        )

        # Build a mock module that looks like vllm.v1.core.block_pool
        mock_mod = types.ModuleType("mock_block_pool")
        mock_mod.BlockPool = MockBlockPool  # type: ignore[attr-defined]
        mock_mod.KVCacheBlock = MockKVCacheBlock  # type: ignore[attr-defined]

        with mock.patch(
            "kvcached.integration.vllm.interfaces.get_kv_cache_manager",
            return_value=manager,
        ):
            from kvcached.integration.vllm.patches import ElasticBlockPoolPatch

            patch = ElasticBlockPoolPatch()
            # Call the injection method directly, bypassing version detection.
            # @version_range only adds metadata, doesn't wrap the function.
            patch.inject_elastic_block_pool(mock_mod)

        ElasticBlockPool = mock_mod.ElasticBlockPool  # type: ignore[attr-defined]

        with mock.patch(
            "kvcached.integration.vllm.interfaces.get_kv_cache_manager",
            return_value=manager,
        ):
            pool = ElasticBlockPool(
                num_gpu_blocks=num_blocks,
                block_size=16,
                cell_size=1024,
                num_layers=1,
                enable_caching=enable_caching,
            )

        return pool, manager

    return _make


@pytest.fixture
def pool_and_manager(pool_factory):
    """Convenience fixture: 100-block pool with caching enabled."""
    return pool_factory(100, True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_request(pool, hashes: list, blocks=None):
    """Simulate a full request lifecycle: alloc -> cache -> return blocks.

    Returns the list of KVCacheBlock objects.
    """
    n = len(hashes)
    new_blocks = pool.get_new_blocks(n)
    req = MockRequest(hashes)
    pool.cache_full_blocks(req, new_blocks, 0, n, 16, 0)
    return new_blocks


def _finish_request(pool, blocks):
    """Simulate request completion: free all blocks."""
    pool.free_blocks(blocks)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLazyEviction:
    """Step 2a: blocks with ref_cnt==0 stay in cache for cross-request reuse."""

    def test_cross_request_cache_hit(self, pool_and_manager):
        """After request A finishes, request B with same prefix gets a cache hit."""
        pool, mgr = pool_and_manager
        hashes = ["h0", "h1", "h2", "h3"]

        # Request A
        blocks_a = _simulate_request(pool, hashes)
        assert all(block.block_hash is not None for block in blocks_a)
        _finish_request(pool, blocks_a)

        # All blocks should be in evictable pool, NOT freed to kvcached
        assert len(pool._evictable_blocks) == 4
        assert len(pool._cached_blocks) == 4

        # Request B: same prefix -> cache hit
        for h in hashes:
            result = pool.get_cached_block(h, [0])
            assert result is not None, f"Expected cache hit for {h}"

    def test_evictable_blocks_not_freed_to_kvcached(self, pool_and_manager):
        """Evictable blocks hold their kvcached allocation (pages stay mapped)."""
        pool, mgr = pool_and_manager
        initial_free = mgr.available_size()

        blocks = _simulate_request(pool, ["h0", "h1"])
        alloc_free = mgr.available_size()
        assert alloc_free == initial_free - 2

        _finish_request(pool, blocks)
        # Blocks in evictable pool: kvcached free should NOT increase
        assert mgr.available_size() == alloc_free

    def test_uncached_blocks_freed_immediately(self, pool_and_manager):
        """Blocks that were never cached (no hash) are freed on ref_cnt==0."""
        pool, mgr = pool_and_manager
        initial_free = mgr.available_size()

        # Allocate blocks but do NOT call cache_full_blocks
        blocks = pool.get_new_blocks(3)
        assert mgr.available_size() == initial_free - 3

        pool.free_blocks(blocks)
        # Not cached -> freed immediately
        assert len(pool._evictable_blocks) == 0
        assert mgr.available_size() == initial_free
        assert all(block.block_hash is None for block in blocks)

    def test_touch_removes_from_evictable_pool(self, pool_and_manager):
        """Touching an evictable block reactivates it (removes from pool)."""
        pool, mgr = pool_and_manager
        hashes = ["h0", "h1"]

        blocks = _simulate_request(pool, hashes)
        _finish_request(pool, blocks)
        assert len(pool._evictable_blocks) == 2

        # Simulate cache hit -> touch
        hit_block = pool.get_cached_block("h0", [0])[0]
        pool.touch([hit_block])

        assert hit_block.block_id not in pool._evictable_blocks
        assert len(pool._evictable_blocks) == 1
        assert hit_block.ref_cnt == 1

    def test_touch_tuple_form(self, pool_and_manager):
        """Touch works with tuple-of-lists form (multi-group compat)."""
        pool, mgr = pool_and_manager
        blocks = _simulate_request(pool, ["h0", "h1"])
        _finish_request(pool, blocks)

        hit = pool.get_cached_block("h0", [0])[0]
        pool.touch(([hit],))

        assert hit.block_id not in pool._evictable_blocks
        assert hit.ref_cnt == 1

    def test_ref_cnt_shared_prefix(self, pool_and_manager):
        """Two concurrent requests sharing a prefix: ref_cnt tracks correctly."""
        pool, mgr = pool_and_manager
        hashes = ["h0", "h1"]

        # Request A allocates and caches blocks
        blocks_a = _simulate_request(pool, hashes)
        assert all(b.ref_cnt == 1 for b in blocks_a)

        # Request B hits the cache
        hit_blocks = []
        for h in hashes:
            result = pool.get_cached_block(h, [0])
            assert result is not None
            hit_blocks.append(result[0])
        pool.touch(hit_blocks)
        assert all(b.ref_cnt == 2 for b in blocks_a)

        # Request A finishes
        _finish_request(pool, blocks_a)
        assert all(b.ref_cnt == 1 for b in blocks_a)
        assert len(pool._evictable_blocks) == 0  # still active via B

        # Request B finishes
        _finish_request(pool, hit_blocks)
        assert all(b.ref_cnt == 0 for b in blocks_a)
        assert len(pool._evictable_blocks) == 2  # now evictable


class TestEvictOnDemand:
    """Eviction from the evictable pool when get_new_blocks needs space."""

    def test_evict_on_alloc_pressure(self, pool_factory):
        """When kvcached is full, get_new_blocks evicts from the pool."""
        pool, mgr = pool_factory(11)  # +1 for null_block

        # Fill all 10 blocks and cache them
        blocks = _simulate_request(pool, [f"h{i}" for i in range(10)])
        _finish_request(pool, blocks)
        assert mgr.available_size() == 0
        assert len(pool._evictable_blocks) == 10

        # get_num_free_blocks counts evictable as available
        assert pool.get_num_free_blocks() == 10

        # Allocate 4 new blocks -- must evict 4 from pool
        new_blocks = pool.get_new_blocks(4)
        assert len(new_blocks) == 4
        assert len(pool._evictable_blocks) == 6
        assert mgr.available_size() == 0  # 4 evicted, 4 re-allocated

    def test_evict_lru_order(self, pool_factory):
        """Oldest blocks (first inserted) are evicted first."""
        pool, mgr = pool_factory(11)  # +1 for null_block

        # Cache blocks h0..h9 in order
        blocks = _simulate_request(pool, [f"h{i}" for i in range(10)])
        _finish_request(pool, blocks)

        # Evict 3 -> should evict h0, h1, h2 (oldest)
        evicted_blocks = blocks[:3]
        pool._evict_blocks_from_pool(3)

        assert pool.get_cached_block("h0", [0]) is None
        assert pool.get_cached_block("h1", [0]) is None
        assert pool.get_cached_block("h2", [0]) is None
        assert pool.get_cached_block("h3", [0]) is not None  # still cached
        assert all(block.block_hash is None for block in evicted_blocks)

    def test_evict_all_then_alloc(self, pool_factory):
        """Can evict entire pool and allocate fresh blocks."""
        pool, mgr = pool_factory(6)  # +1 for null_block

        blocks = _simulate_request(pool, ["a", "b", "c", "d", "e"])
        _finish_request(pool, blocks)
        assert mgr.available_size() == 0
        assert len(pool._evictable_blocks) == 5

        new_blocks = pool.get_new_blocks(5)
        assert len(new_blocks) == 5
        assert len(pool._evictable_blocks) == 0
        assert len(pool._cached_blocks) == 0

    def test_partial_eviction(self, pool_factory):
        """Evict only as many as needed, keep the rest."""
        pool, mgr = pool_factory(11)  # +1 for null_block

        blocks = _simulate_request(pool, [f"h{i}" for i in range(10)])
        _finish_request(pool, blocks)

        # 10 in pool, kvcached has 0 free. Need 2 new blocks.
        new_blocks = pool.get_new_blocks(2)
        assert len(pool._evictable_blocks) == 8  # 10 - 2 evicted
        assert len(new_blocks) == 2

    def test_no_eviction_when_kvcached_has_space(self, pool_factory):
        """Don't evict if kvcached already has enough free blocks."""
        pool, mgr = pool_factory(21)  # +1 for null_block

        # Use 5 blocks, cache them, free them
        blocks = _simulate_request(pool, [f"h{i}" for i in range(5)])
        _finish_request(pool, blocks)
        assert len(pool._evictable_blocks) == 5
        assert mgr.available_size() == 15  # 21-1(null)-5 = 15

        # Allocate 3 -- kvcached has 15 free, no eviction needed
        new_blocks = pool.get_new_blocks(3)
        assert len(pool._evictable_blocks) == 5  # unchanged
        assert len(new_blocks) == 3

    def test_get_num_free_blocks_includes_evictable(self, pool_factory):
        """get_num_free_blocks = kvcached free + evictable pool size."""
        pool, mgr = pool_factory(20)

        blocks = _simulate_request(pool, [f"h{i}" for i in range(8)])
        _finish_request(pool, blocks)

        kvcached_free = mgr.available_size()  # 20 - 8 = 12
        evictable = len(pool._evictable_blocks)  # 8
        assert pool.get_num_free_blocks() == kvcached_free + evictable  # 20


class TestResetAndExplicitEviction:
    """Tests for reset_prefix_cache and evict_blocks."""

    def test_reset_frees_evictable_blocks(self, pool_factory):
        """reset_prefix_cache frees all evictable blocks to kvcached."""
        pool, mgr = pool_factory(21)  # +1 for null_block

        blocks = _simulate_request(pool, ["h0", "h1", "h2"])
        _finish_request(pool, blocks)
        assert mgr.available_size() == 17  # 21-1(null)-3 = 17

        pool.reset_prefix_cache()
        assert len(pool._evictable_blocks) == 0
        assert len(pool._cached_blocks) == 0
        assert len(pool._block_id_to_key) == 0
        assert mgr.available_size() == 20  # all freed (null_block still allocated)
        assert all(block.block_hash is None for block in blocks)

    def test_reset_with_no_evictable(self, pool_and_manager):
        """reset_prefix_cache works even when evictable pool is empty."""
        pool, mgr = pool_and_manager
        result = pool.reset_prefix_cache()
        assert result is True

    def test_evict_blocks_explicit(self, pool_factory):
        """evict_blocks(set) removes specified blocks from cache and pool."""
        pool, mgr = pool_factory(20)

        blocks = _simulate_request(pool, ["h0", "h1", "h2", "h3"])
        _finish_request(pool, blocks)
        block_ids = {b.block_id for b in blocks[:2]}  # evict first 2

        freed_before = mgr.available_size()
        pool.evict_blocks(block_ids)

        assert mgr.available_size() == freed_before + 2
        assert len(pool._evictable_blocks) == 2  # 2 remain
        assert pool.get_cached_block("h0", [0]) is None
        assert pool.get_cached_block("h1", [0]) is None
        assert pool.get_cached_block("h2", [0]) is not None
        assert all(block.block_hash is None for block in blocks[:2])
        assert all(block.block_hash is not None for block in blocks[2:])

    def test_evict_blocks_active_not_freed(self, pool_factory):
        """evict_blocks on active (ref_cnt>0) blocks removes cache entry
        but does NOT free to kvcached (block is still in use)."""
        pool, mgr = pool_factory(20)

        blocks = _simulate_request(pool, ["h0", "h1"])
        # blocks are still active (ref_cnt=1), NOT in evictable pool
        block_ids = {blocks[0].block_id}

        freed_before = mgr.available_size()
        pool.evict_blocks(block_ids)

        # Cache entry removed, but block NOT freed (still active)
        assert pool.get_cached_block("h0", [0]) is None
        assert mgr.available_size() == freed_before  # no kvcached free


class TestCacheDisabled:
    """When enable_caching=False, behavior matches step 1 (no cache)."""

    def test_no_cache_free_immediately(self, pool_factory):
        """With caching disabled, blocks are freed immediately."""
        pool, mgr = pool_factory(20, enable_caching=False)
        initial_free = mgr.available_size()

        blocks = pool.get_new_blocks(5)
        assert mgr.available_size() == initial_free - 5

        pool.free_blocks(blocks)
        assert mgr.available_size() == initial_free
        assert len(pool._evictable_blocks) == 0

    def test_no_cache_hit(self, pool_factory):
        """With caching disabled, get_cached_block always returns None."""
        pool, mgr = pool_factory(20, enable_caching=False)
        result = pool.get_cached_block("h0", [0])
        assert result is None

    def test_get_num_free_blocks_no_cache(self, pool_factory):
        """With caching disabled, get_num_free_blocks = kvcached only."""
        pool, mgr = pool_factory(20, enable_caching=False)
        assert pool.get_num_free_blocks() == mgr.available_size()


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_get_new_blocks_raises_when_alloc_returns_none(self, pool_and_manager):
        pool, _ = pool_and_manager
        with mock.patch.object(pool.kv_cache_manager, "alloc", return_value=None):
            with pytest.raises(ValueError, match="Unable to allocate KV cache blocks"):
                pool.get_new_blocks(1)

    def test_get_new_blocks_retries_after_eviction_when_alloc_returns_none(self, pool_factory):
        pool, _ = pool_factory(5)

        blocks = _simulate_request(pool, ["h0", "h1", "h2", "h3"])
        _finish_request(pool, blocks)

        original_alloc = pool.kv_cache_manager.alloc
        call_count = {"n": 0}

        def flaky_alloc(n: int):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return None
            return original_alloc(n)

        with mock.patch.object(pool.kv_cache_manager, "alloc", side_effect=flaky_alloc):
            new_blocks = pool.get_new_blocks(2)

        assert len(new_blocks) == 2
        assert call_count["n"] == 2

    def test_free_none_blocks(self, pool_and_manager):
        """free_blocks handles None entries in the list."""
        pool, mgr = pool_and_manager
        blocks = pool.get_new_blocks(2)
        pool.free_blocks([None, blocks[0], None, blocks[1]])
        # Should not raise

    def test_evict_more_than_pool_size(self, pool_factory):
        """Requesting eviction of more blocks than in the pool is safe."""
        pool, mgr = pool_factory(5)
        blocks = _simulate_request(pool, ["h0", "h1"])
        _finish_request(pool, blocks)
        assert len(pool._evictable_blocks) == 2

        evicted = pool._evict_blocks_from_pool(100)
        assert evicted == 2
        assert len(pool._evictable_blocks) == 0

    def test_evict_empty_pool(self, pool_and_manager):
        """Evicting from empty pool is a no-op."""
        pool, mgr = pool_and_manager
        evicted = pool._evict_blocks_from_pool(5)
        assert evicted == 0

    def test_cache_full_blocks_idempotent(self, pool_and_manager):
        """Calling cache_full_blocks twice with same hashes is safe."""
        pool, mgr = pool_and_manager
        blocks = pool.get_new_blocks(2)
        req = MockRequest(["h0", "h1"])
        pool.cache_full_blocks(req, blocks, 0, 2, 16, 0)
        pool.cache_full_blocks(req, blocks, 0, 2, 16, 0)
        assert len(pool._cached_blocks) == 2

    def test_heterogeneous_block_size_uses_final_fine_grained_hash(
        self, pool_and_manager, monkeypatch
    ):
        """A 64-token block uses the final hash from its four 16-token chunks."""
        pool, _ = pool_and_manager
        blocks = pool.get_new_blocks(2)
        req = MockRequest([f"h{i}" for i in range(8)])

        kv_cache_utils = sys.modules["vllm.v1.core.kv_cache_utils"]
        converter = mock.Mock(side_effect=MockBlockHashListWithBlockSize)
        monkeypatch.setattr(
            kv_cache_utils, "BlockHashListWithBlockSize", converter
        )

        pool.cache_full_blocks(req, blocks, 0, 2, 64, 1)

        converter.assert_called_once_with(req.block_hashes, 16, 64)
        assert pool.get_cached_block("h0", [1]) is None
        assert pool.get_cached_block("h1", [1]) is None
        assert pool.get_cached_block("h3", [1]) == [blocks[0]]
        assert pool.get_cached_block("h7", [1]) == [blocks[1]]

    def test_heterogeneous_block_size_conversion_respects_cached_offset(
        self, pool_and_manager
    ):
        """Incremental caching indexes the converted hash list by physical block."""
        pool, _ = pool_and_manager
        blocks = pool.get_new_blocks(2)
        req = MockRequest([f"h{i}" for i in range(8)])

        pool.cache_full_blocks(req, blocks, 0, 1, 64, 1)
        pool.cache_full_blocks(req, blocks, 1, 2, 64, 1)

        assert pool.get_cached_block("h3", [1]) == [blocks[0]]
        assert pool.get_cached_block("h7", [1]) == [blocks[1]]

    def test_heterogeneous_block_size_requires_vllm_converter(
        self, pool_and_manager, monkeypatch
    ):
        pool, _ = pool_and_manager
        blocks = pool.get_new_blocks(1)
        req = MockRequest([f"h{i}" for i in range(4)])

        kv_cache_utils = sys.modules["vllm.v1.core.kv_cache_utils"]
        monkeypatch.delattr(kv_cache_utils, "BlockHashListWithBlockSize")

        with pytest.raises(
            RuntimeError, match="does not support heterogeneous block-hash conversion"
        ):
            pool.cache_full_blocks(req, blocks, 0, 1, 64, 1)

    def test_cache_full_blocks_honors_block_mask(self, pool_and_manager):
        """Masked physical blocks must not become reusable prefix entries."""
        pool, _ = pool_and_manager
        blocks = pool.get_new_blocks(2)
        req = MockRequest(["h0", "h1"])

        pool.cache_full_blocks(
            request=req,
            blocks=blocks,
            num_cached_blocks=0,
            num_full_blocks=2,
            block_size=16,
            kv_cache_group_id=0,
            block_mask=[False, True],
        )

        assert pool.get_cached_block("h0", [0]) is None
        assert pool.get_cached_block("h1", [0]) == [blocks[1]]

    def test_duplicate_hashes_keep_each_block_metadata(self, pool_and_manager):
        """Concurrent requests may materialize the same prefix twice."""
        pool, _ = pool_and_manager
        first = pool.get_new_blocks(1)
        second = pool.get_new_blocks(1)
        req = MockRequest(["shared-prefix"])

        pool.cache_full_blocks(req, first, 0, 1, 16, 0)
        pool.cache_full_blocks(req, second, 0, 1, 16, 0)

        assert first[0].block_hash is not None
        assert second[0].block_hash == first[0].block_hash
        assert len(pool._cached_blocks) == 1
        assert pool.get_cached_block("shared-prefix", [0]) is not None

        _finish_request(pool, first)
        _finish_request(pool, second)
        pool._evict_blocks_from_pool(1)

        assert first[0].block_hash is None
        assert second[0].block_hash is not None
        assert pool.get_cached_block("shared-prefix", [0]) == [second[0]]

    def test_reuse_after_eviction_and_realloc(self, pool_factory):
        """After eviction, block IDs can be reallocated and recached."""
        pool, mgr = pool_factory(5)  # +1 for null_block

        # Fill cache
        blocks = _simulate_request(pool, ["h0", "h1", "h2", "h3"])
        _finish_request(pool, blocks)

        # Evict all
        pool._evict_blocks_from_pool(4)
        assert len(pool._cached_blocks) == 0
        assert all(block.block_hash is None for block in blocks)

        # Reallocate -- may get same block IDs
        new_blocks = pool.get_new_blocks(4)
        new_req = MockRequest(["x0", "x1", "x2", "x3"])
        pool.cache_full_blocks(new_req, new_blocks, 0, 4, 16, 0)
        assert len(pool._cached_blocks) == 4
        for h in ["x0", "x1", "x2", "x3"]:
            assert pool.get_cached_block(h, [0]) is not None

    def test_mixed_cached_and_uncached_free(self, pool_and_manager):
        """free_blocks with a mix of cached and uncached blocks."""
        pool, mgr = pool_and_manager
        initial_free = mgr.available_size()

        # Allocate 4 blocks, cache only first 2
        all_blocks = pool.get_new_blocks(4)
        req = MockRequest(["h0", "h1", "h2", "h3"])
        pool.cache_full_blocks(req, all_blocks, 0, 2, 16, 0)  # cache first 2 only

        _finish_request(pool, all_blocks)
        # First 2: cached -> evictable pool. Last 2: uncached -> freed.
        assert len(pool._evictable_blocks) == 2
        assert mgr.available_size() == initial_free - 2  # 2 held by pool

    def test_free_blocks_accepts_prepend_kwarg(self, pool_and_manager):
        """vLLM >= 0.23 calls free_blocks(..., prepend=...); must not raise
        and must free identically for both values (#438)."""
        pool, mgr = pool_and_manager
        initial_free = mgr.available_size()

        blocks = pool.get_new_blocks(2)
        for block in blocks:
            block.ref_cnt = 1
        pool.free_blocks(blocks, prepend=True)
        assert mgr.available_size() == initial_free

        blocks = pool.get_new_blocks(2)
        for block in blocks:
            block.ref_cnt = 1
        pool.free_blocks(blocks, prepend=False)
        assert mgr.available_size() == initial_free

    def test_free_blocks_prepend_with_caching_disabled(self, pool_factory):
        """The caching-disabled fast path must accept prepend too (#438)."""
        pool, mgr = pool_factory(enable_caching=False)
        initial_free = mgr.available_size()

        blocks = pool.get_new_blocks(3)
        pool.free_blocks(blocks, prepend=True)
        assert mgr.available_size() == initial_free

    def test_get_usage(self, pool_factory):
        """get_usage reflects the fraction of blocks in use."""
        pool, mgr = pool_factory(100)
        # null_block occupies 1/100, so usage is ~0.01
        assert pool.get_usage() == pytest.approx(0.01, abs=0.001)

        pool.get_new_blocks(50)
        # 50+1(null) allocated, 0 evictable -> 49 free from kvcached
        assert pool.get_usage() == pytest.approx(0.51)


class DrainedPoolManager(MockKVCacheManager):
    """Leave the pool in the state a colocated peer produces.

    ``available_size()`` reads device-wide free memory, so it is a snapshot of
    state shared with every colocated engine, not a reservation: a peer can
    take the last pages between the pool reading it and the pages being
    claimed.
    """

    def available_size(self) -> int:
        return 1000

    def alloc(self, n: int):
        return None


def test_exhaustion_raises_the_type_the_integration_translates(pool_factory):
    """The other half of this fix lives in KVCacheManagerAllocateSlotsPatch,
    which turns exactly this exception into a scheduling miss. Widening it back
    to a plain ValueError would silently restore the EngineCore crash."""
    from kvcached.utils import KVCachePoolExhausted

    pool, _ = pool_factory(manager=DrainedPoolManager(100))

    with pytest.raises(KVCachePoolExhausted,
                       match="Unable to allocate KV cache blocks"):
        pool.get_new_blocks(4)
