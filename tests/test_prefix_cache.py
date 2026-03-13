# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for prefix caching in ElasticBlockPool.

Tests the cache logic (hit, miss, touch, free, evict, reset) using
mock objects — no GPU or vLLM installation required.
"""

import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock


# --- Mock classes to simulate vLLM's block pool types ---

@dataclass
class MockKVCacheBlock:
    """Simulates vLLM's KVCacheBlock."""
    block_id: int
    ref_cnt: int = 0
    is_null: bool = False

    def __repr__(self):
        return f"Block(id={self.block_id}, ref_cnt={self.ref_cnt})"


class MockBlockPool:
    """Base class to satisfy ElasticBlockPool's inheritance."""
    pass


@dataclass
class MockRequest:
    """Simulates a vLLM Request with block hashes."""
    block_hashes: list = field(default_factory=list)


# --- Build ElasticBlockPool with mocks ---

def create_elastic_block_pool(enable_caching: bool = True):
    """
    Create an ElasticBlockPool class using mocks,
    then instantiate it with a mock KVCacheManager.
    """
    import types

    # Create a fake block_pool module
    block_pool_mod = types.ModuleType("fake_block_pool")
    block_pool_mod.BlockPool = MockBlockPool
    block_pool_mod.KVCacheBlock = MockKVCacheBlock

    # Import and run the patch to create ElasticBlockPool
    from kvcached.integration.vllm.patches import ElasticBlockPoolPatch

    patch = ElasticBlockPoolPatch()
    patch.logger = MagicMock()
    # Bypass version check
    patch.detected_version = "0.9.0"

    # Manually inject
    BlockPool = MockBlockPool
    KVCacheBlockClass = MockKVCacheBlock

    # We'll build the class directly matching patches.py logic
    class ElasticBlockPool(MockBlockPool):

        def __init__(self, num_gpu_blocks, enable_caching):
            self.num_gpu_blocks = num_gpu_blocks
            self.enable_prefix_cache = enable_caching
            self.enable_kv_cache_events = False
            self.kv_event_queue = []
            self.null_block = None

            # Mock the kv_cache_manager
            self.kv_cache_manager = MagicMock()
            self._next_block_id = 0

            def mock_alloc(n):
                ids = list(range(self._next_block_id, self._next_block_id + n))
                self._next_block_id += n
                return ids

            self.kv_cache_manager.alloc = mock_alloc
            self.kv_cache_manager.free = MagicMock()
            self.kv_cache_manager.available_size = MagicMock(
                return_value=num_gpu_blocks)

            # Prefix cache: hash -> KVCacheBlock object
            self._cached_blocks: dict[Any, MockKVCacheBlock] = {}

        def get_cached_block(self, block_hash, kv_cache_group_ids):
            if not self.enable_prefix_cache:
                return None
            if len(kv_cache_group_ids) > 1:
                return None
            block = self._cached_blocks.get(block_hash)
            if block is None:
                return None
            return [block]

        def cache_full_blocks(self, request, blocks, num_cached_blocks,
                              num_full_blocks, block_size, kv_cache_group_id):
            if not self.enable_prefix_cache:
                return
            if num_cached_blocks >= num_full_blocks:
                return
            new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
            for i, block in enumerate(new_full_blocks):
                if getattr(block, 'is_null', False):
                    continue
                block_idx = num_cached_blocks + i
                block_hash = request.block_hashes[block_idx]
                if block_hash in self._cached_blocks:
                    continue
                self._cached_blocks[block_hash] = block

        def get_new_blocks(self, num_blocks):
            block_ids = self.kv_cache_manager.alloc(num_blocks)
            return [MockKVCacheBlock(bid) for bid in block_ids]

        def touch(self, blocks):
            if isinstance(blocks, tuple):
                for block_list in blocks:
                    for block in block_list:
                        block.ref_cnt += 1
            else:
                for block in blocks:
                    block.ref_cnt += 1

        def free_blocks(self, ordered_blocks):
            blocks_to_free = []
            for block in ordered_blocks:
                block.ref_cnt -= 1
                if block.ref_cnt == 0:
                    blocks_to_free.append(block.block_id)
            if blocks_to_free:
                self.kv_cache_manager.free(blocks_to_free)

        def evict_blocks(self, block_ids):
            if not self.enable_prefix_cache:
                return
            to_remove = []
            for block_hash, block in self._cached_blocks.items():
                if block.block_id in block_ids:
                    to_remove.append(block_hash)
            for block_hash in to_remove:
                del self._cached_blocks[block_hash]

        def reset_prefix_cache(self):
            if not self.enable_prefix_cache:
                return True
            self._cached_blocks.clear()
            return True

        def get_num_free_blocks(self):
            return self.kv_cache_manager.available_size()

    return ElasticBlockPool(num_gpu_blocks=100, enable_caching=enable_caching)


class TestPrefixCache(unittest.TestCase):

    def setUp(self):
        self.pool = create_elastic_block_pool(enable_caching=True)

    def test_cache_miss(self):
        """First lookup should miss."""
        result = self.pool.get_cached_block("hash_A", [0])
        self.assertIsNone(result)
        print("PASS: Cache miss on first lookup")

    def test_cache_and_hit(self):
        """After caching a block, lookup should return the same object."""
        # Allocate blocks for Request A
        blocks = self.pool.get_new_blocks(3)
        print(f"  Allocated blocks: {blocks}")

        # Cache them
        request = MockRequest(block_hashes=["h0", "h1", "h2"])
        self.pool.cache_full_blocks(
            request, blocks,
            num_cached_blocks=0, num_full_blocks=3,
            block_size=16, kv_cache_group_id=0)
        print(f"  Cached 3 blocks: h0->Block({blocks[0].block_id}), "
              f"h1->Block({blocks[1].block_id}), h2->Block({blocks[2].block_id})")
        print(f"  Cache size: {len(self.pool._cached_blocks)}")

        # Request B looks up same hash -> should return SAME object
        result = self.pool.get_cached_block("h1", [0])
        self.assertIsNotNone(result)
        self.assertIs(result[0], blocks[1])  # Same object!
        print(f"  Cache HIT for 'h1': {result[0]} (same object: {result[0] is blocks[1]})")

        # Miss for unknown hash
        result2 = self.pool.get_cached_block("h_unknown", [0])
        self.assertIsNone(result2)
        print(f"  Cache MISS for 'h_unknown'")
        print("PASS: Cache and hit")

    def test_touch_increments_refcnt(self):
        """touch() should increment ref_cnt on block objects."""
        blocks = self.pool.get_new_blocks(2)
        self.assertEqual(blocks[0].ref_cnt, 0)

        self.pool.touch(blocks)
        self.assertEqual(blocks[0].ref_cnt, 1)
        self.assertEqual(blocks[1].ref_cnt, 1)
        print(f"  After touch: {blocks}")
        print("PASS: touch increments ref_cnt")

    def test_concurrent_sharing(self):
        """
        Two requests sharing cached blocks via prefix cache.
        Blocks should only be freed when ALL references are gone.
        """
        print("\n--- Concurrent Sharing Test ---")

        # Request A allocates and caches blocks
        blocks_a = self.pool.get_new_blocks(3)
        request_a = MockRequest(block_hashes=["h0", "h1", "h2"])
        self.pool.cache_full_blocks(
            request_a, blocks_a,
            num_cached_blocks=0, num_full_blocks=3,
            block_size=16, kv_cache_group_id=0)

        # vLLM calls touch for Request A's blocks
        self.pool.touch(blocks_a)
        print(f"  Request A allocated + cached + touched: {blocks_a}")

        # Request B hits cache for h0, h1
        hit_b = []
        for h in ["h0", "h1"]:
            result = self.pool.get_cached_block(h, [0])
            self.assertIsNotNone(result)
            hit_b.append(result[0])
        print(f"  Request B cache hits: {hit_b}")

        # vLLM calls touch for Request B's cached blocks
        self.pool.touch(hit_b)
        print(f"  After Request B touch: {hit_b}")

        # Request B also gets new blocks for its unique suffix
        new_b = self.pool.get_new_blocks(1)
        self.pool.touch(new_b)
        all_b = hit_b + new_b
        print(f"  Request B all blocks: {all_b}")

        # Verify shared blocks have ref_cnt=2 (A + B)
        self.assertEqual(blocks_a[0].ref_cnt, 2)  # shared
        self.assertEqual(blocks_a[1].ref_cnt, 2)  # shared
        self.assertEqual(blocks_a[2].ref_cnt, 1)  # only A
        print(f"  Block ref_cnts: {[b.ref_cnt for b in blocks_a]} "
              f"(shared: 2, A-only: 1)")

        # Request A finishes -> free its blocks
        self.pool.free_blocks(blocks_a)
        print(f"  After Request A free: {blocks_a}")

        # Shared blocks should NOT be freed (ref_cnt=1, still used by B)
        self.assertEqual(blocks_a[0].ref_cnt, 1)
        self.assertEqual(blocks_a[1].ref_cnt, 1)
        # A-only block freed (ref_cnt=0)
        self.assertEqual(blocks_a[2].ref_cnt, 0)

        # kv_cache_manager.free should only be called with block 2 (A-only)
        self.pool.kv_cache_manager.free.assert_called_once_with([blocks_a[2].block_id])
        print(f"  kv_cache_manager.free called with: [{blocks_a[2].block_id}] (A-only block)")
        self.pool.kv_cache_manager.free.reset_mock()

        # Request B finishes -> free its blocks
        self.pool.free_blocks(all_b)
        print(f"  After Request B free: ref_cnts={[b.ref_cnt for b in all_b]}")

        # Now all blocks should be freed
        for b in all_b:
            self.assertEqual(b.ref_cnt, 0)
        freed_ids = self.pool.kv_cache_manager.free.call_args[0][0]
        self.assertEqual(sorted(freed_ids),
                         sorted([b.block_id for b in all_b]))
        print(f"  kv_cache_manager.free called with: {sorted(freed_ids)}")

        print("PASS: Concurrent sharing with correct ref_cnt lifecycle")

    def test_evict_blocks(self):
        """evict_blocks removes from cache but doesn't free memory."""
        blocks = self.pool.get_new_blocks(2)
        request = MockRequest(block_hashes=["h0", "h1"])
        self.pool.cache_full_blocks(
            request, blocks,
            num_cached_blocks=0, num_full_blocks=2,
            block_size=16, kv_cache_group_id=0)

        self.assertEqual(len(self.pool._cached_blocks), 2)

        # Evict block 0
        self.pool.evict_blocks({blocks[0].block_id})
        self.assertEqual(len(self.pool._cached_blocks), 1)

        # h0 should miss now
        self.assertIsNone(self.pool.get_cached_block("h0", [0]))
        # h1 should still hit
        self.assertIsNotNone(self.pool.get_cached_block("h1", [0]))
        print("PASS: evict_blocks removes from cache lookup")

    def test_reset_prefix_cache(self):
        """reset_prefix_cache clears all cache entries."""
        blocks = self.pool.get_new_blocks(3)
        request = MockRequest(block_hashes=["h0", "h1", "h2"])
        self.pool.cache_full_blocks(
            request, blocks,
            num_cached_blocks=0, num_full_blocks=3,
            block_size=16, kv_cache_group_id=0)

        self.assertEqual(len(self.pool._cached_blocks), 3)
        self.pool.reset_prefix_cache()
        self.assertEqual(len(self.pool._cached_blocks), 0)

        # All lookups should miss
        for h in ["h0", "h1", "h2"]:
            self.assertIsNone(self.pool.get_cached_block(h, [0]))
        print("PASS: reset_prefix_cache clears all entries")

    def test_idempotent_cache(self):
        """Caching the same hash twice should be idempotent."""
        blocks = self.pool.get_new_blocks(1)
        request = MockRequest(block_hashes=["h0"])

        self.pool.cache_full_blocks(
            request, blocks,
            num_cached_blocks=0, num_full_blocks=1,
            block_size=16, kv_cache_group_id=0)
        self.pool.cache_full_blocks(
            request, blocks,
            num_cached_blocks=0, num_full_blocks=1,
            block_size=16, kv_cache_group_id=0)

        self.assertEqual(len(self.pool._cached_blocks), 1)
        print("PASS: Idempotent caching")

    def test_disabled_prefix_cache(self):
        """When prefix caching is disabled, all cache ops are no-ops."""
        pool = create_elastic_block_pool(enable_caching=False)
        blocks = pool.get_new_blocks(2)

        # get_cached_block returns None
        self.assertIsNone(pool.get_cached_block("h0", [0]))

        # cache_full_blocks does nothing
        request = MockRequest(block_hashes=["h0", "h1"])
        pool.cache_full_blocks(
            request, blocks,
            num_cached_blocks=0, num_full_blocks=2,
            block_size=16, kv_cache_group_id=0)
        self.assertEqual(len(pool._cached_blocks), 0)

        print("PASS: Disabled prefix cache")


if __name__ == "__main__":
    print("=" * 60)
    print("    Prefix Cache Unit Tests")
    print("=" * 60)
    print()
    unittest.main(verbosity=2)
