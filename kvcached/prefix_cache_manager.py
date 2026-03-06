# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Prefix Cache Manager for kvcached

This module implements prefix caching for KV cache blocks, allowing blocks
with the same hash (representing identical prompt prefixes) to be reused
across different requests.

Key features:
- Hash-based block lookup and storage
- LRU (Least Recently Used) eviction policy
- Reference counting to prevent premature block freeing
"""

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


@dataclass
class CachedBlockEntry:
    """
    Represents a cached block entry.

    Each block has its own hash (computed from block content + parent block hash),
    enabling partial prefix matches where only some blocks in a sequence are reused.

    Attributes:
        block_id: The cached block ID
        last_access_time: Timestamp of last access (for LRU tracking)
        refcount: Number of active sequences using this cached block
    """
    block_id: int
    last_access_time: float
    refcount: int


class PrefixCacheManager:
    """
    Manages prefix caching for KV cache blocks.

    This class provides a hash-based cache with LRU eviction for KV cache blocks.
    Blocks are stored with a hash key and can be retrieved when new requests
    share the same prefix.
    """

    def __init__(
        self,
        max_cache_size: int = 1000,
        on_evict_callback: Optional[Callable[[List[int]], None]] = None,
    ):
        """
        Initialize the Prefix Cache Manager.

        Args:
            max_cache_size: Maximum number of cached block sequences
            on_evict_callback: Optional callback function called when blocks
                are evicted. Receives list of block IDs as argument.
        """
        self.max_cache_size = max_cache_size
        self.on_evict_callback = on_evict_callback

        # Hash to cached block entry mapping
        self.cache_dict: Dict[Any, CachedBlockEntry] = {}

        # Reverse index: block_id to hash for O(1) lookup in touch()
        self.block_to_hash: Dict[int, Any] = {}

        # LRU tracking: maps hash to access timestamp
        # OrderedDict maintains insertion order, making LRU tracking efficient
        self.lru_list: OrderedDict[Any, float] = OrderedDict()

        # Current number of cached blocks
        self.current_cache_size: int = 0

        logger.info(f"PrefixCacheManager initialized with max_cache_size={max_cache_size}")

    def get_cached_block(self, block_hash: Any) -> Optional[int]:
        """
        Look up a cached block by hash.

        Each block has its own individual hash computed from:
        hash = hash(block_content + parent_block_hash)

        Args:
            block_hash: Hash key for the block

        Returns:
            Block ID if found, None otherwise
        """
        if block_hash not in self.cache_dict:
            return None

        entry = self.cache_dict[block_hash]

        # Update LRU tracking
        self._touch_hash(block_hash)

        # Increment refcount since block is being reused
        entry.refcount += 1

        logger.debug(f"Cache hit for hash {block_hash}: block_id={entry.block_id}, "
                    f"refcount={entry.refcount}")

        return entry.block_id

    def cache_block(self, block_hash: Any, block_id: int) -> None:
        """
        Store a single block in the cache with the given hash.

        Each block has its own hash computed from block content + parent hash.
        This enables partial prefix matching where only some blocks match.

        If the cache is full, evicts the least recently used entry first.

        Args:
            block_hash: Hash key for the block
            block_id: Block ID to cache
        """
        # If hash already exists, update it
        if block_hash in self.cache_dict:
            entry = self.cache_dict[block_hash]
            old_block_id = entry.block_id
            entry.block_id = block_id
            entry.last_access_time = time.time()
            entry.refcount += 1
            self._touch_hash(block_hash)

            # Update reverse index
            if old_block_id != block_id:
                if old_block_id in self.block_to_hash:
                    del self.block_to_hash[old_block_id]
                self.block_to_hash[block_id] = block_hash

            logger.debug(f"Updated cache entry for hash {block_hash}: block_id={block_id}")
            return

        # Check if we need to evict before adding new entry
        while self.current_cache_size >= self.max_cache_size:
            evicted = self._evict_lru()
            if not evicted:
                logger.warning("Failed to evict any entries, cache may be full with "
                             "in-use blocks")
                break

        # Add new entry
        current_time = time.time()
        entry = CachedBlockEntry(
            block_id=block_id,
            last_access_time=current_time,
            refcount=1  # Initial reference
        )

        self.cache_dict[block_hash] = entry
        self.lru_list[block_hash] = current_time
        self.block_to_hash[block_id] = block_hash
        self.current_cache_size += 1

        logger.debug(f"Cached block {block_id} with hash {block_hash}, "
                    f"cache_size={self.current_cache_size}/{self.max_cache_size}")

    def touch(self, block_ids: List[int]) -> None:
        """
        Update LRU timestamp for blocks.

        This should be called when blocks are accessed to keep them
        from being evicted.

        Args:
            block_ids: List of block IDs being accessed
        """
        if not block_ids:
            return

        current_time = time.time()
        for block_id in block_ids:
            # Use reverse index for O(1) lookup
            if block_id in self.block_to_hash:
                block_hash = self.block_to_hash[block_id]
                entry = self.cache_dict[block_hash]
                entry.last_access_time = current_time
                self._touch_hash(block_hash)

    def decrement_refcount(self, block_hash: Any) -> bool:
        """
        Decrement reference count for a cached entry.

        Should be called when a sequence using cached blocks finishes.

        Args:
            block_hash: Hash key for the block sequence

        Returns:
            True if blocks can now be freed (refcount reached 0), False otherwise
        """
        if block_hash not in self.cache_dict:
            return True  # Not cached, can be freed

        entry = self.cache_dict[block_hash]
        entry.refcount -= 1

        logger.debug(f"Decremented refcount for hash {block_hash}: "
                    f"refcount={entry.refcount}")

        # If refcount reaches 0, blocks can be freed
        return entry.refcount <= 0

    def increment_refcount(self, block_hash: Any) -> None:
        """
        Increment reference count for a cached entry.

        Should be called when a new sequence starts using cached blocks.

        Args:
            block_hash: Hash key for the block sequence
        """
        if block_hash not in self.cache_dict:
            logger.warning(f"Attempted to increment refcount for non-existent hash {block_hash}")
            return

        entry = self.cache_dict[block_hash]
        entry.refcount += 1

        logger.debug(f"Incremented refcount for hash {block_hash}: "
                    f"refcount={entry.refcount}")

    def reset(self) -> None:
        """
        Clear all cached blocks.

        This will trigger the eviction callback for all cached entries
        with refcount = 0.
        """
        logger.info("Resetting prefix cache")

        # Evict all entries with refcount = 0
        hashes_to_remove = []
        for block_hash, entry in self.cache_dict.items():
            if entry.refcount <= 0:
                if self.on_evict_callback:
                    self.on_evict_callback([entry.block_id])
                hashes_to_remove.append(block_hash)

        # Remove evicted entries
        for block_hash in hashes_to_remove:
            entry = self.cache_dict[block_hash]
            # Clean up reverse index
            if entry.block_id in self.block_to_hash:
                del self.block_to_hash[entry.block_id]
            del self.cache_dict[block_hash]
            if block_hash in self.lru_list:
                del self.lru_list[block_hash]

        self.current_cache_size = len(self.cache_dict)

        if self.cache_dict:
            logger.warning(f"After reset, {self.current_cache_size} entries still in cache "
                         f"(in use by active sequences)")

    def _touch_hash(self, block_hash: Any) -> None:
        """
        Update LRU position for a hash.

        Moves the hash to the end of the LRU list (most recently used).

        Args:
            block_hash: Hash key to update
        """
        if block_hash in self.lru_list:
            # Move to end (most recently used)
            self.lru_list.move_to_end(block_hash)
            self.lru_list[block_hash] = time.time()

    def _evict_lru(self) -> bool:
        """
        Evict the least recently used cached entry.

        Skips entries that are still in use (refcount > 0).

        Returns:
            True if an entry was evicted, False otherwise
        """
        if not self.lru_list:
            return False

        # Try to find an entry with refcount = 0 to evict
        # Start from the beginning (least recently used)
        for block_hash in list(self.lru_list.keys()):
            entry = self.cache_dict[block_hash]

            if entry.refcount <= 0:
                # This entry can be evicted
                logger.debug(f"Evicting LRU entry with hash {block_hash}: "
                           f"block_id={entry.block_id}")

                # Notify callback if provided
                if self.on_evict_callback:
                    self.on_evict_callback([entry.block_id])

                # Clean up reverse index
                if entry.block_id in self.block_to_hash:
                    del self.block_to_hash[entry.block_id]

                # Remove from cache
                del self.cache_dict[block_hash]
                del self.lru_list[block_hash]
                self.current_cache_size -= 1

                return True

        # All entries are in use
        logger.debug("All cached entries are in use, cannot evict")
        return False

    def evict_blocks_by_id(self, block_ids: List[int]) -> None:
        """
        Evict specific blocks from the cache by their block IDs. No refcount decrement is performed.

        This removes blocks from the prefix cache hash table without triggering
        the eviction callback. Used when vLLM explicitly requests eviction.

        Args:
            block_ids: List of block IDs to evict from cache
        """
        for block_id in block_ids:
            # Look up the hash for this block
            if block_id not in self.block_to_hash:
                # Block not in cache, nothing to evict
                continue

            block_hash = self.block_to_hash[block_id]

            # Remove from cache structures
            if block_hash in self.cache_dict:
                del self.cache_dict[block_hash]
                self.current_cache_size -= 1

            if block_hash in self.lru_list:
                del self.lru_list[block_hash]

            del self.block_to_hash[block_id]

            logger.debug(f"Evicted block {block_id} with hash {block_hash} from prefix cache")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        total_refcount = sum(entry.refcount for entry in self.cache_dict.values())

        return {
            "current_cache_size": self.current_cache_size,
            "max_cache_size": self.max_cache_size,
            "total_cached_blocks": self.current_cache_size,  # Each entry is one block
            "total_refcount": total_refcount,
            "cache_utilization": self.current_cache_size / self.max_cache_size if self.max_cache_size > 0 else 0.0,
        }
