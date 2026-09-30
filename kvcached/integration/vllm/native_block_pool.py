# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Native vLLM hash metadata and block lifetimes for the elastic physical pool."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from typing import Any


class NativeBlockPoolMixin:
    """Combine native hash indexing with elastic ownership and copy lifetimes."""

    # Members supplied by the dynamically composed ElasticBlockPool.
    kv_block_pool: list[Any]
    kv_cache_manager: Any
    enable_prefix_cache: bool
    _block_id_to_key: dict[int, Any]
    _evictable_blocks: OrderedDict[int, Any]
    _cached_blocks: dict[Any, dict[int, Any]]
    _evict_blocks_from_pool: Callable[[int], int]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool

        self._native_block_pool = BlockPool
        self.blocks = self.kv_block_pool
        self.enable_caching = self.enable_prefix_cache
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self.cached_block_hashes_by_block: dict[int, set[Any]] = {}
        self.metrics_collector = None

    def get_cached_block(self, block_hash: Any, kv_cache_group_ids: list[int]) -> Any:
        if not self.enable_prefix_cache:
            return None
        return self._native_block_pool.get_cached_block(self, block_hash, kv_cache_group_ids)

    def cache_full_blocks(
        self,
        request: Any,
        blocks: list[Any],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        if self.enable_prefix_cache:
            self._native_block_pool.cache_full_blocks(
                self, request, blocks, num_cached_blocks, num_full_blocks,
                block_size, kv_cache_group_id, block_mask,
            )

    def _insert_block_hash(
        self, block_hash_with_group_id: Any, block: Any, num_tokens: int | None
    ) -> None:
        self._native_block_pool._insert_block_hash(
            self, block_hash_with_group_id, block, num_tokens
        )
        # The existing eviction/retention code only needs one ownership marker.
        # Native primary + alias indexes own the complete set of lookup keys.
        self._block_id_to_key[block.block_id] = block.block_hash

    def _remove_cached_block_hashes(self, block: Any) -> list[Any]:
        removed = self._native_block_pool._remove_cached_block_hashes(self, block)
        self._block_id_to_key.pop(block.block_id, None)
        return removed

    def _remove_cached_block(self, key: Any, block_id: int) -> Any:
        # Elastic eviction calls this after removing its primary ownership marker.
        # Evict every native alias before the physical slot can be reused.
        block = self.blocks[block_id]
        return block if self._remove_cached_block_hashes(block) else None

    def touch(self, blocks: Sequence[Any]) -> None:
        for block in blocks:
            if block.is_null:
                continue
            block.ref_cnt += 1
            self._evictable_blocks.pop(block.block_id, None)

    def free_blocks(self, ordered_blocks: Iterable[Any], prepend: bool = False) -> None:
        if self.enable_prefix_cache:
            # Retains the existing LRU/page-aware policy, including cache caps.
            return super().free_blocks(ordered_blocks, prepend=prepend)  # type: ignore[misc]

        # A retained copy endpoint must stay mapped until its last reference,
        # even when prefix caching is disabled. The older path frees immediately.
        block_ids = []
        for block in ordered_blocks:
            if block.is_null:
                continue
            assert block.ref_cnt > 0, "Cannot release an unreferenced KV block"
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                block.reset_hash()
                block_ids.append(block.block_id)
        if block_ids:
            self.kv_cache_manager.free(block_ids)

    def reset_prefix_cache(self) -> bool:
        if not self.enable_prefix_cache:
            return True
        # Physical availability is not a count of logical free slots: use live
        # references, including pending worker copies, to decide whether reset is safe.
        if any(block.ref_cnt > 0 for block in self.blocks if not block.is_null):
            return False
        self._evict_blocks_from_pool(len(self._evictable_blocks))
        self.cached_block_hash_to_block = type(self.cached_block_hash_to_block)()
        self.cached_block_hashes_by_block.clear()
        self._block_id_to_key.clear()
        self._cached_blocks.clear()
        for block in self.blocks:
            block.reset_hash()
        return True
