# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import heapq
import types
from typing import Any

import pytest
import torch

import kvcached.integration.sglang.patches as sglang_patches
from kvcached.integration.sglang.patches import (
    RadixCacheLimitPatch,
    _evict_radix_cache_page_aware,
    _select_page_aware_radix_plan,
)

BLOCKS_PER_PAGE = 4


def _selected_nodes(radix_cache: Any, token_budget: int):
    return _select_page_aware_radix_plan(radix_cache, token_budget)[0]


class FakeNode:

    def __init__(self, block_ids, priority):
        if isinstance(block_ids, int):
            block_ids = [block_ids]
        self.value = torch.tensor(block_ids, dtype=torch.int64)
        self.priority = priority
        self.lock_ref = 0
        self.children = {}
        self.parent = None


class FakeStrategy:

    def get_priority(self, node):
        return node.priority


class FakeTupleStrategy:

    def get_priority(self, node):
        return (node.priority, node.priority)


class FakePageAllocator:

    def group_indices_by_page(self, indices, block_mem_size):
        pages: dict[int, list[int]] = {}
        for index in indices:
            pages.setdefault(index // BLOCKS_PER_PAGE, []).append(index)
        return pages


class FakeManager:

    def __init__(self, allocated):
        self.allocated = set(allocated)
        self.block_mem_size = 1
        self.page_size = BLOCKS_PER_PAGE
        self.page_allocator = FakePageAllocator()

    def get_page_occupancy(self, page_ids):
        return {
            page_id: sum(
                block_id // BLOCKS_PER_PAGE == page_id
                for block_id in self.allocated
            )
            for page_id in page_ids
        }

    def free(self, block_ids):
        self.allocated.difference_update(block_ids)


class FakeAllocator:

    def __init__(self, manager):
        self.kvcached_allocator = manager
        self.page_size = 1


class FakeEvictParams:

    def __init__(self, num_tokens):
        self.num_tokens = num_tokens


def _make_cache(block_ids, priorities=None, allocated=None):
    if priorities is None:
        priorities = list(range(len(block_ids)))
    if allocated is None:
        allocated = block_ids
    manager = FakeManager(allocated)
    nodes = [
        FakeNode(block_id, priority)
        for block_id, priority in zip(block_ids, priorities)
    ]
    root = types.SimpleNamespace(
        children={int(node.value[0]): node for node in nodes},
        lock_ref=1,
        parent=None,
    )
    for node in nodes:
        node.parent = root
    cache = types.SimpleNamespace(
        root_node=root,
        evictable_leaves=set(nodes),
        eviction_strategy=FakeStrategy(),
        token_to_kv_pool_allocator=FakeAllocator(manager),
    )
    return cache, manager, nodes


def test_selector_empties_one_page_instead_of_scattering_victims():
    # LRU order alternates between pages 1 and 2. Pure LRU would leave both
    # pages pinned after four evictions; page-aware selection empties page 1.
    block_ids = [4, 8, 5, 9, 6, 10, 7, 11]
    cache, _manager, nodes = _make_cache(block_ids)

    selected = _selected_nodes(cache, token_budget=4)

    assert {int(node.value[0]) for node in selected} == {4, 5, 6, 7}
    assert len(selected) == 4
    assert set(nodes[:4]) != selected


def test_selector_skips_page_pinned_by_active_block():
    # Block 4 is active and therefore absent from the radix tree. Occupancy
    # still sees it, so page 1 cannot be reclaimed by cached-prefix eviction.
    cache, _manager, _nodes = _make_cache(
        [5, 6, 7, 8, 9, 10, 11],
        allocated=[4, 5, 6, 7, 8, 9, 10, 11],
    )

    selected = _selected_nodes(cache, token_budget=4)

    assert {int(node.value[0]) for node in selected} == {8, 9, 10, 11}


def test_selector_includes_reachable_internal_node():
    cache, _manager, nodes = _make_cache([4, 5])
    parent, leaf = nodes
    parent.children = {5: leaf}
    leaf.parent = parent
    cache.root_node.children = {4: parent}
    cache.evictable_leaves = {leaf}

    selected = _selected_nodes(cache, token_budget=2)

    assert selected == {parent, leaf}


def test_selector_counts_only_new_nodes_from_overlapping_closures():
    manager = FakeManager(range(4, 12))
    parent = FakeNode(range(4, 8), priority=10)
    leaf = FakeNode(range(8, 12), priority=0)
    root = types.SimpleNamespace(children={4: parent}, lock_ref=1, parent=None)
    parent.parent = root
    parent.children = {8: leaf}
    leaf.parent = parent
    cache = types.SimpleNamespace(
        root_node=root,
        evictable_leaves={leaf},
        eviction_strategy=FakeStrategy(),
        token_to_kv_pool_allocator=FakeAllocator(manager),
    )

    selected, eviction_budget = _select_page_aware_radix_plan(
        cache,
        token_budget=5,
    )

    assert selected == {parent, leaf}
    assert eviction_budget == 8


def test_selector_amortizes_long_node_over_all_pages_it_releases():
    manager = FakeManager(range(4, 16))
    long_node = FakeNode(range(4, 12), priority=10)
    short_node = FakeNode(range(12, 16), priority=0)
    root = types.SimpleNamespace(
        children={4: long_node, 12: short_node},
        lock_ref=1,
        parent=None,
    )
    long_node.parent = root
    short_node.parent = root
    cache = types.SimpleNamespace(
        root_node=root,
        evictable_leaves={long_node, short_node},
        eviction_strategy=FakeStrategy(),
        token_to_kv_pool_allocator=FakeAllocator(manager),
    )

    selected = _selected_nodes(cache, token_budget=8)

    assert selected == {long_node}


def test_plan_expands_budget_to_complete_first_candidate():
    cache, _manager, nodes = _make_cache([4, 5, 6, 7])

    selected, eviction_budget = _select_page_aware_radix_plan(
        cache,
        token_budget=1,
    )

    assert selected == set(nodes)
    assert eviction_budget == 4


def test_plan_selects_next_candidate_when_it_crosses_budget():
    cache, _manager, nodes = _make_cache(range(4, 12))

    selected, eviction_budget = _select_page_aware_radix_plan(
        cache,
        token_budget=5,
    )

    assert selected == set(nodes)
    assert eviction_budget == 8


def test_selector_prefers_fewer_tokens_even_when_it_requires_more_nodes():
    manager = FakeManager([4, 5, 6, 7, 8, 9, 12, 13, 14, 15])
    large_node = FakeNode([4, 5, 6, 7, 8], priority=0)
    small_nodes = [FakeNode(block_id, priority=10) for block_id in range(12, 16)]
    nodes = [large_node, *small_nodes]
    root = types.SimpleNamespace(
        children={int(node.value[0]): node for node in nodes},
        lock_ref=1,
        parent=None,
    )
    for node in nodes:
        node.parent = root
    cache = types.SimpleNamespace(
        root_node=root,
        evictable_leaves=set(nodes),
        eviction_strategy=FakeStrategy(),
        token_to_kv_pool_allocator=FakeAllocator(manager),
    )

    selected, eviction_budget = _select_page_aware_radix_plan(
        cache,
        token_budget=1,
    )

    assert selected == set(small_nodes)
    assert eviction_budget == 4


def test_selector_caches_block_ids_on_node_values(monkeypatch):
    cache, manager, nodes = _make_cache([4, 5, 6, 7])
    original_cat = torch.cat
    cat_calls = 0

    def counting_cat(*args, **kwargs):
        nonlocal cat_calls
        cat_calls += 1
        return original_cat(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", counting_cat)
    _selected_nodes(cache, token_budget=4)
    index = cache._kvcached_radix_block_index
    block_owners = index.block_owners
    _selected_nodes(cache, token_budget=4)
    assert cat_calls == 1
    assert cache._kvcached_radix_block_index is index
    assert index.block_owners is block_owners

    manager.free([4])
    manager.allocated.add(12)
    nodes[0].value = nodes[0].value.clone()
    nodes[0].value[0] = 12
    _selected_nodes(cache, token_budget=4)
    assert cat_calls == 2
    assert 4 not in index.block_owners
    assert index.block_owners[12] is nodes[0]


def test_selector_removes_nodes_from_persistent_index():
    cache, manager, nodes = _make_cache([4, 5, 6, 7])
    _selected_nodes(cache, token_budget=4)
    index = cache._kvcached_radix_block_index

    removed = nodes[0]
    cache.evictable_leaves.remove(removed)
    del cache.root_node.children[4]
    manager.free([4])

    _selected_nodes(cache, token_budget=3)

    assert removed not in index.node_blocks
    assert 4 not in index.block_owners


def test_page_aware_evict_uses_native_radix_eviction():
    cache, manager, nodes = _make_cache([4, 8, 5, 9, 6, 10, 7, 11])
    cache.evicted = []

    def evict(params):
        count = params.num_tokens
        heap = [
            (cache.eviction_strategy.get_priority(node), index, node)
            for index, node in enumerate(nodes)
        ]
        heapq.heapify(heap)
        while count > 0 and heap:
            _priority, _index, node = heapq.heappop(heap)
            block_id = int(node.value[0])
            cache.evicted.append(block_id)
            manager.free([block_id])
            count -= len(node.value)
            cache.evictable_leaves.remove(node)
            del cache.root_node.children[block_id]

    cache.evict = evict
    _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=4,
        evict_params_cls=FakeEvictParams,
    )

    assert set(cache.evicted) == {4, 5, 6, 7}
    assert manager.allocated == {8, 9, 10, 11}
    index = cache._kvcached_radix_block_index
    assert set(index.block_owners) == {8, 9, 10, 11}


def test_page_aware_evict_expands_budget_to_complete_node():
    cache, manager, nodes = _make_cache(
        [range(4, 8)],
        allocated=range(4, 8),
    )
    cache.evicted = []
    cache.eviction_budgets = []

    def evict(params):
        cache.eviction_budgets.append(params.num_tokens)
        count = params.num_tokens
        heap = [
            (cache.eviction_strategy.get_priority(node), index, node)
            for index, node in enumerate(nodes)
        ]
        heapq.heapify(heap)
        while count > 0 and heap:
            _priority, _index, node = heapq.heappop(heap)
            block_ids = [int(block_id) for block_id in node.value]
            cache.evicted.append(node)
            manager.free(block_ids)
            count -= len(node.value)
            cache.evictable_leaves.remove(node)
            del cache.root_node.children[block_ids[0]]

    cache.evict = evict
    _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=1,
        evict_params_cls=FakeEvictParams,
    )

    assert cache.eviction_budgets == [4]
    assert cache.evicted == nodes
    assert manager.allocated == set()


def test_page_aware_evict_reaches_internal_node():
    manager = FakeManager(range(4, 12))
    parent = FakeNode([4, 5], priority=10)
    leaf = FakeNode([6, 7], priority=10)
    old_a = FakeNode(8, priority=0)
    old_b = FakeNode(9, priority=1)
    root = types.SimpleNamespace(children={}, lock_ref=1, parent=None)
    root.children = {4: parent, 8: old_a, 9: old_b}
    parent.parent = root
    parent.children = {6: leaf}
    leaf.parent = parent
    old_a.parent = root
    old_b.parent = root

    cache = types.SimpleNamespace(
        root_node=root,
        evictable_leaves={leaf, old_a, old_b},
        eviction_strategy=FakeStrategy(),
        token_to_kv_pool_allocator=FakeAllocator(manager),
        evicted=[],
    )

    def evict(params):
        heap = [
            (cache.eviction_strategy.get_priority(node), index, node)
            for index, node in enumerate(cache.evictable_leaves)
        ]
        heapq.heapify(heap)
        next_index = len(heap)
        num_evicted = 0
        while num_evicted < params.num_tokens and heap:
            _priority, _index, node = heapq.heappop(heap)
            block_ids = [int(block_id) for block_id in node.value]
            cache.evicted.append(node)
            manager.free(block_ids)
            num_evicted += len(block_ids)

            parent_node = node.parent
            for key, child in list(parent_node.children.items()):
                if child is node:
                    del parent_node.children[key]
                    break
            if (
                parent_node is not root
                and not parent_node.children
                and parent_node.lock_ref == 0
            ):
                heapq.heappush(
                    heap,
                    (
                        cache.eviction_strategy.get_priority(parent_node),
                        next_index,
                        parent_node,
                    ),
                )
                next_index += 1

    cache.evict = evict
    _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=4,
        evict_params_cls=FakeEvictParams,
    )

    assert cache.evicted == [leaf, parent]
    assert manager.allocated == {8, 9, 10, 11}


def test_radix_patch_uses_page_aware_order_for_cache_cap(monkeypatch):
    radix_module: Any = types.ModuleType("sglang.srt.mem_cache.radix_cache")
    radix_module.EvictParams = FakeEvictParams
    monkeypatch.setattr(sglang_patches, "MAX_CACHED_TOKENS", 4)

    class FakeRadixCache:

        def __init__(self):
            cache, self.manager, self.nodes = _make_cache(
                [4, 8, 5, 9, 6, 10, 7, 11]
            )
            self.root_node = cache.root_node
            self.evictable_leaves = set(cache.evictable_leaves)
            self.eviction_strategy = cache.eviction_strategy
            self.eviction_policy = "lru"
            self.token_to_kv_pool_allocator = cache.token_to_kv_pool_allocator
            self.evictable_size_ = 8
            self.evicted = []
            self._kvcached_radix_block_index: Any = None

        def cache_finished_req(self, *args, **kwargs):
            pass

        def reset(self):
            pass

        def evict(self, params):
            count = params.num_tokens
            heap = [
                (self.eviction_strategy.get_priority(node), index, node)
                for index, node in enumerate(self.root_node.children.values())
            ]
            heapq.heapify(heap)
            while count > 0 and heap:
                _priority, _index, node = heapq.heappop(heap)
                block_id = int(node.value[0])
                self.evicted.append(block_id)
                self.manager.free([block_id])
                count -= len(node.value)
                self.evictable_size_ -= len(node.value)

    radix_module.RadixCache = FakeRadixCache
    assert RadixCacheLimitPatch().patch_radix_cache_limit(radix_module)

    cache = FakeRadixCache()
    cache.cache_finished_req(None)

    assert set(cache.evicted[:4]) == {4, 5, 6, 7}
    assert cache.manager.allocated == {8, 9, 10, 11}

    priority_cache = FakeRadixCache()
    priority_cache.eviction_policy = "priority"
    priority_cache.eviction_strategy = FakeTupleStrategy()
    priority_cache.cache_finished_req(None)
    assert set(priority_cache.evicted[:4]) == {4, 5, 6, 7}

    priority_cache.reset()
    assert priority_cache._kvcached_radix_block_index is None

    class FakeHiRadixCache(FakeRadixCache):
        pass

    hicache = FakeHiRadixCache()
    hicache.cache_finished_req(None)
    assert hicache.evicted[:4] == [4, 8, 5, 9]


def test_radix_patch_uses_native_eviction_for_legacy_sglang(monkeypatch):
    radix_module: Any = types.ModuleType("sglang.srt.mem_cache.radix_cache")
    monkeypatch.setattr(sglang_patches, "MAX_CACHED_TOKENS", 4)

    class FakeLegacyRadixCache:

        def __init__(self):
            self.evictable_size_ = 8
            self.eviction_args = []

        def cache_finished_req(self, *args, **kwargs):
            pass

        def reset(self):
            pass

        def evict(self, num_tokens):
            self.eviction_args.append(num_tokens)

    radix_module.RadixCache = FakeLegacyRadixCache
    assert RadixCacheLimitPatch().patch_radix_cache_limit(radix_module)

    cache = FakeLegacyRadixCache()
    cache.cache_finished_req(None)

    assert cache.eviction_args == [4]
    assert not hasattr(cache, "_kvcached_radix_block_index")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
