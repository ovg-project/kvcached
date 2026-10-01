# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import heapq
import types
from array import array
from typing import Any

import pytest
import torch

import kvcached.integration.sglang.patches as sglang_patches
from kvcached.integration.sglang.patches import (
    RadixCacheLimitPatch,
    _evict_radix_cache_page_aware,
    _select_page_aware_plan,
)

BLOCKS_PER_PAGE = 4


def _whole_node_suffixes(selection):
    return {
        leaf
        for leaf, split_len in selection.suffix_plans
        if split_len == 0
    }


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

    def __init__(self):
        self.group_calls = 0

    def group_indices_by_page(self, indices, block_mem_size):
        self.group_calls += 1
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

    def __init__(self, manager, page_size=1):
        self.kvcached_allocator = manager
        self.page_size = page_size
        self.device = "cpu"

    def free(self, token_indices):
        self.kvcached_allocator.free(
            {int(index) // self.page_size for index in token_indices}
        )

    def free_segment(self, token_indices, start_pos):
        self.free(token_indices)


class FakeEvictParams:

    def __init__(self, num_tokens):
        self.num_tokens = num_tokens


def _make_cache(block_ids, priorities=None, allocated=None, logical_page_size=1):
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
        evictable_size_=sum(int(node.value.numel()) for node in nodes),
        eviction_strategy=FakeStrategy(),
        page_size=logical_page_size,
        token_to_kv_pool_allocator=FakeAllocator(manager, logical_page_size),
    )
    return cache, manager, nodes


def _native_radix_types():
    radix_cache = pytest.importorskip("sglang.srt.mem_cache.radix_cache")
    base_prefix_cache = pytest.importorskip(
        "sglang.srt.mem_cache.base_prefix_cache"
    )
    return (
        radix_cache.RadixCache,
        radix_cache.RadixKey,
        base_prefix_cache.EvictParams,
        base_prefix_cache.InsertParams,
        base_prefix_cache.MatchPrefixParams,
    )


def _native_insert(
    cache, radix_key_cls, insert_params_cls, tokens, block_ids, priority=0
):
    cache.insert(
        insert_params_cls(
            key=radix_key_cls(token_ids=array("q", tokens)),
            value=torch.tensor(block_ids, dtype=torch.int64),
            priority=priority,
        )
    )


def _native_match(cache, radix_key_cls, match_params_cls, tokens):
    result = cache.match_prefix(
        match_params_cls(key=radix_key_cls(token_ids=array("q", tokens)))
    )
    return result.device_indices.tolist(), result.last_device_node


def test_selector_empties_one_page_instead_of_scattering_victims():
    # LRU order alternates between pages 1 and 2. Pure LRU would leave both
    # pages pinned after four evictions; page-aware selection empties page 1.
    block_ids = [4, 8, 5, 9, 6, 10, 7, 11]
    cache, _manager, nodes = _make_cache(block_ids)

    selected = _whole_node_suffixes(
        _select_page_aware_plan(cache, token_budget=4)
    )

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

    selected = _whole_node_suffixes(
        _select_page_aware_plan(cache, token_budget=4)
    )

    assert {int(node.value[0]) for node in selected} == {8, 9, 10, 11}


def test_selector_includes_internal_node_suffix_closure():
    cache, _manager, nodes = _make_cache([4, 5])
    parent, leaf = nodes
    parent.children = {5: leaf}
    leaf.parent = parent
    cache.root_node.children = {4: parent}
    cache.evictable_leaves = {leaf}

    selection = _select_page_aware_plan(cache, token_budget=2)

    assert selection.suffix_plans == [(parent, 0)]
    assert selection.token_count == 2


def test_selector_prefers_internal_closure_reclaiming_two_pages():
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

    selection = _select_page_aware_plan(
        cache,
        token_budget=8,
    )

    assert selection.suffix_plans == [(parent, 0)]
    assert selection.token_count == 8


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

    selected = _whole_node_suffixes(
        _select_page_aware_plan(cache, token_budget=8)
    )

    assert selected == {long_node}


def test_plan_rejects_first_candidate_that_exceeds_budget():
    cache, _manager, _nodes = _make_cache([4, 5, 6, 7])

    selection = _select_page_aware_plan(
        cache,
        token_budget=1,
    )

    assert selection.suffix_plans == []
    assert selection.token_count == 0


def test_plan_does_not_cross_remaining_budget():
    cache, _manager, nodes = _make_cache(range(4, 12))

    selection = _select_page_aware_plan(
        cache,
        token_budget=5,
    )

    assert _whole_node_suffixes(selection) == set(nodes[:4])
    assert selection.token_count == 4


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

    selection = _select_page_aware_plan(
        cache,
        token_budget=4,
    )

    assert _whole_node_suffixes(selection) == set(small_nodes)
    assert selection.token_count == 4


def test_leaf_suffix_can_reclaim_page_while_touching_partial_page():
    (
        radix_cache_cls,
        radix_key_cls,
        _evict_params_cls,
        insert_params_cls,
        _match_params_cls,
    ) = _native_radix_types()
    manager = FakeManager([4, 5, 8, 9, 10, 11])
    cache = radix_cache_cls.create_simulated(
        mock_allocator=FakeAllocator(manager),
        page_size=1,
    )
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=list(range(6)),
        block_ids=[4, 8, 9, 10, 11, 5],
    )

    selection = _select_page_aware_plan(
        cache,
        token_budget=5,
    )

    assert len(selection.suffix_plans) == 1
    assert selection.suffix_plans[0][1] == 1
    assert selection.token_count == 5


def test_leaf_suffix_split_uses_radix_logical_page_alignment():
    (
        radix_cache_cls,
        radix_key_cls,
        _evict_params_cls,
        insert_params_cls,
        _match_params_cls,
    ) = _native_radix_types()
    manager = FakeManager(range(4, 12))
    cache = radix_cache_cls.create_simulated(
        mock_allocator=FakeAllocator(manager, page_size=2),
        page_size=2,
    )
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=list(range(16)),
        block_ids=list(range(8, 24)),
    )

    selection = _select_page_aware_plan(
        cache,
        token_budget=8,
    )

    assert len(selection.suffix_plans) == 1
    assert selection.suffix_plans[0][1] == 8
    assert selection.suffix_plans[0][1] % cache.page_size == 0
    assert selection.token_count == 8


def test_leaf_suffix_uses_budget_to_reclaim_equal_cost_pages():
    """Equal-efficiency suffixes should grow instead of stopping at one page."""
    manager = FakeManager([4, 8, 12, 16, 100])
    leaf = FakeNode([100, 4, 8, 12, 16], priority=0)
    root = types.SimpleNamespace(children={100: leaf}, lock_ref=1, parent=None)
    leaf.parent = root
    cache = types.SimpleNamespace(
        root_node=root,
        evictable_leaves={leaf},
        eviction_strategy=FakeStrategy(),
        page_size=1,
        token_to_kv_pool_allocator=FakeAllocator(manager),
        _split_node=lambda *_args: None,
    )

    selection = _select_page_aware_plan(
        cache,
        token_budget=4,
    )

    assert selection.suffix_plans == [(leaf, 1)]
    assert selection.token_count == 4


def test_page_candidate_combines_suffixes_from_multiple_leaves():
    manager = FakeManager([4, 5, 6, 7, 100, 101])
    first = FakeNode([100, 4, 5], priority=0)
    second = FakeNode([101, 6, 7], priority=1)
    root = types.SimpleNamespace(
        children={100: first, 101: second}, lock_ref=1, parent=None
    )
    first.parent = root
    second.parent = root
    cache = types.SimpleNamespace(
        root_node=root,
        evictable_leaves={first, second},
        eviction_strategy=FakeStrategy(),
        page_size=1,
        token_to_kv_pool_allocator=FakeAllocator(manager),
        _split_node=lambda *_args: None,
    )

    selection = _select_page_aware_plan(cache, token_budget=4)

    assert set(selection.suffix_plans) == {(first, 1), (second, 1)}
    assert selection.token_count == 4


def test_planner_compares_single_and_multi_leaf_suffix_candidates():
    """A costly suffix must not hide several cheaper reclaimable pages."""
    manager = FakeManager([4, 5, 6, 7, 8, 12, 16, 20, 100])
    long_leaf = FakeNode([100, 4, 5, 6, 7], priority=0)
    short_leaves = [
        FakeNode(block_id, priority=10 + index)
        for index, block_id in enumerate([8, 12, 16, 20])
    ]
    nodes = [long_leaf, *short_leaves]
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
        page_size=1,
        token_to_kv_pool_allocator=FakeAllocator(manager),
        _split_node=lambda *_args: None,
    )
    selection = _select_page_aware_plan(
        radix_cache=cache,
        token_budget=4,
    )

    whole_leaf_suffixes = {
        leaf for leaf, split_len in selection.suffix_plans if split_len == 0
    }
    assert whole_leaf_suffixes == set(short_leaves)
    assert selection.token_count == 4


def test_selector_converts_multi_token_logical_pages_to_block_ids(monkeypatch):
    logical_page_size = 2
    block_ids = [4, 8, 5, 9, 6, 10, 7, 11]
    token_indices = [
        range(block_id * logical_page_size, (block_id + 1) * logical_page_size)
        for block_id in block_ids
    ]
    cache, _manager, nodes = _make_cache(
        token_indices,
        allocated=block_ids,
        logical_page_size=logical_page_size,
    )
    original_cat = torch.cat
    concatenated_elements = []

    def record_cat(tensors, *args, **kwargs):
        concatenated_elements.append(sum(int(tensor.numel()) for tensor in tensors))
        return original_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", record_cat)

    selection = _select_page_aware_plan(
        cache,
        token_budget=8,
    )

    assert _whole_node_suffixes(selection) == {
        nodes[0],
        nodes[2],
        nodes[4],
        nodes[6],
    }
    assert selection.token_count == 8
    assert concatenated_elements == [len(block_ids)]


def test_selector_caches_block_ids_on_node_values(monkeypatch):
    cache, manager, nodes = _make_cache([4, 5, 6, 7])
    original_cat = torch.cat
    cat_calls = 0

    def counting_cat(*args, **kwargs):
        nonlocal cat_calls
        cat_calls += 1
        return original_cat(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", counting_cat)
    _select_page_aware_plan(cache, token_budget=4)
    index = cache._kvcached_radix_block_index
    block_owners = index.block_owners
    _select_page_aware_plan(cache, token_budget=4)
    assert cat_calls == 1
    assert cache._kvcached_radix_block_index is index
    assert index.block_owners is block_owners

    manager.free([4])
    manager.allocated.add(12)
    nodes[0].value = nodes[0].value.clone()
    nodes[0].value[0] = 12
    _select_page_aware_plan(cache, token_budget=4)
    assert cat_calls == 2
    assert 4 not in index.block_owners
    assert index.block_owners[12] is nodes[0]


def test_selector_removes_nodes_from_persistent_index():
    cache, manager, nodes = _make_cache([4, 5, 6, 7])
    _select_page_aware_plan(cache, token_budget=4)
    index = cache._kvcached_radix_block_index

    removed = nodes[0]
    cache.evictable_leaves.remove(removed)
    del cache.root_node.children[4]
    manager.free([4])

    _select_page_aware_plan(cache, token_budget=3)

    assert removed not in index.node_blocks
    assert 4 not in index.block_owners


def test_native_split_refreshes_index_for_internal_suffix_closure():
    (
        radix_cache_cls,
        radix_key_cls,
        evict_params_cls,
        insert_params_cls,
        match_params_cls,
    ) = _native_radix_types()
    manager = FakeManager(range(4, 8))
    cache = radix_cache_cls.create_simulated(
        mock_allocator=FakeAllocator(manager),
        page_size=1,
    )
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=[10, 11, 12, 13],
        block_ids=[4, 5, 6, 7],
    )

    _select_page_aware_plan(cache, token_budget=4)
    old_node = cache.root_node.children[10]
    matched, split_node = _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=[10, 11],
    )
    selection = _select_page_aware_plan(cache, token_budget=4)
    selected = _whole_node_suffixes(selection)
    index = cache._kvcached_radix_block_index

    assert matched == [4, 5]
    assert split_node is not old_node
    assert index.node_blocks[split_node] == (4, 5)
    assert index.node_blocks[old_node] == (6, 7)
    assert selected == {split_node}

    result = _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=4,
        evict_params_cls=evict_params_cls,
    )

    assert result.num_tokens_evicted == 4
    assert manager.allocated == set()
    assert index.node_blocks == {}
    assert index.block_owners == {}


def test_native_page_aware_evict_trims_reclaimable_leaf_suffix():
    (
        radix_cache_cls,
        radix_key_cls,
        evict_params_cls,
        insert_params_cls,
        match_params_cls,
    ) = _native_radix_types()
    manager = FakeManager(range(4, 12))
    cache = radix_cache_cls.create_simulated(
        mock_allocator=FakeAllocator(manager),
        page_size=1,
    )
    tokens = list(range(10, 18))
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=tokens,
        block_ids=list(range(4, 12)),
    )

    result = _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=4,
        evict_params_cls=evict_params_cls,
    )

    retained_values, retained_node = _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=tokens,
    )
    assert result.num_tokens_evicted == 4
    assert retained_values == [4, 5, 6, 7]
    assert retained_node.value.tolist() == [4, 5, 6, 7]
    assert manager.allocated == {4, 5, 6, 7}
    assert cache.evictable_size_ == 4
    index = cache._kvcached_radix_block_index
    assert index.node_blocks == {retained_node: (4, 5, 6, 7)}
    assert set(index.block_owners) == {4, 5, 6, 7}


def test_native_page_aware_evict_combines_multiple_leaf_suffixes():
    (
        radix_cache_cls,
        radix_key_cls,
        evict_params_cls,
        insert_params_cls,
        match_params_cls,
    ) = _native_radix_types()
    manager = FakeManager([4, 5, 6, 7, 100, 101])
    cache = radix_cache_cls.create_simulated(
        mock_allocator=FakeAllocator(manager),
        page_size=1,
    )
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=[10, 11, 12],
        block_ids=[100, 4, 5],
    )
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=[20, 21, 22],
        block_ids=[101, 6, 7],
    )

    result = _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=4,
        evict_params_cls=evict_params_cls,
    )

    assert result.num_tokens_evicted == 4
    assert manager.allocated == {100, 101}
    assert _native_match(
        cache, radix_key_cls, match_params_cls, tokens=[10, 11, 12]
    )[0] == [100]
    assert _native_match(
        cache, radix_key_cls, match_params_cls, tokens=[20, 21, 22]
    )[0] == [101]


def test_native_page_aware_evict_splits_internal_prefix_and_removes_children():
    (
        radix_cache_cls,
        radix_key_cls,
        evict_params_cls,
        insert_params_cls,
        match_params_cls,
    ) = _native_radix_types()
    manager = FakeManager([4, 5, 6, 7, 100])
    cache = radix_cache_cls.create_simulated(
        mock_allocator=FakeAllocator(manager, page_size=2),
        page_size=2,
    )
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=[10, 11, 12, 13, 14, 15, 20, 21],
        block_ids=[200, 201, 8, 9, 10, 11, 12, 13],
    )
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=[10, 11, 12, 13, 14, 15, 30, 31],
        block_ids=[200, 201, 8, 9, 10, 11, 14, 15],
    )

    shared_prefix = next(iter(cache.root_node.children.values()))
    selection = _select_page_aware_plan(cache, token_budget=8)

    assert selection.suffix_plans == [(shared_prefix, 2)]
    assert selection.token_count == 8

    result = _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=8,
        evict_params_cls=evict_params_cls,
    )

    assert result.num_tokens_evicted == 8
    assert manager.allocated == {100}
    assert _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=[10, 11, 12, 13, 14, 15, 20, 21],
    )[0] == [200, 201]
    assert _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=[10, 11, 12, 13, 14, 15, 30, 31],
    )[0] == [200, 201]


def test_native_page_aware_evict_runs_exact_lru_after_suffix_plan():
    (
        radix_cache_cls,
        radix_key_cls,
        evict_params_cls,
        insert_params_cls,
        match_params_cls,
    ) = _native_radix_types()
    manager = FakeManager(range(4, 16))
    cache = radix_cache_cls.create_simulated(
        mock_allocator=FakeAllocator(manager),
        page_size=1,
    )
    long_tokens = list(range(10, 18))
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=long_tokens,
        block_ids=list(range(4, 12)),
    )
    for offset, block_id in enumerate(range(12, 16)):
        _native_insert(
            cache,
            radix_key_cls,
            insert_params_cls,
            tokens=[100 + offset],
            block_ids=[block_id],
        )

    result = _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=8,
        evict_params_cls=evict_params_cls,
    )

    assert result.num_tokens_evicted == 8
    assert manager.allocated == {12, 13, 14, 15}
    assert _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=long_tokens,
    )[0] == []


def test_native_exact_lru_splits_final_leaf_at_logical_page_boundary():
    (
        radix_cache_cls,
        radix_key_cls,
        evict_params_cls,
        insert_params_cls,
        match_params_cls,
    ) = _native_radix_types()
    manager = FakeManager([4, 5, 6, 7])
    cache = radix_cache_cls.create_simulated(
        mock_allocator=FakeAllocator(manager, page_size=2),
        page_size=2,
    )
    tokens = list(range(10, 16))
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=tokens,
        block_ids=list(range(8, 14)),
    )

    result = _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=2,
        evict_params_cls=evict_params_cls,
    )

    assert result.num_tokens_evicted == 2
    assert _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=tokens,
    )[0] == [8, 9, 10, 11]
    assert manager.allocated == {4, 5, 7}


@pytest.mark.parametrize(
    ("cached_tokens", "requested_tokens", "expected_evicted"),
    [
        (1024, 24, 32),
        (16, 8, 16),
    ],
)
def test_native_page_aware_evict_aligns_fractional_page_budget(
    cached_tokens, requested_tokens, expected_evicted
):
    (
        radix_cache_cls,
        radix_key_cls,
        evict_params_cls,
        insert_params_cls,
        match_params_cls,
    ) = _native_radix_types()
    logical_page_size = 16
    manager = FakeManager(range(cached_tokens // logical_page_size))
    cache = radix_cache_cls.create_simulated(
        mock_allocator=FakeAllocator(manager, page_size=logical_page_size),
        page_size=logical_page_size,
    )
    tokens = list(range(10_000, 10_000 + cached_tokens))
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=tokens,
        block_ids=list(range(cached_tokens)),
    )

    result = _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=requested_tokens,
        evict_params_cls=evict_params_cls,
    )

    retained_values, _ = _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=tokens,
    )
    assert result.num_tokens_evicted == expected_evicted
    assert len(retained_values) == cached_tokens - expected_evicted
    assert cache.evictable_size_ == cached_tokens - expected_evicted


def test_native_cancel_keeps_prefix_locked_by_another_request():
    (
        radix_cache_cls,
        radix_key_cls,
        evict_params_cls,
        insert_params_cls,
        match_params_cls,
    ) = _native_radix_types()
    manager = FakeManager([4, 5, 6, 8, 9])
    cache = radix_cache_cls.create_simulated(
        mock_allocator=FakeAllocator(manager),
        page_size=1,
    )
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=[10, 11, 12],
        block_ids=[4, 5, 6],
    )
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=[20],
        block_ids=[8],
        priority=0,
    )
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=[21],
        block_ids=[9],
        priority=1,
    )

    shared_values, shared_node = _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=[10, 11, 12],
    )
    cache.inc_lock_ref(shared_node)
    cache.inc_lock_ref(shared_node)

    cache.dec_lock_ref(shared_node)
    assert shared_node.lock_ref == 1

    result = _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=2,
        evict_params_cls=evict_params_cls,
    )
    retained_values, retained_node = _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=[10, 11, 12],
    )

    assert result.num_tokens_evicted == 2
    assert shared_values == retained_values == [4, 5, 6]
    assert retained_node is shared_node
    assert manager.allocated == {4, 5, 6}

    cache.dec_lock_ref(shared_node)
    assert shared_node.lock_ref == 0
    assert _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=[10, 11, 12],
    )[0] == [4, 5, 6]


def test_native_partial_eviction_failure_restores_policy_and_syncs_index():
    (
        radix_cache_cls,
        radix_key_cls,
        evict_params_cls,
        insert_params_cls,
        match_params_cls,
    ) = _native_radix_types()
    manager = FakeManager([4, 5, 6, 7, 8])

    class FaultingAllocator(FakeAllocator):
        def __init__(self):
            super().__init__(manager)
            self.free_calls = 0
            self.fault_hits = 0

        def free(self, token_indices):
            self.free_calls += 1
            if self.free_calls == 2:
                self.fault_hits += 1
                raise RuntimeError("injected second-free failure")
            super().free(token_indices)

    allocator = FaultingAllocator()
    cache = radix_cache_cls.create_simulated(
        mock_allocator=allocator,
        page_size=1,
    )
    for offset, block_id in enumerate(range(4, 8)):
        _native_insert(
            cache,
            radix_key_cls,
            insert_params_cls,
            tokens=[100 + offset],
            block_ids=[block_id],
            priority=offset,
        )
    _native_insert(
        cache,
        radix_key_cls,
        insert_params_cls,
        tokens=[999],
        block_ids=[8],
    )
    locked_values, locked_node = _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=[999],
    )
    cache.inc_lock_ref(locked_node)
    original_strategy = cache.eviction_strategy

    with pytest.raises(RuntimeError, match="injected second-free failure"):
        _evict_radix_cache_page_aware(
            radix_cache=cache,
            num_tokens=2,
            evict_params_cls=evict_params_cls,
        )

    index = cache._kvcached_radix_block_index
    surviving_values, _ = _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=[101],
    )
    retained_locked_values, retained_locked_node = _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=[999],
    )

    assert allocator.free_calls == 2
    assert allocator.fault_hits == 1
    assert cache.eviction_strategy is original_strategy
    assert 4 not in manager.allocated
    assert 4 not in index.block_owners
    assert set(index.block_owners) == {5, 6, 7}
    assert surviving_values == [5]
    assert retained_locked_values == locked_values == [8]
    assert retained_locked_node is locked_node

    manager.allocated.add(12)
    insert_result = cache.insert(
        insert_params_cls(
            key=radix_key_cls(token_ids=array("q", [101, 201])),
            value=torch.tensor([5, 12], dtype=torch.int64),
        )
    )
    assert insert_result.prefix_len == 1
    assert _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=[101, 201],
    )[0] == [5, 12]

    retry_result = _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=1,
        evict_params_cls=evict_params_cls,
    )
    assert retry_result.num_tokens_evicted == 1
    assert allocator.free_calls == 3
    assert _native_match(
        cache,
        radix_key_cls,
        match_params_cls,
        tokens=[101],
    )[0] == [5]

    cache.dec_lock_ref(locked_node)

def test_page_aware_evict_uses_native_radix_eviction():
    cache, manager, nodes = _make_cache([4, 8, 5, 9, 6, 10, 7, 11])
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

    assert cache.eviction_budgets == [4]
    assert manager.page_allocator.group_calls == 1
    assert set(cache.evicted) == {4, 5, 6, 7}
    assert manager.allocated == {8, 9, 10, 11}
    index = cache._kvcached_radix_block_index
    assert set(index.block_owners) == {8, 9, 10, 11}


def test_page_aware_evict_preserves_budget_when_page_does_not_fit():
    cache, _manager, _nodes = _make_cache([4, 5, 6, 7])
    original_strategy = cache.eviction_strategy
    eviction_calls = []

    def evict(params):
        eviction_calls.append((params.num_tokens, cache.eviction_strategy))

    cache.evict = evict
    _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=1,
        evict_params_cls=FakeEvictParams,
    )

    assert len(eviction_calls) == 1
    assert eviction_calls[0][0] == 1
    assert eviction_calls[0][1] is not original_strategy
    assert cache.eviction_strategy is original_strategy


def test_page_aware_evict_falls_back_when_no_page_is_reclaimable():
    cache, _manager, _nodes = _make_cache(
        [5, 6, 7],
        allocated=[4, 5, 6, 7],
    )
    original_strategy = cache.eviction_strategy
    eviction_calls = []

    def evict(params):
        eviction_calls.append((params.num_tokens, cache.eviction_strategy))

    cache.evict = evict
    _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=2,
        evict_params_cls=FakeEvictParams,
    )

    assert [budget for budget, _strategy in eviction_calls] == [2]
    assert all(strategy is not original_strategy for _, strategy in eviction_calls)
    assert cache.eviction_strategy is original_strategy


def test_page_aware_evict_batches_no_reclaimable_page_fallback():
    num_pages = 400
    cached_blocks = [
        4 * page_id + offset
        for page_id in range(num_pages)
        for offset in (1, 2, 3)
    ]
    pinned_blocks = [4 * page_id for page_id in range(num_pages)]
    cache, _manager, _nodes = _make_cache(
        cached_blocks,
        allocated=[*cached_blocks, *pinned_blocks],
    )
    original_strategy = cache.eviction_strategy
    eviction_calls = []

    def evict(params):
        eviction_calls.append((params.num_tokens, cache.eviction_strategy))

    cache.evict = evict
    _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=600,
        evict_params_cls=FakeEvictParams,
    )

    assert len(eviction_calls) == 1
    assert eviction_calls[0][0] == 600
    assert eviction_calls[0][1] is not original_strategy
    assert cache.eviction_strategy is original_strategy


def test_page_aware_evict_skips_planning_when_evicting_all_tokens(monkeypatch):
    cache, _manager, _nodes = _make_cache([4, 5, 6, 7])
    eviction_calls = []

    def fail_if_planned(*args, **kwargs):
        raise AssertionError("full eviction should not build a page-aware plan")

    def evict(params):
        eviction_calls.append(params.num_tokens)

    monkeypatch.setattr(
        cache.token_to_kv_pool_allocator.kvcached_allocator.page_allocator,
        "group_indices_by_page",
        fail_if_planned,
    )
    cache.evict = evict

    _evict_radix_cache_page_aware(
        radix_cache=cache,
        num_tokens=cache.evictable_size_,
        evict_params_cls=FakeEvictParams,
    )

    assert eviction_calls == [cache.evictable_size_]


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
            self.page_size = cache.page_size
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

        def evict(self, num_tokens):
            self.eviction_args.append(num_tokens)

    radix_module.RadixCache = FakeLegacyRadixCache
    assert RadixCacheLimitPatch().patch_radix_cache_limit(radix_module)

    cache = FakeLegacyRadixCache()
    cache.cache_finished_req(None)

    assert cache.eviction_args == [4]
    assert not hasattr(cache, "_kvcached_radix_block_index")


def test_radix_patch_enforces_limit_after_each_finished_request(monkeypatch):
    radix_module: Any = types.ModuleType("sglang.srt.mem_cache.radix_cache")
    monkeypatch.setattr(sglang_patches, "MAX_CACHED_TOKENS", 4)

    class FakeRadixCache:

        def __init__(self):
            self.evictable_size_ = 4
            self.eviction_args = []

        def cache_finished_req(self, *args, **kwargs):
            self.evictable_size_ += 1

        def evict(self, num_tokens):
            self.eviction_args.append(num_tokens)
            self.evictable_size_ -= num_tokens

    radix_module.RadixCache = FakeRadixCache
    assert RadixCacheLimitPatch().patch_radix_cache_limit(radix_module)

    cache = FakeRadixCache()
    cache.cache_finished_req(None)

    assert cache.evictable_size_ == 4
    assert cache.eviction_args == [1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
