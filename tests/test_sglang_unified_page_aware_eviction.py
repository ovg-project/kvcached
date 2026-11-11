# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Real SGLang FULL-only trees must release pages while retaining prefixes."""

import types
from array import array

import pytest
import torch

from kvcached.integration.sglang.page_eviction import (
    evict_prefix_cache,
    supports_page_aware_eviction,
)
from kvcached.integration.sglang.patches import UnifiedRadixCacheLimitPatch

pytest_plugins = ["test_sglang_auto_page_size_native"]


class Manager:
    block_mem_size = 1
    page_size = 4

    def __init__(self, blocks):
        self.allocated = set(blocks)
        self.page_allocator = self

    def group_indices_by_page(self, indices, block_mem_size):
        pages: dict[int, list[int]] = {}
        for block in indices:
            pages.setdefault(block // self.page_size, []).append(block)
        return pages

    def get_page_occupancy(self, pages):
        return {
            page: sum(block // self.page_size == page for block in self.allocated) for page in pages
        }

    def free(self, indices):
        self.allocated.difference_update(indices)


class Allocator:
    device = "cpu"

    def __init__(self, manager, page_size):
        self.kvcached_allocator = manager
        self.page_size = page_size
        self.free_group = None

    def free(self, indices):
        if self.free_group is not None:
            self.free_group.append(indices.clone())
        else:
            self.kvcached_allocator.free([int(index) // self.page_size for index in indices])

    def free_segment(self, indices, start_pos):
        self.free(indices)


@pytest.fixture
def native():
    unified = pytest.importorskip("sglang.srt.mem_cache.unified_radix_cache")
    base = pytest.importorskip("sglang.srt.mem_cache.base_prefix_cache")
    init = pytest.importorskip("sglang.srt.mem_cache.cache_init_params")
    return types.SimpleNamespace(unified=unified, base=base, init=init)


def make_cache(native, blocks, page_size=1, policy="lru"):
    manager = Manager(blocks)
    allocator = Allocator(manager, page_size)
    cache = native.unified.UnifiedRadixCache(
        native.init.CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            tree_components=(native.unified.BASE_COMPONENT_TYPE,),
            eviction_policy=policy,
        )
    )
    return cache, manager


def insert(native, cache, tokens, indices):
    cache.insert(
        native.base.InsertParams(
            key=native.unified.RadixKey(token_ids=array("q", tokens)),
            value=torch.tensor(indices, dtype=torch.int64, device=cache.device),
        )
    )


def match(native, cache, tokens):
    result = cache.match_prefix(
        native.base.MatchPrefixParams(
            key=native.unified.RadixKey(token_ids=array("q", tokens)),
        )
    )
    return result.device_indices.tolist(), result.last_device_node


def core(cache):
    return getattr(cache, "tree_core", cache)


def check_tree(cache):
    # Check the native tree/LRU/lock counters, not just freed block IDs.
    cache.sanity_check()


@pytest.mark.parametrize("policy", ["lru", "lfu"])
@pytest.mark.parametrize("session_field", [True, False], ids=["current", "pre_0_5_17"])
def test_page_selection_retains_scattered_old_prefixes(native, policy, session_field):
    cache, manager = make_cache(native, range(4, 12), policy=policy)
    if not session_field and hasattr(cache, "enable_session_radix_cache"):
        # SGLang <= 0.5.16 has no enable_session_radix_cache field.
        del cache.enable_session_radix_cache
    # Native eviction picks 4,8,5,9: both pages stay mapped. Page selection
    # picks one complete page instead, retaining the other four prefixes.
    order = [4, 8, 5, 9, 6, 10, 7, 11]
    for block in order:
        insert(native, cache, [100 + block], [block])

    evict_prefix_cache(cache, native.base.EvictParams(num_tokens=4))

    assert len(manager.allocated) == 4
    assert len({block // 4 for block in manager.allocated}) == 1
    assert cache.full_evictable_size() == 4
    for block in order:
        assert match(native, cache, [100 + block])[0] == (
            [block] if block in manager.allocated else []
        )
    check_tree(cache)


@pytest.mark.parametrize("page_size,budget", [(1, 4), (2, 7)])
def test_suffix_split_keeps_reusable_prefix(native, page_size, budget):
    cache, manager = make_cache(native, range(4, 12), page_size)
    tokens = list(range(100, 100 + 8 * page_size))
    indices = list(range(4 * page_size, 12 * page_size))
    insert(native, cache, tokens, indices)

    evict_prefix_cache(cache, native.base.EvictParams(num_tokens=budget))

    assert manager.allocated == {4, 5, 6, 7}
    assert match(native, cache, tokens)[0] == indices[: 4 * page_size]
    assert cache.full_evictable_size() == 4 * page_size
    assert set(cache._kvcached_radix_block_index.block_owners) == manager.allocated
    check_tree(cache)


def test_internal_suffix_closure_removes_children(native):
    cache, manager = make_cache(native, [4, 5, 6, 7, 100, 150, 151, 152], page_size=2)
    first = [10, 11, 12, 13, 14, 15, 20, 21]
    second = [10, 11, 12, 13, 14, 15, 30, 31]
    insert(native, cache, first, [200, 201, 8, 9, 10, 11, 12, 13])
    insert(native, cache, second, [300, 301, 302, 303, 304, 305, 14, 15])

    evict_prefix_cache(cache, native.base.EvictParams(num_tokens=8))

    assert manager.allocated == {100}
    assert match(native, cache, first)[0] == [200, 201]
    assert match(native, cache, second)[0] == [200, 201]
    check_tree(cache)


def test_locked_prefix_pins_its_physical_page(native):
    cache, manager = make_cache(native, range(4, 12))
    for block in range(4, 12):
        insert(native, cache, [100 + block], [block])
    _, locked = match(native, cache, [104])
    receipt = cache.inc_lock_ref(locked)

    evict_prefix_cache(cache, native.base.EvictParams(num_tokens=4))

    assert manager.allocated == {4, 5, 6, 7}
    assert match(native, cache, [104])[0] == [4]
    cache.dec_lock_ref(locked, receipt.to_dec_params())
    check_tree(cache)


def test_no_reclaimable_page_uses_native_lru(native):
    # Every cached block shares a page with a running block.
    cache, manager = make_cache(native, [4, 5, 8, 9])
    insert(native, cache, [100], [4])
    insert(native, cache, [101], [8])

    evict_prefix_cache(cache, native.base.EvictParams(num_tokens=1))

    assert manager.allocated == {5, 8, 9}
    assert match(native, cache, [101])[0] == [8]
    check_tree(cache)


@pytest.mark.parametrize("page_size", [1, 2])
def test_selected_and_native_fallback_count_actual_frees(native, monkeypatch, page_size):
    cache, manager = make_cache(native, range(4, 12), page_size)
    insert(
        native,
        cache,
        list(range(100, 100 + 8 * page_size)),
        list(range(4 * page_size, 12 * page_size)),
    )
    calls = []
    original = cache.evict

    def evict(params):
        calls.append(params)
        return original(params)

    monkeypatch.setattr(cache, "evict", evict)
    params = native.base.EvictParams(num_tokens=5 * page_size)
    result = evict_prefix_cache(cache, params)

    assert calls == [
        native.base.EvictParams(num_tokens=4 * page_size),
        native.base.EvictParams(num_tokens=page_size),
    ]
    # The fallback removes the whole retained node, overshooting its demand.
    assert result.num_tokens_evicted == 8 * page_size
    assert result.swa_num_tokens_evicted == result.mamba_num_evicted == 0
    assert params == native.base.EvictParams(num_tokens=5 * page_size)
    assert manager.allocated == set()
    assert cache._kvcached_radix_block_index.block_owners == {}
    check_tree(cache)


def test_failure_restores_cached_priority_and_allows_retry(native, monkeypatch):
    cache, manager = make_cache(native, range(4, 12))
    # Populate FullComponent's bound strategy before page-aware eviction.
    manager.allocated.add(100)
    insert(native, cache, [500], [100])
    cache.evict(native.base.EvictParams(num_tokens=1))
    for block in [4, 8, 5, 9, 6, 10, 7, 11]:
        insert(native, cache, [100 + block], [block])
    tree = core(cache)
    full = cache.components[native.unified.BASE_COMPONENT_TYPE]
    strategy = tree.eviction_strategy
    callback = getattr(full, "session_ref_eviction_strategy", None)
    original = cache.evict

    def failing_evict(params):
        raise RuntimeError("injected eviction failure")

    monkeypatch.setattr(cache, "evict", failing_evict)
    with pytest.raises(RuntimeError, match="injected eviction failure"):
        evict_prefix_cache(cache, native.base.EvictParams(num_tokens=4))
    assert tree.eviction_strategy is strategy
    assert getattr(full, "session_ref_eviction_strategy", None) is callback

    monkeypatch.setattr(cache, "evict", original)
    evict_prefix_cache(cache, native.base.EvictParams(num_tokens=4))
    assert len({block // 4 for block in manager.allocated}) == 1
    assert tree.eviction_strategy is strategy
    assert getattr(full, "session_ref_eviction_strategy", None) is callback
    check_tree(cache)


@pytest.mark.parametrize("failure", ["split", "selected", "fallback", "budget_mismatch"])
def test_partial_failure_resyncs_index_and_restores_policy(native, monkeypatch, failure):
    cache, manager = make_cache(native, range(4, 12))
    tokens = list(range(100, 108))
    # Bind the native callback before the operation to verify identity recovery.
    manager.allocated.add(100)
    insert(native, cache, [500], [100])
    cache.evict(native.base.EvictParams(num_tokens=1))
    if failure == "budget_mismatch":
        for block in [4, 8, 5, 9, 6, 10, 7, 11]:
            insert(native, cache, [100 + block], [block])
    else:
        insert(native, cache, tokens, list(range(4, 12)))
    tree = core(cache)
    full = cache.components[native.unified.BASE_COMPONENT_TYPE]
    strategy = tree.eviction_strategy
    callback = getattr(full, "session_ref_eviction_strategy", None)
    original_split = tree._split_node
    original_evict = cache.evict
    calls = []

    def split_then_fail(*args):
        original_split(*args)
        raise RuntimeError("injected partial failure")

    def evict_then_fail(params):
        calls.append(params.num_tokens)
        if failure == "budget_mismatch":
            return original_evict(native.base.EvictParams(num_tokens=1))
        result = original_evict(params)
        if failure == "selected" or len(calls) == 2:
            raise RuntimeError("injected partial failure")
        return result

    if failure == "split":
        monkeypatch.setattr(tree, "_split_node", split_then_fail)
    else:
        monkeypatch.setattr(cache, "evict", evict_then_fail)
    params = native.base.EvictParams(num_tokens=5 if failure == "fallback" else 4)
    error = (
        "exact selected-node budget" if failure == "budget_mismatch" else "injected partial failure"
    )
    with pytest.raises(RuntimeError, match=error):
        evict_prefix_cache(cache, params)

    assert manager.allocated == (
        set(range(4, 12))
        if failure == "split"
        else set()
        if failure == "fallback"
        else set(range(5, 12))
        if failure == "budget_mismatch"
        else {4, 5, 6, 7}
    )
    assert tree.eviction_strategy is strategy
    assert getattr(full, "session_ref_eviction_strategy", None) is callback
    assert set(cache._kvcached_radix_block_index.block_owners) == manager.allocated
    assert cache.full_evictable_size() == len(manager.allocated)
    if failure == "fallback":
        assert calls == [4, 1]
    check_tree(cache)

    monkeypatch.setattr(tree, "_split_node", original_split)
    monkeypatch.setattr(cache, "evict", original_evict)
    result = evict_prefix_cache(cache, native.base.EvictParams(num_tokens=4))
    assert result.num_tokens_evicted == (4 if failure != "fallback" else 0)
    assert result.swa_num_tokens_evicted == result.mamba_num_evicted == 0
    assert manager.allocated == (
        {4, 5, 6, 7}
        if failure == "split"
        else {9, 10, 11}
        if failure == "budget_mismatch"
        else set()
    )
    assert set(cache._kvcached_radix_block_index.block_owners) == manager.allocated
    assert tree.eviction_strategy is strategy
    check_tree(cache)


def test_deferred_frees_keep_pages_mapped_until_group_flush(native):
    cache, manager = make_cache(native, range(4, 12))
    insert(native, cache, list(range(100, 108)), list(range(4, 12)))
    allocator = cache.token_to_kv_pool_allocator
    allocator.free_group = []

    evict_prefix_cache(cache, native.base.EvictParams(num_tokens=4))

    assert manager.allocated == set(range(4, 12))
    assert cache.full_evictable_size() == 4
    pending, allocator.free_group = allocator.free_group, None
    for indices in pending:
        allocator.free(indices)
    assert manager.allocated == {4, 5, 6, 7}
    check_tree(cache)


@pytest.mark.parametrize(
    "unsupported",
    [
        "swa",
        "mamba",
        "hicache",
        "session",
        "custom_core",
        "custom_component",
        "linker",
    ],
)
def test_unsupported_configurations_keep_native_eviction(native, monkeypatch, unsupported):
    cache, manager = make_cache(native, range(4, 12))
    if unsupported in ("swa", "mamba"):
        cache.tree_components = (
            *cache.tree_components,
            getattr(native.unified.ComponentType, unsupported.upper()),
        )
    elif unsupported == "hicache":
        cache.cache_controller = object()
    elif unsupported == "session":
        cache.enable_session_radix_cache = True
    elif unsupported == "custom_core":
        cache.tree_core = types.SimpleNamespace()
    elif unsupported == "custom_component":
        cache.components[native.unified.BASE_COMPONENT_TYPE] = object()
    else:
        core(cache).enable_external_cache_linker = True
    assert not supports_page_aware_eviction(cache)
    calls = []
    sentinel = object()

    def native_evict(params):
        calls.append(params)
        return sentinel

    monkeypatch.setattr(cache, "evict", native_evict)
    params = native.base.EvictParams(num_tokens=4)
    assert evict_prefix_cache(cache, params) is sentinel
    assert calls == [params]
    assert manager.allocated == set(range(4, 12))


def test_retention_hook_and_reset_use_unified_page_planner(native, monkeypatch):
    import kvcached.prefix_cache_retention as retention

    monkeypatch.setattr(retention, "_policy", retention._PrefixCacheRetentionPolicy(4))
    cache_cls = native.unified.UnifiedRadixCache
    original_init = cache_cls.__init__
    original_reset = cache_cls.reset
    monkeypatch.setattr(cache_cls, "cache_finished_req", lambda self, req: None)
    # Record patch mutations so pytest restores the real engine afterwards.
    monkeypatch.setattr(cache_cls, "__init__", original_init)
    monkeypatch.setattr(cache_cls, "reset", original_reset)
    monkeypatch.setattr(cache_cls, "_enforce_prefix_cache_retention", None, raising=False)
    patch = UnifiedRadixCacheLimitPatch()
    monkeypatch.setattr(patch.version_manager, "detect_version", lambda _: "0.5.20")
    assert patch.apply(native.unified)
    cache, manager = make_cache(native, range(4, 12))
    for block in [4, 8, 5, 9, 6, 10, 7, 11]:
        insert(native, cache, [100 + block], [block])

    cache.cache_finished_req(None)

    assert len({block // 4 for block in manager.allocated}) == 1
    assert cache._kvcached_radix_block_index is not None
    retention._policy.set_limit(0)
    assert manager.allocated == set()
    cache.reset()
    assert cache._kvcached_radix_block_index is None
    check_tree(cache)


@pytest.mark.parametrize("attention,buffers", [("MHA", 2), ("MLA", 1)])
def test_cuda_page_release_preserves_retained_kv(native, pools, attention, buffers):
    iface, managers = pools
    cell_bytes = 256 * 1024
    raw = iface.alloc_kv_cache(
        (32, 1, cell_bytes // 2),
        torch.bfloat16,
        "cuda:0",
        2,
        page_size=2,
        attention_type=attention,
    )
    tensors = raw if attention == "MLA" else raw[0] + raw[1]
    manager = iface.get_kv_cache_manager(16, 2, cell_bytes, 2, num_kv_buffers=buffers)
    managers.append(manager)
    manager.wait_ready(timeout=10)
    allocated = manager.alloc(11)
    assert allocated == list(range(1, 12))
    allocator = Allocator(manager, page_size=2)
    allocator.device = "cuda:0"
    cache = native.unified.UnifiedRadixCache(
        native.init.CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=allocator,
            page_size=2,
            tree_components=(native.unified.BASE_COMPONENT_TYPE,),
        )
    )
    for block in allocated:
        for tensor in tensors:
            tensor[block * 2 : block * 2 + 2].fill_(block)
    torch.cuda.synchronize()
    for block in [4, 8, 5, 9, 6, 10, 7, 11]:
        insert(native, cache, [100 + block * 2, 101 + block * 2], [block * 2, block * 2 + 1])
    before = manager.page_allocator.get_num_inuse_pages()

    evict_prefix_cache(cache, native.base.EvictParams(num_tokens=8))

    torch.cuda.synchronize()
    assert manager.page_allocator.get_num_inuse_pages() == before - 1
    retained = [
        block
        for block in range(4, 12)
        if match(native, cache, [100 + block * 2, 101 + block * 2])[0]
    ]
    assert len(retained) == 4
    assert len({block // 4 for block in retained}) == 1
    for block in [1, 2, 3, *retained]:
        for tensor in tensors:
            assert torch.all(tensor[block * 2 : block * 2 + 2] == block).item()
    torch.cuda.synchronize()
    evict_prefix_cache(cache, native.base.EvictParams(num_tokens=cache.full_evictable_size()))
    manager.free([1, 2, 3])
    check_tree(cache)
