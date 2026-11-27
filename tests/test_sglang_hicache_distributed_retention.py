# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Exercise retention votes with separate Gloo attention and pipeline groups."""

import gc
import sys
from array import array
from dataclasses import dataclass
from datetime import timedelta
from types import ModuleType, SimpleNamespace
from weakref import WeakKeyDictionary, ref

import pytest
import torch
import torch.distributed as dist

from kvcached.integration.sglang import hicache_retention as retention, rank_groups


class Leaf:
    def __init__(self, token, parent, salt=None):
        self.parent = parent
        self.key = Key([token], salt)
        self.backuped = True
        self.children = {}
        self.evicted = False
        self.lock_ref = 0


class Key(list):
    is_bigram = False
    extra_key = None

    def __init__(self, tokens, salt):
        super().__init__(tokens)
        self.cache_salt = salt

    def raw_token_ids(self):
        return array("q", self)


def _priority_cache(parent_locked):
    root = Leaf(0, None)
    parent = Leaf(10, root)
    cold = Leaf(11, parent)
    hot = Leaf(20, root)
    active = Leaf(30, root)
    parent.children[11] = cold
    parent.lock_ref, active.lock_ref = int(parent_locked), 1
    priorities = {cold: 0, parent: 1, hot: 2, active: -1}
    order = []
    cache = SimpleNamespace(
        root_node=root, evictable_leaves={cold, hot}, evictable_size_=3 - int(parent_locked), protected_size_=0,
        eviction_strategy=SimpleNamespace(get_priority=priorities.__getitem__),
        cache_controller=SimpleNamespace(write_policy="write_through"), pp_size=1,
        update_eviction_metrics=lambda *_: None)

    def demote(node):
        assert node in cache.evictable_leaves and not node.lock_ref
        node.evicted = True
        cache.evictable_leaves.remove(node)
        cache.evictable_size_ -= len(node.key)
        order.append(node.key[0])
        if node is cold and not parent.lock_ref:
            cache.evictable_leaves.add(parent)
        return len(node.key)

    cache._evict_backuped = demote
    return cache, order, (parent, cold, hot, active)


def _poll_to_limit(cache, limit):
    """Drive the scheduler's bounded victim plan to a verdict for protocol tests."""
    plan = retention._VictimPlan()
    demand, _, proposer = retention._group_demand(cache, limit, True)
    while demand:
        result = plan.poll(cache, demand, proposer, False)
        if result is not None:
            return result
        demand, _, proposer = retention._group_demand(cache, limit, False)
    return True


@pytest.mark.parametrize("locked, expected, resident", [
    (False, [11, 10], [20, 30]), (True, [11, 20], [10, 30])])
def test_retention_reconsiders_new_device_leaves(locked, expected, resident):
    cache, order, nodes = _priority_cache(locked)
    _poll_to_limit(cache, cache.evictable_size_ - 2)
    assert order == expected
    assert [node.key[0] for node in nodes if not node.evicted] == resident


def test_limit_shrink_evicts_a_token_count_through_the_victim_plan():
    """A memory-limit shrink asks for a count; the plan picks the victims."""
    cache, order, _ = _priority_cache(False)
    # Without retention's leaf tracking no rank can enumerate victims.
    assert not retention.evict_parallel_hicache(cache, 2)
    assert order == []

    retention._track_device_leaves(cache)
    result = retention.evict_parallel_hicache(cache, 2)
    while result is None:
        result = retention.evict_parallel_hicache(cache, 0)
    assert result
    assert order == [11, 10]
    assert not retention.parallel_hicache_trim_pending(cache)


def test_limit_shrink_polls_once_per_call(monkeypatch):
    """Each call does one bounded poll and keeps the plan for the next idle
    pass; the target is fixed when the trim starts."""
    cache, order, _ = _priority_cache(False)
    retention._track_device_leaves(cache)
    limits = []
    group_demand = retention._group_demand

    def count_demand(cache, limit, changed):
        limits.append(limit)
        return group_demand(cache, limit, changed)

    monkeypatch.setattr(retention, "_group_demand", count_demand)
    target = cache.evictable_size_ - 2
    calls = 0
    result = None
    while result is None:
        result = retention.evict_parallel_hicache(cache, 2 if calls == 0 else 5)
        calls += 1
        assert len(limits) == calls
    assert result and order == [11, 10]
    assert set(limits) == {target}


@pytest.mark.parametrize("still_needed", [0, 32])
def test_lockstep_trim_stops_once_no_rank_needs_space(monkeypatch, still_needed):
    """kvctl raises the limit after a 32-token trim indexed its first page.
    The trim then keeps every prefix, unless some rank still needs space."""
    from kvcached.integration.sglang import patches

    cache, order, _ = _priority_cache(False)
    cache.evictable_leaves = {Leaf(token, cache.root_node) for token in range(100, 1124)}
    cache.evictable_size_ = len(cache.evictable_leaves)
    cache.eviction_strategy.get_priority = lambda node: node.key[0]
    retention._track_device_leaves(cache)
    assert retention.evict_parallel_hicache(cache, 32) is None
    assert order == []

    monkeypatch.setattr(patches, "reduce_counts", lambda counts, groups, op: counts)
    reclaimer = SimpleNamespace(swa=False, shrink_tokens=lambda: still_needed,
                                manager=SimpleNamespace(in_shrink=False))
    scheduler = SimpleNamespace(_kvcached_shrink_hicache=cache)
    # Stay between two lockstep rounds, so every pass only continues the trim.
    for _ in range(patches._LOCKSTEP_SHRINK_INTERVAL - 1):
        if not retention.parallel_hicache_trim_pending(cache):
            break
        patches._shrink_in_lockstep(scheduler, [reclaimer], ["group"])

    assert not retention.parallel_hicache_trim_pending(cache)
    assert order == list(range(100, 100 + still_needed))


def test_unbacked_parent_above_cap_retries_after_host_child_leaves():
    cache, order, (parent, cold, _, _) = _priority_cache(False)
    cache._evict_backuped(cold)
    parent.backuped = False
    cache.evictable_leaves = {parent}
    cache.evictable_size_ = 1
    assert not _poll_to_limit(cache, 0)
    assert order == [11]
    parent.children.clear()
    cache._evict_regular = cache._evict_backuped
    assert _poll_to_limit(cache, 0)
    assert order == [11, 10]


@pytest.mark.parametrize("paused,storage", [(True, False), (False, False), (False, True)])
def test_large_trim_reuses_its_plan_between_quiet_polls(monkeypatch, paused, storage):
    root = Leaf(-1, None)
    leaves = []
    for token in range(0, 2048, 2):
        parent = Leaf(token, root)
        child = Leaf(token + 1, parent)
        parent.children[token + 1] = child
        leaves.append(child)
    votes: list[int] = []
    evicted: list[int] = []
    evaluations = 0

    def priority(node):
        nonlocal evaluations
        evaluations += 1
        return node.key[0]

    cache = SimpleNamespace(
        pp_size=1, tp_world_size=2, cache_controller=SimpleNamespace(write_policy="write_through"),
        enable_storage=storage,
        evictable_size_=0, protected_size_=2048, evictable_leaves=set(),
        eviction_strategy=SimpleNamespace(get_priority=priority),
        check_hicache_events=lambda: None, _all_reduce=lambda *_: None,
        _enforce_prefix_cache_retention=lambda limit: None, update_eviction_metrics=lambda *_: None)
    monkeypatch.setattr(retention, "_reduce_victims", lambda vote, op: votes.append(vote.numel()))

    def demote(node):
        cache.evictable_leaves.remove(node)
        cache.evictable_size_ -= 1
        node.evicted = True
        if node.parent is not root:
            cache.evictable_leaves.add(node.parent)
        evicted.append(node.key[0])
        return 1

    def dispatch(requests):
        if "pause" in requests:
            scheduler._engine_paused = True
        if "retract" in requests:
            cache.evictable_leaves.update(leaves)
            cache.evictable_size_, cache.protected_size_ = 2048, 0
            cache._enforce_prefix_cache_retention(0)

    cache._evict_backuped = demote
    scheduler = SimpleNamespace(tree_cache=cache, enable_hierarchical_cache=True, is_fully_idle=lambda: True,
                                process_input_requests=dispatch, _engine_paused=False)
    sleeps = []
    scheduler.maybe_sleep_on_idle = lambda: sleeps.append(True)
    def native_poll():
        cache._all_reduce(torch.zeros(2, dtype=torch.int), dist.ReduceOp.MIN)
        cache._enforce_prefix_cache_retention(0)

    cache.check_hicache_events = native_poll
    monkeypatch.setattr(retention, "get_prefix_cache_retention_limit", lambda: 0)
    retention.install_parallel_hicache_retention(scheduler)
    scheduler.process_input_requests(["pause"] if paused else [])
    assert not evicted
    scheduler.process_input_requests(["retract"])
    if not paused:
        cache.check_hicache_events()
    assert not evicted and evaluations == retention._INDEX_PER_POLL
    for _ in range(2048):
        if not cache.evictable_size_:
            break
        scheduler.maybe_sleep_on_idle()
        assert not sleeps
        before = len(votes)
        remaining = cache.evictable_size_
        scheduler.process_input_requests([])
        if not paused:
            cache.check_hicache_events()
        assert 0 < len(votes) - before <= retention._ROUNDS_PER_POLL
        assert cache.evictable_size_ <= remaining
    assert evicted == [key for token in range(0, 2048, 2) for key in (token + 1, token)]
    assert evaluations == 2048
    scheduler.maybe_sleep_on_idle()
    assert sleeps == [True]
    before = len(votes)
    scheduler.process_input_requests([])
    if not paused:
        cache.check_hicache_events()
    assert len(votes) == before


@pytest.mark.parametrize("polls", [1, 4])
def test_busy_trim_pages_remaining_work_and_releases_plan_when_disabled(monkeypatch, polls):
    root = Leaf(-1, None)
    count = 16000
    priorities = []
    cache, _, _ = _priority_cache(False)
    cache.tp_world_size, cache.protected_size_ = 2, 0
    cache.evictable_size_ = count
    cache.evictable_leaves = {Leaf(i, root) for i in range(count)}
    refs = [ref(node) for node in cache.evictable_leaves]
    def priority(node):
        priorities.append(node.key[0])
        return node.key[0]

    def evict(node):
        cache.evictable_leaves.remove(node)
        cache.evictable_size_ -= 1
        return 1

    cache.eviction_strategy.get_priority = priority
    cache._evict_backuped = evict
    cache._enforce_prefix_cache_retention = cache._all_reduce = lambda *_: None
    cache.check_hicache_events = lambda: cache._all_reduce(torch.zeros(2), dist.ReduceOp.MIN)
    scheduler = SimpleNamespace(tree_cache=cache, enable_hierarchical_cache=True, _engine_paused=False,
                                is_fully_idle=lambda: False, process_input_requests=lambda _: None)
    policy = retention.prefix_cache_retention._policy
    monkeypatch.setattr(policy, "limit_tokens", 0)
    monkeypatch.setattr(policy, "applied_revision", None)
    retention.install_parallel_hicache_retention(scheduler)
    for i in range(polls):
        priorities.clear()
        cache.check_hicache_events()
        assert len(priorities) <= retention._INDEX_PER_POLL
        assert cache.evictable_size_ == count
    cache.evictable_leaves.clear()
    gc.collect()
    monkeypatch.setattr(policy, "limit_tokens", -1)
    monkeypatch.setattr(policy, "applied_revision", (2,))
    scheduler.process_input_requests([])
    assert all(r() is None for r in refs)


def test_shared_search_refreshes_a_leaf_hit_between_polls(monkeypatch):
    cache, _, _ = _priority_cache(False)
    root = Leaf(-1, None)
    leaves = {Leaf(i, root) for i in range(66)}
    cache.evictable_leaves = leaves
    hot = next(iter(leaves))  # In the first index page.
    bumped = False
    cache.eviction_strategy.get_priority = lambda n: (5000 if bumped else -1) if n is hot else n.key[0]
    freed = []
    def evict(node):
        freed.append(node)
        return 1

    cache._evict_backuped = evict
    clock = [0.0]
    metrics = []
    def vote(*_):
        clock[0] += 2.0

    monkeypatch.setattr(retention.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(retention, "_reduce_victims", vote)
    cache.update_eviction_metrics = lambda n, start: metrics.append((n, clock[0] - start))
    plan = retention._VictimPlan()
    plan.remaining = 1
    plan.search = plan._shared_victims(cache, 0)
    assert plan.poll(cache, 1, 0, False) is None
    clock[0] += 1000  # Time spent scheduling other work between the votes.
    bumped = True
    assert plan.poll(cache, 1, 0, True) is True
    assert freed == [min(leaves, key=cache.eviction_strategy.get_priority)]
    assert metrics == [(1, 6.0)]


def test_eviction_plans_bound_sibling_reads_on_wide_prefixes(monkeypatch):
    reads = 0

    class Children(dict):
        def values(self):
            nonlocal reads
            for child in super().values():
                reads += 1
                yield child

    cache, _, _ = _priority_cache(False)
    parent = Leaf(-1, cache.root_node)
    page_size = retention._VICTIMS_PER_BATCH
    parent.children = Children((i, Leaf(i, parent)) for i in range(page_size ** 2))
    cache.evictable_leaves = set(dict.values(parent.children))
    cache.eviction_strategy.get_priority = lambda n: n.key[0]

    def reject(vote, op):
        if op == dist.ReduceOp.MIN:
            vote.zero_()

    monkeypatch.setattr(retention, "_reduce_victims", reject)
    plan = retention._VictimPlan()
    plan.remaining = 1
    for verdict in plan._shared_victims(cache, 0):
        if verdict is not None:
            break
    assert verdict is False
    assert reads <= page_size * (page_size + 1)

    # Device demotion preserves the host entries in insertion order.
    # Later batches must still fill even with host-only siblings at the front.
    sizes = []
    for batch, _ in retention._victim_batches(cache, len(cache.evictable_leaves)):
        if not batch:
            continue
        sizes.append(len(batch))
        for node in batch:
            node.evicted = True
            cache.evictable_leaves.remove(node)
    assert sizes == [page_size] * page_size

    # Only the final two siblings are still device-resident. An ordered
    # trim must not walk all the host-only entries to predict promotion.
    for child in dict.values(parent.children):
        child.evicted = child.key[0] < len(parent.children) - 2
    cache.evictable_leaves = {n for n in dict.values(parent.children) if not n.evicted}
    reads = 0
    assert next(retention._victim_batches(cache, 1))[1] and reads == 0
    ordered = []
    for batch, done in retention._victim_batches(cache, 2):
        for node in batch:
            node.evicted = True
            cache.evictable_leaves.remove(node)
            ordered.append(node.key[0])
    assert done and ordered == [len(parent.children) - 2, len(parent.children) - 1]
    assert reads <= page_size + 1


def test_first_poll_bounds_priority_reads_on_a_large_tree():
    cache, _, _ = _priority_cache(False)
    size = 100_000
    cache.evictable_leaves = {Leaf(i, cache.root_node) for i in range(size)}
    retention._track_device_leaves(cache)
    reads = 0

    def priority(node):
        nonlocal reads
        reads += 1
        return node.key[0]

    cache.eviction_strategy.get_priority = priority
    plan = retention._VictimPlan()
    assert plan.poll(cache, 1, 0, False) is None
    assert reads == retention._INDEX_PER_POLL
    assert len(cache.evictable_leaves) == size


def test_shared_search_reuses_prefix_digest_as_parents_become_leaves(monkeypatch):
    depth = 128
    cache, _, _ = _priority_cache(False)
    parent = cache.root_node
    for token in range(depth):
        child = Leaf(token, parent)
        parent.children[token] = child
        parent = child
    cache.evictable_leaves = {parent}
    cache.eviction_strategy.get_priority = lambda node: node.key[0]
    reads = 0
    raw = Key.raw_token_ids

    def count_reads(key):
        nonlocal reads
        reads += 1
        return raw(key)

    def demote(node):
        cache.evictable_leaves.remove(node)
        if node.parent is not cache.root_node:
            cache.evictable_leaves.add(node.parent)
        return len(node.key)

    cache._evict_backuped = demote
    monkeypatch.setattr(Key, "raw_token_ids", count_reads)
    plan = retention._VictimPlan()
    plan.remaining = depth
    assert list(plan._shared_victims(cache, 0))[-1] is True
    assert not cache.evictable_leaves and reads <= 5 * depth


def test_shared_search_indexes_deep_disjoint_paths_in_bounded_steps(monkeypatch):
    root = Leaf(-1, None)
    leaves = set()
    for branch in range(4):
        parent = root
        for token in range(700):
            node = Leaf(branch * 1000 + token, parent)
            parent.children[node.key[0]] = node
            parent = node
        leaves.add(parent)
    calls = 0
    original = Key.raw_token_ids

    def count_reads(key):
        nonlocal calls
        calls += 1
        return original(key)

    monkeypatch.setattr(Key, "raw_token_ids", count_reads)
    pages = retention._signature_pages(leaves, WeakKeyDictionary())
    indexed: set[Leaf] = set()
    for entries, more, _ in pages:
        assert calls <= retention._INDEX_PER_POLL
        calls = 0
        indexed.update(node for node, _ in entries)
        if not more:
            break
    assert indexed == leaves


def test_shared_search_batches_common_victims_with_resident_siblings(monkeypatch):
    cache, freed, _ = _priority_cache(False)
    cache.evictable_leaves.clear()
    for i in range(200):
        parent = Leaf(i, cache.root_node)
        for token in (i, i + 200):
            child = Leaf(token, parent)
            parent.children[token] = child
            cache.evictable_leaves.add(child)
    retention._track_device_leaves(cache)
    cache.eviction_strategy.get_priority = lambda node: node.key[0]
    ballots = []

    def vote(data, op):
        if op == dist.ReduceOp.MIN:
            ballots.append(data.numel())

    monkeypatch.setattr(retention, "_reduce_victims", vote)
    plan = retention._VictimPlan()
    plan.remaining = 199
    assert list(plan._shared_victims(cache, 0))[-1] is True
    assert freed == list(range(199))
    assert len(ballots) == 4


def test_shared_search_proposes_when_every_lru_priority_changes_between_polls():
    cache, _, _ = _priority_cache(False)
    page = retention._VICTIMS_PER_BATCH
    leaves = {Leaf(i, cache.root_node) for i in range(page + 1)}
    cache.evictable_leaves = leaves
    clock = 0
    cache.eviction_strategy.get_priority = lambda n: clock + n.key[0]
    def evict(node):
        leaves.remove(node)
        return 1

    cache._evict_backuped = evict
    plan = retention._VictimPlan()
    plan.remaining = len(leaves)
    sweep = plan._shared_victims(cache, 0)
    assert next(sweep) is None
    for _ in range(page + 1):
        clock += page + 1
        before = len(leaves)
        done = next(sweep)
        assert len(leaves) < before
        if done:
            break
    assert not leaves


@pytest.mark.parametrize("unified", [False, True])
def test_locked_parents_allow_full_victim_pages(monkeypatch, unified):
    page = retention._VICTIMS_PER_BATCH
    cache, _, _ = _priority_cache(False)
    if unified:
        module = ModuleType("sglang.srt.mem_cache.unified_radix_cache")
        setattr(module, "BASE_COMPONENT_TYPE", 0)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    leaves = set()
    for i in range(4 * page):
        parent = Leaf(i, cache.root_node)
        if unified:
            del parent.lock_ref
            setattr(parent, "component_data", [SimpleNamespace(lock_ref=1)])
        else:
            parent.lock_ref = 1
        child = Leaf(i, parent)
        parent.children[i] = child
        leaves.add(child)
    cache.evictable_leaves = leaves.copy()
    cache.eviction_strategy.get_priority = lambda n: n.key[0]
    batches = []
    for batch, _ in retention._victim_batches(cache, len(leaves)):
        if not batch:
            continue
        batches.append(len(batch))
        cache.evictable_leaves.difference_update(batch)
    assert batches == [page] * 4

    accepted = []
    def count(vote, op):
        if op == dist.ReduceOp.MIN:
            accepted.append(vote.numel())

    monkeypatch.setattr(retention, "_reduce_victims", count)
    cache.evictable_leaves = leaves
    def evict(node):
        leaves.remove(node)
        return 1

    cache._evict_backuped = evict
    plan = retention._VictimPlan()
    plan.remaining = len(leaves)
    assert list(plan._shared_victims(cache, 0))[-1] is True
    assert accepted == [page] * 4 and not leaves


def test_busy_exhausted_search_backs_off_across_pages(monkeypatch):
    cache, _, _ = _priority_cache(False)
    cache.tp_world_size, cache.protected_size_ = 2, 0
    cache.evictable_size_ = 100
    root = Leaf(-1, None)
    cache.evictable_leaves = {Leaf(i, root) for i in range(100)}
    cache.eviction_strategy.get_priority = lambda n: n.key[0]
    attempts = []
    ready = torch.zeros(2)
    def dissent(vote, op):
        if vote.numel() == 2 * 32 * retention._VICTIMS_PER_BATCH + 4:
            attempts.append(tick)
            vote[0] = vote[32 * retention._VICTIMS_PER_BATCH] = 0
        elif op == dist.ReduceOp.MIN:
            vote.zero_()  # No proposed victim exists on the peer.

    monkeypatch.setattr(retention, "_reduce_victims", dissent)
    cache._enforce_prefix_cache_retention = cache._all_reduce = lambda *_: None
    cache.check_hicache_events = lambda: cache._all_reduce(ready.clone(), dist.ReduceOp.MIN)
    scheduler = SimpleNamespace(tree_cache=cache, enable_hierarchical_cache=True, _engine_paused=False,
                                is_fully_idle=lambda: False, process_input_requests=lambda _: None)
    monkeypatch.setattr(retention, "get_prefix_cache_retention_limit", lambda: 1)
    retention.install_parallel_hicache_retention(scheduler)
    for tick in range(80):
        cache.check_hicache_events()
    gaps = [b - a for a, b in zip(attempts, attempts[1:])]
    assert len(gaps) >= 4 and gaps[:4] == sorted(set(gaps[:4]))
    assert gaps[3] - gaps[2] == 8
    tick += 1
    ready[0] = 1
    cache.check_hicache_events()
    assert attempts[-1] == tick


def test_unified_reclaim_digest_does_not_restart_an_exhausted_idle_search(monkeypatch):
    cache, _, _ = _priority_cache(False)
    cache.tp_world_size = 2
    cache.tree_core = SimpleNamespace(evictable_device_leaves=set(), write_back_duplicate_reclaim_digest=74)
    cache.full_evictable_size, cache.full_protected_size = lambda: 1, lambda: 0
    cache._enforce_prefix_cache_retention = cache._all_reduce = lambda *_: None
    counts = torch.tensor([0, 0, 74, -74])
    cache.check_hicache_events = lambda: cache._all_reduce(counts.clone(), dist.ReduceOp.MIN)
    scheduler = SimpleNamespace(tree_cache=cache, enable_hierarchical_cache=True, _engine_paused=False,
                                is_fully_idle=lambda: True, process_input_requests=lambda _: None)
    builds = []
    def empty_plan(*_):
        builds.append(True)
        return iter(())

    monkeypatch.setattr(retention, "_victim_batches", empty_plan)
    monkeypatch.setattr(retention, "get_prefix_cache_retention_limit", lambda: 0)
    retention.install_parallel_hicache_retention(scheduler)
    for _ in range(130):
        cache.check_hicache_events()
    assert len(builds) == 1


def test_zero_limit_matches_only_new_requests_cold(monkeypatch):
    @dataclass
    class Match:
        key: tuple
        req: object

    cache, _, _ = _priority_cache(False)
    cache.tp_world_size, cache.protected_size_ = 2, 0
    cache._all_reduce = cache._enforce_prefix_cache_retention = lambda *_: None
    cache.check_hicache_events = lambda: cache._all_reduce(torch.zeros(2), dist.ReduceOp.MIN)
    cache.match_prefix = lambda params: params.key
    cache.session = SimpleNamespace(slots={"held": SimpleNamespace(kv=object()),
                                           "released": SimpleNamespace(kv=SimpleNamespace(holds_kv=False))})
    scheduler = SimpleNamespace(tree_cache=cache, enable_hierarchical_cache=True, _engine_paused=False,
                                process_input_requests=lambda _: None)
    policy = retention.prefix_cache_retention
    monkeypatch.setattr(policy, "_policy", policy._PrefixCacheRetentionPolicy(0))
    retention.install_parallel_hicache_retention(scheduler)
    cache.check_hicache_events()
    key = (10, 11)
    fields: dict = dict(session=None, req_pool_idx=None, extend_batch_idx=0, retracted_stain=False, output_ids=[])
    fresh = Match(key, SimpleNamespace(**fields))
    assert cache.match_prefix(fresh) == () and fresh.key == key
    assert cache.match_prefix(Match(key, None)) == key
    for owned in (dict(req_pool_idx=0), dict(extend_batch_idx=1), dict(retracted_stain=True),
                  dict(output_ids=[4]), dict(session=SimpleNamespace(streaming=True, session_id="held"))):
        assert cache.match_prefix(Match(key, SimpleNamespace(**(fields | owned)))) == key
    for session_id in ("new", "released"):
        first_turn = SimpleNamespace(**(fields | dict(session=SimpleNamespace(streaming=True, session_id=session_id))))
        assert cache.match_prefix(Match(key, first_turn)) == ()
    newer = SimpleNamespace(**{k: v for k, v in fields.items() if k != "req_pool_idx"},
                            kv=SimpleNamespace(req_pool_idx=0))
    assert cache.match_prefix(Match(key, newer)) == key
    policy._apply_prefix_cache_retention_update({"revision": (1,), "limit_tokens": 10})
    scheduler.process_input_requests([])
    assert cache.match_prefix(fresh) == key


def test_storage_prefetch_observes_zero_at_dispatch_and_drops_a_previous_hold(monkeypatch):
    cache, _, _ = _priority_cache(False)
    cache.tp_world_size = 2
    cache.protected_size_ = 0
    cache._all_reduce = cache._enforce_prefix_cache_retention = lambda *_: None
    cache.check_hicache_events = lambda: cache._all_reduce(torch.zeros(2), dist.ReduceOp.MIN)
    issued: list[str] = []
    staged = {"repeat": (4, 5)}
    loaded = {"repeat": 2}
    cache.pop_prefetch_loaded_span = loaded.pop
    def consume(req):
        issued.append("consume")
        return True

    reasons = []
    def release(handle, *, reason):
        reasons.append(reason)
        staged.pop(handle)

    pipeline = SimpleNamespace(has_staged=staged.__contains__, release_staged_hold=release,
                               prepare_staged_prefetch=consume,
                               _clear_storage_hit=lambda req: setattr(req, "storage_hit_length", 0))
    cache.buffer_pipeline = pipeline
    req = SimpleNamespace(req_pool_idx=None, extend_batch_idx=0, retracted_stain=False, output_ids=[], session=None,
                          cache_request_handle="repeat", staged_prefetch_plan=object(), storage_hit_length=2)
    scheduler = SimpleNamespace(tree_cache=cache, _engine_paused=False, enable_hierarchical_cache=True,
                                enable_hicache_storage=True, _prefetch_kvcache=lambda r: issued.append("fetch"))
    scheduler.process_input_requests = lambda reqs: [scheduler._prefetch_kvcache(r) for r in reqs]
    policy = retention.prefix_cache_retention._policy
    monkeypatch.setattr(policy, "limit_tokens", 0)
    monkeypatch.setattr(policy, "applied_revision", None)
    retention.install_parallel_hicache_retention(scheduler)
    scheduler.process_input_requests([req])
    assert not issued
    monkeypatch.setattr(policy, "limit_tokens", 20)
    monkeypatch.setattr(policy, "applied_revision", (1,))
    scheduler.process_input_requests([req])
    assert issued == ["fetch"]
    monkeypatch.setattr(policy, "limit_tokens", 0)
    monkeypatch.setattr(policy, "applied_revision", (2,))
    scheduler.process_input_requests([req])
    assert pipeline.prepare_staged_prefetch(req) and issued == ["fetch"]
    assert not staged and not loaded and req.staged_prefetch_plan is None
    assert reasons == ["dropped"]
    assert req.host_hit_length == req.swa_host_hit_length == req.storage_hit_length == 0


def test_zero_limit_skips_decode_storage_reservation(monkeypatch):
    module = ModuleType("sglang.srt.disaggregation.decode")
    setattr(module, "DecodePrefixMatch", SimpleNamespace)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cache, _, _ = _priority_cache(False)
    cache.tp_world_size = 2
    cache._all_reduce = cache._enforce_prefix_cache_retention = cache.check_hicache_events = lambda *_: None
    native = SimpleNamespace(l3_storage_hit_length=32)
    queue = SimpleNamespace(_build_decode_prefix_match=lambda req, result: native)
    scheduler = SimpleNamespace(tree_cache=cache, enable_hierarchical_cache=True, enable_decode_hicache=True,
                                disagg_decode_prealloc_queue=queue, _engine_paused=False, process_input_requests=lambda _: None)
    req = SimpleNamespace(req_pool_idx=None, extend_batch_idx=0, retracted_stain=False, session=None, output_ids=[])
    root = SimpleNamespace(device_indices=(), last_device_node=0)
    monkeypatch.setattr(retention, "get_prefix_cache_retention_limit", lambda: 0)
    retention.install_parallel_hicache_retention(scheduler)
    match = queue._build_decode_prefix_match(req, root)
    assert match.prefix_indices == () and match.l2_host_hit_length == match.l3_storage_hit_length == 0
    assert match.last_device_node == 0
    req.req_pool_idx = 0
    assert queue._build_decode_prefix_match(req, root) is native


def test_rid_based_buffer_pipeline_releases_only_cold_request_before_splice(monkeypatch):
    @dataclass
    class Match:
        key: tuple
        req: SimpleNamespace

    cache, _, _ = _priority_cache(False)
    cache.tp_world_size = 2
    cache.match_prefix = lambda params: params.key
    cache._all_reduce = cache._enforce_prefix_cache_retention = cache.check_hicache_events = lambda *_: None
    held = {"fresh": 2, "owned": 3}
    pipeline = SimpleNamespace(staged_prefetches=held, release_staged_hold=held.pop,
                               plan_staged_splice=lambda rid, _: (held.get(rid, 0), 0))
    cache.buffer_pipeline = pipeline
    scheduler = SimpleNamespace(tree_cache=cache, enable_hierarchical_cache=True, enable_hicache_storage=True,
                                _engine_paused=False, process_input_requests=lambda _: None)
    monkeypatch.setattr(retention, "get_prefix_cache_retention_limit", lambda: 0)
    retention.install_parallel_hicache_retention(scheduler)
    for rid, slot, expected in (("fresh", None, 0), ("owned", 0, 3)):
        req = SimpleNamespace(rid=rid, req_pool_idx=slot, session=None, output_ids=[], extend_batch_idx=0,
                              retracted_stain=False, storage_hit_length=held[rid])
        hit = cache.match_prefix(Match((1, 2), req))
        assert hit == (() if slot is None else (1, 2))
        assert pipeline.plan_staged_splice(rid, len(hit)) == (expected, 0)
        assert req.storage_hit_length == expected


@pytest.mark.parametrize("unified", [False, True])
def test_lowering_limit_terminates_only_unowned_waiting_prefetches(monkeypatch, unified):
    cache, _, _ = _priority_cache(False)
    cache.tp_world_size = 2
    cache._all_reduce = cache._enforce_prefix_cache_retention = cache.check_hicache_events = lambda *_: None
    if unified:
        cache.tree_core = SimpleNamespace(evictable_device_leaves=set())
    waiting = [SimpleNamespace(rid=str(i), cache_request_handle=(i, 1), req_pool_idx=slot, extend_batch_idx=0,
                               retracted_stain=False, session=None, output_ids=[]) for i, slot in enumerate((None, 0, None))]
    cache.ongoing_prefetch = {
        req.cache_request_handle if unified else req.rid:
        (None, None, None, req.rid) if unified else (None, None, req.rid) for req in waiting}
    terminated: list[str] = []
    cache.cache_controller.terminate_prefetch = terminated.append
    scheduler = SimpleNamespace(tree_cache=cache, _engine_paused=False, enable_hierarchical_cache=True,
                                enable_hicache_storage=True, waiting_queue=waiting[:2],
                                disagg_prefill_bootstrap_queue=SimpleNamespace(queue=waiting[2:]),
                                process_input_requests=lambda _: None)
    policy = retention.prefix_cache_retention._policy
    monkeypatch.setattr(policy, "limit_tokens", 10)
    monkeypatch.setattr(policy, "applied_revision", None)
    retention.install_parallel_hicache_retention(scheduler)
    monkeypatch.setattr(policy, "limit_tokens", 0)
    monkeypatch.setattr(policy, "applied_revision", (3,))
    scheduler.process_input_requests([])
    scheduler.process_input_requests([])
    assert terminated == ["0", "2"]


@pytest.mark.parametrize("change", ["session", "flush", "request"])
def test_paused_continuation_refreshes_after_tree_changes(monkeypatch, change):
    root = Leaf(-1, None)
    children = []
    for token in range(0, 32, 2):
        parent = Leaf(token, root)
        child = Leaf(token + 1, parent)
        parent.children[token + 1] = child
        children.append(child)
    evicted = []
    target = children[-1]
    target_priority = target.key[0]
    cache = SimpleNamespace(
        root_node=root, tp_world_size=2, pp_size=1, evictable_leaves=set(children), evictable_size_=32, protected_size_=0,
        cache_controller=SimpleNamespace(write_policy="write_through"),
        eviction_strategy=SimpleNamespace(get_priority=lambda n: target_priority if n is target else n.key[0]),
        _enforce_prefix_cache_retention=lambda _: None, _all_reduce=lambda *_: None,
        check_hicache_events=lambda: None, update_eviction_metrics=lambda *_: None)

    def evict(node):
        evicted.append(node)
        cache.evictable_leaves.remove(node)
        node.evicted = True
        cache.evictable_size_ -= 1
        if node in children:
            cache.evictable_leaves.add(node.parent)
        return 1

    sessions = {"s": object()}

    def dispatch(_):
        nonlocal target_priority
        if evicted:
            target_priority = -1
            if change == "session":
                sessions.clear()
            elif change == "flush":
                cache.root_node = Leaf(-1, None)
                for child in children:
                    child.parent.parent = cache.root_node

    cache._evict_backuped = evict
    scheduler = SimpleNamespace(tree_cache=cache, enable_hierarchical_cache=True, _engine_paused=True,
                                session_controller=SimpleNamespace(sessions=sessions), process_input_requests=dispatch)
    monkeypatch.setattr(retention, "get_prefix_cache_retention_limit", lambda: 0)
    retention.install_parallel_hicache_retention(scheduler)
    scheduler.process_input_requests([])
    first_round = len(evicted)
    assert 0 < first_round < len(children) and target not in evicted
    scheduler.process_input_requests(["change"] if change == "request" else [])
    assert evicted[first_round] is target


def test_shared_prefix_hashed_once_per_plan(monkeypatch):
    root = SimpleNamespace(parent=None)
    parent = Leaf(1, root)
    parent.key = Key(range(32768), None)
    siblings = [Leaf(i, parent) for i in range(64)]
    parent.children = {leaf.key[0]: leaf for leaf in siblings}
    prefix_bytes = []
    read_tokens = parent.key.raw_token_ids

    def read_prefix():
        raw = read_tokens()
        prefix_bytes.append(len(raw) * raw.itemsize)
        return raw

    monkeypatch.setattr(parent.key, "raw_token_ids", read_prefix)
    cache = SimpleNamespace(
        evictable_leaves=set(siblings), evictable_size_=len(siblings), pp_size=1,
        eviction_strategy=SimpleNamespace(get_priority=lambda node: node.key[0]),
        cache_controller=SimpleNamespace(write_policy="write_through"),
        _evict_backuped=lambda node: len(node.key), update_eviction_metrics=lambda *_: None)
    votes = []
    monkeypatch.setattr(retention, "_reduce_victims", lambda vote, op: votes.append(vote[:8].tolist()))
    _poll_to_limit(cache, 0)
    parent.key.cache_salt = "new namespace"
    _poll_to_limit(cache, 0)
    assert prefix_bytes == [32768 * 8] * 2
    proofs = [words for words in votes if len(words) == 8]
    assert len(proofs) == 2 and proofs[0] != proofs[1]


@pytest.mark.parametrize("bigram", [False, True])
def test_victim_identity_survives_a_split_above_the_leaf(bigram):
    def victim(edges):
        parent = Leaf(0, None)
        for tokens in edges:
            leaf = Leaf(0, parent)
            leaf.key = Key(tokens, "tenant")
            leaf.key.is_bigram = bigram
            parent = leaf
        return retention._leaf_signature(parent, {})

    if bigram:
        assert victim([[1, 2, 3], [3, 4]]) == victim([[1, 2], [2, 3], [3, 4]])
    else:
        assert victim([[1, 2], [3]]) == victim([[1], [2], [3]])
        assert victim([[1, 2], [3]]) != victim([[1], [2, 3]])


def _use_groups(attention, pipeline):
    """Point kvcached's shared rank groups at these Gloo groups; None has no peer."""
    def coordinator(group):
        return SimpleNamespace(world_size=1 if group is None else dist.get_world_size(group), cpu_group=group)

    state = SimpleNamespace(get_attn_tp_group=lambda: coordinator(attention),
                            get_pp_group=lambda: coordinator(pipeline))
    rank_groups._parallel_state = lambda: state


def _exercise_mixed_limits(stage, lane, attention, pipeline):
    @dataclass
    class Match:
        key: tuple
        req: object

    request = SimpleNamespace(session=None, req_pool_idx=None, extend_batch_idx=0, retracted_stain=False, output_ids=[])
    for pp in (False, True):
        caps = ((-1, 2), (2, -1), (0, 2), (2, 0), (0, -1), (-1, 0), (0, 0), (-1, -1))
        cases = [(pair, None) for pair in caps] + [((2, 0), idx) for idx in (0, 1)]
        for limits, write_back_rank in cases:
            limit = limits[stage if pp else lane]
            cache, order, _ = _priority_cache(False)
            if write_back_rank == (stage if pp else lane):
                cache.cache_controller.write_policy = "write_back"
            cache.tp_world_size, cache.pp_size, cache.pp_rank = 2, 2 if pp else 1, stage if pp else 0
            cache.protected_size_ = 1
            _use_groups(attention, pipeline if pp else None)

            def native_ack(t, op):
                if not pp or stage == 0:
                    dist.all_reduce(t, op=op, group=attention)
                if pp:
                    dist.broadcast(t, src=lane, group=pipeline)

            cache._all_reduce = native_ack
            cache.check_hicache_events = lambda: cache._all_reduce(torch.zeros(2), dist.ReduceOp.MIN)
            local_evictions: list[int] = []
            cache._enforce_prefix_cache_retention = local_evictions.append
            cache.match_prefix = lambda params: params.key
            prefetched: list[SimpleNamespace] = []
            def route(item):
                if hasattr(item, "hicache_write_policy"):
                    if (stage if pp else lane) != item.reject:
                        cache.cache_controller.write_policy = item.hicache_write_policy
                else:
                    scheduler._prefetch_kvcache(item)

            def dispatch(items):
                for item in items:
                    scheduler._request_dispatcher(item)

            scheduler = SimpleNamespace(tree_cache=cache, enable_hierarchical_cache=True, enable_hicache_storage=True,
                                        _prefetch_kvcache=prefetched.append,
                                        waiting_queue=[request],
                                        process_input_requests=dispatch, _request_dispatcher=route, _engine_paused=False,
                                        ps=SimpleNamespace(attn_dp_size=1), pp_group=SimpleNamespace(is_last_rank=bool(stage)),
                                        pp_outputs=None, disaggregation_mode=SimpleNamespace(value="null"))
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(retention, "get_prefix_cache_retention_limit", lambda: limit)
                retention.install_parallel_hicache_retention(scheduler)
                cold = 0 in limits and write_back_rank is None
                expected = () if cold else (7,)
                resumed = SimpleNamespace(**(vars(request) | {"retracted_stain": True}))
                def check_admission():
                    assert cache.match_prefix(Match((7,), request)) == expected
                    assert cache.match_prefix(Match((7,), resumed)) == (7,)
                    prefetched.clear()
                    scheduler._prefetch_kvcache(request)
                    scheduler._prefetch_kvcache(resumed)
                    assert prefetched == ([resumed] if cold else [request, resumed])
                check_admission()
                for _ in range(retention._RECHECK_POLLS + 1):
                    cache.check_hicache_events()
                check_admission()
                if write_back_rank is not None:
                    cache._enforce_prefix_cache_retention(0)
                    assert not local_evictions
                    assert not order
                    continue
                finite = [v for v in limits if v >= 0]
                assert len(order) == (3 - min(finite) if finite else 0)
                if limits == (2, 0):
                    changes = (("write_back", 1, False, False), ("write_through", None, True, False),
                               ("write_back", 0, False, False), ("write_back", None, False, True),
                               ("write_through", 1, False, False), ("write_through", None, True, False))
                    for policy, reject, next_cold, all_back in changes:
                        prefetched.clear()
                        change = SimpleNamespace(hicache_write_policy=policy, reject=reject)
                        scheduler.process_input_requests([change, request, resumed])
                        same_batch_cold = cold if pp else next_cold
                        assert prefetched == ([resumed] if same_batch_cold else [request, resumed])
                        assert cache.match_prefix(Match((7,), request)) == (() if same_batch_cold else (7,))
                        local_evictions.clear()
                        cache._enforce_prefix_cache_retention(0)
                        assert bool(local_evictions) == (all_back and not pp)
                        cache.check_hicache_events()
                        if same_batch_cold:
                            assert prefetched == ([resumed] if next_cold else [resumed, request])
                        local_evictions.clear()
                        cache._enforce_prefix_cache_retention(0)
                        assert bool(local_evictions) == all_back
                        cold, expected = next_cold, () if next_cold else (7,)
                        check_admission()


def _exercise_unequal_batches(rank, attention, pipeline):
    cache, _, _ = _priority_cache(False)
    cache.pp_size = 2
    _use_groups(attention, pipeline)
    cache.evictable_leaves = set()
    count = 2 * retention._VICTIMS_PER_BATCH
    for token in range(count):
        parent = Leaf(token, cache.root_node)
        keys = (token, count + token) if rank == 3 else (token,)
        for key in keys:
            leaf = Leaf(key, parent)
            parent.children[key] = leaf
            cache.evictable_leaves.add(leaf)
    initial = cache.evictable_leaves.copy()
    cache.evictable_size_ = len(initial) + count
    evaluations = set()
    freed = []

    def priority(node):
        assert node not in evaluations
        evaluations.add(node)
        return (node.parent is cache.root_node, node.key[0])

    def evict(node):
        node.evicted = True
        cache.evictable_leaves.remove(node)
        cache.evictable_size_ -= 1
        freed.append(node.key[0])
        if all(child.evicted for child in node.parent.children.values()):
            cache.evictable_leaves.add(node.parent)
        return 1

    cache.eviction_strategy.get_priority = priority
    cache._evict_backuped = evict
    assert _poll_to_limit(cache, len(initial))
    assert freed == list(range(count))
    assert initial <= evaluations


def _exercise(rank, uri):
    dist.init_process_group("gloo", init_method=uri, rank=rank, world_size=4,
                            timeout=timedelta(seconds=30))
    try:
        groups = {members: dist.new_group(list(members))
                  for members in [(0, 1), (2, 3), (0, 2), (1, 3)]}
        stage, lane = divmod(rank, 2)
        attention = groups[(stage * 2, stage * 2 + 1)]
        pipeline = groups[(lane, lane + 2)]
        _exercise_mixed_limits(stage, lane, attention, pipeline)
        _exercise_unequal_batches(rank, attention, pipeline)
        # A peer with no candidate blocks the ordered vote immediately.
        # Once a shared leaf becomes available the next plan can commit it.
        cache, _, _ = _priority_cache(False)
        cache.pp_size = 2
        _use_groups(attention, pipeline)
        size = retention._VICTIMS_PER_BATCH ** 2
        cache.evictable_leaves = {Leaf(i, cache.root_node) for i in range(size) if rank != 3}
        cache.eviction_strategy.get_priority = lambda n: n.key[0]
        plan = retention._VictimPlan()
        assert plan.poll(cache, 1, 0, True) is False and plan.search is None
        if rank == 3:
            cache.evictable_leaves.add(Leaf(0, cache.root_node))
        def release(node):
            assert node.key[0] == 0
            cache.evictable_leaves.remove(node)
            return 1

        cache._evict_backuped = release
        plan = retention._VictimPlan()
        for _ in range(size // retention._INDEX_PER_POLL):
            verdict = plan.poll(cache, 1, 0, False)
            if verdict is not None:
                break
        assert verdict is True
        # A lock on one rank prevents promotion there. Every rank can demote
        # the cold child, then their proposed next victims disagree.
        cache, order, nodes = _priority_cache(rank == 3)
        cache.pp_size = 2
        _use_groups(attention, pipeline)
        _poll_to_limit(cache, 0)
        assert order == [11, 20]
        assert [node.key[0] for node in nodes if not node.evicted] == [10, 30]

        # Native first choices differ, but demoting their common child
        # promotes a shared parent that ranks ahead of the unrelated hot leaf.
        cache, order, nodes = _priority_cache(False)
        cache.pp_size = 2
        _use_groups(attention, pipeline)
        if rank == 3:
            priority = cache.eviction_strategy.get_priority
            hot_leaf = nodes[2]
            cache.eviction_strategy.get_priority = lambda n: -1 if n is hot_leaf else priority(n)
        assert _poll_to_limit(cache, 1)
        assert order == [11, 10]

        # An entire page of rank-local prefixes precedes the only common
        # child. Its newly eligible parent should be proposed by the next
        # poll, without rebuilding the index of already rejected prefixes.
        cache, order, (parent, child, hot, _) = _priority_cache(False)
        cache.pp_size = 2
        _use_groups(attention, pipeline)
        cache.evictable_leaves.remove(hot)
        page = retention._VICTIMS_PER_BATCH
        cache.evictable_leaves.update(Leaf(-i - 1, parent.parent, str(rank)) for i in range(page))
        cache.evictable_size_ = page + 2
        cache.eviction_strategy.get_priority = lambda n: n.key[0]
        plan = retention._VictimPlan()
        for _ in range(3):
            assert plan.poll(cache, 2, 0, True) is None
        assert order == [child.key[0]]
        assert plan.poll(cache, 1, 0, True) is True
        assert order == [child.key[0], parent.key[0]]

        # Same two candidates but opposite priorities must not commit a
        # different sequence on one rank.
        cache, order, nodes = _priority_cache(False)
        cold, hot = nodes[1:3]
        cache._evict_backuped(cold)
        cache.eviction_strategy.get_priority = lambda n: int((n is hot) != (rank == 3))
        cache.pp_size = 2
        _use_groups(attention, pipeline)
        _poll_to_limit(cache, 0)
        assert order == [11, 10, 20]
        assert [node.key[0] for node in nodes if not node.evicted] == [30]

        root = SimpleNamespace(parent=None)
        # All can release both shared leaves. The rank holding the extra
        # third leaf sets the group demand, even if a peer falls below its cap.
        leaves = [Leaf(i, root) for i in (10, 20, 30)]
        eligible = leaves[:2] if rank == 3 else leaves
        cache.evictable_size_ = len(eligible)
        cache.evictable_leaves = set(eligible)
        cache.eviction_strategy = SimpleNamespace(get_priority=lambda n: n.key[0])
        freed_tokens = []

        def evict_common(node):
            freed_tokens.append(node.key[0])
            cache.evictable_leaves.remove(node)
            cache.evictable_size_ -= 1
            return 1

        cache._evict_backuped = evict_common
        _poll_to_limit(cache, 1)
        assert freed_tokens == [10, 20]

        # Disagreeing native priorities choose the proposer's order, even
        # for a subgroup that does not contain global rank zero.
        for pp_size, local_group, first_rank in ((1, attention, stage * 2), (2, None, lane), (2, attention, 0)):
            cache.pp_size, cache.pp_rank = pp_size, stage if pp_size > 1 else 0
            _use_groups(local_group, pipeline if pp_size > 1 else None)
            cache.evictable_leaves = {Leaf(i, root) for i in (10, 20, 30)}
            cache.evictable_size_ = 3
            reverse = bool(first_rank % 2) if rank == first_rank else not bool(first_rank % 2)
            cache.eviction_strategy.get_priority = lambda n: -n.key[0] if reverse else n.key[0]
            freed_tokens.clear()
            # An explicitly unlimited PP follower still participates when
            # a finite stage asks it to release a common prefix.
            limit = -1 if pp_size > 1 and stage else 2
            assert _poll_to_limit(cache, limit)
            assert freed_tokens == [30 if first_rank % 2 else 10]
            if pp_size > 1:
                if stage:
                    cache.evictable_leaves.add(Leaf(40, root))
                    cache.evictable_size_ += 1
                # The finite stage is now at its cap. An unlimited follower
                # with an extra leaf has no further demand of its own.
                assert _poll_to_limit(cache, limit)
                assert len(freed_tokens) == 1
        cache.eviction_strategy.get_priority = lambda n: n.key[0]

        # The next proposal page was snapshotted with both leaves available.
        # Rank 3 then locks one and changes the namespace of the other.
        # Revalidate on every rank, including ranks with no local mutation.
        page = retention._VICTIMS_PER_BATCH
        all_leaves = [Leaf(i, root, str(rank) if i < page else None) for i in range(page + 3)]
        cache.evictable_size_ = len(all_leaves)
        cache.evictable_leaves = set(all_leaves)
        plan = retention._VictimPlan()
        freed_tokens.clear()
        # Indexing spans two polls, then the proposer sends two pages.
        assert plan.poll(cache, 1, 0, True) is None
        assert plan.poll(cache, 1, 0, True) is None
        if rank == 3:
            cache.evictable_leaves.remove(all_leaves[page])
            all_leaves[page + 1].key.cache_salt = "different"
        assert plan.poll(cache, 1, 0, True) is True
        assert freed_tokens == [page + 2]

        # An MRU hit moves a previously indexed leaf to the front of the
        # live order while the distributed sweep is between index pages.
        size = page + 2
        cache.evictable_leaves = {Leaf(i, root) for i in range(size)}
        cache.evictable_size_ = size
        # Choose a leaf from the proposer's first index page.
        touched = torch.tensor(next(iter(cache.evictable_leaves)).key[0])
        dist.broadcast(touched, src=0)
        hit = int(touched.item())
        newest = (hit + 1) % size
        access_times = list(range(size))
        access_times[hit] = -1
        access_times[newest] = size
        if rank == 3:
            access_times[(hit + 2) % size] = size + 1
        cache.eviction_strategy.get_priority = lambda node: -access_times[node.key[0]]
        plan = retention._VictimPlan()
        freed_tokens.clear()
        assert plan.poll(cache, 1, 0, True) is None
        assert not freed_tokens
        access_times[hit] = size + 2
        assert plan.poll(cache, 1, 0, True) is True
        assert freed_tokens == [newest] and cache.evictable_size_ == size - 1
        assert any(node.key[0] == hit for node in cache.evictable_leaves)
        cache.eviction_strategy.get_priority = lambda n: n.key[0]

        # Only the follower needs several index pages. The proposer must
        # keep its first page until that rank can confirm membership.
        common = Leaf(1000, root)
        cache.evictable_leaves = {common}
        if rank == 3:
            cache.evictable_leaves.update(Leaf(i, root) for i in range(2 * page))
        calls = 0
        signature = retention._leaf_signature
        def count_hash(*args):
            nonlocal calls
            calls += 1
            return signature(*args)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(retention, "_leaf_signature", count_hash)
            plan = retention._VictimPlan()
            freed_tokens.clear()
            for verdict in (None, None, True):
                calls = 0
                assert plan.poll(cache, 1, 0, True) is verdict
                # At most one index page plus an ordered or membership check.
                assert calls <= page + 1
        assert freed_tokens == [1000]

        # A disagreement in the last slot of a full batch needs one victim
        # ordered vote. A proposal confirms there is no shared tail.
        limit = retention._VICTIMS_PER_BATCH
        cache.evictable_leaves = {Leaf(i, root) for i in range(limit - 1)}
        cache.evictable_leaves.add(Leaf(limit + rank, root))
        cache.evictable_size_ = limit
        rounds = []
        reduce_victims = retention._reduce_victims
        def count_round(tensor, op):
            rounds.append(tensor.numel())
            reduce_victims(tensor, op)
        freed_tokens.clear()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(retention, "_reduce_victims", count_round)
            assert _poll_to_limit(cache, 0) is False
        assert len(rounds) == 4 and freed_tokens == list(range(limit - 1))

        # PP0 has exactly the cap. Only the far stage needs to reclaim. Its
        # excess must eventually reach PP0 despite the downstream ACK relay.
        common, extra = Leaf(1, root), Leaf(2, root)
        cache.evictable_leaves = {common, extra} if stage else {common}
        cache.evictable_size_, cache.protected_size_ = len(cache.evictable_leaves), 0
        cache.check_hicache_events = lambda: cache._all_reduce(torch.zeros(2), dist.ReduceOp.MIN)
        cache._enforce_prefix_cache_retention = lambda _: None

        def downstream(counts, op):
            if not stage:
                dist.all_reduce(counts, op=op, group=attention)
            dist.broadcast(counts, src=lane, group=pipeline)

        cache._all_reduce = downstream
        scheduler = SimpleNamespace(tree_cache=cache, enable_hierarchical_cache=True, ps=SimpleNamespace(attn_dp_size=1),
                                    pp_group=SimpleNamespace(is_last_rank=bool(stage)), pp_outputs=None,
                                    disaggregation_mode=SimpleNamespace(value="null"))
        freed_tokens.clear()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(retention, "get_prefix_cache_retention_limit", lambda: 1)
            retention.install_parallel_hicache_retention(scheduler)
            for _ in range(retention._RECHECK_POLLS):
                cache.check_hicache_events()
        assert freed_tokens == [1] and cache.evictable_size_ == stage

        for topology in ("TP", "PP", "TP+PP"):
            for difference in ("capacity", "namespace", "ancestor", "split", "shared-parent", None):
                # Each participating group has one disagreeing member.
                odd = lane if topology == "TP" else stage
                if topology == "TP+PP":
                    odd = rank == 3
                parent = Leaf(11 if difference == "ancestor" and odd else 10, root)
                leaf = Leaf(12, parent, "other" if difference == "namespace" and odd else None)
                if difference == "split":
                    parent.key = Key([10] if odd else [10, 11], None)
                    leaf.key = Key([11, 12] if odd else [12], None)
                parent.children[leaf.key[0]] = leaf
                freed, vote_sizes = [], []
                delegated: list[int] = []

                def evict(node):
                    freed.append(node)
                    cache.evictable_size_ -= len(node.key)
                    cache.evictable_leaves.remove(node)
                    if difference == "shared-parent" and node is leaf:
                        cache.evictable_leaves.add(parent)
                    return len(node.key)

                size = 1 + (difference == "shared-parent")
                cache = SimpleNamespace(
                    cache_controller=SimpleNamespace(write_policy="write_through"),
                    protected_size_=0, evictable_size_=0 if difference == "capacity" and odd else size,
                    evictable_leaves=set() if difference == "capacity" and odd else {leaf},
                    eviction_strategy=SimpleNamespace(get_priority=lambda _: 0),
                    _evict_backuped=evict,
                    update_eviction_metrics=lambda *args: None,
                    tp_world_size=1 if topology == "PP" else 2,
                    pp_size=1 if topology == "TP" else 2,
                    _enforce_prefix_cache_retention=delegated.append)

                def reduce_attention(vote, op):
                    vote_sizes.append(vote.numel())
                    if topology != "PP":
                        dist.all_reduce(vote, op=op, group=attention)

                ack_sizes = []

                def ready_counts(vote, op):
                    ack_sizes.append(vote.numel())
                    if topology == "TP" or stage == 0:
                        reduce_attention(vote, op)
                    if topology != "TP":
                        dist.broadcast(vote, src=lane, group=pipeline)

                cache.match_prefix = lambda p: p.key
                cache._all_reduce = ready_counts
                cache.check_hicache_events = lambda: cache._all_reduce(torch.zeros(2, dtype=torch.int), dist.ReduceOp.MIN)
                def dispatch_policy(request):
                    cache.cache_controller.write_policy = request.hicache_write_policy

                scheduler = SimpleNamespace(tree_cache=cache, enable_hierarchical_cache=True,
                                            _request_dispatcher=dispatch_policy, _engine_paused=False,
                                            process_input_requests=lambda items: [scheduler._request_dispatcher(r) for r in items],
                                            is_fully_idle=lambda: True,
                                            ps=SimpleNamespace(attn_dp_size=1),
                                            disaggregation_mode=SimpleNamespace(value="null"),
                                            pp_group=SimpleNamespace(is_last_rank=bool(stage)), pp_outputs=None)
                _use_groups(None if topology == "PP" else attention, None if topology == "TP" else pipeline)
                reduce_victims = retention._reduce_victims

                def record_vote(vote, op):
                    vote_sizes.append(vote.numel())
                    reduce_victims(vote, op)

                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(retention, "_reduce_victims", record_vote)
                    policy = retention.prefix_cache_retention._policy
                    patch.setattr(policy, "limit_tokens", -1 if topology == "TP" else 0)
                    patch.setattr(policy, "applied_revision", None)
                    cache.cache_controller.write_policy = "write_back"
                    retention.install_parallel_hicache_retention(scheduler)
                    assert vote_sizes == [4]  # One startup vote before the first match.
                    vote_sizes.clear()
                    cache.check_hicache_events()
                    assert not freed and all(n == 3 for n in vote_sizes)
                    if topology == "TP":
                        # The shared startup poll determined that the limit is off.
                        cache.check_hicache_events()
                        assert ack_sizes == [3, 2]
                        retention.prefix_cache_retention._apply_prefix_cache_retention_update(
                            {"revision": (1,), "limit_tokens": 0})
                        scheduler.process_input_requests([])
                    cache._enforce_prefix_cache_retention(0)
                    cache.check_hicache_events()
                    assert delegated == [0] and not freed
                    if topology == "PP" and difference is None:
                        ack_sizes.clear()
                        vote_sizes.clear()
                        for _ in range(retention._RECHECK_POLLS):
                            cache.check_hicache_events()
                        assert len(ack_sizes) == retention._RECHECK_POLLS
                        assert vote_sizes == ([] if stage else ack_sizes)
                    scheduler.process_input_requests([SimpleNamespace(hicache_write_policy="write_through")])
                    if rank == 0:
                        cache._enforce_prefix_cache_retention(0)
                    cache.check_hicache_events()
                    expected = {"shared-parent": [leaf, parent], None: [leaf]}.get(difference, [])
                    assert freed == expected
                    if difference == "namespace":
                        attempts = 0
                        original_batches = retention._victim_batches

                        def count_attempts(*args):
                            nonlocal attempts
                            attempts += 1
                            return original_batches(*args)

                        patch.setattr(retention, "_victim_batches", count_attempts)
                        for _ in range(16):
                            cache.check_hicache_events()
                        assert attempts == 0 and not freed
                        if odd:
                            leaf.key.cache_salt = None
                            cache.match_prefix(SimpleNamespace(req=None, key=leaf.key))
                        for _ in range(33):
                            cache.check_hicache_events()
                        assert freed == [leaf]
                    if topology == "TP" and difference is None:
                        scheduler._engine_paused = True
                        for revision, limit in [(2, -1), (3, 0)]:
                            if limit < 0:
                                cache.evictable_leaves.add(leaf)
                                cache.evictable_size_ = 1
                            retention.prefix_cache_retention._apply_prefix_cache_retention_update(
                                {"revision": (revision,), "limit_tokens": limit})
                            scheduler.process_input_requests([])
                            if limit < 0:
                                cache.check_hicache_events()
                                assert ack_sizes[-1] == 2 and freed == [leaf]
                        assert freed == [leaf, leaf]
                        scheduler._engine_paused = False
                        scheduler.process_input_requests([])
                        cache.evictable_leaves.add(leaf)
                        cache.evictable_size_ = 1
                        scheduler._engine_paused = True
                        scheduler.process_input_requests([])
                        assert freed == [leaf, leaf, leaf]
                        # An idle dispatch can expire sessions at different
                        # times on each rank. Only shared retries can evict.
                        for i in range(1, 2 * retention._RECHECK_POLLS + 1):
                            if i == lane * retention._RECHECK_POLLS + 1:
                                cache.evictable_size_ = 1
                                cache.evictable_leaves.add(leaf)
                            scheduler.process_input_requests([])
                            if i < 2 * retention._RECHECK_POLLS:
                                assert freed == [leaf, leaf, leaf]
                        assert freed == [leaf, leaf, leaf, leaf]
    finally:
        dist.destroy_process_group()


def test_retention_agrees_across_attention_and_pipeline_groups(tmp_path):
    torch.multiprocessing.spawn(_exercise, args=((tmp_path / "store").as_uri(),), nprocs=4)


def test_unlimited_pipeline_keeps_native_scheduler_and_leaf_set(monkeypatch):
    cache = SimpleNamespace(pp_size=4, pp_rank=0, cache_controller=SimpleNamespace(write_policy="write_through"),
                            evictable_leaves=set(), check_hicache_events=lambda: None, _all_reduce=lambda *_: None)
    scheduler = SimpleNamespace(tree_cache=cache, enable_hierarchical_cache=True, ps=SimpleNamespace(attn_dp_size=1),
                                disaggregation_mode=SimpleNamespace(value="null"))
    votes = []
    monkeypatch.setattr(retention, "_reduce_victims", lambda data, op: votes.append(data.tolist()))
    monkeypatch.setattr(retention, "get_prefix_cache_retention_limit", lambda: -1)
    native_cache, native_scheduler = vars(cache).copy(), vars(scheduler).copy()
    retention.install_parallel_hicache_retention(scheduler)
    assert votes == [[0, 0, 1, 0]]
    assert vars(scheduler) == native_scheduler and vars(cache) == native_cache
    assert cache.evictable_leaves is native_cache["evictable_leaves"]


@pytest.mark.parametrize("mode", [dict(enable_pdmux=True), dict(dllm_config=object())])
def test_schedules_without_a_cache_poll_preserve_request_finish_retention(mode):
    calls: list[int] = []
    cache = SimpleNamespace(cache_controller=SimpleNamespace(write_policy="write_through"),
                            tp_world_size=2, pp_size=1, _enforce_prefix_cache_retention=calls.append)
    scheduler = SimpleNamespace(enable_hierarchical_cache=True, tree_cache=cache, **mode)
    retention.install_parallel_hicache_retention(scheduler)
    cache._enforce_prefix_cache_retention(3)
    assert calls == [3]


@pytest.mark.parametrize("enabled", [False, True])
def test_scheduler_installs_hook_after_cache_creation(monkeypatch, enabled):
    class Scheduler:
        def __init__(self):
            self.tree_cache = object()

    module = ModuleType("scheduler")
    setattr(module, "Scheduler", Scheduler)
    patch = retention.DistributedHiCacheRetentionPatch()
    monkeypatch.setattr(patch.version_manager, "detect_version", lambda library: "0.5.20")
    monkeypatch.setattr(retention, "enable_kvcached", lambda: enabled)
    caches = []
    monkeypatch.setattr(retention, "install_parallel_hicache_retention",
                        lambda scheduler: caches.append(scheduler.tree_cache))
    assert patch.apply(module)
    assert patch.apply(module)
    scheduler = Scheduler()
    assert caches == ([scheduler.tree_cache] if enabled else [])
