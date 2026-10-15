# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Native hash semantics in vLLM 0.26 through 0.30, and 0.30 event support.

Run this file separately in that image: older tests install process-wide vLLM
stubs, which cannot validate inherited metadata methods.
"""

from __future__ import annotations

import importlib
import sys
import types
from importlib.metadata import PackageNotFoundError, version as distribution_version

import pytest


class PhysicalPool:
    def __init__(self, size):
        self.free_ids = list(range(1, size))
        self.allocated = set()
        self.reported_free = None
        self.block_mem_size = 16 * 32
        self.page_size = 4 * self.block_mem_size
        self.page_allocator = object()

    def get_page_occupancy(self, pages):
        return {page: sum(bid // 4 == page for bid in self.allocated | {0})
                for page in pages}

    def alloc(self, count):
        if count > len(self.free_ids):
            return None
        ids, self.free_ids = self.free_ids[:count], self.free_ids[count:]
        self.allocated.update(ids)
        return ids

    def free(self, ids):
        for block_id in ids:
            assert block_id in self.allocated
            self.allocated.remove(block_id)
            self.free_ids.append(block_id)

    def available_size(self):
        return len(self.free_ids) if self.reported_free is None else self.reported_free


@pytest.fixture
def native_pool_factory(monkeypatch):
    try:
        installed_version = distribution_version("vllm")
    except PackageNotFoundError:
        pytest.skip("Native metadata checks require vLLM 0.26 through 0.30")
    supported_versions = ("0.26.", "0.27.", "0.28.", "0.29.", "0.30.")
    if not installed_version.startswith(supported_versions):
        pytest.skip("Native metadata checks target vLLM 0.26 through 0.30")
    vllm = importlib.import_module("vllm")
    version = getattr(vllm, "__version__", "")
    if not isinstance(version, str) or not version.startswith(supported_versions):
        pytest.skip("Native metadata checks require real vLLM, not test stubs")
    native = importlib.import_module("vllm.v1.core.block_pool")
    if not isinstance(getattr(native, "__file__", None), str):
        pytest.skip("Run this file separately: other tests installed a vLLM stub")

    from kvcached.integration.vllm.native_block_pool import NativeBlockPoolMixin
    from kvcached.integration.vllm.patches import ElasticBlockPoolPatch

    interface = types.ModuleType("kvcached.integration.vllm.interfaces")
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    package = importlib.import_module("kvcached.integration.vllm")
    monkeypatch.setattr(package, "interfaces", interface, raising=False)

    def make(size=8, cache_cap=-1, enable_caching=True, events=False):
        if events and not installed_version.startswith("0.30."):
            pytest.skip("Elastic KV event checks target vLLM 0.30")
        physical = PhysicalPool(size)
        setattr(interface, "get_kv_cache_manager", lambda *args, **kwargs: physical)
        target = types.ModuleType("_native_metadata_elastic_pool")
        setattr(target, "BlockPool", native.BlockPool)
        setattr(target, "KVCacheBlock", native.KVCacheBlock)
        assert ElasticBlockPoolPatch().apply(target)
        cls = target.ElasticBlockPool
        assert issubclass(cls, NativeBlockPoolMixin)
        pool = cls(
            num_gpu_blocks=size,
            block_size=16,
            cell_size=32,
            num_layers=1,
            enable_caching=enable_caching,
            enable_kv_cache_events=events,
            max_cached_blocks=cache_cap,
            hash_block_size=4,
        )
        assert pool.cache_partial_block.__func__ is native.BlockPool.cache_partial_block
        assert pool.move_block_hashes.__func__ is native.BlockPool.move_block_hashes
        return pool, physical

    return make


def request():
    return types.SimpleNamespace(block_hashes=[b"a", b"b", b"c", b"d"])


def cache_full(pool, block):
    pool.cache_full_blocks(request(), [block], 0, 1, 16, 0)


def cache_partial(pool, block, num_tokens):
    return pool.cache_partial_block(request(), block, num_tokens, 0, 16)


def test_native_alias_eviction_preserves_duplicate_and_reuses_clean_slot(native_pool_factory):
    pool, physical = native_pool_factory(size=3)
    first, second = pool.get_new_blocks(2)
    cache_full(pool, first)
    cache_full(pool, second)
    cache_partial(pool, first, 8)
    assert first.block_hash_num_tokens == 16
    assert pool.get_cached_block(b"b", [0]) == [first]

    pool.free_blocks([first, second])
    pool.evict_blocks({first.block_id})
    assert pool.get_cached_block(b"b", [0]) is None
    assert pool.get_cached_block(b"d", [0]) == [second]
    assert first.block_id not in pool.cached_block_hashes_by_block
    assert physical.allocated == {second.block_id}

    reused, = pool.get_new_blocks(1)
    assert reused is first and reused.block_hash is None
    assert pool.get_cached_block(b"d", [0]) == [second]


def test_native_partial_promotion_drops_old_aliases_and_sets_boundary(native_pool_factory):
    pool, _ = native_pool_factory()
    block, = pool.get_new_blocks(1)
    cache_partial(pool, block, 8)
    cache_partial(pool, block, 4)
    assert pool.get_cached_block(b"a", [0]) == [block]
    assert pool.get_cached_block(b"b", [0]) == [block]

    cache_full(pool, block)
    assert block.block_hash_num_tokens == 16
    assert pool.get_cached_block(b"a", [0]) is None
    assert pool.get_cached_block(b"b", [0]) is None
    assert pool.get_cached_block(b"d", [0]) == [block]
    assert pool._block_id_to_key[block.block_id] == block.block_hash


def test_native_cow_move_keeps_copy_endpoints_alive_then_reset_frees_cache(native_pool_factory):
    pool, physical = native_pool_factory()
    source, destination = pool.get_new_blocks(2)
    cache_full(pool, source)
    cache_partial(pool, source, 8)
    # Match the producer COW path: source keeps its request and worker-copy
    # references, destination's allocation reference belongs to the pending copy.
    source.ref_cnt += 1
    pool.move_block_hashes(source, destination)
    assert source.block_hash is None
    assert destination.block_hash_num_tokens == 16
    assert pool.get_cached_block(b"b", [0]) == [destination]
    assert pool.get_cached_block(b"d", [0]) == [destination]

    pool.free_blocks([source])  # request cancellation before worker copy
    assert physical.allocated == {source.block_id, destination.block_id}
    assert pool.reset_prefix_cache() is False
    pool.free_blocks([source, destination])  # copy finished
    assert physical.allocated == {destination.block_id}

    # A colocated peer can exhaust physical availability even when all local
    # requests are finished; that must not make an idle cache impossible to reset.
    physical.reported_free = 0
    assert pool.reset_prefix_cache() is True
    assert not physical.allocated
    assert pool.get_cached_block(b"b", [0]) is None
    assert pool.get_cached_block(b"d", [0]) is None
    assert not pool.cached_block_hashes_by_block
    assert not pool._block_id_to_key


def test_native_cache_cap_evicts_all_aliases_through_existing_lru(native_pool_factory):
    pool, physical = native_pool_factory(cache_cap=0)
    block, = pool.get_new_blocks(1)
    cache_full(pool, block)
    cache_partial(pool, block, 8)
    pool.free_blocks([block])
    assert not physical.allocated
    assert pool.get_cached_block(b"b", [0]) is None
    assert pool.get_cached_block(b"d", [0]) is None
    assert block.block_hash is None


@pytest.fixture
def event_pool_factory(native_pool_factory):
    def make(**kwargs):
        pool, _ = native_pool_factory(events=True, **kwargs)
        return pool
    return make


def event_request(session="session-a"):
    return types.SimpleNamespace(
        block_hashes=[b"a", b"b", b"c", b"d"], all_token_ids=list(range(16)),
        session_id=session, lora_request=None, mm_features=[], cache_salt=None,
        prompt_embeds=None,
    )


def test_native_event_store_reuse_and_wire_format(event_pool_factory):
    import msgspec
    from vllm.distributed.kv_events import BlockStored, KVEventBatch
    from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

    pool = event_pool_factory()
    block, = pool.get_new_blocks(1)
    pool.cache_full_blocks(event_request(), [block], 0, 1, 16, 0)
    event, = pool.take_events()
    assert isinstance(event, BlockStored)
    assert event.block_hashes == [maybe_convert_block_hash(b"d")]
    assert event.parent_block_hash is None and event.token_ids == list(range(16))
    assert event.group_idx == 0 and event.session_id == "session-a"
    batch = KVEventBatch(ts=1.0, events=[event])
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(batch), type=KVEventBatch) == batch
    assert pool.take_events() == []
    pool.free_blocks([block])
    pool.touch([block])
    pool.emit_cached_block_events(event_request("session-b"), 1, 16, 0)
    reused, = pool.take_events()
    assert reused.session_id == "session-b" and reused.block_hashes == event.block_hashes
    assert block.ref_cnt == 1
    pool.free_blocks([block])


def test_native_event_cap_eviction_removes_primary_and_alias(event_pool_factory):
    from vllm.distributed.kv_events import BlockRemoved
    from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

    pool = event_pool_factory(cache_cap=0)
    block, = pool.get_new_blocks(1)
    pool.cache_full_blocks(event_request(), [block], 0, 1, 16, 0)
    pool.cache_partial_block(event_request(), block, 8, 0, 16)
    pool.take_events()
    pool.free_blocks([block])
    events = pool.take_events()
    assert len(events) == 2 and all(isinstance(e, BlockRemoved) for e in events)
    assert {h for e in events for h in e.block_hashes} == {
        maybe_convert_block_hash(b"b"), maybe_convert_block_hash(b"d")}
    assert all(e.medium == "GPU" and e.group_idx == 0 for e in events)
    assert pool.get_cached_block(b"b", [0]) is None
    assert pool.get_cached_block(b"d", [0]) is None
    assert block.block_hash is None


def test_native_event_reset_waits_for_shared_references(event_pool_factory):
    from vllm.distributed.kv_events import AllBlocksCleared

    pool = event_pool_factory()
    block, = pool.get_new_blocks(1)
    pool.cache_full_blocks(event_request(), [block], 0, 1, 16, 0)
    pool.take_events()
    pool.touch([block])
    pool.free_blocks([block])
    assert block.ref_cnt == 1 and pool.reset_prefix_cache() is False
    assert pool.take_events() == []
    pool.free_blocks([block])
    assert pool.reset_prefix_cache() is True
    assert isinstance(pool.take_events()[-1], AllBlocksCleared)
    assert pool.get_cached_block(b"d", [0]) is None


def test_native_event_reset_without_prefix_caching(event_pool_factory):
    from vllm.distributed.kv_events import AllBlocksCleared

    pool = event_pool_factory(enable_caching=False)
    blocks = pool.get_new_blocks(2)
    pool.free_blocks(blocks)
    assert pool.take_events() == []
    assert pool.reset_prefix_cache() is True
    assert pool.take_events() == [AllBlocksCleared()]
    assert pool.take_events() == []


@pytest.mark.parametrize("failures", [1, 20])
def test_native_event_capacity_recovery_preserves_shared_prefix(
    native_pool_factory, monkeypatch, failures,
):
    from vllm.distributed.kv_events import BlockStored

    from kvcached.utils import KVCachePoolExhausted

    pool, physical = native_pool_factory(events=True)
    source, = pool.get_new_blocks(1)
    pool.cache_full_blocks(event_request(), [source], 0, 1, 16, 0)
    pool.touch([source])
    pool.take_events()
    allocate = physical.alloc
    calls = []

    def capacity_miss(count):
        calls.append(count)
        return None if len(calls) <= failures else allocate(count)

    monkeypatch.setattr(physical, "alloc", capacity_miss)
    if failures == 1:
        destination, = pool.get_new_blocks(1)
        assert calls == [1, 1]
    else:
        for _ in range(10):
            with pytest.raises(KVCachePoolExhausted):
                pool.get_new_blocks(1)
            assert physical.allocated == {source.block_id}
            assert source.ref_cnt == 2
            assert pool.get_cached_block(b"d", [0]) == [source]
            assert pool.take_events() == []
        assert calls == [1] * 20
        monkeypatch.setattr(physical, "alloc", allocate)
        destination, = pool.get_new_blocks(1)

    assert pool.take_events() == []
    pool.cache_full_blocks(event_request(), [destination], 0, 1, 16, 1)
    event, = pool.take_events()
    assert isinstance(event, BlockStored) and event.group_idx == 1
    assert source.ref_cnt == 2 and pool.get_cached_block(b"d", [0]) == [source]
    pool.free_blocks([source, source, destination])
    assert pool.reset_prefix_cache() is True
    assert not physical.allocated


def test_native_event_allocation_errors_propagate(native_pool_factory, monkeypatch):
    pool, physical = native_pool_factory(events=True)

    def invariant_failure(count):
        raise ValueError("physical allocator invariant failure")

    monkeypatch.setattr(physical, "alloc", invariant_failure)
    with pytest.raises(ValueError, match="invariant failure"):
        pool.get_new_blocks(1)
    assert not physical.allocated and pool.take_events() == []


def test_native_event_allocation_failure_keeps_real_evictions(native_pool_factory, monkeypatch):
    from vllm.distributed.kv_events import BlockRemoved

    from kvcached.utils import KVCachePoolExhausted

    pool, physical = native_pool_factory(size=5, events=True)
    cached = pool.get_new_blocks(3)
    live, = pool.get_new_blocks(1)
    pool.cache_full_blocks(event_request(), cached[:1], 0, 1, 16, 0)
    for group, block in enumerate(cached[1:], start=1):
        pool.cache_full_blocks(event_request(), [block], 0, 1, 16, group)
    pool.free_blocks(cached)
    pool.take_events()
    physical.reported_free = 0
    monkeypatch.setattr(physical, "alloc", lambda count: None)

    with pytest.raises(KVCachePoolExhausted):
        pool.get_new_blocks(3)
    events = pool.take_events()
    assert len(events) == 3 and all(isinstance(event, BlockRemoved) for event in events)
    assert {event.group_idx for event in events} == {0, 1, 2}
    assert physical.allocated == {live.block_id} and live.ref_cnt == 1
    assert all(pool.get_cached_block(b"d", [group]) is None for group in range(3))
    pool.free_blocks([live])
