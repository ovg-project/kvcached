# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Native hash semantics in the vLLM 0.29 image.

Run this file separately in that image: older tests install process-wide vLLM
stubs, which cannot validate inherited 0.29 metadata methods.
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
        pytest.skip("Native metadata checks require the vLLM 0.29 image")
    if not installed_version.startswith("0.29."):
        pytest.skip("Native metadata checks target vLLM 0.29")
    vllm = importlib.import_module("vllm")
    version = getattr(vllm, "__version__", "")
    if not isinstance(version, str) or not version.startswith("0.29."):
        pytest.skip("Native metadata checks require real vLLM 0.29, not test stubs")
    native = importlib.import_module("vllm.v1.core.block_pool")
    if not isinstance(getattr(native, "__file__", None), str):
        pytest.skip("Run this file separately: other tests installed a vLLM stub")

    from kvcached.integration.vllm.native_block_pool import NativeBlockPoolMixin
    from kvcached.integration.vllm.patches import ElasticBlockPoolPatch

    interface = types.ModuleType("kvcached.integration.vllm.interfaces")
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    package = importlib.import_module("kvcached.integration.vllm")
    monkeypatch.setattr(package, "interfaces", interface, raising=False)

    def make(size=8, cache_cap=-1, enable_caching=True):
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
