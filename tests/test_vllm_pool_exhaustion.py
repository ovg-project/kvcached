# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""An exhausted shared KV pool must reach vLLM as a scheduling miss.

vLLM's scheduler has one channel for "this request cannot be served right
now": ``allocate_slots()`` returns None, and it preempts a running request and
retries on the next step. It has no channel for an exception -- ``schedule()``
contains no exception handler, and EngineCore's own handler wraps only
``execute_model`` -- so an exception from the block pool terminates the engine
along with every in-flight request.

Under kvcached the pool can legitimately fail to back an allocation: colocated
engines share one physical pool, and a peer can take the last pages between the
moment availability is observed and the moment they are claimed. These tests
pin the translation, and pin that it stays narrow.
"""
from __future__ import annotations

import importlib
import sys
import types
from typing import Any
from unittest import mock

import pytest


@pytest.fixture
def vllm_patches(monkeypatch):
    torch = mock.MagicMock()
    torch.__version__ = "2.6.0"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", torch.cuda)
    monkeypatch.setitem(sys.modules, "torch.utils", torch.utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension",
                        torch.utils.cpp_extension)
    monkeypatch.setitem(sys.modules, "posix_ipc", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())
    monkeypatch.delitem(sys.modules, "kvcached.integration.vllm.patches",
                        raising=False)
    return importlib.import_module("kvcached.integration.vllm.patches")


def _module_with_manager(raises: BaseException | None):
    """Build a stand-in ``vllm.v1.core.kv_cache_manager`` module.

    ``allocate_slots`` either raises what the block pool would raise, or
    returns a sentinel so the success path stays observable.
    """

    class KVCacheManager:
        def allocate_slots(self, *args: Any, **kwargs: Any) -> Any:
            if raises is not None:
                raise raises
            return ("blocks", args, kwargs)

    module = types.ModuleType("vllm.v1.core.kv_cache_manager")
    module.KVCacheManager = KVCacheManager  # type: ignore[attr-defined]
    return module


def _apply(vllm_patches, monkeypatch, module, *, kvcached_enabled=True):
    patch = vllm_patches.KVCacheManagerAllocateSlotsPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    monkeypatch.setattr(vllm_patches, "enable_kvcached",
                        lambda: kvcached_enabled)
    assert patch.patch_allocate_slots(module) is True
    return module.KVCacheManager()


def test_pool_exhaustion_becomes_a_scheduling_miss(vllm_patches, monkeypatch):
    from kvcached.utils import KVCachePoolExhausted

    manager = _apply(
        vllm_patches, monkeypatch,
        _module_with_manager(KVCachePoolExhausted("physical pool empty")))

    assert manager.allocate_slots("request", 8) is None


def test_contract_violations_still_terminate(vllm_patches, monkeypatch):
    """Unrelated ValueError exceptions must not become scheduling misses."""
    manager = _apply(
        vllm_patches, monkeypatch,
        _module_with_manager(ValueError("invalid block metadata")))

    with pytest.raises(ValueError, match="invalid block metadata"):
        manager.allocate_slots("request", 8)


def test_unknown_map_outcome_is_not_downgraded(vllm_patches, monkeypatch):
    from kvcached.tp_ipc_util import MapTransactionOutcomeUnknownError

    manager = _apply(vllm_patches, monkeypatch, _module_with_manager(
        MapTransactionOutcomeUnknownError("restart after unresolved map")))
    with pytest.raises(MapTransactionOutcomeUnknownError, match="restart"):
        manager.allocate_slots("request", 8)


@pytest.fixture
def pool_factory(vllm_patches, monkeypatch):
    """Exercise the injected pool without importing vLLM or a CUDA allocator."""
    interface = types.ModuleType("kvcached.integration.vllm.interfaces")
    monkeypatch.setitem(sys.modules, interface.__name__, interface)

    def make(enable_caching):
        free_ids = list(range(1, 8))  # Block zero is reserved for null.

        def alloc(count):
            if count > len(free_ids):
                return None
            ids = free_ids[:count]
            del free_ids[:count]
            return ids

        physical = types.SimpleNamespace(
            available_size=mock.Mock(side_effect=lambda: len(free_ids)),
            alloc=mock.Mock(side_effect=alloc),
            free=mock.Mock(side_effect=free_ids.extend),
        )
        setattr(interface, "get_kv_cache_manager", lambda *a, **kw: physical)
        module = types.ModuleType("test_block_pool")
        setattr(module, "BlockPool", type("BlockPool", (), {}))
        setattr(module, "KVCacheBlock", lambda bid: types.SimpleNamespace(
            block_id=bid, ref_cnt=0, is_null=False, block_hash=None))
        assert vllm_patches.ElasticBlockPoolPatch().inject_elastic_block_pool(module)
        pool = module.ElasticBlockPool(8, 16, 32, 1, enable_caching)
        return pool, physical

    return make


@pytest.mark.parametrize("enable_caching", [False, True])
def test_capacity_recheck_recovers_without_mutation(pool_factory, enable_caching):
    from kvcached.utils import KVCachePoolExhausted

    pool, physical = pool_factory(enable_caching)
    active = pool.get_new_blocks(1)[0]
    if enable_caching:
        cached = pool.get_new_blocks(1)[0]
        request = types.SimpleNamespace(block_hashes=[b"prefix"])
        pool.cache_full_blocks(request, [cached], 0, 1, 16, 0)
        pool.free_blocks([cached])

    count = len(pool._evictable_blocks) + 1
    assert pool.get_num_free_blocks() >= count  # Scheduler preflight.
    refs = [block.ref_cnt for block in pool.kv_block_pool]
    cached_blocks = {key: dict(value) for key, value in pool._cached_blocks.items()}
    reverse = dict(pool._block_id_to_key)
    evictable = list(pool._evictable_blocks.items())
    physical.alloc.reset_mock()
    physical.free.reset_mock()
    available = physical.available_size.side_effect
    physical.available_size.side_effect = lambda: 0

    with pytest.raises(KVCachePoolExhausted):
        pool.get_new_blocks(count)

    physical.alloc.assert_not_called()
    physical.free.assert_not_called()
    assert [block.ref_cnt for block in pool.kv_block_pool] == refs
    assert pool._cached_blocks == cached_blocks
    assert pool._block_id_to_key == reverse
    assert list(pool._evictable_blocks.items()) == evictable

    physical.available_size.side_effect = available
    blocks = pool.get_new_blocks(count)
    physical.alloc.assert_called_once_with(count)
    assert len({block.block_id for block in blocks}) == count
    assert all(block is pool.kv_block_pool[block.block_id] for block in blocks)
    assert all(block.ref_cnt == 1 and not block.is_null for block in blocks)
    assert active not in blocks and active.ref_cnt == 1
    pool.free_blocks(blocks)
    physical.free.assert_called_once_with([block.block_id for block in blocks])


@pytest.mark.parametrize("enable_caching", [False, True])
@pytest.mark.parametrize("kvcached_enabled", [False, True])
def test_capacity_recheck_through_scheduling_wrapper(
    vllm_patches, monkeypatch, pool_factory, enable_caching, kvcached_enabled,
):
    from kvcached.utils import KVCachePoolExhausted

    pool, physical = pool_factory(enable_caching)
    module = _module_with_manager(None)
    module.KVCacheManager.allocate_slots = lambda self, count: pool.get_new_blocks(count)
    manager = _apply(vllm_patches, monkeypatch, module,
                     kvcached_enabled=kvcached_enabled)
    assert pool.get_num_free_blocks() >= 1
    available = physical.available_size.side_effect
    physical.available_size.side_effect = lambda: 0
    if kvcached_enabled:
        assert manager.allocate_slots(1) is None
    else:
        with pytest.raises(KVCachePoolExhausted):
            manager.allocate_slots(1)
    physical.alloc.assert_not_called()
    physical.available_size.side_effect = available
    blocks = manager.allocate_slots(1)
    assert len(blocks) == 1 and blocks[0].ref_cnt == 1


@pytest.mark.parametrize("source", ["available_size", "alloc"])
@pytest.mark.parametrize("error", [ValueError("invalid allocation"),
                                   RuntimeError("CUDA illegal memory access")])
def test_pool_errors_escape_scheduling_wrapper(
    vllm_patches, monkeypatch, pool_factory, source, error,
):
    pool, physical = pool_factory(False)
    getattr(physical, source).side_effect = error
    module = _module_with_manager(None)
    module.KVCacheManager.allocate_slots = lambda self: pool.get_new_blocks(1)
    manager = _apply(vllm_patches, monkeypatch, module)
    with pytest.raises(type(error)) as caught:
        manager.allocate_slots()
    assert caught.value is error
