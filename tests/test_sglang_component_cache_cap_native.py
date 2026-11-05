# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Real SGLang cache/allocator eviction on small CUDA pools, without a model."""

import importlib
import importlib.metadata
from array import array

import pytest


@pytest.fixture(params=["unified", "swa"])
def native_cache(monkeypatch, request):
    pytest.importorskip("sglang")
    if importlib.metadata.version("sglang") != "0.5.20":
        pytest.skip("Native cache fixture is pinned to SGLang 0.5.20")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the native SWA allocator")
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.mem_cache.unified_cache.components import ComponentType

    monkeypatch.setenv("SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND", "python")
    pool = SWAKVPool(size=64, size_swa=64, page_size=1, dtype=torch.float16,
                     head_num=1, head_dim=16, swa_attention_layer_ids=[1],
                     full_attention_layer_ids=[0], device="cuda")
    req_pool = ReqToTokenPool(2, 64, "cuda", enable_memory_saver=False)
    allocator = SWATokenToKVPoolAllocator(
        64, 64, 1, torch.float16, "cuda", pool, need_sort=False)
    module_name, class_name, patch_name = (
        ("unified_radix_cache", "UnifiedRadixCache", "UnifiedRadixCacheLimitPatch")
        if request.param == "unified" else
        ("swa_radix_cache", "SWARadixCache", "SWARadixCacheLimitPatch"))
    module = importlib.import_module("sglang.srt.mem_cache." + module_name)
    cache_cls = getattr(module, class_name)
    # Restore the real class after each parameterized fixture.
    monkeypatch.setattr(cache_cls, "cache_finished_req", cache_cls.cache_finished_req)
    cache = cache_cls(CacheInitParams(
        disable=False, req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator, page_size=1,
        sliding_window_size=4, tree_components=(ComponentType.FULL, ComponentType.SWA),
    ))
    yield cache, allocator, pool, req_pool, module, patch_name
    torch.cuda.synchronize()


@pytest.mark.parametrize("full_cap,swa_cap", [(-1, -1), (0, -1), (-1, 0), (0, 0), (4, 4)])
@pytest.mark.parametrize("grouped", [False, True], ids=["immediate-free", "grouped-free"])
def test_native_caps_preserve_active_full_prefix_and_swa_window(
    monkeypatch, native_cache, full_cap, swa_cap, grouped,
):
    import torch
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.sampling.sampling_params import SamplingParams

    from kvcached.integration.sglang import patches

    cache, allocator, pool, req_pool, module, patch_name = native_cache
    monkeypatch.setattr(patches, "MAX_CACHED_TOKENS", full_cap)
    monkeypatch.setattr(patches, "MAX_CACHED_SWA_TOKENS", swa_cap)
    assert getattr(patches, patch_name)().apply(module)

    # The old prefix stays FULL-locked while only its last four SWA tokens are
    # active. A separate request completes and triggers the cache-cap policy.
    values = allocator.alloc(12)
    assert values is not None
    tokens = list(range(12))
    cache.insert(InsertParams(key=RadixKey(tokens), value=values))
    cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens[:8])))
    node = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens))).last_device_node
    receipt = cache.inc_lock_ref(node)
    assert cache.full_evictable_size() == 0
    assert cache.swa_evictable_size() == 8
    active_swa = allocator.translate_loc_from_full_to_swa(values[-4:]).clone()
    pool.get_key_buffer(0)[values] = 42
    pool.get_key_buffer(1)[active_swa] = 84

    req = Req("finished", "", array("q", range(100, 108)), SamplingParams(max_new_tokens=1))
    # A cache miss still owns the root lock receipt in the scheduler lifecycle.
    req.last_node = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(req.origin_input_ids))).last_device_node
    req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
    assert req_pool.alloc([req]) is not None
    completed_values = allocator.alloc(8)
    assert completed_values is not None
    req_pool.req_to_token[req.kv.req_pool_idx, :8] = completed_values
    req.kv.kv_allocated_len = 8
    req.kv.kv_committed_len = 8
    before = (allocator.full_available_size(), allocator.swa_available_size())
    if grouped:
        allocator.free_group_begin()
    try:
        cache.cache_finished_req(req, kv_len_to_handle=8)
        if full_cap >= 0:
            assert cache.full_evictable_size() <= full_cap
        if swa_cap >= 0:
            assert cache.swa_evictable_size() <= swa_cap
        if full_cap < 0 and swa_cap < 0:
            assert cache.full_evictable_size() == 8
            assert cache.swa_evictable_size() == 16
        if grouped:
            assert (allocator.full_available_size(), allocator.swa_available_size()) == before
    finally:
        if grouped:
            allocator.free_group_end()
    try:
        torch.cuda.synchronize()
        assert torch.all(pool.get_key_buffer(0)[values] == 42).item()
        assert torch.all(pool.get_key_buffer(1)[active_swa] == 84).item()
        # Native eviction may split or drop an unprotected FULL leaf to release
        # SWA. All remaining FULL slots belong to the locked prefix or the cache.
        assert allocator.full_available_size() == 64 - 12 - cache.full_evictable_size()
        if swa_cap == 0:
            assert allocator.swa_available_size() == 60
    finally:
        cache.dec_lock_ref(node, receipt.to_dec_params())
        cache.evict(EvictParams(num_tokens=64, swa_num_tokens=64))
        req_pool.free(req)
    assert allocator.full_available_size() == allocator.swa_available_size() == 64
