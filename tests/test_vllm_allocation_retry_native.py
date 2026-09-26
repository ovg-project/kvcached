# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Run separately with real vLLM 0.28/0.29, not the CPU import stubs."""

from types import SimpleNamespace

import pytest
from test_vllm_native_block_pool import native_pool_factory  # noqa: F401


@pytest.fixture(params=["native", "elastic"])
def manager_factory(monkeypatch, request, native_pool_factory):  # noqa: F811
    vllm = pytest.importorskip("vllm")
    if not isinstance(vllm.__version__, str) or not vllm.__version__.startswith(("0.28.", "0.29.")):
        pytest.skip("Requires native vLLM 0.28 or 0.29")
    import torch
    from vllm.v1.core import kv_cache_manager as native
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )

    from kvcached.integration.vllm import patches

    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(native.KVCacheManager, "allocate_slots", native.KVCacheManager.allocate_slots)
    assert patches.KVCacheManagerAllocateSlotsPatch().apply(native)
    from vllm.v1.core import single_type_kv_cache_manager as single_type

    monkeypatch.setattr(single_type.MambaManager, "_cache_partial_tail_block",
                        single_type.MambaManager._cache_partial_tail_block)
    assert patches.MambaPartialTailPatch().apply(single_type)

    def make(*, caching=True, mamba_first=False, speculative=0):
        attention = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16)
        mamba = MambaSpec(block_size=16, shapes=((2, 8), (2, 8)),
                          dtypes=(torch.float16, torch.float16), mamba_cache_mode="align",
                          num_speculative_blocks=speculative)
        specs = [mamba, attention] if mamba_first else [attention, mamba]
        config = KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[f"layer{i}"], kv_cache_spec=spec)
            for i, spec in enumerate(specs)])
        manager = native.KVCacheManager(config, max_model_len=256,
                                        scheduler_block_size=16, hash_block_size=4,
                                        enable_caching=caching)
        if request.param == "elastic":
            pool, _ = native_pool_factory(size=64, enable_caching=caching)
            manager.block_pool = manager.coordinator.block_pool = pool
            for group in manager.coordinator.single_type_managers:
                group.block_pool = pool
                group._null_block = pool.null_block
        return manager
    return make


def request(status=None):
    from vllm.v1.request import RequestStatus

    return SimpleNamespace(request_id="retry", status=status or RequestStatus.WAITING,
                           num_computed_tokens=0, num_in_flight_tokens=0,
                           num_prompt_tokens=128, num_tokens=128,
                           shared_prefix_boundary=0,
                           block_hashes=[bytes([i]) for i in range(32)])


def inject(pool, monkeypatch, fail_at):
    from kvcached.utils import KVCachePoolExhausted

    original = pool.get_new_blocks
    calls = []

    def allocate(count):
        calls.append(count)
        if len(calls) == fail_at:
            raise KVCachePoolExhausted("injected physical admission failure")
        return original(count)
    monkeypatch.setattr(pool, "get_new_blocks", allocate)
    return calls, original


@pytest.mark.parametrize("hit_tokens,fail_at", [(0, 1), (0, 2), (8, 1), (8, 2), (8, 3), (16, 2)])
@pytest.mark.parametrize("preempted", [False, True])
def test_waiting_retry_releases_only_unpublished_ownership(
    manager_factory, monkeypatch, hit_tokens, fail_at, preempted,
):
    from vllm.v1.request import RequestStatus

    manager = manager_factory()
    pool = manager.block_pool
    req = request(RequestStatus.PREEMPTED if preempted else None)
    sources = pool.get_new_blocks(2)
    computed = manager.create_kv_cache_blocks(([sources[0]], [sources[1]])) if hit_tokens else None
    refs = [b.ref_cnt for b in pool.blocks]
    calls, original = inject(pool, monkeypatch, fail_at)
    kwargs = dict(num_new_computed_tokens=hit_tokens, new_computed_blocks=computed,
                  delay_cache_blocks=True)
    assert manager.allocate_slots(req, 24, **kwargs) is None
    assert len(calls) == fail_at
    assert [b.ref_cnt for b in pool.blocks] == refs
    for group in manager.coordinator.single_type_managers:
        assert not group.req_to_blocks.get(req.request_id)
        assert req.request_id not in group.num_cached_block
        assert req.request_id not in group._partial_hit_reqs
        assert group._pending_cow_copies == []
        assert group.new_block_ids == []
    monkeypatch.setattr(pool, "get_new_blocks", original)
    assert manager.allocate_slots(req, 24, **kwargs) is not None


@pytest.mark.parametrize("caching", [False, True])
@pytest.mark.parametrize("mamba_first", [False, True])
def test_running_retry_keeps_existing_blocks_and_other_requests(
    manager_factory, monkeypatch, caching, mamba_first,
):
    from vllm.v1.request import RequestStatus

    manager = manager_factory(caching=caching, mamba_first=mamba_first)
    pool = manager.block_pool
    req = request()
    assert manager.allocate_slots(req, 16, delay_cache_blocks=True) is not None
    req.status = RequestStatus.RUNNING
    req.num_computed_tokens = 16
    groups = manager.coordinator.single_type_managers
    before = [list(g.req_to_blocks[req.request_id]) for g in groups]
    unrelated = tuple(pool.get_new_blocks(2))
    for group in groups:
        pool.touch(unrelated)
        group._pending_cow_copies.append(unrelated)
    refs = [b.ref_cnt for b in pool.blocks]
    calls, original = inject(pool, monkeypatch, 2)
    assert manager.allocate_slots(request=req, num_new_tokens=32, delay_cache_blocks=True) is None
    assert len(calls) == 2
    assert [list(g.req_to_blocks[req.request_id]) for g in groups] == before
    assert [b.ref_cnt for b in pool.blocks] == refs
    assert all(g._pending_cow_copies == [unrelated] for g in groups)
    monkeypatch.setattr(pool, "get_new_blocks", original)
    assert manager.allocate_slots(req, 32, delay_cache_blocks=True) is not None


def test_running_producer_hashes_and_handoff_restored(manager_factory, monkeypatch):
    from vllm.v1.request import RequestStatus

    manager = manager_factory(mamba_first=True)
    pool = manager.block_pool
    req = request()
    assert manager.allocate_slots(req, 8, delay_cache_blocks=True) is not None
    req.status = RequestStatus.RUNNING
    req.num_computed_tokens = 8
    group = manager.coordinator.single_type_managers[0]
    source = group.req_to_blocks[req.request_id][0]
    key = pool.cache_partial_block(req, source, 8, 0, 16)
    pool.cache_partial_block(req, source, 4, 0, 16)
    group._partial_hit_reqs[req.request_id] = (0, source)
    group._producer_partial_tail_reqs[req.request_id] = 8
    original_aliases = set(pool.cached_block_hashes_by_block[source.block_id])
    refs = [b.ref_cnt for b in pool.blocks]
    calls, original = inject(pool, monkeypatch, 2)
    assert manager.allocate_slots(req, 24, delay_cache_blocks=True) is None
    assert len(calls) == 2
    assert source.block_hash == key
    assert set(pool.cached_block_hashes_by_block[source.block_id]) == original_aliases
    assert pool.get_cached_block(req.block_hashes[1], [0]) == [source]
    assert group._partial_hit_reqs[req.request_id] == (0, source)
    assert group._producer_partial_tail_reqs[req.request_id] == 8
    assert group.cached_blocks_this_step == set()
    assert [b.ref_cnt for b in pool.blocks] == refs
    for name in ("_pending_boundary_state_offloads", "_pending_partial_tail_offloads"):
        if hasattr(group, name):
            assert getattr(group, name) == []
    monkeypatch.setattr(pool, "get_new_blocks", original)
    assert manager.allocate_slots(req, 24, delay_cache_blocks=True) is not None


@pytest.mark.parametrize("speculative", [0, 2])
def test_skipped_release_and_speculative_relocation(manager_factory, monkeypatch, speculative):
    from vllm.v1.request import RequestStatus

    manager = manager_factory(mamba_first=True, speculative=speculative)
    req = request()
    assert manager.allocate_slots(req, 16, delay_cache_blocks=True) is not None
    req.status = RequestStatus.RUNNING
    req.num_computed_tokens = 16
    assert manager.allocate_slots(req, 16, delay_cache_blocks=True) is not None
    req.num_computed_tokens = 32
    group = manager.coordinator.single_type_managers[0]
    # Take the expected snapshot after the native skipped-block release.
    manager.coordinator.remove_skipped_blocks(req.request_id, 32, num_prompt_tokens=128)
    before = list(group.req_to_blocks[req.request_id])
    refs = [b.ref_cnt for b in manager.block_pool.blocks]
    calls, original = inject(manager.block_pool, monkeypatch, 2)
    assert manager.allocate_slots(req, 48, delay_cache_blocks=True) is None
    assert len(calls) == 2
    assert group.req_to_blocks[req.request_id] == before
    assert [b.ref_cnt for b in manager.block_pool.blocks] == refs
    monkeypatch.setattr(manager.block_pool, "get_new_blocks", original)
    assert manager.allocate_slots(req, 48, delay_cache_blocks=True) is not None


@pytest.mark.parametrize("fail_at", [1, 2])
def test_external_computed_blocks_survive_repeated_admission_miss(
    manager_factory, monkeypatch, fail_at,
):
    manager = manager_factory()
    req = request()
    pool = manager.block_pool
    original = pool.get_new_blocks
    refs = [b.ref_cnt for b in pool.blocks]
    for _ in range(3):
        calls, _ = inject(pool, monkeypatch, fail_at)
        assert manager.allocate_slots(req, 16, num_external_computed_tokens=16,
                                      delay_cache_blocks=True) is None
        assert len(calls) == fail_at
        assert [b.ref_cnt for b in pool.blocks] == refs
        for group in manager.coordinator.single_type_managers:
            assert not group.req_to_blocks.get(req.request_id)
            assert req.request_id not in group.num_cached_block
        monkeypatch.setattr(pool, "get_new_blocks", original)
    assert manager.allocate_slots(req, 16, num_external_computed_tokens=16,
                                  delay_cache_blocks=True) is not None


@pytest.mark.parametrize("prompt_tokens", [8, 12, 24])
def test_async_decode_does_not_republish_producer_boundary(manager_factory, prompt_tokens):
    from vllm.v1.request import RequestStatus

    manager = manager_factory(mamba_first=True)
    req = request()
    req.num_prompt_tokens = req.num_tokens = prompt_tokens
    group = manager.coordinator.single_type_managers[0]
    assert manager.allocate_slots(req, prompt_tokens) is not None
    req.status = RequestStatus.RUNNING
    req.num_computed_tokens = req.num_in_flight_tokens = prompt_tokens
    assert manager.allocate_slots(req, 1) is not None
    [(source, checkpoint)] = group.take_pending_cow_copies()
    assert checkpoint.block_hash_num_tokens == prompt_tokens
    assert manager.block_pool.get_cached_block(
        req.block_hashes[prompt_tokens // 4 - 1], [0]) == [checkpoint]
    assert source.block_hash is None
    assert req.request_id not in group._partial_hit_reqs

    req.num_tokens = req.num_computed_tokens = prompt_tokens + 1
    req.num_in_flight_tokens = 1
    assert manager.allocate_slots(req, 1) is not None
    assert group.take_pending_cow_copies() == []


def test_remote_completion_still_registers_first_partial_boundary(manager_factory):
    from vllm.v1.request import RequestStatus

    manager = manager_factory(mamba_first=True)
    req = request()
    req.num_prompt_tokens = req.num_tokens = 8
    group = manager.coordinator.single_type_managers[0]
    assert manager.allocate_slots(req, 8, delay_cache_blocks=True) is not None
    req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    req.num_computed_tokens = 8
    manager.cache_blocks(req, 8)
    source = group.req_to_blocks[req.request_id][0]
    assert source.block_hash_num_tokens == 8
    assert group._partial_hit_reqs[req.request_id] == (0, source)


def test_preempted_producer_can_publish_boundary_again(manager_factory):
    from vllm.v1.request import RequestStatus

    manager = manager_factory(mamba_first=True)
    req = request()
    req.num_prompt_tokens = req.num_tokens = 8
    group = manager.coordinator.single_type_managers[0]
    assert manager.allocate_slots(req, 8) is not None
    manager.free(req)
    req.status = RequestStatus.PREEMPTED
    req.num_computed_tokens = 0
    assert manager.allocate_slots(req, 8) is not None
    source = group.req_to_blocks[req.request_id][0]
    assert source.block_hash_num_tokens == 8
    assert group._partial_hit_reqs[req.request_id] == (0, source)
