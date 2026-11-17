# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Asymmetric prepare outcomes through the real coordinator and backing adapter."""

import sys
from types import SimpleNamespace as NS
from typing import Any

import pytest
import test_sglang_rank_coordination as consensus
from test_sglang_dsv4_native_backing import MODULE, Composite, Tensor, decode_setup, make_backing


class RequestRows:
    _aux_cache = None

    def __init__(self):
        self.free_rows = [1, 2, 3, 4]

    def available_size(self):
        return len(self.free_rows)

    def alloc(self, reqs):
        need = [req for req in reqs if req.kv.req_pool_idx is None]
        if len(need) > len(self.free_rows):
            return None
        for req in need:
            req.kv.req_pool_idx = self.free_rows.pop()
        return [req.kv.req_pool_idx for req in reqs]

    def free(self, req):
        self.free_rows.append(req.kv.req_pool_idx)
        req.kv.req_pool_idx = None


def preparation_case(world=None, *, failing_rank=3, chunk=False, fatal=False):
    case = consensus.CoordinationTests()
    case.setUp()
    try:
        if world is not None:
            case.world = world
        torch: Any = sys.modules["torch"]
        torch.distributed.all_gather_object = case.world.gather
        torch.tensor = lambda values, **_: Tensor(values)
        torch.int64 = object()
        torch.device = lambda value: NS(type=str(value).split(":")[0])
        schedulers = [case.Scheduler(rank, 65536) for rank in range(4)]
        allocators, backings, pools = [], [], []
        for rank, scheduler in enumerate(schedulers):
            allocator, backing, pool = Composite(4), make_backing(), RequestRows()
            assert backing.acquire({1})
            allocator.full_attn_allocator.output = [8, 20]  # Borrowed prefix tail and a new page.
            allocator.swa_attn_allocator.output = [4, 12]
            MODULE.bind_native_swa(allocator, backing)
            if rank == failing_rank:
                if fatal:
                    allocator.swa_attn_allocator.failure = RuntimeError(
                        "unexpected allocator error"
                    )
                else:
                    allocator.swa_attn_allocator.output = None
            req = scheduler.waiting_queue[0]
            req.kv.req_pool_idx = 4 if chunk else None
            req.kv.kv_allocated_len = 4 if chunk else 0
            if chunk:
                pool.free_rows.remove(4)
            scheduler.token_to_kv_pool_allocator = allocator
            scheduler.req_to_token_pool = pool
            scheduler.tree_cache = NS(dec_lock_ref=lambda *args: None)

            def allocate(batch):
                batch.req_to_token_pool.alloc(batch.reqs)
                batch.token_to_kv_pool_allocator.alloc_extend()
                for request in batch.reqs:
                    request.kv.kv_allocated_len += 4
                    request.kv.kv_committed_len += 4

            scheduler.allocate = allocate
            allocators.append(allocator)
            backings.append(backing)
            pools.append(pool)

        def worker(rank):
            scheduler = schedulers[rank]
            running = case.Batch([consensus.Req("decode", tokens=100, host=0, output=1)])
            first = scheduler.get_next_batch_to_run(running, None)
            if first.batch_to_run is not None:
                scheduler.run_batch(first.batch_to_run)
            allocator, backing, pool = allocators[rank], backings[rank], pools[rank]
            assert allocator.full_attn_allocator.live == {2}
            assert allocator.swa_attn_allocator.live == {1}
            assert backing.live == {1}
            assert len(pool.free_rows) == (3 if chunk else 4)
            req = scheduler.waiting_queue[0]
            assert req.kv.kv_allocated_len == (4 if chunk else 0)
            assert req.kv.req_pool_idx == (4 if chunk else None)
            # The next ordinary scheduler pass can make progress after capacity returns.
            allocator.swa_attn_allocator.output = [4, 12]
            second = scheduler.get_next_batch_to_run(running, first.batch_to_run)
            assert second.batch_to_run.forward_mode.name == "EXTEND"
            scheduler.run_batch(second.batch_to_run)
            return scheduler.forwards

        results = case.world.run(worker)
        if fatal:
            assert all(
                isinstance(value, consensus.coord.RankConsensusError) for value in results
            ), results
            assert all(scheduler.forwards == 0 for scheduler in schedulers)
        else:
            assert results == [2] * 4, results
        return results
    finally:
        case.doCleanups()


@pytest.mark.parametrize("failing_rank", [0, 3])
@pytest.mark.parametrize("chunk", [False, True])
def test_peer_prepare_miss_rolls_back_new_pages_and_retries(failing_rank, chunk):
    preparation_case(failing_rank=failing_rank, chunk=chunk)


def test_unexpected_prepare_error_stops_all_forward():
    preparation_case(fatal=True)


def decode_case(world=None, *, failing_rank=3, rollback_error=False):
    case = consensus.CoordinationTests()
    case.setUp()
    try:
        if world is not None:
            case.world = world
        torch: Any = sys.modules["torch"]
        torch.distributed.all_gather_object = case.world.gather
        torch.tensor = lambda values, **_: Tensor(values)
        torch.int64 = object()
        torch.device = lambda value: NS(type=str(value).split(":")[0])
        owners = [decode_setup((10,) if rank == failing_rank else (3, 7, 9))
                  for rank in range(4)]
        allocator, backing = owners[failing_rank]
        assert backing.acquire({2})
        allocator.swa_attn_allocator.live.add(2)
        backing.manager.budget = len(backing.manager.live)
        if rollback_error:
            original = owners[0][1].release

            def release(pages):
                if pages:
                    raise RuntimeError("decode reservation rollback failed")
                return original(pages)

            owners[0][1].release = release

        def worker(rank):
            allocator, backing = owners[rank]
            scheduler = case.Scheduler(rank, 65536)
            runtime = consensus.coord.RankCoordinator(scheduler)
            req = consensus.Req("decode", tokens=4, host=0)
            req.kv.kv_committed_len = 4
            batch = case.Batch([req])
            batch.token_to_kv_pool_allocator = allocator
            batch.capacity_check = lambda: allocator.check_decode_capacity(num_tokens=4)
            token = consensus.coord._ACTIVE.set(runtime)
            try:
                assert not batch.check_decode_mem()
                assert backing.live == ({1, 2} if rank == failing_rank else {1})
                allocator.swa_attn_allocator.choices = [3, 7, 9]
                assert batch.check_decode_mem()
                backing.manager.budget = len(backing.manager.live)
                assert allocator.alloc_decode().tolist() == [20]
                runtime.selected(batch, stage="forward")
                return True
            finally:
                consensus.coord._ACTIVE.reset(token)

        results = case.world.run(worker)
        if rollback_error:
            assert all(isinstance(result, consensus.coord.RankConsensusError)
                       for result in results), results
        else:
            assert results == [True] * 4, results
    finally:
        case.doCleanups()


@pytest.mark.parametrize("failing_rank", [0, 3])
def test_peer_decode_shortfall_releases_reservations_and_recovers(failing_rank):
    decode_case(failing_rank=failing_rank)


def test_decode_reservation_rollback_failure_stops_every_rank():
    decode_case(rollback_error=True)
