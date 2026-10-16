# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Exercise admission failure before SGLang publishes request rows or KV."""

from enum import Enum
from types import SimpleNamespace as NS
from typing import Any

import pytest

from kvcached.integration.sglang.admission import _install_prefill_guard


class Manager:
    world_size = 1

    def __init__(self, capacity=100, error=None):
        self.capacity = capacity
        self.error = error
        self.reserved_blocks = []
        self.calls = []

    def available_size(self):
        return self.capacity

    def try_to_reserve(self, count):
        self.calls.append(count)
        if self.error:
            raise self.error
        if count > self.capacity:
            return False
        self.reserved_blocks = list(range(count))
        return True

    def free_reserved(self):
        self.reserved_blocks.clear()


class Slot:
    def unsqueeze(self, _dimension):
        return self


class Request:
    session = None

    def __init__(self, *, chunk=False, mamba=False):
        self.kv = NS(holds_kv=chunk, holds_mamba=mamba,
                     mamba_pool_idx=Slot() if mamba else None,
                     mamba_ping_pong_track_buffer=None)
        self.prefix_indices = [10, 11]
        self.last_node = object()
        self.lock_receipt = object()
        self.inflight_middle_chunks = 2 if chunk else 0
        self.reset_count = 0

    def get_fill_ids(self):
        return [0] * 6

    def reset_for_retract(self):
        assert not self.kv.holds_kv
        self.reset_count += 1
        self.prefix_indices = []
        self.inflight_middle_chunks = 0


def fixture(*, kv_capacity=100, mamba_capacity=None, chunk=False,
            reserve_error=None, prepare_error=None, final_chunk=False):
    manager = Manager(kv_capacity, reserve_error)
    mamba_manager = Manager(mamba_capacity) if mamba_capacity is not None else None
    fresh = Request(mamba=mamba_manager is not None)
    old = Request(chunk=True, mamba=mamba_manager is not None) if chunk else None
    unlocked: list[Any] = []
    freed_mamba: list[Any] = []
    evicted: list[Any] = []
    pool = NS(enable_mamba_extra_buffer=mamba_manager is not None,
              enable_mamba_extra_buffer_lazy=False,
              mamba_ping_pong_track_buffer_size=2,
              mamba_allocator=NS(free=freed_mamba.append))
    if mamba_manager is not None:
        pool.mamba_pool = NS(kvcached_allocator=mamba_manager)
    tree = NS(dec_lock_ref=lambda node, receipt: unlocked.append((node, receipt)),
              supports_mamba=lambda: mamba_manager is not None,
              is_chunk_cache=lambda: False,
              evict_for_alloc=evicted.append)

    class Batch:
        prepare_calls = 0

        def __init__(self, reqs):
            self.reqs = reqs
            self.token_to_kv_pool_allocator = NS(page_size=1, kvcached_allocator=manager)
            self.req_to_token_pool = pool
            self.tree_cache = tree

        def is_dllm(self):
            return False

        def prepare_for_extend(self):
            self.prepare_calls += 1
            if prepare_error:
                raise prepare_error
            expected = sum(len(r.get_fill_ids()) - len(r.prefix_indices) for r in self.reqs)
            assert len(manager.reserved_blocks) == expected
            manager.reserved_blocks.clear()
            if mamba_manager is not None:
                assert len(mamba_manager.reserved_blocks) == 2 * len(self.reqs)
                mamba_manager.reserved_blocks.clear()
            for req in self.reqs:
                req.kv.holds_kv = True

    class DisaggregationMode(Enum):
        NULL = "null"

    class Scheduler:
        # Match the real SGLang 0.5.20 contract, not a convenient string fake.
        disaggregation_mode = DisaggregationMode.NULL
        spec_algorithm = NS(is_none=lambda: True)

        def __init__(self):
            self.chunked_req = old
            self.waiting_queue = [fresh]
            self.select_calls = 0
            self.batch = None

        def _add_request_to_queue(self, req, is_retracted=False):
            assert is_retracted
            self.waiting_queue.append(req)

        def _get_new_batch_prefill_raw(self, _delayer, running):
            self.select_calls += 1
            reqs, self.waiting_queue = self.waiting_queue, []
            if old is not None:
                if final_chunk:
                    self.chunked_req = None
                else:
                    old.inflight_middle_chunks += 1
                reqs = [old] + reqs
            else:
                self.chunked_req = fresh  # New chunk selection is unpublished.
                fresh.inflight_middle_chunks += 1
            running.batch_is_full = True
            self.batch = Batch(reqs)
            self.batch.prepare_for_extend()
            return self.batch, running

    _install_prefill_guard(Scheduler, Batch,
                           lambda cache, count: evicted.append(count),
                           lambda **kwargs: NS(**kwargs))
    scheduler = Scheduler()
    return NS(scheduler=scheduler, manager=manager, mamba_manager=mamba_manager,
              fresh=fresh, old=old, unlocked=unlocked, freed_mamba=freed_mamba,
              running=NS(reqs=[object()], batch_is_full=False))


def test_capacity_race_returns_unpublished_request_and_can_later_progress():
    f = fixture(kv_capacity=0)
    assert f.scheduler._get_new_batch_prefill_raw(None, f.running) == (None, f.running)
    assert f.scheduler.batch.prepare_calls == 0
    assert f.scheduler.waiting_queue == [f.fresh]
    assert f.scheduler.chunked_req is None
    assert not f.running.batch_is_full
    assert not f.fresh.kv.holds_kv
    assert f.fresh.reset_count == 1 and len(f.unlocked) == 1
    assert f.manager.reserved_blocks == []
    f.scheduler._get_new_batch_prefill_raw(None, f.running)
    assert f.scheduler.select_calls == 1  # No busy retry while capacity is unchanged.
    f.manager.capacity = 100
    f.scheduler._kvcached_prefill_retry_at = 0
    batch, running = f.scheduler._get_new_batch_prefill_raw(None, f.running)
    assert batch.prepare_calls == 1 and running is f.running
    assert f.fresh.kv.holds_kv and f.scheduler.waiting_queue == []


def test_mamba_miss_releases_kv_reservation_and_unpublished_matched_state():
    f = fixture(mamba_capacity=0)
    slot = f.fresh.kv.mamba_pool_idx
    assert f.scheduler._get_new_batch_prefill_raw(None, f.running) == (None, f.running)
    assert f.scheduler.batch.prepare_calls == 0
    assert f.manager.calls == [4] and f.mamba_manager.calls == [2]
    assert f.manager.reserved_blocks == f.mamba_manager.reserved_blocks == []
    assert f.freed_mamba == [slot] and f.fresh.kv.mamba_pool_idx is None
    assert f.scheduler.waiting_queue == [f.fresh]


def test_success_reserves_both_groups_before_any_request_row_is_published():
    f = fixture(mamba_capacity=100)
    batch, running = f.scheduler._get_new_batch_prefill_raw(None, f.running)
    assert batch.prepare_calls == 1 and running is f.running
    assert f.manager.calls == [4] and f.mamba_manager.calls == [2]
    assert f.manager.reserved_blocks == f.mamba_manager.reserved_blocks == []
    assert not f.unlocked and not f.freed_mamba


@pytest.mark.parametrize("final_chunk", [False, True])
def test_existing_chunk_keeps_its_row_state_and_previous_inflight_count(final_chunk):
    f = fixture(kv_capacity=0, mamba_capacity=100, chunk=True, final_chunk=final_chunk)
    slot = f.old.kv.mamba_pool_idx
    assert f.scheduler._get_new_batch_prefill_raw(None, f.running) == (None, f.running)
    assert f.scheduler.chunked_req is f.old
    assert f.old.kv.holds_kv and f.old.kv.mamba_pool_idx is slot
    assert f.old.inflight_middle_chunks == 2 and f.old.reset_count == 0
    assert f.scheduler.waiting_queue == [f.fresh]
    assert len(f.unlocked) == 1 and f.unlocked[0][0] is f.fresh.last_node
    assert slot not in f.freed_mamba


@pytest.mark.parametrize("where", ["reservation", "prepare"])
def test_unexpected_or_consistency_failure_is_never_converted_to_requeue(where):
    error = RuntimeError("unsafe native state" if where == "reservation" else "published failure")
    f = fixture(reserve_error=error if where == "reservation" else None,
                prepare_error=error if where == "prepare" else None)
    with pytest.raises(RuntimeError) as caught:
        f.scheduler._get_new_batch_prefill_raw(None, f.running)
    assert caught.value is error
    assert not f.unlocked and f.fresh.reset_count == 0
    assert f.scheduler.waiting_queue == []
    assert len(f.manager.reserved_blocks) == (4 if where == "prepare" else 0)
