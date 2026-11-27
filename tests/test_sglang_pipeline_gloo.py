# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Real CPU collectives over stage-local TP and column-local PP groups."""

import gc
import sys
import weakref
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_sglang_rank_coordination_gloo import ProcessWorld


class PipelineWorld(ProcessWorld):
    def __init__(self, rank, tp_size, groups):
        super().__init__(rank, dist.group.WORLD)
        self.tp_size = tp_size
        self.tp, self.pp = groups

    def gather(self, output, value, group=None):
        with mock.patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}):
            dist.all_gather_object(output, value, group=group or self.group)

    def configure_scheduler(self, scheduler, running):
        tp, pp = self.tp_size, 4 // self.tp_size
        stage, column = divmod(self.rank, tp)
        scheduler.ps = NS(tp_size=tp, pp_size=pp)
        scheduler.tp_group = NS(
            world_size=tp, rank_in_group=column, ranks=list(range(stage * tp, (stage + 1) * tp))
        )
        scheduler.tp_cpu_group = self.tp[stage]
        scheduler.pp_group = NS(world_size=pp, rank_in_group=stage, cpu_group=self.pp[column])
        scheduler.pp_loop_size = pp
        scheduler.running_mbs = [running] + [object() for _ in range(pp - 1)]
        scheduler._kvcached_pipeline_progress = NS(
            before_admission=lambda: None,
            after_agreement=lambda: None,
            release_storage=lambda: None,
        )


def _exercise(rank, tp_size, world):
    import importlib

    from test_sglang_rank_coordination import Req, coord, mode, package
    from test_sglang_rank_prepare import decode_case, preparation_case

    pipeline = importlib.import_module(package.__name__ + ".pipeline")
    startup = NS()
    world.configure_scheduler(startup, None)
    assert pipeline.sum_pipeline_counts(startup, rank) == sum(range(4))

    for failed in (0, 3):
        for chunk in (False, True):
            preparation_case(world, failing_rank=failed, chunk=chunk)
        decode_case(world, failing_rank=failed)
    preparation_case(world, fatal=True)
    decode_case(world, rollback_error=True)

    # One stage has C128 layers; the other stages use native request rows.
    from test_sglang_dsv4_request_state import NativeRows, bind, load_request_module

    with pytest.MonkeyPatch.context() as patch:
        module = load_request_module(patch)
        owner = None
        if rank < tp_size:
            pool, owner, backings = bind(module)
            owner.reserve(1)
            pool.alloc_rows(1)
            for backing in backings:
                backing.manager.budget = len(backing.manager.live)
        else:
            pool = NativeRows()
            pool.alloc_rows(1)
        scheduler = NS(
            req_to_token_pool=pool,
            waiting_queue=[],
            chunked_req=NS(rid="chunk", kv=NS(req_pool_idx=8)),
        )
        world.configure_scheduler(scheduler, None)
        runtime = coord.RankCoordinator(scheduler)
        # Full backing on the C128 stage: the held chunk must still progress.
        before = list(pool.free_slots)
        budget = runtime.reserve_slots(
            lambda: min(4, pool.available_size()),
            demand_preview=lambda slots: {"p": {"rows": [{"rid": "chunk"}]}},
        )
        assert budget == 1
        assert runtime.new_slot_budget == 0
        assert pool.free_slots == before
        runtime.release_slots()
        if owner is not None:
            assert owner.active == {8} and not owner.standby
            for backing in backings:
                backing.manager.budget += 2

        # Stages without C128 layers must join the token-capacity exchanges
        # both after a fresh row reservation and after returning it unused.
        updates: list[tuple[int, int]] = []
        scheduler.token_to_kv_pool_allocator = NS(
            _kvcached_invalidate_token_capacity=lambda: None,
            _kvcached_budget_counts=lambda: (10, 10),
            _kvcached_update_budget=updates.append,
        )
        runtime.token_budget = (10, 10)
        scheduler.waiting_queue = [NS(rid="fresh", kv=NS(req_pool_idx=None))]
        runtime.reserve_slots(
            lambda: min(4, pool.available_size()),
            demand_preview=lambda slots: {"p": {"rows": [{"rid": "chunk"}, {"rid": "fresh"}]}},
        )
        assert runtime.new_slot_budget == 1 and runtime.leading_row_reused
        runtime.release_slots(refresh=True)
        assert updates == [(10, 10), (10, 10)]
        assert pool.free_slots == before

        # A genuine ownership mismatch is still fatal across stages.
        if rank == 3:
            scheduler.chunked_req.kv.req_pool_idx = None
        with pytest.raises(coord.RankConsensusError, match="ownership"):
            runtime.reserve_slots(
                lambda: min(4, pool.available_size()), demand_preview=lambda slots: {"p": None}
            )
        runtime.release_slots()

    scheduler = NS()
    world.configure_scheduler(scheduler, None)
    runtime = coord.RankCoordinator(scheduler)
    assert runtime.exchange("order", rank) == list(range(4))
    with pytest.raises(coord.RankConsensusError):
        runtime.exchange("different" if rank == 3 else "phase", None)

    batch = NS(
        reqs=[Req("decode", tokens=8, host=0)], forward_mode=mode("DECODE"), seq_lens_cpu=[8]
    )
    runtime.selected(batch)
    # Only stage zero enters forward: a PP collective here would time out.
    if rank < tp_size:
        runtime.selected(batch, stage="forward")
        if rank == 0:
            batch.seq_lens_cpu = [9]
        with pytest.raises(coord.RankConsensusError):
            runtime.selected(batch, stage="forward")
    dist.barrier()
    scheduler.forward_ct = 0
    scheduler.waiting_queue = []
    if rank == 3:
        scheduler.running_mbs = [object(), None] + [object()] * (4 // tp_size - 2)
    with pytest.raises(coord.RankConsensusError, match="mismatch"):
        runtime.start(None, None)


def _worker(rank, store_path, tp_size):
    dist.init_process_group(
        "gloo",
        store=dist.FileStore(store_path, 4),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    tp, pp, references = [], [], []
    completed = False
    try:
        tp = [
            dist.new_group(list(range(s * tp_size, (s + 1) * tp_size))) for s in range(4 // tp_size)
        ]
        pp = [dist.new_group(list(range(c, 4, tp_size))) for c in range(tp_size)]
        references = [
            weakref.ref(group) for group in tp + pp if group != dist.GroupMember.NON_GROUP_MEMBER
        ]
        _exercise(rank, tp_size, PipelineWorld(rank, tp_size, (tp, pp)))
        dist.barrier()
        completed = True
    finally:
        # Expected-error tracebacks and scheduler/coordinator cycles can keep
        # subgroup backends alive after destroy_process_group drops its registry.
        # Release those owners while Python can still service Gloo worker threads.
        tp.clear()
        pp.clear()
        gc.collect()
        dist.destroy_process_group()
        gc.collect()
        if completed:
            assert all(reference() is None for reference in references)


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
@pytest.mark.parametrize("tp_size", [1, 2])
def test_pipeline_groups_prepare_recovery_and_forward_boundary(tmp_path, tp_size):
    mp.spawn(_worker, args=(str(tmp_path / "pp-store"), tp_size), nprocs=4, join=True)
