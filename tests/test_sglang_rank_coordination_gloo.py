# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Four CPU processes using Gloo, without model weights or CUDA initialization."""

import sys
from datetime import timedelta
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class ProcessWorld:
    def __init__(self, rank, group):
        self.rank = rank
        self.group = group

    def gather(self, output, value, group=None):
        # Admission fixtures replace torch's tensor surface. Gloo itself uses
        # the real module, including C-extension lookups through sys.modules.
        with mock.patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}):
            dist.all_gather_object(output, value, group=self.group)

    def run(self, worker):
        try:
            value = worker(self.rank)
        except Exception as exc:
            value = exc
        values = [None] * 4
        self.gather(values, value)
        return values


def _worker(rank, store_path):
    store = dist.FileStore(store_path, 4)
    dist.init_process_group(
        "gloo", store=store, rank=rank, world_size=4, timeout=timedelta(seconds=45)
    )
    try:
        from types import SimpleNamespace as NS

        from test_sglang_rank_coordination import Req, coord, mode
        from test_sglang_rank_prepare import decode_case, preparation_case

        world = ProcessWorld(rank, dist.group.WORLD)
        for chunk in (False, True):
            for failed_rank in (0, 3):
                preparation_case(world, failing_rank=failed_rank, chunk=chunk)
        preparation_case(world, fatal=True)
        for failed_rank in (0, 3):
            decode_case(world, failing_rank=failed_rank)
        decode_case(world, rollback_error=True)

        runtime = coord.RankCoordinator(
            NS(
                tp_group=NS(world_size=4, rank_in_group=rank, ranks=list(range(4))),
                tp_cpu_group=dist.group.WORLD,
                ps=NS(pp_size=1, tp_size=4),
            )
        )
        batch = NS(
            reqs=[Req("decode", tokens=8, host=0)],
            forward_mode=mode("DECODE"),
            seq_lens_cpu=[8 if rank != 3 else 9],
        )
        with pytest.raises(coord.RankConsensusError):
            runtime.selected(batch, stage="forward")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="This PyTorch build has no CPU/Gloo backend",
)
def test_four_process_prepare_recovery_and_forward_guard(tmp_path: Path):
    # FileStore uses an owned local path and does not need a listening TCP port.
    mp.spawn(_worker, args=(str(tmp_path / "rank-store"),), nprocs=4, join=True)
