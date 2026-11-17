# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Keep the native PP output ring progressing across admission collectives."""

from collections import deque
from functools import wraps
from typing import Any


class PipelineProgress:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.size = scheduler.ps.pp_size
        self.round = 0
        self.pending: deque[tuple[int, list[Any]]] = deque()

    def defer_output_work(self):
        scheduler = self.scheduler
        if scheduler.send_output_work:
            # Retain P2PWork.payload as well as its work handle until receipt.
            self.pending.append((self.round, scheduler.send_output_work))
            scheduler.send_output_work = []
        if len(self.pending) > 2 * (self.size + 2):
            raise RuntimeError("PP output receipts exceeded the bounded scheduling window")

    def before_admission(self):
        scheduler = self.scheduler
        if not scheduler.pp_group.is_last_rank and scheduler.pp_outputs is not None:
            self.defer_output_work()
            scheduler.send_output_work = scheduler._pp_send_dict_to_next_stage(
                scheduler.pp_outputs.tensors, async_send=True, msg_type="output"
            )
            scheduler.pp_outputs = None

    def after_agreement(self):
        # Only wait for sends whose receivers have already traversed the ring.
        # Waiting on the current output before admission would deadlock PP>2.
        while self.pending and self.round - self.pending[0][0] > self.size:
            _, work = self.pending[0]
            self.scheduler._pp_commit_comm_work(work)
            self.pending.popleft()
        self.round += 1

    def fence_storage(self):
        # NCCL output sends may await future receives on their own streams.
        # A device-wide synchronize here would block pipeline progress.
        self.scheduler.forward_stream.synchronize()
        self.scheduler.schedule_stream.synchronize()

    def release_storage(self):
        from kvcached.pool_registry import get_registered_kv_cache_pools

        for manager, _ in get_registered_kv_cache_pools(integration="sglang"):
            manager.release_retired_pages_through(manager.capture_physical_release_marker())


def patch_pipeline_progress(scheduler_class):
    # Small CPU fixtures and TP-only integrations need not implement PP hooks.
    original_init = getattr(scheduler_class, "init_pp_loop_state", None)
    original_output = getattr(
        scheduler_class, "_pp_commit_send_output_work_and_preprocess_output_tensors", None
    )
    if original_init is None or original_output is None:
        return

    @wraps(original_init)
    def initialize(self):
        from kvcached.integration.sglang.interfaces import set_pipeline_release_barrier

        original_init(self)
        if self.ps.pp_size > 1:
            progress = PipelineProgress(self)
            set_pipeline_release_barrier(progress.fence_storage)
            self._kvcached_pipeline_progress = progress

    @wraps(original_output)
    def output(self, *args, **kwargs):
        progress = getattr(self, "_kvcached_pipeline_progress", None)
        if progress is not None:
            progress.defer_output_work()
        return original_output(self, *args, **kwargs)

    scheduler_class.init_pp_loop_state = initialize
    scheduler_class._pp_commit_send_output_work_and_preprocess_output_tensors = output
