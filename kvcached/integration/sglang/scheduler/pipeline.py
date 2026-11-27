# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Keep the native PP output ring progressing across admission collectives."""

from collections import deque
from functools import wraps
from typing import Any


def sum_pipeline_counts(scheduler, local_count):
    """Sum rank-local counts within this TP/PP serving instance."""
    import torch
    import torch.distributed as dist

    count = torch.tensor(local_count, dtype=torch.int64, device="cpu")
    if scheduler.ps.tp_size > 1:
        dist.all_reduce(count, group=scheduler.tp_cpu_group)
    dist.all_reduce(count, group=scheduler.pp_group.cpu_group)
    return int(count.item())


class PipelineProgress:
    def __init__(self, scheduler, allocator=None):
        self.scheduler = scheduler
        self.allocator = allocator
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

    def initialize_output_link(self):
        # The last-to-first edge can be cold even after activation traffic.
        # Match the first NCCL send before either endpoint enters admission.
        import torch

        group = self.scheduler.pp_group
        if group.is_last_rank:
            group.send_tensor_dict({"warmup": torch.empty(1, device=group.device)}, dst=0)
        elif group.is_first_rank:
            group.recv_tensor_dict(src=self.size - 1)

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
        if getattr(self.allocator, "_kvcached_parked_cuda_probe", False):
            from kvcached.integration.sglang.patches import _probe_parked_dsv4_cuda_cache

            # Peers are at the start agreement. Poll NCCL only: a CPU metadata
            # send can wait for the next receive without blocking a CUDA drain.
            groups = [self.scheduler.send_output_work, self.scheduler.send_proxy_work]
            groups.extend(work for _, work in self.pending)
            if all(item.work is None or not item.payload.is_cuda or item.work.is_completed()
                   for group in groups for item in group):
                self.allocator._kvcached_parked_cuda_probe = False
                _probe_parked_dsv4_cuda_cache(self.allocator)

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
        from kvcached.integration.sglang.patches import _sglang_dsv4_demand_allocator

        original_init(self)
        if self.ps.pp_size > 1:
            allocator = _sglang_dsv4_demand_allocator(self)
            managed = sum_pipeline_counts(self, int(allocator is not None))
            if managed == 0:
                self._kvcached_native_pipeline = True
                return
            if managed != self.ps.tp_size * self.ps.pp_size:
                raise RuntimeError("Managed DSV4 storage must be available on every TP/PP rank")
            from kvcached.integration.sglang.interfaces import set_pipeline_release_barrier

            progress = PipelineProgress(self, allocator)
            set_pipeline_release_barrier(progress.fence_storage)
            progress.initialize_output_link()
            self._kvcached_pipeline_progress = progress

    @wraps(original_output)
    def output(self, *args, **kwargs):
        progress = getattr(self, "_kvcached_pipeline_progress", None)
        if progress is not None:
            progress.defer_output_work()
        return original_output(self, *args, **kwargs)

    scheduler_class.init_pp_loop_state = initialize
    scheduler_class._pp_commit_send_output_work_and_preprocess_output_tensors = output
