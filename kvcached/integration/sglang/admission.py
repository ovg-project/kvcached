# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Reserve elastic prefill memory before SGLang publishes request slots.

This guard covers the SGLang 0.5.20 single-rank, page-size-one prefill path.
It deliberately does not catch errors from prepare_for_extend: after that
boundary, SGLang owns partially published request state and must fail loudly.
"""

import functools
import time
from contextvars import ContextVar
from typing import Any

from kvcached.integration.patch_base import BasePatch
from kvcached.integration.version_utils import VersionAwarePatch
from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()
_prefill_context: ContextVar[Any] = ContextVar("kvcached_prefill", default=None)


class _PrefillCapacityMiss(Exception):
    def __init__(self, batch):
        self.batch = batch


def _managers(batch, context):
    scheduler, old_chunk, _old_inflight = context
    allocator = batch.token_to_kv_pool_allocator
    manager = getattr(allocator, "kvcached_allocator", None)
    mode = getattr(scheduler, "disaggregation_mode", "null")
    mode = getattr(mode, "value", mode)
    if (manager is None or allocator.page_size != 1
            or getattr(manager, "world_size", 1) != 1
            or getattr(scheduler, "enable_priority_preemption", False)
            or getattr(scheduler, "enable_hierarchical_cache", False)
            or getattr(scheduler, "enable_unified_cache_external_linker", False)
            or getattr(scheduler, "pp_size", 1) != 1
            or mode != "null"
            or not scheduler.spec_algorithm.is_none()
            or batch.is_dllm()):
        return None
    # Sessions and retained request rows have a different rollback contract.
    # The old chunk is retained in place; it is never freed or requeued here.
    for req in batch.reqs:
        if getattr(req, "session", None):
            return None
        if req is not old_chunk and (req.kv.holds_kv
                or req.kv.mamba_ping_pong_track_buffer is not None):
            return None
    mamba = getattr(batch.req_to_token_pool, "mamba_pool", None)
    mamba_manager = getattr(mamba, "kvcached_allocator", None)
    managers = [manager] + ([mamba_manager] if mamba_manager is not None else [])
    if any(m.reserved_blocks for m in managers):
        return None  # Do not consume another reservation owner's blocks.
    return managers


def _mamba_slots(batch):
    pool = batch.req_to_token_pool
    extra = getattr(pool, "enable_mamba_extra_buffer", False)
    extra_count = (1 if getattr(pool, "enable_mamba_extra_buffer_lazy", False)
                   else getattr(pool, "mamba_ping_pong_track_buffer_size", 0))
    return sum(
        int(not req.kv.holds_mamba)
        + (extra_count if extra and req.kv.mamba_ping_pong_track_buffer is None else 0)
        for req in batch.reqs
    )


def _reset_for_admission_retry(req):
    """Discard selection state without recording a model-work retraction."""
    retraction_state = (req.is_retracted, req.retracted_stain, req.retraction_count)
    req.reset_for_retract()
    req.is_retracted, req.retracted_stain, req.retraction_count = retraction_state


def _requeue_unpublished(scheduler, batch, old_chunk, old_inflight):
    """Undo selection only; no request row or new token KV was published."""
    for req in batch.reqs:
        if req is old_chunk:
            # add_chunked_req retains the previous lock and row. This failed
            # attempt never ran a forward pass, so undo only its inflight debit.
            req.inflight_middle_chunks = old_inflight
            continue
        assert not req.kv.holds_kv
        if (not getattr(batch.tree_cache, "disable", False)
                and not batch.tree_cache.is_chunk_cache()):
            batch.tree_cache.dec_lock_ref(req.last_node, req.lock_receipt)
        if req.kv.holds_mamba:
            batch.req_to_token_pool.mamba_allocator.free(
                req.kv.mamba_pool_idx.unsqueeze(-1))
            req.kv.mamba_pool_idx = None
        _reset_for_admission_retry(req)
        scheduler._add_request_to_queue(req, is_retracted=req.is_retracted)
    scheduler.chunked_req = old_chunk
    # Yield to the scheduler's normal loop and existing running work. There is
    # no in-call retry, sleep under a lock, or loop around allocation failure.
    scheduler._kvcached_prefill_retry_at = time.monotonic() + 0.01
    logger.info("Elastic prefill reservation unavailable; returned %d requests to scheduling",
                sum(req is not old_chunk for req in batch.reqs))


def _install_prefill_guard(Scheduler, ScheduleBatch, evict, EvictParams):
    if getattr(Scheduler, "_kvcached_prefill_reservation", False):
        return
    original_select = Scheduler._get_new_batch_prefill_raw
    original_prepare = ScheduleBatch.prepare_for_extend

    @functools.wraps(original_select)
    def select(self, prefill_delayer_single_pass, running_batch):
        if time.monotonic() < getattr(self, "_kvcached_prefill_retry_at", 0):
            return None, running_batch
        old_chunk = self.chunked_req
        old_inflight = old_chunk.inflight_middle_chunks if old_chunk is not None else None
        old_full = running_batch.batch_is_full
        token = _prefill_context.set((self, old_chunk, old_inflight))
        try:
            return original_select(self, prefill_delayer_single_pass, running_batch)
        except _PrefillCapacityMiss as miss:
            _requeue_unpublished(self, miss.batch, old_chunk, old_inflight)
            running_batch.batch_is_full = old_full
            return None, running_batch
        finally:
            _prefill_context.reset(token)

    @functools.wraps(original_prepare)
    def prepare(self):
        context = _prefill_context.get()
        managers = _managers(self, context) if context is not None else None
        if managers is None:
            return original_prepare(self)
        need = sum(len(req.get_fill_ids()) - len(req.prefix_indices) for req in self.reqs)
        try:
            evict(self.tree_cache, need)
            if not managers[0].try_to_reserve(need):
                raise _PrefillCapacityMiss(self)
            if len(managers) == 2:
                mamba_need = _mamba_slots(self)
                shortfall = max(0, mamba_need - managers[1].available_size())
                if shortfall and self.tree_cache.supports_mamba():
                    self.tree_cache.evict_for_alloc(
                        EvictParams(num_tokens=0, mamba_num=shortfall))
                if not managers[1].try_to_reserve(mamba_need):
                    raise _PrefillCapacityMiss(self)
            # These are physically backed blocks, not an availability estimate.
            # Both groups are reserved before request rows/Mamba mappings change.
            result = original_prepare(self)
        except _PrefillCapacityMiss:
            for manager in reversed(managers):
                manager.free_reserved()
            raise
        else:
            for manager in reversed(managers):
                manager.free_reserved()
            return result
        # Unexpected/unsafe errors retain ownership for process teardown.
        # Do not touch an uncertain native pool in an unconditional finally.

    Scheduler._get_new_batch_prefill_raw = select
    ScheduleBatch.prepare_for_extend = prepare
    Scheduler._kvcached_prefill_reservation = True


class SchedulerPrefillAdmissionPatch(VersionAwarePatch, BasePatch):
    library = "sglang"
    target_module = "sglang.srt.managers.scheduler"
    target_class = "Scheduler"
    patch_name = "elastic_prefill_admission"

    def apply(self, module) -> bool:
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        if not self.initialize_version_info():
            return False
        _install_prefill_guard(module.Scheduler, module.ScheduleBatch,
                               evict_from_tree_cache, EvictParams)
        return True
