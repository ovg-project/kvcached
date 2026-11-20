# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Instance-local scheduling consensus, independent of performance policies.

No cost tables, peer-instance state, CUDA collectives or barriers.
Native resource checks remain authoritative; actual disagreements fail closed
before model forward, rather than pretending a preview reserved physical KV.
"""

from __future__ import annotations

import functools
import importlib.metadata
import logging
import os
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any

from .prefill_transaction import (
    PrefillAllocationMiss,
    PrefillAllocationTransaction,
    PrefillPrepareDeferred,
    PrefillScheduleSnapshot,
)
from .rank_plan import batch_signature, common_plan, preview, request_slot

LOG = logging.getLogger(__name__)
_ACTIVE: ContextVar[Any] = ContextVar("kvcached_rank_coordination", default=None)
_PATCHED = False


def active():
    return _ACTIVE.get()


def _enabled(name):
    return os.getenv(name, "0").lower() in {"1", "true", "yes", "on"}


class RankConsensusError(RuntimeError):
    """Fatal mismatch; never retry a collective after this exception."""


class _PrefillDeferred(RuntimeError):
    """Internal control flow: retry this prefill after running decode."""


class RankCoordinator:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.group = getattr(scheduler, "tp_group", None)
        ps = getattr(scheduler, "ps", None)
        self.tp_size = int(getattr(self.group, "world_size", getattr(ps, "tp_size", 1)))
        self.tp_rank = int(getattr(self.group, "rank_in_group", 0))
        self.pp_size = int(getattr(ps, "pp_size", 1))
        pp_group = getattr(scheduler, "pp_group", None)
        self.pp_rank = int(getattr(pp_group, "rank_in_group", 0))
        self.pp_cpu_group = getattr(pp_group, "cpu_group", None)
        if self.pp_size > 1 and (
            self.pp_cpu_group is None or getattr(pp_group, "world_size", None) != self.pp_size
            or getattr(scheduler, "_kvcached_pipeline_progress", None) is None
        ):
            raise RankConsensusError("PP coordination requires the native pipeline progress hooks")
        self.size = self.tp_size * self.pp_size
        self.rank = self.pp_rank * self.tp_size + self.tp_rank
        self.committed_signature = None
        self.cpu_group = getattr(scheduler, "tp_cpu_group", None)
        if self.tp_size > 1 and (self.group is None or self.cpu_group is None):
            raise RankConsensusError("rank coordination requires the instance TP CPU group")
        self.round = 0
        self.last_restriction_log = float("-inf")
        self.restriction_count = 0
        self.has_running_decode = False
        self.prefill_backoff_until = 0.0
        self.prefill_backoff_failures = 0
        try:
            self.prefill_backoff_base = max(
                float(os.getenv("KVCACHED_RANK_COORDINATION_PREFILL_BACKOFF_MS", "10")) / 1000.0,
                0.0,
            )
        except ValueError as exc:
            raise RankConsensusError(
                "KVCACHED_RANK_COORDINATION_PREFILL_BACKOFF_MS must be numeric"
            ) from exc
        self.slot_budget = None
        self.new_slot_budget = None
        self.prefill_plan = None
        self.admission_materialization = None
        self.outer_required = None
        self.token_budget = None
        self.prefill_attempt = None
        self.row_reservation: Any = None
        self.planned = self.grammar_polled = self.priority_sorted = False
        error = None
        if bool(getattr(scheduler, "enable_dp_attention", False)):
            error = "rank coordination requires attention-DP disabled"
        spec = getattr(scheduler, "spec_algorithm", None)
        if spec is not None and (not spec.is_none()):
            error = "rank coordination requires non-speculative generation"
        if any(
            (
                getattr(scheduler, name, None)
                for name in (
                    "is_mixed_chunk",
                    "dllm_config",
                    "dynamic_chunk_sizer",
                    "enable_dynamic_chunking",
                    "enable_hisparse",
                )
            )
        ):
            error = "rank coordination requires ordinary P/D batches with fixed chunking"
        cp = getattr(scheduler, "attn_cp_group", None)
        if (
            cp is not None
            and self.group is not None
            and (not set(cp.ranks) <= set(self.group.ranks))
        ):
            error = "instance TP consensus must contain every CP rank"
        topology = self.exchange("init", (self.tp_size, self.pp_size, self.rank), error=error)
        expected = [(self.tp_size, self.pp_size, rank) for rank in range(self.size)]
        if topology != expected:
            raise RankConsensusError(f"invalid TP/PP rank layout: {topology}")
        if self.rank == 0:
            LOG.info(
                "[kvcached-rank-consensus] enabled size=%s ranks=%s; no cross-instance collective",
                self.size,
                getattr(self.group, "ranks", None),
            )

    def exchange(self, stage, value, *, error=None, tp_only=False):
        """Every boundary uses the same tagged CPU collective protocol.

        A peer taking a different branch meets a phase/round mismatch rather
        than interpreting another boundary's payload as its own decision.
        """
        local = {"round": self.round, "stage": stage, "value": value, "error": error}
        import torch.distributed as dist

        states: list[Any] = [local]
        if self.tp_size > 1:
            states = [None] * self.tp_size
            dist.all_gather_object(states, local, group=self.cpu_group)
        if self.pp_size > 1 and not tp_only:
            stages: list[Any] = [None] * self.pp_size
            dist.all_gather_object(stages, states, group=self.pp_cpu_group)
            states = [state for peers in stages for state in peers]
        if any((s["round"] != self.round or s["stage"] != stage or s["error"] for s in states)):
            raise RankConsensusError(
                f"rank consensus boundary failed: local={stage} states={states}"
            )
        return [s["value"] for s in states]

    def agree(self, stage, value, *, error=None, tp_only=False):
        values = self.exchange(stage, value, error=error, tp_only=tp_only)
        if any((v != values[0] for v in values[1:])):
            raise RankConsensusError(
                f"rank consensus mismatch before forward: round={self.round} stage={stage} values={values}"
            )
        return values[0]

    def report_restriction(self, stage, values, decision):
        """Bound diagnostics under sustained memory pressure; never log tokens."""
        if self.rank != 0:
            return
        self.restriction_count += 1
        now = time.monotonic()
        if now - self.last_restriction_log >= 5:
            LOG.warning(
                "[kvcached-rank-consensus] restricted round=%s stage=%s count=%s values=%s decision=%s",
                self.round,
                stage,
                self.restriction_count,
                values,
                decision,
            )
            self.last_restriction_log = now

    def start(self, running, last):
        self.round += 1
        self.prefill_plan = None
        self.slot_budget = None
        self.new_slot_budget = None
        self.planned = self.grammar_polled = self.priority_sorted = False
        self.has_running_decode = bool(
            running is not None
            and (not bool(getattr(running, "is_prefill_only", False)))
            and any(
                (
                    not req.finished() and (not bool(getattr(req, "is_retracted", False)))
                    for req in getattr(running, "reqs", ())
                )
            )
        )

        def requests(reqs):
            return [
                (str(r.rid), len(r.origin_input_ids), len(r.output_ids), bool(r.finished()))
                for r in reqs
            ]

        def state(batch):
            return (
                None
                if batch is None
                else {
                    "reqs": requests(batch.reqs),
                    "full": bool(batch.batch_is_full),
                    "mode": getattr(getattr(batch, "forward_mode", None), "name", None),
                }
            )

        chunk = getattr(self.scheduler, "chunked_req", None)
        error, value = (None, None)
        try:
            value = {
                "running": state(running),
                "last": state(last),
                "waiting": requests(self.scheduler.waiting_queue),
                "chunk": str(chunk.rid) if chunk is not None else None,
                "forward_ct": self.scheduler.forward_ct,
                "defer": getattr(self.scheduler, "_prefill_decode_interval_remaining", 0),
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        allocator = getattr(self.scheduler, "token_to_kv_pool_allocator", None)
        counts = getattr(allocator, "_kvcached_budget_counts", None)
        if value is not None:
            value["native_backing"] = counts is not None
        progress = getattr(self.scheduler, "_kvcached_pipeline_progress", None)
        if self.pp_size > 1 and value is not None:
            try:
                value["pipeline"] = (
                    self.scheduler.pp_loop_size,
                    next(i for i, batch in enumerate(self.scheduler.running_mbs) if batch is running),
                )
            except Exception as exc:
                error = f"invalid PP microbatch: {exc}"
        self.agree("start", value, error=error)
        if progress is not None:
            release_error = None
            try:
                progress.after_agreement()
                progress.release_storage()
            except Exception as exc:
                release_error = f"{type(exc).__name__}: {exc}"
            self.exchange("pipeline_release", None, error=release_error)
        self.token_budget = None
        if counts is not None:
            local, count_error = None, None
            try:
                local = counts()
            except Exception as exc:
                count_error = f"{type(exc).__name__}: {exc}"
            peers = self.exchange("token_capacity", local, error=count_error)
            self.token_budget = tuple(min(values) for values in zip(*peers))

    def capacity_context(self):
        if self.token_budget is None:
            return nullcontext()
        return self.scheduler.token_to_kv_pool_allocator._kvcached_budget_context(self.token_budget)

    def reserve_slots(self, query, *, demand_preview=None):
        """One tagged boundary combines physical C128 readiness and slot MIN.

        All ranks participate, including those with a standby, zero logical
        rows, or a previously blocked allocator. No conditional inner AND can
        collide with the scheduler's next collective.
        """
        pool: Any = getattr(self.scheduler, "req_to_token_pool", None)
        allocator: Any = getattr(pool, "_kvcached_dsv4_request_state", None)
        reservation, local_slots, error = (None, 0, None)
        ownership, reused = (None, 0)

        @contextmanager
        def capacity(count):
            if allocator is not None:
                with allocator.capacity_query(count):
                    yield
            else:
                # Expose held chunk rows as logical capacity without editing
                # native free lists. The scheduler owns this synchronous query.
                previous = pool.__dict__.get("available_size")
                pool.available_size = lambda: count
                try:
                    yield
                finally:
                    if previous is None:
                        del pool.available_size
                    else:
                        pool.available_size = previous

        try:
            logical = allocator.logical_available() if allocator is not None else None
            if demand_preview is not None:
                if logical is None:
                    logical = pool.available_size()
                chunk = getattr(self.scheduler, "chunked_req", None)
                reqs = ([chunk] if chunk is not None else []) + list(self.scheduler.waiting_queue)
                by_id = {str(req.rid): req for req in reqs}
                if len(by_id) != len(reqs):
                    raise ValueError("duplicate prefill request IDs")
                ownership = [(str(req.rid), request_slot(req) is not None) for req in reqs]
                if allocator is not None:
                    for req in reqs:
                        slot = request_slot(req)
                        if slot is not None and int(slot) not in allocator.active:
                            raise ValueError(f"request {req.rid} does not own a backed C128 row")
                with capacity(logical + sum(held for _, held in ownership)):
                    logical_slots = int(query())
                candidate = demand_preview(logical_slots)["p"]
                rows = [] if candidate is None else candidate["rows"]
                reused = sum(request_slot(by_id[row["rid"]]) is not None for row in rows)
                target = len(rows) - reused
            elif allocator is not None:
                with capacity(logical):
                    target = max(0, int(query()))
            if allocator is not None:
                reservation = allocator.prepare_admission(target)
                self.row_reservation = (allocator, reservation)
                with capacity(reservation.available + reused):
                    local_slots = int(query())
            elif demand_preview is not None:
                with capacity(logical + reused):
                    local_slots = int(query())
            else:
                local_slots = int(query())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        value = {
            "slots": local_slots,
            "ownership": ownership,
            "c128": None
            if reservation is None
            else {"available": reservation.available, "target": reservation.target},
        }
        try:
            states = self.exchange("slots", value, error=error)
            for offset in range(0, self.size, self.tp_size):
                peers = states[offset:offset + self.tp_size]
                if any((s["c128"] is None) != (peers[0]["c128"] is None) for s in peers):
                    raise RankConsensusError(f"C128 allocator presence differs within TP: {states}")
            if any((s["ownership"] != states[0]["ownership"] for s in states)):
                raise RankConsensusError(f"C128 request row ownership differs: {states}")
        except Exception as exc:
            if reservation is not None:
                try:
                    allocator.finish_admission(reservation, 0)
                except Exception:
                    LOG.exception("C128 rollback failed after rank consensus failure")
            if isinstance(exc, RankConsensusError):
                raise
            raise RankConsensusError(
                "C128 slot exchange failed; refusing collective retry"
            ) from exc
        self.slot_budget = max(0, min((s["slots"] for s in states)))
        managed = [s["c128"]["available"] for s in states if s["c128"] is not None]
        if managed:
            self.new_slot_budget = max(0, min(managed))
            if allocator is not None:
                allocator.finish_admission(reservation, self.new_slot_budget)
        return self.slot_budget

    def release_slots(self):
        """Release this attempt's unused leases; active and older rows survive."""
        pending, self.row_reservation = self.row_reservation, None
        if pending is not None:
            allocator, reservation = pending
            try:
                allocator.finish_admission(reservation, 0)
            finally:
                allocator.end_admission()

    def prepare(self, candidates, *, error=None):
        states = self.exchange("candidates", candidates, error=error)
        decision, decision_error = (None, None)
        if self.rank == 0:
            try:
                decision = {"p": common_plan([s["p"] for s in states], "P")}
            except Exception as exc:
                decision_error = f"{type(exc).__name__}: {exc}"
        self.prefill_plan = self.exchange("decision", decision, error=decision_error)[0]["p"]
        self.planned = True
        if any((s != states[0] for s in states[1:])):
            self.report_restriction("preview", states, self.prefill_plan)

    def selected(self, batch, *, error=None, stage="selected"):
        signature = None
        if error is None:
            try:
                signature = batch_signature(batch)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        tp_only = self.pp_size > 1 and stage == "forward"
        if tp_only and signature != self.committed_signature:
            error = error or "PP batch changed after admission consensus"
        self.agree(stage, signature, error=error, tp_only=tp_only)
        if stage == "selected":
            self.committed_signature = signature

    def begin_admission(self, req, states):
        """Accept tier-placement differences for one logical prefix.

        A rank which completed an earlier deferred H2D can have the prefix on
        device while a peer still sees the same prefix in host cache.  The
        request identity and logical input must remain identical; only the
        local materialization state may differ.
        """
        stable_keys = ("rid", "input", "context", "max_new", "output", "admitted")
        reference = {key: states[0][key] for key in stable_keys}
        if any(({key: state[key] for key in stable_keys} != reference for state in states[1:])):
            raise RankConsensusError(
                f"rank admission input mismatch before materialization: {states}"
            )
        if reference["rid"] != str(req.rid):
            raise RankConsensusError(
                f"rank admission request mismatch: local={req.rid} states={states}"
            )
        self.admission_materialization = {
            "rid": str(req.rid),
            "required": any((state["needs_load"] for state in states)),
            "done": False,
            "accepted": None,
        }

    def materialized(self, req, *, ready, context, error=None):
        """AND actual load-back completion before any rank commits prefill.

        This is deliberately after the real allocator call, rather than a
        capacity preview.  On rejection, a successful peer may have populated
        an ordinary cache node, but no request state or prefill batch has been
        committed.  The waiting request is re-matched on the next scheduler
        pass and the current pass can continue with decode.
        """
        state = self.admission_materialization
        if state is None or state["rid"] != str(req.rid):
            raise RankConsensusError(f"missing materialization state for request {req.rid}")
        if state["done"]:
            return bool(state["accepted"])
        values = self.exchange(
            "admission_materialize",
            {"rid": str(req.rid), "ready": bool(ready), "context": int(context)},
            error=error,
        )
        accepted = all(
            (value["rid"] == str(req.rid) and value["ready"] for value in values)
        ) and all((value["context"] == values[0]["context"] for value in values[1:]))
        state["done"] = True
        state["accepted"] = accepted
        if accepted:
            self.prefill_backoff_failures = 0
            self.prefill_backoff_until = 0.0
        else:
            self.defer_prefill("admission_materialize", values)
        return accepted

    def defer_prefill(self, stage, values):
        if self.has_running_decode:
            self.prefill_backoff_failures = 0
            self.prefill_backoff_until = 0.0
        else:
            self.prefill_backoff_failures = min(self.prefill_backoff_failures + 1, 5)
            delay = min(self.prefill_backoff_base * 2 ** (self.prefill_backoff_failures - 1), 0.2)
            self.prefill_backoff_until = time.monotonic() + delay
        decision = (
            "defer_prefill_run_decode" if self.has_running_decode else "defer_prefill_backoff"
        )
        self.report_restriction(stage, values, decision)

    def prefill_backoff_active(self):
        """Back off allocation retries only when there is no decode work."""
        return not self.has_running_decode and time.monotonic() < self.prefill_backoff_until

    def row(self, req):
        return next(
            (r for r in (self.prefill_plan or {}).get("rows", []) if r["rid"] == str(req.rid)), None
        )


def _runtime(scheduler):
    runtime = getattr(scheduler, "_kvcached_rank_coordinator", None)
    if runtime is None:
        runtime = RankCoordinator(scheduler)
        scheduler._kvcached_rank_coordinator = runtime
    return runtime


def patch_rank_coordination():
    global _PATCHED
    if (
        _PATCHED
        or not _enabled("KVCACHED_RANK_COORDINATION_ENABLE")
        or not _enabled("ENABLE_KVCACHED")
    ):
        return []
    version = importlib.metadata.version("sglang")
    if version not in {"0.5.20"}:
        raise RuntimeError("rank coordination requires SGLang 0.5.20")
    from sglang.srt.constrained.grammar_manager import GrammarManager
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder, SchedulePolicy
    from sglang.srt.managers.scheduler import Scheduler

    original_next = Scheduler.get_next_batch_to_run
    original_new = Scheduler.get_new_batch_prefill
    native_add = PrefillAdder.add_one_req
    original_add = native_add
    original_select = PrefillAdder._select_prefill_admission
    original_commit = PrefillAdder._commit_prefill_admission
    original_chunk = PrefillAdder.add_chunked_req
    original_sort = SchedulePolicy.calc_priority
    original_grammar = GrammarManager.get_ready_grammar_requests
    original_check = ScheduleBatch.check_decode_mem
    original_slots = Scheduler.get_num_allocatable_reqs
    original_run = Scheduler.run_batch

    def empty_prefill(running_batch):
        from sglang.srt.managers.scheduler import NextBatchPlan

        return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

    def admission_shape(admission):
        return {
            name: getattr(admission, name)
            for name in ("prefix_len", "extend_len", "max_new_tokens", "is_chunked")
        }

    budget_properties = {
        name: getattr(PrefillAdder, name)
        for name in ("rem_total_tokens", "cur_rem_tokens", "rem_swa_tokens")
    }
    for name, prop in budget_properties.items():
        if not isinstance(prop, property):
            raise RuntimeError(f"unsupported PrefillAdder budget property: {name}")
    for name, prop in budget_properties.items():

        def coordinated_budget(original, stage):

            @functools.wraps(original)
            def getter(self):
                runtime = _ACTIVE.get()
                if runtime is None:
                    return original(self)
                value, error = (None, None)
                try:
                    value = original(self)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                if stage == "rem_total_tokens" and runtime.outer_required is not None:
                    required = runtime.outer_required
                    runtime.outer_required = None
                    values = runtime.exchange(
                        "admission_total_gate",
                        {"required": required, "available": value},
                        error=error,
                    )
                    # Make the native local-demand comparison test the largest
                    # peer demand, including a host-only prefix.
                    return (
                        min(v["available"] for v in values)
                        + required
                        - max(v["required"] for v in values)
                    )
                values = runtime.exchange("budget." + stage, value, error=error)
                decision = min(values)
                if any((v != values[0] for v in values[1:])):
                    runtime.report_restriction("budget." + stage, values, decision)
                return decision

            return getter

        setattr(
            PrefillAdder,
            name,
            property(coordinated_budget(prop.fget, name), prop.fset, prop.fdel, prop.__doc__),
        )

    @functools.wraps(original_next)
    def next_batch(self, running_batch, last_batch):
        progress = getattr(self, "_kvcached_pipeline_progress", None)
        if progress is not None:
            progress.before_admission()
        runtime = _runtime(self)
        if runtime.size == 1:
            return original_next(self, running_batch, last_batch)
        runtime.start(running_batch, last_batch)
        token = _ACTIVE.set(runtime)
        try:
            try:
                with runtime.capacity_context():
                    result = original_next(self, running_batch, last_batch)
            except RankConsensusError:
                raise
            except Exception as exc:
                runtime.selected(None, error=f"{type(exc).__name__}: {exc}")
                raise
            runtime.selected(result.batch_to_run)
            return result
        finally:
            _ACTIVE.reset(token)

    @functools.wraps(original_grammar)
    def grammar(self, *args, **kwargs):
        runtime = _ACTIVE.get()
        if runtime is not None:
            if runtime.grammar_polled:
                return []
            runtime.grammar_polled = True
        return original_grammar(self, *args, **kwargs)

    @functools.wraps(original_sort)
    def sort(self, *args, **kwargs):
        runtime = _ACTIVE.get()
        if runtime is not None and runtime.priority_sorted:
            return None
        return original_sort(self, *args, **kwargs)

    @functools.wraps(original_new)
    def new(self, running_batch):
        runtime = _ACTIVE.get()
        if runtime is None:
            return original_new(self, running_batch)
        allocator = getattr(
            getattr(self, "req_to_token_pool", None), "_kvcached_dsv4_request_state", None
        )
        try:
            candidates, error = (None, None)
            try:
                if self.grammar_manager.has_waiting_grammars():
                    for req in self.grammar_manager.get_ready_grammar_requests():
                        self._add_request_to_queue(req)
                runtime.agree(
                    "prefill_input",
                    {
                        "waiting": [str(r.rid) for r in self.waiting_queue],
                        "chunk": str(self.chunked_req.rid)
                        if self.chunked_req is not None
                        else None,
                    },
                )
                self.policy.calc_priority(
                    self.waiting_queue,
                    running_batch,
                    **{"processed_tokens": self.processed_tokens_counter},
                )
                runtime.priority_sorted = True
                backoff_states = runtime.exchange(
                    "prefill_backoff", {"active": runtime.prefill_backoff_active()}
                )
                if self.chunked_req is not None and any(
                    (state["active"] for state in backoff_states)
                ):
                    runtime.prepare({"p": None})
                    return empty_prefill(running_batch)
                if self.chunked_req is None and any((state["active"] for state in backoff_states)):
                    runtime.slot_budget = 0
                    runtime.new_slot_budget = 0
                    candidates = {"p": None}
                    runtime.prepare(candidates)
                    result = original_new(self, running_batch)
                    running_batch.batch_is_full = False
                    return result
                preview_matches: dict[str, Any] = {}

                def shadow(slots, **kwargs):
                    token = _ACTIVE.set(None)
                    try:
                        return preview(
                            self, running_batch, slots=slots, match_cache=preview_matches, **kwargs
                        )
                    finally:
                        _ACTIVE.reset(token)

                runtime.reserve_slots(
                    lambda: original_slots(
                        self, len(running_batch.reqs), **{"running_batch": running_batch}
                    ),
                    demand_preview=shadow if allocator is not None or runtime.pp_size > 1 else None,
                )
                candidates = shadow(
                    runtime.slot_budget,
                    **{"new_slots": runtime.new_slot_budget} if runtime.new_slot_budget is not None else {},
                )
            except RankConsensusError:
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            runtime.prepare(candidates, error=error)
            attempt = PrefillScheduleSnapshot(self)
            runtime.prefill_attempt = attempt
            try:
                return original_new(self, running_batch)
            except PrefillPrepareDeferred:
                rollback_error = None
                try:
                    attempt.rollback(running_batch)
                except Exception as exc:
                    rollback_error = f"{type(exc).__name__}: {exc}"
                runtime.exchange("prepare_extend.schedule_rollback", True, error=rollback_error)
                return empty_prefill(running_batch)
            finally:
                runtime.prefill_attempt = None
        finally:
            runtime.release_slots()

    @functools.wraps(original_slots)
    def slots(self, *args, **kwargs):
        runtime = _ACTIVE.get()
        if runtime is not None and runtime.slot_budget is not None:
            return runtime.slot_budget
        return original_slots(self, *args, **kwargs)

    @functools.wraps(original_add)
    def add(self, req, *args, **kwargs):
        runtime = _ACTIVE.get()
        if runtime is None:
            return native_add(self, req, *args, **kwargs)
        local_input = {
            "rid": str(req.rid),
            "input": len(req.full_untruncated_fill_ids),
            "context": len(req.prefix_indices) + int(req.host_hit_length),
            "prefix": len(req.prefix_indices),
            "host": int(req.host_hit_length),
            "swa_host": int(getattr(req, "swa_host_hit_length", 0)),
            "needs_load": bool(req.needs_host_load_back()),
            "max_new": req.sampling_params.max_new_tokens,
            "output": len(req.output_ids),
            "admitted": [str(r.rid) for r in self.can_run_list],
        }
        states = runtime.exchange("admission_input", local_input)
        runtime.begin_admission(req, states)
        if runtime.planned and runtime.row(req) is None:
            runtime.admission_materialization = None
            return AddReqResult.OTHER
        error, result = (None, None)
        cache: Any = getattr(self, "tree_cache", None)
        original_init_load_back: Any = (
            getattr(cache, "init_load_back", None) if cache is not None else None
        )
        original_load_back = getattr(cache, "load_back", None) if cache is not None else None
        own_init_load_back = (
            getattr(cache, "__dict__", {}).get("init_load_back") if cache is not None else None
        )
        own_load_back = (
            getattr(cache, "__dict__", {}).get("load_back") if cache is not None else None
        )
        load_state = {"called": False, "success": None}
        if original_init_load_back is not None and original_load_back is not None:

            def tracked_load_back(*load_args, **load_kwargs):
                load_state["called"] = True
                load_state["success"] = bool(original_load_back(*load_args, **load_kwargs))
                return load_state["success"]

            cache.load_back = tracked_load_back

        def coordinated_init_load_back(params):
            loaded, load_error = (None, None)
            try:
                loaded = original_init_load_back(params)
            except Exception as exc:
                load_error = f"{type(exc).__name__}: {exc}"
            ready = loaded is not None
            if load_state["called"]:
                ready = ready and bool(load_state["success"])
            loaded_count = 0
            if loaded is not None:
                try:
                    loaded_count = len(loaded[0])
                except Exception as exc:
                    load_error = f"{type(exc).__name__}: {exc}"
                    ready = False
            context = len(req.prefix_indices) + loaded_count
            if not runtime.materialized(req, ready=ready, context=context, error=load_error):
                return None
            return loaded

        if original_init_load_back is not None:
            cache.init_load_back = coordinated_init_load_back
        try:
            try:
                if hasattr(self, "_mamba_gap_budget_for_req"):
                    from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS

                    runtime.outer_required = (
                        len(req.full_untruncated_fill_ids)
                        - len(req.prefix_indices)
                        + min(
                            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
                            CLIP_MAX_NEW_TOKENS,
                        )
                        + self.page_size
                        + self._mamba_gap_budget_for_req(req)
                    )
                result = original_add(self, req, *args, **kwargs)
            except _PrefillDeferred:
                result = AddReqResult.OTHER
        except RankConsensusError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if original_init_load_back is not None:
                if own_init_load_back is None:
                    del cache.init_load_back
                else:
                    cache.init_load_back = own_init_load_back
            if original_load_back is not None:
                if own_load_back is None:
                    del cache.load_back
                else:
                    cache.load_back = own_load_back
        try:
            runtime.agree(
                "admission_result",
                {
                    "result": getattr(result, "name", None),
                    "admitted": [str(r.rid) for r in self.can_run_list],
                },
                error=error,
            )
            return result
        finally:
            runtime.admission_materialization = None
            runtime.outer_required = None

    @functools.wraps(original_select)
    def select(self, req, **kwargs):
        runtime = _ACTIVE.get()
        if runtime is None:
            return original_select(self, req, **kwargs)
        admission: Any = None
        error = None
        token = _ACTIVE.set(None)
        try:
            admission = original_select(self, req, **kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            _ACTIVE.reset(token)
        if runtime is not None and runtime.planned and (not isinstance(admission, AddReqResult)):
            row = runtime.row(req)
            if error is None and (
                row is None
                or (
                    not row.get("host_load", False)
                    and (
                        admission.prefix_len != row["context"]
                        or admission.extend_len > row["tokens"]
                    )
                )
            ):
                admission = AddReqResult.OTHER
        value = (
            None
            if error
            else {"reject": admission.name}
            if isinstance(admission, AddReqResult)
            else {"shape": admission_shape(admission)}
        )
        values = runtime.exchange("admission_select", value, error=error)
        if any((v != values[0] for v in values[1:])):
            runtime.report_restriction("admission_select", values, "reject_prefill")
        if any((v.get("reject") == "NO_TOKEN" for v in values)):
            return AddReqResult.NO_TOKEN
        if any(("reject" in v for v in values)) or any((v != values[0] for v in values[1:])):
            return AddReqResult.OTHER
        return admission

    @functools.wraps(original_commit)
    def commit(self, req, admission, *args, **kwargs):
        runtime = _ACTIVE.get()
        if runtime is not None:
            materialization = runtime.admission_materialization
            if materialization is not None and materialization["required"]:
                if not runtime.materialized(req, ready=True, context=len(req.prefix_indices)):
                    raise _PrefillDeferred()
            runtime.agree(
                "admission_commit",
                {
                    "rid": str(req.rid),
                    "prefix": len(req.prefix_indices),
                    "shape": admission_shape(admission),
                },
            )
        result = original_commit(self, req, admission, *args, **kwargs)
        if runtime is not None and runtime.prefill_attempt is not None:
            if hasattr(req, "lock_receipt"):
                runtime.prefill_attempt.record_admission(req)
        return result

    @functools.wraps(original_chunk)
    def chunk(self, req):
        runtime = _ACTIVE.get()
        if runtime is None:
            return original_chunk(self, req)
        runtime.agree(
            "chunk_input",
            {
                "rid": str(req.rid),
                "prefix": len(req.prefix_indices),
                "input": len(req.full_untruncated_fill_ids),
                "max_new": req.sampling_params.max_new_tokens,
            },
        )
        if runtime.planned and runtime.row(req) is None:
            req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices))
            return req
        result, error = (None, None)
        try:
            result = original_chunk(self, req)
        except RankConsensusError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        runtime.agree(
            "chunk_result",
            {
                "pending": result is not None,
                "start": req.extend_range.start,
                "tokens": req.extend_range.length,
                "admitted": [str(r.rid) for r in self.can_run_list],
            },
            error=error,
        )
        return result

    @functools.wraps(original_check)
    def check(self, selected_indices=None):
        runtime = _ACTIVE.get()
        if runtime is None:
            return original_check(self, selected_indices=selected_indices)
        local, error = (False, None)
        try:
            local = original_check(self, selected_indices=selected_indices)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        indices = range(len(self.reqs)) if selected_indices is None else selected_indices
        ids = [str(self.reqs[i].rid) for i in indices]
        states = runtime.exchange(
            "decode_capacity", {"ids": ids, "can_run": bool(local)}, error=error
        )
        if any((s["ids"] != states[0]["ids"] for s in states[1:])):
            raise RankConsensusError(f"decode retraction membership differs: {states}")
        accepted = all((s["can_run"] for s in states))
        if not accepted:
            rollback_error = None
            try:
                release = getattr(getattr(self, "token_to_kv_pool_allocator", None),
                                  "_kvcached_release_decode_reservation", None)
                if release is not None:
                    release()
            except Exception as exc:
                rollback_error = f"{type(exc).__name__}: {exc}"
            runtime.exchange("decode_capacity.release", True, error=rollback_error)
        return accepted

    def wrap_prepare(original, name):

        @functools.wraps(original)
        def prepare(self, *args, **kwargs):
            runtime = _ACTIVE.get()
            if runtime is None:
                return original(self, *args, **kwargs)
            transaction = None
            if name == "prepare_extend" and runtime.prefill_attempt is not None:
                setup_error = None
                try:
                    transaction = PrefillAllocationTransaction.create(self)
                except Exception as exc:
                    setup_error = f"{type(exc).__name__}: {exc}"
                runtime.agree(
                    name + ".enter",
                    {
                        "ids": [str(r.rid) for r in self.reqs],
                        "transaction": transaction is not None,
                    },
                    error=setup_error,
                )
                if transaction is not None:
                    error, miss, result = (None, None, None)
                    try:
                        with transaction:
                            result = original(self, *args, **kwargs)
                    except PrefillAllocationMiss as exc:
                        miss = str(exc)
                    except RankConsensusError:
                        raise
                    except Exception as exc:
                        LOG.exception(
                            "[kvcached-rank-consensus] local %s failed before forward; reqs=%s",
                            name,
                            [str(req.rid) for req in self.reqs],
                        )
                        error = f"{type(exc).__name__}: {exc}"
                    states = runtime.exchange(
                        name + ".ready", {"ready": miss is None, "miss": miss}, error=error
                    )
                    if all((state["ready"] for state in states)):
                        return result
                    rollback_error = None
                    try:
                        transaction.rollback()
                    except Exception as exc:
                        rollback_error = f"{type(exc).__name__}: {exc}"
                    runtime.exchange(name + ".allocation_rollback", True, error=rollback_error)
                    runtime.defer_prefill(name + ".allocation", states)
                    raise PrefillPrepareDeferred()
            else:
                runtime.agree(name + ".enter", [str(r.rid) for r in self.reqs])
            error, result = (None, None)
            try:
                result = original(self, *args, **kwargs)
            except RankConsensusError:
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            runtime.exchange(name + ".ready", True, error=error)
            return result

        return prepare

    @functools.wraps(original_run)
    def run(self, batch, *args, **kwargs):
        runtime = _runtime(self)
        if runtime.size > 1:
            runtime.selected(batch, stage="forward")
        return original_run(self, batch, *args, **kwargs)

    Scheduler.get_next_batch_to_run = next_batch
    Scheduler.get_new_batch_prefill = new
    Scheduler.get_num_allocatable_reqs = slots
    PrefillAdder.add_one_req = add
    PrefillAdder.add_chunked_req = chunk
    PrefillAdder._select_prefill_admission = select
    PrefillAdder._commit_prefill_admission = commit
    SchedulePolicy.calc_priority = sort
    GrammarManager.get_ready_grammar_requests = grammar
    ScheduleBatch.check_decode_mem = check
    ScheduleBatch.prepare_for_extend = wrap_prepare(
        ScheduleBatch.prepare_for_extend, "prepare_extend"
    )
    ScheduleBatch.prepare_for_decode = wrap_prepare(
        ScheduleBatch.prepare_for_decode, "prepare_decode"
    )
    Scheduler.run_batch = run
    from .pipeline import patch_pipeline_progress

    patch_pipeline_progress(Scheduler)
    _PATCHED = True
    return ["rank-coordination"]
