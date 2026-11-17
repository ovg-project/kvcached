# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Policy-free, CPU-only plans for ordinary SGLang 0.5.20 scheduling.

Previews do not reserve physical memory: the caller must also compare actual
batches before model forward.
"""

from __future__ import annotations

import copy
from contextlib import ExitStack
from typing import Any


def request_row(req, mode, *, start=None, tokens=None):
    if mode == "P":
        start = int(req.extend_range.start if start is None else start)
        tokens = int(req.extend_range.length if tokens is None else tokens)
    else:
        start = int(
            (req.kv_committed_len if hasattr(req, "kv_committed_len") else req.kv.kv_committed_len)
            if start is None
            else start
        )
        tokens = 0
    return {"rid": str(req.rid), "context": start, "tokens": tokens}


def plan(mode, rows):
    return {"mode": mode, "rows": rows} if rows else None


def common_plan(plans, mode):
    """P admits only the common ordered prefix; D requires the whole batch."""
    if not plans or any((p is None for p in plans)):
        return None
    rows = []
    for candidates in zip(*(p["rows"] for p in plans)):
        logical = [
            {key: value for key, value in row.items() if key != "host_load"} for row in candidates
        ]
        if any((row != logical[0] for row in logical[1:])):
            break
        row = dict(logical[0])
        if any(("host_load" in candidate for candidate in candidates)):
            row["host_load"] = any(
                (bool(candidate.get("host_load", False)) for candidate in candidates)
            )
        rows.append(row)
    if mode == "D" and any((len(rows) != len(p["rows"]) for p in plans)):
        return None
    return plan(mode, rows)


def _cpu_list(value):
    if value is None:
        return None
    if hasattr(value, "device") and value.device.type != "cpu":
        raise ValueError("rank coordination must not materialize a GPU tensor")
    return value.tolist() if hasattr(value, "tolist") else list(value)


def batch_signature(batch):
    """Full ordered CPU shape, including the overlap scheduler's actual cursor."""
    if batch is None or not batch.reqs:
        return None
    mode = batch.forward_mode
    if not (mode.is_decode() or mode.is_extend()):
        raise ValueError(f"rank coordination does not support batch mode {mode}")
    mode_name = getattr(mode, "name", "D" if mode.is_decode() else "P")
    return {
        "mode": mode_name,
        "ids": [str(r.rid) for r in batch.reqs],
        "seq_lens": _cpu_list(getattr(batch, "seq_lens_cpu", None)),
        "prefix_lens": _cpu_list(getattr(batch, "prefix_lens", None)) if mode.is_extend() else None,
        "extend_lens": _cpu_list(getattr(batch, "extend_lens", None)) if mode.is_extend() else None,
        "extend_tokens": getattr(batch, "extend_num_tokens", None) if mode.is_extend() else None,
        "extend_logprob_start_lens": _cpu_list(getattr(batch, "extend_logprob_start_lens", None))
        if mode.is_extend()
        else None,
        "return_logprob": bool(getattr(batch, "return_logprob", False)),
        "return_hidden_states": bool(getattr(batch, "return_hidden_states", False)),
        "hidden_mode": getattr(getattr(batch, "return_hidden_states_mode", None), "name", None),
        "prefill_only": bool(getattr(batch, "is_prefill_only", False)),
        "input_embeds": getattr(batch, "input_embeds", None) is not None,
        "rows": [request_row(r, "P" if mode.is_extend() else "D") for r in batch.reqs],
    }


def _clone_request(req):
    clone = copy.copy(req)
    clone.kv = copy.copy(req.kv)
    clone.full_untruncated_fill_ids = req.full_untruncated_fill_ids[:]
    return clone


def request_slot(req):
    """0.5.18 owns rows on Req; 0.5.20 owns them on ReqKvInfo."""
    if hasattr(req, "req_pool_idx"):
        return req.req_pool_idx
    return getattr(getattr(req, "kv", None), "req_pool_idx", None)


def match_preview_requests(scheduler, running_batch):
    """Match the whole ordered queue before any rank-local resource gate.

    Native match_prefix is NOT a read-only query: it refreshes Full/SWA LRU
    and may split radix nodes. Matching only the locally affordable candidates
    makes future host eviction choose different victims on different ranks.
    Match private requests in a capacity-independent order; the actual budget
    preview below only consumes these snapshots and temporarily pins nodes.
    """
    matched: dict[str, Any] = {}
    limit = max(0, scheduler.max_running_requests - len(running_batch.reqs))
    for original in scheduler.waiting_queue:
        if len(matched) >= limit:
            break
        if original.finished():
            continue
        req = _clone_request(original)
        req.init_next_round_input(scheduler.tree_cache, cow_mamba=False)
        matched[str(original.rid)] = req
    return matched


def preview(scheduler, running_batch, *, slots=None, new_slots=None, match_cache=None):
    """Preview native resource gates on private requests in native queue order.

    Radix matching runs in a fixed queue order before local gates, since it
    updates eviction metadata. Afterwards only temporary radix pins are taken.
    No ScheduleBatch preparation, H2D,
    request-slot allocation, eviction or speculative GPU execution is allowed.
    Unsupported admission modes fail closed instead of silently going native.
    """
    from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS, AddReqResult, PrefillAdder
    from sglang.srt.runtime_context import get_schedule

    running = [r for r in running_batch.reqs if not r.finished() and (not r.is_retracted)]
    queue = scheduler.waiting_queue
    chunk = scheduler.chunked_req
    requests = [*running, *queue, *([chunk] if chunk is not None else [])]
    if any(
        (
            getattr(r, "beam_group", None) is not None
            or getattr(r, "multimodal_inputs", None) is not None
            for r in requests
        )
    ):
        raise ValueError("rank coordination supports ordinary text requests, not beam/multimodal")
    if (
        getattr(scheduler, "enable_lora", False)
        or getattr(scheduler, "enable_priority_preemption", False)
        or scheduler.tree_cache.supports_mamba()
        or getattr(scheduler, "enable_hicache_storage", False)
        or (getattr(scheduler, "min_free_slots_delayer", None) is not None)
        or (getattr(scheduler, "prefill_delayer", None) is not None)
    ):
        raise ValueError("unsupported rank-coordinated admission configuration")
    if match_cache is None:
        match_cache = {}
    if "requests" not in match_cache:
        match_cache["requests"] = match_preview_requests(scheduler, running_batch)
    matched = match_cache["requests"]
    result = {"p": None, "d": None}
    if running and (not running_batch.is_prefill_only):
        indices = [i for i, r in enumerate(running_batch.reqs) if r in running]
        required = running_batch.new_tokens_required_next_decode(indices)
        if required <= scheduler.token_to_kv_pool_allocator.available_size():
            result["d"] = plan("D", [request_row(r, "D") for r in running])
    full = running_batch.batch_is_full and (not getattr(scheduler, "is_hybrid_swa", False))
    if (full or not queue) and chunk is None:
        return result
    running_bs = len(running_batch.reqs)
    if slots is None:
        slots = scheduler.get_num_allocatable_reqs(running_bs, **{"running_batch": running_batch})
    if slots <= 0 and chunk is None:
        return result
    backend = scheduler.tp_worker.model_runner.attn_backend
    adder = PrefillAdder(
        scheduler.page_size,
        scheduler.tree_cache,
        scheduler.token_to_kv_pool_allocator,
        running_batch,
        scheduler.new_token_ratio_tracker.current,
        scheduler.max_prefill_tokens,
        scheduler.chunked_prefill_size,
        max_prefill_bs=int(scheduler.max_prefill_bs),
        max_running_requests=scheduler.max_running_requests,
        prefill_max_requests=get_schedule().prefill_max_requests,
        waiting_queue_len=len(queue),
        prefill_tile_block_m=getattr(backend, "extend_attention_block_m", 64),
    )
    rows = []
    used_new_slots = 0
    with ExitStack() as pins:
        if chunk is not None:
            if new_slots is not None and new_slots <= 0 and (request_slot(chunk) is None):
                return result
            shadow = _clone_request(chunk)
            shadow.init_next_round_input()
            adder.add_chunked_req(shadow)
            if adder.can_run_list:
                rows.append(request_row(shadow, "P"))
                used_new_slots += request_slot(chunk) is None
        for original in queue:
            if len(adder.can_run_list) >= slots:
                break
            if original.finished():
                continue
            if False and adder.dsa_prefill_cp_in_seq_split and adder.can_run_list:
                break
            needs_row = request_slot(original) is None
            if new_slots is not None and needs_row and (used_new_slots >= new_slots):
                break
            req = _clone_request(matched[str(original.rid)])
            if req.sampling_params.ignore_eos and getattr(scheduler.tree_cache, "disable", True):
                raise ValueError(
                    "rank coordination does not support ignore_eos without radix cache"
                )
            if (
                adder.prefill_max_requests is not None
                and len(adder.can_run_list) >= adder.prefill_max_requests
            ):
                break
            max_new = min(
                max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
                CLIP_MAX_NEW_TOKENS,
            )
            gap = adder._mamba_gap_budget_for_req(req)
            total = (
                len(req.full_untruncated_fill_ids)
                - len(req.prefix_indices)
                + max_new
                + adder.page_size
                + gap
            )
            if total >= adder.rem_total_tokens:
                break
            pins.enter_context(adder._lock_node(req.last_node))
            host = max(int(req.host_hit_length), 0)
            swa_host = max(int(getattr(req, "swa_host_hit_length", 0)), 0)
            admission = adder._select_prefill_admission(
                req,
                total_tokens=total,
                host_hit_length=host,
                swa_host_hit_length=swa_host,
                truncation_align_size=scheduler.truncation_align_size,
            )
            if isinstance(admission, AddReqResult):
                break
            req.set_extend_range(admission.prefix_len, admission.prefix_len + admission.extend_len)
            adder.can_run_list.append(req)
            adder._update_prefill_budget(
                admission.prefix_len,
                admission.extend_len,
                admission.max_new_tokens,
                req.retracted_stain,
                mamba_gap_reserve=gap,
                **{"compute_charge": admission.extend_len if adder.exact_chunk_fill else None},
            )
            adder.rem_total_token_offset += adder.ceil_paged_tokens(host)
            adder.cur_rem_token_offset += adder.ceil_paged_tokens(host)
            if adder.is_hybrid_swa and (not getattr(adder, "_swa_req_ring", False)):
                adder.rem_swa_token_offset += adder.ceil_paged_tokens(swa_host)
            row = request_row(req, "P")
            row["host_load"] = bool(req.needs_host_load_back())
            rows.append(row)
            used_new_slots += needs_row
            if adder.budget_state() != AddReqResult.CONTINUE:
                break
    result["p"] = plan("P", rows)
    return result
