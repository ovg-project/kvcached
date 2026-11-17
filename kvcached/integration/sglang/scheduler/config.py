# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Opt-in topology contract for rank-local DSV4 backing."""

import os


def enabled() -> bool:
    return os.getenv("KVCACHED_RANK_COORDINATION_ENABLE", "0").lower() in {"1", "true", "yes", "on"}


def validate_runtime_config() -> None:
    """Reject unsupported serving modes before creating managed KV storage."""
    from sglang.srt.runtime_context import (
        get_disagg,
        get_exec,
        get_lora,
        get_memory,
        get_parallel,
        get_schedule,
        get_spec,
    )

    schedule, parallel = get_schedule(), get_parallel()
    unsupported = (
        parallel.pp_size != 1
        or parallel.enable_dp_attention
        or get_spec().speculative_algorithm is not None
        or get_disagg().disaggregation_mode != "null"
        or get_exec().dllm.dllm_algorithm is not None
        or get_lora().enable_lora
        or get_memory().hicache_storage_backend is not None
        or get_memory().enable_hisparse
        or schedule.enable_mixed_chunk
        or schedule.enable_dynamic_chunking
        or (schedule.enable_priority_scheduling and not schedule.disable_priority_preemption)
    )
    if unsupported:
        raise ValueError(
            "KVCached rank coordination requires PP=1, attention-DP disabled, "
            "non-speculative text generation, fixed separate prefill/decode batches, "
            "and no preemption, LoRA, storage-tier cache or disaggregation. "
            "Use those settings, or unset KVCACHED_RANK_COORDINATION_ENABLE "
            "to retain native multi-rank DSV4 storage."
        )
