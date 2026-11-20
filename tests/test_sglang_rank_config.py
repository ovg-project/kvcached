# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Unsupported runtime settings reject before backing constructors are called."""

import importlib
import sys
from types import SimpleNamespace as NS

import pytest
import test_sglang_rank_coordination as consensus

config = importlib.import_module(consensus.package.__name__ + ".config")


@pytest.fixture
def runtime(monkeypatch):
    values = dict(
        parallel=NS(pp_size=1, enable_dp_attention=False),
        spec=NS(speculative_algorithm=None),
        disagg=NS(disaggregation_mode="null"),
        exec=NS(dllm=NS(dllm_algorithm=None)),
        lora=NS(enable_lora=False),
        memory=NS(hicache_storage_backend=None, enable_hisparse=False),
        schedule=NS(
            enable_mixed_chunk=False,
            enable_dynamic_chunking=False,
            enable_priority_scheduling=False,
            disable_priority_preemption=True,
            enable_prefill_delayer=False,
            min_free_slots_delay=None,
            max_running_requests=None,
        ),
    )
    module = NS(**{f"get_{name}": lambda value=value: value for name, value in values.items()})
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", module)
    return values


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("parallel", "pp_size", 2),
        ("parallel", "enable_dp_attention", True),
        ("spec", "speculative_algorithm", "EAGLE"),
        ("disagg", "disaggregation_mode", "prefill"),
        ("lora", "enable_lora", True),
        ("memory", "hicache_storage_backend", "configured"),
        ("memory", "enable_hisparse", True),
        ("schedule", "enable_mixed_chunk", True),
        ("schedule", "enable_dynamic_chunking", True),
        ("schedule", "enable_prefill_delayer", True),
        ("schedule", "min_free_slots_delay", 2),
    ],
)
def test_unsupported_config_has_native_fallback_instruction(runtime, section, field, value):
    setattr(runtime[section], field, value)
    with pytest.raises(ValueError, match="unset KVCACHED_RANK_COORDINATION_ENABLE"):
        config.validate_runtime_config()


def test_supported_configuration(runtime):
    config.validate_runtime_config()


@pytest.mark.parametrize("delay,limit", [(None, None), (0, None), (1, None), (8, 1)])
def test_disabled_slot_delay_preserves_supported_configuration(runtime, delay, limit):
    runtime["schedule"].min_free_slots_delay = delay
    runtime["schedule"].max_running_requests = limit
    config.validate_runtime_config()


@pytest.mark.parametrize("section,field,value", [
    ("parallel", "enable_dp_attention", True),
    ("schedule", "enable_prefill_delayer", True),
    ("schedule", "min_free_slots_delay", 2),
])
def test_unsupported_opt_in_rejects_before_managed_pool_construction(
    runtime, monkeypatch, section, field, value
):
    from test_sglang_dsv4_kv_pool_patch import _load_modules, _make_dsv4_module

    interfaces, patches, _ = _load_modules(monkeypatch)
    monkeypatch.setenv("KVCACHED_RANK_COORDINATION_ENABLE", "true")
    distributed = sys.modules["sglang.srt.distributed"]
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: 4)
    setattr(runtime[section], field, value)
    monkeypatch.setattr(interfaces, "alloc_dsv4_swa_cache",
                        lambda **kwargs: pytest.fail("allocated before configuration rejection"))
    module = _make_dsv4_module()
    assert patches.DeepSeekV4KVPoolPatch().inject_dsv4_swa_pool(module)
    with pytest.raises(ValueError, match="unset KVCACHED_RANK_COORDINATION_ENABLE"):
        module.DeepSeekV4TokenToKVPool()


def test_enable_flag_does_not_override_disabled_library(monkeypatch):
    monkeypatch.setenv("KVCACHED_RANK_COORDINATION_ENABLE", "true")
    monkeypatch.setenv("ENABLE_KVCACHED", "false")
    monkeypatch.setattr(consensus.coord, "_PATCHED", False)
    assert consensus.coord.patch_rank_coordination() == []
