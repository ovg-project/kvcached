#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Offline 0.29 MRV2 GPT-OSS/MLA native and elastic validation.

FP16 eager, single GPU, no TP/PP/MPS, quantization, model-quality or performance
qualification. MLA uses dense FFNs to isolate the pre-V4 attention path.
"""
import hashlib
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path

import engine_compat_gpu_probe as probe
from engine_compat_sharing_probe import collect_outputs, validate_diversity


@contextmanager
def reject_two_allocations(manager_class):
    """Exercise recoverable scheduler admission, not distributed VMM rollback."""
    original = manager_class.alloc
    counts = {"injected": 0, "subsequent_successes": 0}

    def allocate(self, need_size):
        if need_size > 0 and counts["injected"] < 2:
            counts["injected"] += 1
            print(f"family_admission_miss hit={counts['injected']}", flush=True)
            return None
        value = original(self, need_size)
        if need_size > 0 and value is not None and counts["injected"] == 2:
            counts["subsequent_successes"] += 1
        return value

    manager_class.alloc = allocate
    try:
        yield counts
    finally:
        manager_class.alloc = original


def worker(family, args, result):
    if family not in ("gptoss", "mla"):
        raise probe.Blocked("Unknown fixed model family")
    if args.version != "0.29.0" or args.runner != "v2" or args.model is not None:
        raise probe.Blocked("Requires fixed tiny fixture on 0.29.0 MRV2")
    os.environ["VLLM_KV_CACHE_LAYOUT"] = (
        "BLNHC" if args.layout == "contiguous" else "LBNHC")
    probe.version(args.version, result)
    if args._stage == "oom":
        return probe.worker(args, result)
    import torch
    if args._stage == "prepare":
        from transformers import (
            DeepseekV2Config,
            DeepseekV2ForCausalLM,
            GptOssConfig,
            GptOssForCausalLM,
        )
        if family == "gptoss":
            config = GptOssConfig(
                vocab_size=256, hidden_size=128, intermediate_size=256,
                num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                head_dim=64, num_local_experts=4, num_experts_per_tok=2,
                sliding_window=64, max_position_embeddings=512,
                layer_types=["sliding_attention", "full_attention"] * 2,
                architectures=["GptOssForCausalLM"],
            )
            cls = GptOssForCausalLM
        else:
            config = DeepseekV2Config(
                vocab_size=256, hidden_size=128, intermediate_size=256,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                kv_lora_rank=512, q_lora_rank=128, qk_nope_head_dim=128,
                qk_rope_head_dim=64, v_head_dim=128, n_routed_experts=4,
                n_shared_experts=1, num_experts_per_tok=2, first_k_dense_replace=2,
                max_position_embeddings=512,
                architectures=["DeepseekV2ForCausalLM"],
            )
            cls = DeepseekV2ForCausalLM
        torch.manual_seed(20260926)
        model = cls(config).to(device="cpu", dtype=torch.float16)
        model.save_pretrained(args.output / "model", safe_serialization=True)
        probe.check(not torch.cuda.is_initialized(), "CPU fixture initialized CUDA")
        return
    probe.check(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                "Requires one visible CUDA GPU")
    torch.cuda.set_device(0)
    patched = args._stage == "patched"
    if patched:
        import kvcached.integration.vllm.autopatch  # noqa: F401
        from kvcached.integration.vllm import patches
        probe.check(Path(patches.__file__).resolve().is_relative_to(args.source),
                    "Wrong candidate source")
    import vllm
    from vllm.v1.worker.gpu_worker import Worker
    active = bool(getattr(Worker.determine_available_memory,
                          "determine_available_memory", False))
    probe.check(active == patched, "Wrong patch activation")
    options = dict(
        model=str(args.output / "model"), load_format="safetensors",
        skip_tokenizer_init=True, trust_remote_code=False,
        distributed_executor_backend="uni", tensor_parallel_size=1,
        pipeline_parallel_size=1, dtype="float16", max_model_len=512,
        max_num_seqs=4, max_num_batched_tokens=256, gpu_memory_utilization=0.4,
        enforce_eager=True, enable_prefix_caching=True, async_scheduling=False,
        attention_config={"backend": "TRITON_ATTN" if family == "gptoss" else "TRITON_MLA"},
        seed=20260926,
    )
    result.update(family=family, options=options, worker_patch_active=active,
                  gpu=torch.cuda.get_device_name(0), torch_version=str(torch.__version__),
                  cuda_version=torch.version.cuda,
                  config_sha256=hashlib.sha256((args.output / "model/config.json").read_bytes()).hexdigest(),
                  weights_sha256=hashlib.sha256((args.output / "model/model.safetensors").read_bytes()).hexdigest())
    llm = None
    try:
        llm = vllm.LLM(**options)
        result["loaded_runner"] = probe.runner_identity(llm, args.runner)
        runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        result["resolved_layout"] = runner.vllm_config.cache_config.kv_cache_layout
        probe.check(result["resolved_layout"] == os.environ["VLLM_KV_CACHE_LAYOUT"],
                    "Wrong native layout")
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
        specs = []
        for group in runner.kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            specs.extend(spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else [spec])
        result["specs"] = [dict(type=type(s).__name__, block_size=s.block_size,
                                page_size_bytes=s.page_size_bytes) for s in specs]
        expected = {"FullAttentionSpec", "SlidingWindowSpec"} if family == "gptoss" else {"MLAAttentionSpec"}
        probe.check({s["type"] for s in result["specs"]} == expected,
                    "Wrong attention cache types")
        views = []
        for name, layer in runner.compilation_config.static_forward_context.items():
            cache = getattr(layer, "kv_cache", None)
            if torch.is_tensor(cache):
                views.append(dict(layer=name, shape=list(cache.shape), stride=list(cache.stride()),
                                  dtype=str(cache.dtype), backend=layer.attn_backend.get_name()))
        result["views"] = views
        probe.check(len(views) == (4 if family == "gptoss" else 2), "Missing layer views")
        probe.check(all(v["backend"] == options["attention_config"]["backend"]
                        and v["dtype"] == "torch.float16" for v in views),
                    "Wrong attention backend or cache dtype")
        if family == "mla":
            probe.check(all(v["shape"][-1] == 576 for v in views), "Lost MLA latent/RoPE geometry")
        prompts = [dict(prompt_token_ids=[token] * n)
                   for n in (31, 64, 65, 127, 128, 129, 383) for token in (17, 23)]
        sampling = vllm.SamplingParams(temperature=0, max_tokens=32, ignore_eos=True,
                                      detokenize=False, logprobs=5)
        result["prompt_token_ids"], result["token_ids"] = [], []
        for round_id in range(3):
            if round_id == 2:
                probe.check(llm.reset_prefix_cache(), "Prefix reset rejected")
                result["prefix_reset_completed"] = True
            if patched and round_id == 2:
                from kvcached.kv_cache_manager import KVCacheManager

                fault = reject_two_allocations(KVCacheManager)
            else:
                fault = nullcontext({})
            with fault as counts:
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
            if patched and round_id == 2:
                result["admission_fault"] = counts
                probe.check(counts["injected"] == 2 and counts["subsequent_successes"] > 0,
                            "Admission failure was not hit and recovered")
            result["token_ids"].extend(collect_outputs(prompts, outputs))
            result["prompt_token_ids"].extend(p["prompt_token_ids"] for p in prompts)
        result["requests"] = len(result["token_ids"])
        result["output_tokens"] = sum(map(len, result["token_ids"]))
        probe.check(result["requests"] == 42 and result["output_tokens"] == 1344, "Incomplete output")
        validate_diversity(result["token_ids"])
        if patched:
            from kvcached.observability import get_registered_kv_cache_pool_snapshot_dicts
            result["pools"] = get_registered_kv_cache_pool_snapshot_dicts(integration="vllm")
            probe.check(any(p["virtual_total_bytes"] > p["mapped_bytes"] > 0 for p in result["pools"]),
                        "No physical KV mapping exercised")
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
            if patched:
                from kvcached.integration.vllm.interfaces import shutdown_kvcached
                shutdown_kvcached()
            result["explicit_shutdown_completed"] = True
