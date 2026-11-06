#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Offline tiny Gemma4 cross-layer KV-sharing acceptance, vLLM 0.29 MRV2 only.

Both layouts require a matched native control, two actual borrower aliases,
heterogeneous full/sliding attention, repeated prefix requests, exact native /
elastic token equality and explicit shutdown. The shared supervisor also runs
an oversized CUDA allocation/recovery probe. A native failure is not evidence
of a KVCached regression and does not waive the failing layout.
No full-checkpoint quality, FP8, multimodal, async scheduling, TP/PP, MPS,
distributed transaction chaos or performance qualification. No downloads.
"""

import hashlib
import math
import os
from pathlib import Path

import engine_compat_gpu_probe as probe

LAYOUTS = {"contiguous": "BLNHC", "non-contiguous": "LBNHC"}
LENGTHS = (31, 64, 65, 127, 128, 129, 383)


def validate_alias(left, right):
    probe.check(left.data_ptr() == right.data_ptr(), "Borrower lost its owner's storage")
    probe.check(left.shape == right.shape and left.stride() == right.stride()
                and left.dtype == right.dtype,
                "Borrower view differs from owner")


def collect_outputs(prompts, outputs):
    probe.check(len(outputs) == len(prompts), "Missing requests")
    tokens = []
    for prompt, output in zip(prompts, outputs):
        probe.check(list(output.prompt_token_ids) == prompt["prompt_token_ids"],
                    "Prompt ordering/IDs changed")
        probe.check(output.finished and len(output.outputs) == 1, "Incomplete request")
        completion = output.outputs[0]
        probe.check(len(completion.token_ids) == 32 and completion.finish_reason == "length",
                    "Truncated generation")
        probe.check(completion.logprobs is not None and len(completion.logprobs) == 32,
                    "Missing generation logprobs")
        probe.check(all(step and all(math.isfinite(value.logprob) for value in step.values())
                        for step in completion.logprobs), "Non-finite generation logprobs")
        tokens.append(list(completion.token_ids))
    return tokens


def validate_diversity(token_ids):
    probe.check(len({token for row in token_ids for token in row}) > 1,
                "Degenerate fixture: every completion has the same token")


def fixture_state_dict(model):
    state = model.state_dict()
    # vLLM 0.29 constructs k_norm on borrowers but never applies it. HF omits
    # those unused weights; supply them in this fixed fixture, not in the loader.
    for borrower, owner in ((2, 0), (3, 1)):
        key = f"model.layers.{borrower}.self_attn.k_norm.weight"
        if key not in state:
            state[key] = state[f"model.layers.{owner}.self_attn.k_norm.weight"].clone()
    return state


def worker(args, result):
    if args.version != "0.29.0" or args.runner != "v2" or args.model is not None:
        raise probe.Blocked("Sharing probe requires vLLM 0.29.0 MRV2 and its fixed tiny fixture")
    # Pin the native control as well; its default layout need not match elastic.
    os.environ["VLLM_KV_CACHE_LAYOUT"] = LAYOUTS[args.layout]
    probe.version(args.version, result)
    if args._stage == "oom":
        return probe.worker(args, result)

    import torch

    if args._stage == "prepare":
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig

        config = Gemma4TextConfig(
            vocab_size=256, hidden_size=128, intermediate_size=256,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
            head_dim=64, global_head_dim=128, num_global_key_value_heads=1,
            layer_types=["sliding_attention", "full_attention"] * 2,
            num_kv_shared_layers=2, sliding_window=64,
            max_position_embeddings=512, hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=256, enable_moe_block=False,
            architectures=["Gemma4ForCausalLM"],
        )
        torch.manual_seed(20260925)
        model = Gemma4ForCausalLM(config).to(device="cpu", dtype=torch.float16)
        model.save_pretrained(args.output / "model", safe_serialization=True,
                              state_dict=fixture_state_dict(model))
        probe.check(not torch.cuda.is_initialized(), "Config generation initialized CUDA")
        return

    if not torch.version.cuda or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise probe.Blocked("Exactly one visible CUDA GPU is required")
    torch.cuda.set_device(0)
    patched = args._stage == "patched"
    if patched:
        import kvcached.integration.vllm.autopatch  # noqa: F401
        from kvcached.integration.vllm import patches

        probe.check(Path(patches.__file__).resolve().is_relative_to(args.source),
                    "Patch module is outside candidate checkout")

    import vllm
    from packaging.version import Version
    from vllm.v1.worker.gpu_worker import Worker

    probe.check(Version(vllm.__version__).public == args.version, "Imported vLLM version differs")
    active = bool(getattr(Worker.determine_available_memory, "determine_available_memory", False))
    probe.check(active == patched, "Wrong patch activation")
    options = dict(
        model=str(args.output / "model"), load_format="safetensors", skip_tokenizer_init=True,
        trust_remote_code=False, distributed_executor_backend="uni", tensor_parallel_size=1,
        pipeline_parallel_size=1, dtype="float16", max_model_len=512,
        max_num_seqs=4, max_num_batched_tokens=256, gpu_memory_utilization=0.4,
        enforce_eager=True, enable_prefix_caching=True, async_scheduling=False,
        attention_config={"backend": "TRITON_ATTN"}, seed=20260925,
    )
    result.update(options=options, worker_patch_active=active, gpu=torch.cuda.get_device_name(0),
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  model_config_sha256=hashlib.sha256(
                      (args.output / "model/config.json").read_bytes()).hexdigest(),
                  model_weights_sha256=hashlib.sha256(
                      (args.output / "model/model.safetensors").read_bytes()).hexdigest())
    llm = None
    try:
        llm = vllm.LLM(**options)
        result["loaded_runner"] = probe.runner_identity(llm, args.runner)
        runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        result["resolved_layout"] = runner.vllm_config.cache_config.kv_cache_layout
        probe.check(result["resolved_layout"] == LAYOUTS[args.layout], "Wrong resolved native layout")
        if patched:
            from kvcached.integration.vllm import interfaces

            probe.check(interfaces._contiguous_layout == (args.layout == "contiguous"),
                        "Wrong elastic layout")
        context = runner.compilation_config.static_forward_context
        aliases = []
        for name, layer in context.items():
            target = getattr(layer, "kv_sharing_target_layer_name", None)
            if target is None:
                continue
            left, right = layer.kv_cache, context[target].kv_cache
            probe.check(torch.is_tensor(left) and torch.is_tensor(right), "Expected bound tensor")
            validate_alias(left, right)
            aliases.append(dict(layer=name, target=target, shape=list(left.shape),
                                stride=list(left.stride())))
        result["aliases"] = aliases
        probe.check(len(aliases) == 2, "Expected two actual cross-layer borrowers")
        owners = {alias["target"] for alias in aliases}
        probe.check(len(owners) == 2, "Expected separate full/sliding owners")
        # UniformType wraps the owner specs in contiguous layout. Borrowers do
        # not allocate another physical pool or acquire their own spec entry.
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

        specs = []
        for group in runner.kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            specs.extend(spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs)
                         else [spec])
        result["spec_types"] = sorted({type(spec).__name__ for spec in specs})
        probe.check(set(result["spec_types"]) == {"FullAttentionSpec", "SlidingWindowSpec"},
                    "Missing heterogeneous full/sliding cache specs")
        prompts = [dict(prompt_token_ids=[token] * length)
                   for length in LENGTHS for token in (17, 23)]
        sampling = vllm.SamplingParams(temperature=0, max_tokens=32,
                                      ignore_eos=True, detokenize=False, logprobs=5)
        result["prompt_token_ids"], result["token_ids"] = [], []
        for _ in range(2):
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            result["token_ids"].extend(collect_outputs(prompts, outputs))
            result["prompt_token_ids"].extend(prompt["prompt_token_ids"] for prompt in prompts)
        result["requests"] = len(result["token_ids"])
        result["output_tokens"] = sum(map(len, result["token_ids"]))
        probe.check(result["requests"] == 28 and result["output_tokens"] == 896,
                    "Incomplete request/token totals")
        validate_diversity(result["token_ids"])
        if patched:
            from kvcached.observability import get_registered_kv_cache_pool_snapshot_dicts

            result["pools"] = get_registered_kv_cache_pool_snapshot_dicts(integration="vllm")
            probe.check(any(p["virtual_total_bytes"] > p["mapped_bytes"] > 0 for p in result["pools"]),
                        "No elastic physical mapping")
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
            if patched:
                from kvcached.integration.vllm.interfaces import shutdown_kvcached

                shutdown_kvcached()
            result["explicit_shutdown_completed"] = True


if __name__ == "__main__":
    raise SystemExit(probe.main(worker_fn=worker, script=__file__, description=__doc__))
