# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Engines, models and the case lists each CI profile runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class EngineSpec:
    name: str
    image: str
    # Log line listing the patches kvcached applied.
    patched_re: str
    # Patches that must be in that list for the integration to count as active.
    required_patches: tuple[str, ...]
    # Log line in which the engine reports the KV capacity it sized; with
    # kvcached it shows that kvcached sized the KV cache.
    capacity_re: str
    # Token capacity the server reports, kept in the results for reference.
    tokens_re: str
    env: tuple[tuple[str, str], ...] = ()


ENGINES = {
    "vllm": EngineSpec(
        name="vllm",
        image="vllm/vllm-openai:v0.30.0",
        patched_re=r"Successfully patched vllm: (.*)",
        required_patches=("elastic_block_pool", "model_runner_v2", "kv_layout_v2"),
        capacity_re=r"Using kvcached process-local KV capacity: .*available=(\d+) bytes",
        tokens_re=r"GPU KV cache size: ([\d,]+) tokens",
        # MRV2 explicitly, so an unsupported model fails instead of silently
        # falling back to the V1 runner. DEV_MODE enables /reset_prefix_cache.
        env=(("VLLM_USE_V2_MODEL_RUNNER", "1"), ("VLLM_SERVER_DEV_MODE", "1")),
    ),
    "sglang": EngineSpec(
        name="sglang",
        image="lmsysorg/sglang:v0.5.20",
        patched_re=r"Successfully patched sglang: (.*)",
        required_patches=("elastic_allocator", "elastic_memory_pool"),
        capacity_re=r"max_total_num_tokens=(\d+)",
        tokens_re=r"max_total_num_tokens=(\d+)",
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    # Per-engine server arguments (memory fraction, context length, batch size).
    args: dict[str, tuple[str, ...]]
    # Extra patches required on top of the engine's list, per engine.
    extra_patches: dict[str, tuple[str, ...]] = field(default_factory=dict)


MODELS = {
    "qwen05b": ModelSpec(
        "qwen05b", "Qwen/Qwen2.5-0.5B-Instruct",
        args={
            "vllm": ("--dtype", "bfloat16", "--max-model-len", "8192", "--max-num-seqs", "32",
                     "--gpu-memory-utilization", "0.8"),
            "sglang": ("--dtype", "bfloat16", "--context-length", "8192",
                       "--max-running-requests", "32", "--mem-fraction-static", "0.8"),
        },
    ),
    "qwen35_4b": ModelSpec(
        "qwen35_4b", "Qwen/Qwen3.5-4B",
        args={
            "vllm": ("--dtype", "bfloat16", "--max-model-len", "8192", "--max-num-seqs", "16",
                     "--gpu-memory-utilization", "0.9"),
            "sglang": ("--dtype", "bfloat16", "--context-length", "8192",
                       "--max-running-requests", "16", "--mem-fraction-static", "0.85"),
        },
        extra_patches={"vllm": ("mamba_partial_tail", "hybrid_block_size_align"),
                       "sglang": ("elastic_mamba_pool",)},
    ),
    "gemma4e2b": ModelSpec(
        "gemma4e2b", "google/gemma-4-E2B-it",
        args={
            "vllm": ("--dtype", "bfloat16", "--max-model-len", "8192", "--max-num-seqs", "16",
                     "--gpu-memory-utilization", "0.9"),
            "sglang": ("--dtype", "bfloat16", "--context-length", "8192",
                       "--max-running-requests", "16", "--mem-fraction-static", "0.85"),
        },
    ),
    # Weekly, with TP=2 across two L4s.
    "qwen35_9b": ModelSpec(
        "qwen35_9b", "Qwen/Qwen3.5-9B",
        args={
            "vllm": ("--dtype", "bfloat16", "--max-model-len", "8192", "--max-num-seqs", "16",
                     "--gpu-memory-utilization", "0.9"),
            "sglang": ("--dtype", "bfloat16", "--context-length", "8192",
                       "--max-running-requests", "16", "--mem-fraction-static", "0.85"),
        },
        extra_patches={"vllm": ("mamba_partial_tail", "hybrid_block_size_align"),
                       "sglang": ("elastic_mamba_pool",)},
    ),
    "dsv2lite": ModelSpec(
        "dsv2lite", "deepseek-ai/DeepSeek-V2-Lite-Chat",
        args={
            "vllm": ("--dtype", "bfloat16", "--max-model-len", "8192", "--max-num-seqs", "16",
                     "--gpu-memory-utilization", "0.9", "--trust-remote-code"),
            "sglang": ("--dtype", "bfloat16", "--context-length", "8192",
                       "--max-running-requests", "16", "--mem-fraction-static", "0.85",
                       "--trust-remote-code"),
        },
        extra_patches={"sglang": ("elastic_mla_memory_pool",)},
    ),
    # SGLang cannot serve the unified Gemma 4 12B natively.
    "gemma4_12b": ModelSpec(
        "gemma4_12b", "google/gemma-4-12B-it",
        args={
            "vllm": ("--dtype", "bfloat16", "--max-model-len", "8192", "--max-num-seqs", "16",
                     "--gpu-memory-utilization", "0.9"),
        },
    ),
    # Hopper only: too large for the L4 runs.
    "qwen38_27b": ModelSpec(
        "qwen38_27b", "Qwen/Qwen3.8-27B",
        args={
            "vllm": ("--dtype", "bfloat16", "--max-model-len", "8192", "--max-num-seqs", "16",
                     "--gpu-memory-utilization", "0.92"),
        },
        extra_patches={"vllm": ("mamba_partial_tail", "hybrid_block_size_align")},
    ),
}


TP_FLAG = {"vllm": "--tensor-parallel-size", "sglang": "--tp-size"}
BATCH_FLAG = {"vllm": "--max-num-seqs", "sglang": "--max-running-requests"}
ELASTIC_BATCH = 128
FP8_KV = {"vllm": ("--kv-cache-dtype", "fp8"), "sglang": ("--kv-cache-dtype", "fp8_e4m3")}


@dataclass(frozen=True)
class Case:
    engine: str
    model: str
    # "c1"/"c0" for kvcached with the contiguous / non-contiguous KV layout,
    # None for the native engine.
    layout: Optional[str]
    # Distinguishes servers of the same model in other tests (log directory).
    tag: str = ""
    tp: int = 1
    fp8_kv: bool = False

    @property
    def kvcached(self) -> bool:
        return self.layout is not None

    @property
    def name(self) -> str:
        name = f"{self.engine}-{self.model}-{'kv_' + self.layout if self.layout else 'native'}"
        name += f"-tp{self.tp}" if self.tp > 1 else ""
        name += "-fp8kv" if self.fp8_kv else ""
        return f"{self.tag}-{name}" if self.tag else name

    def args(self) -> tuple[str, ...]:
        """Server arguments for the model, the parallelism and the KV dtype."""
        args = MODELS[self.model].args[self.engine]
        if self.tp > 1:
            args += (TP_FLAG[self.engine], str(self.tp))
        if self.fp8_kv:
            args += FP8_KV[self.engine]
        return args


@dataclass(frozen=True)
class Group:
    """One model on one engine: kvcached in each layout, and the native engine."""
    engine: str
    model: str
    layouts: tuple[str, ...] = ("c1", "c0")
    tp: int = 1
    fp8_kv: bool = False

    def cases(self) -> list[Case]:
        """kvcached first: on vLLM the native case takes the attention block
        size kvcached chose for a hybrid model."""
        kv = [Case(self.engine, self.model, layout, tp=self.tp, fp8_kv=self.fp8_kv)
              for layout in self.layouts]
        return kv + [Case(self.engine, self.model, None, tp=self.tp, fp8_kv=self.fp8_kv)]


@dataclass(frozen=True)
class ElasticPair:
    """Two kvcached servers sharing one GPU. Both run with a batch limit of
    ELASTIC_BATCH, so that their load can fill most of the free GPU memory;
    the extra arguments are appended after it.
    """
    engine_a: str
    model_a: str
    engine_b: str
    model_b: str
    extra_args_a: tuple[str, ...] = ()
    extra_args_b: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return f"elastic-{self.engine_a}-{self.model_a}+{self.engine_b}-{self.model_b}"


@dataclass(frozen=True)
class LimitTest:
    """One kvcached server whose memory limit is lowered while its prefix
    cache holds most of its KV memory (e2e/limit.py). The limit is set on the
    server's first segment, so the model needs a single KV cache group.
    """
    engine: str
    model: str

    @property
    def name(self) -> str:
        return f"limit-{self.engine}-{self.model}"


@dataclass(frozen=True)
class Profile:
    groups: tuple[Group, ...]
    elastic: tuple[ElasticPair, ...] = ()
    limit: tuple[LimitTest, ...] = ()

    @property
    def engines(self) -> tuple[str, ...]:
        names = [g.engine for g in self.groups]
        names += [e for pair in self.elastic for e in (pair.engine_a, pair.engine_b)]
        names += [t.engine for t in self.limit]
        return tuple(dict.fromkeys(names))


def groups(engines: tuple[str, ...], models: tuple[str, ...], **kwargs: Any) -> tuple[Group, ...]:
    """Every model on every engine that can serve it."""
    return tuple(Group(e, m, **kwargs) for e in engines for m in models if e in MODELS[m].args)


PROFILES = {
    # A quick end-to-end check of the CI plumbing.
    "smoke": Profile(groups=groups(("vllm", "sglang"), ("qwen05b",), layouts=("c1",))),
    # Run inside the weekly H100 sandbox, next to the performance runs.
    "hopper": Profile(groups=(Group("vllm", "qwen38_27b", layouts=("c1",)),)),
    "nightly": Profile(
        groups=groups(("vllm", "sglang"), ("qwen05b", "qwen35_4b", "gemma4e2b")),
        elastic=(ElasticPair("vllm", "qwen05b", "vllm", "qwen35_4b"),
                 ElasticPair("sglang", "qwen05b", "sglang", "qwen35_4b")),
        limit=(LimitTest("vllm", "qwen05b"), LimitTest("sglang", "qwen05b")),
    ),
    # Two L4s.
    "weekly": Profile(
        groups=(
            *groups(("vllm", "sglang"), ("qwen05b", "dsv2lite", "gemma4_12b", "qwen35_9b"),
                    tp=2),
            *groups(("vllm", "sglang"), ("qwen05b",), layouts=("c1",), fp8_kv=True),
        ),
        # Two different engines sharing GPU 0.
        elastic=(ElasticPair("vllm", "qwen05b", "sglang", "qwen35_4b"),),
    ),
}
