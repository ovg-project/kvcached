# DGX Spark Demo

## Overview

This demo benchmarks **kvcached** against a static-memory-split baseline on a
single DGX Spark, running two co-located vLLM instances (a main LLM and a
guardrail model) that share one GPU.  The workload simulates the guardrail
pipeline that a real deployment would use:

```
User request  -->  Guardrail (input check)  -->  LLM  -->  Guardrail (output check)  -->  Response
```

**Key message:** Under the same memory constraint, kvcached lowers TTFT tail
latency across the tested concurrency sweep while supporting a much larger
context window than the static-memory-split baseline.

## Hardware

| Component | Spec |
|-----------|------|
| System | NVIDIA DGX Spark |
| GPU | NVIDIA GB10 (128 GB unified CPU/GPU memory) |
| CUDA | 13.0 |
| Driver | 580.126.09 |

## Software

| Component | Version / Detail |
|-----------|-----------------|
| vLLM | 0.19.2.dev0 (custom build with kvcached patches) |
| kvcached | from source (`main` branch) |
| Python | 3.12 (conda env `kvcached`) |

## Models

| Role | Model | Params | Weights (BF16) |
|------|-------|--------|----------------|
| Main LLM | `Qwen/Qwen3.6-35B-A3B` | 35B total / 3B active (MoE) | ~67 GB |
| Guardrail | `meta-llama/Llama-Guard-3-8B` | 8B | ~15 GB |

Qwen3.6-35B-A3B is a hybrid linear-attention MoE model, requiring
`KVCACHED_CONTIGUOUS_LAYOUT=false`.  Thinking is disabled during benchmarking
(`enable_thinking: false`).

## Configurations

### kvcached mode (dynamic memory sharing)

Both vLLM instances register with kvcached and share the GPU memory pool
dynamically.  Neither model is starved of KV-cache budget when the other is
idle.

| Parameter | Main | Guard |
|-----------|------|-------|
| `gpu-memory-utilization` | 0.70 | 0.25 |
| `max-model-len` | 65,536 | 8,192 |
| `KVCACHED_PAGE_SIZE_MB` | 32 | 32 |
| Prefix caching | off | off |
| Eager mode | yes | yes |
| YaRN (2x context) | yes | - |
| Tool calling (Hermes) | yes | - |

### Baseline mode (static memory split)

Each vLLM instance gets a fixed `gpu-memory-utilization` slice.  The split
must be conservative enough that both models fit simultaneously, which limits
the KV-cache budget available to each.

| Parameter | Main | Guard |
|-----------|------|-------|
| `gpu-memory-utilization` | 0.65 | 0.16 |
| `max-model-len` | 8,192 | 8,192 |
| Prefix caching | off | off |
| Eager mode | yes | yes |
| Tool calling (Hermes) | yes | - |

## Benchmark Setup

| Parameter | Value |
|-----------|-------|
| Harness | `workflow_benchmark.py` (Guard -> LLM -> Guard pipeline) |
| Dataset used for reported results | Synthetic random prompts (`DATASET_NAME=random`) |
| Input length | `random-input-len=256` word target (~400 prompt tokens with the current tokenizers) |
| Main output cap | 2,048 tokens |
| Concurrency levels | 1, 2, 4, 8, 16 |
| Prompts per level in reported results | 16, 16, 16, 16, 32 |
| Timeout per level | 1,800 s |

The checked-in result files under `results/` were produced with synthetic
random prompts, not ShareGPT.  The random prompt generator repeats a fixed
enterprise-workflow instruction around a sampled topic; `random-input-len=256`
is a word-count target rather than a tokenizer-level sequence-length filter.

The scripts also support ShareGPT by setting `DATASET_NAME=sharegpt`.  ShareGPT
prompts are filtered by character length (`16 <= prompt chars <= 12000`), so
ShareGPT runs are not directly comparable to the reported table unless the
results are regenerated and the prompt token distribution is recorded.

## Results

### Summary Table

| Mode | Concurrency | Completed | Mean TTFT (ms) | P99 TTFT (ms) | Mean E2E (ms) | P99 E2E (ms) |
|------|:-----------:|:---------:|:--------------:|:-------------:|:-------------:|:------------:|
| kvcached | 1 | 16 | 817 | 2,953 | 10,349 | 18,311 |
| kvcached | 2 | 16 | 1,157 | 1,465 | 14,102 | 25,317 |
| kvcached | 4 | 16 | 1,211 | 1,374 | 20,260 | 41,128 |
| kvcached | 8 | 16 | 1,789 | 2,339 | 28,363 | 60,923 |
| kvcached | 16 | 32 | 2,581 | 4,524 | 47,163 | 96,843 |
| baseline | 1 | 16 | 904 | 4,029 | 10,503 | 18,301 |
| baseline | 2 | 16 | 1,278 | 2,064 | 14,614 | 25,763 |
| baseline | 4 | 16 | 1,219 | 1,445 | 19,957 | 34,770 |
| baseline | 8 | 16 | 2,010 | 3,076 | 30,441 | 63,146 |
| baseline | 16 | 32 | 2,757 | 4,687 | 45,526 | 89,155 |

### Figures

| TTFT vs Concurrency | End-to-End Latency vs Concurrency |
|:---:|:---:|
| ![TTFT](results/ttft_vs_concurrency.png) | ![E2E](results/e2e_vs_concurrency.png) |

### Key Observations

- **P99 TTFT** is lower with kvcached at every tested concurrency.  The largest
  reductions are at C=1 (4.0 s -> 3.0 s, **27%**), C=2 (2.1 s -> 1.5 s,
  **29%**), and C=8 (3.1 s -> 2.3 s, **24%**).
- **P99 E2E latency** is mixed because the 2,048-token output cap makes decode
  time dominate total workflow latency.  kvcached is roughly tied at C=1,
  lower at C=2 and C=8, and higher at C=4 and C=16 in this run.
- Both modes complete all requests without failures through C=16, but kvcached
  achieves this with a much larger context window (65K vs 8K) thanks to dynamic
  memory sharing.

## How to Reproduce

```bash
conda activate kvcached

# Reproduce the reported synthetic-prompt sweep.
export DATASET_NAME=random
export CONCURRENCIES="1 2 4 8 16"
export MIN_NUM_PROMPTS=16
export NUM_PROMPTS_MULTIPLIER=2

# Optional on shared machines with stale kvcached IPC files.
export KVCACHED_MAIN_IPC_NAME="kvcached_main_${USER}"
export KVCACHED_GUARD_IPC_NAME="kvcached_guard_${USER}"

# Run kvcached benchmark (launch both models + sweep).
./run_benchmark.sh kvcached

# Run baseline benchmark.
./run_benchmark.sh baseline

# Or step-by-step:
./stop.sh
./launch_main.sh --mode kvcached
./launch_guard.sh --mode kvcached
./bench.sh kvcached
./stop.sh
```

## File Inventory

```
benchmarks/bench_dgx_spark
├── README.md                 # this file
├── config.sh                 # shared configuration (models, ports, tuning knobs)
├── launch_main.sh            # start main LLM (--mode kvcached|baseline)
├── launch_guard.sh           # start guardrail model (--mode kvcached|baseline)
├── bench.sh                  # concurrency sweep (Guard -> LLM -> Guard)
├── workflow_benchmark.py     # Python harness for the full pipeline benchmark
├── plot_results.py           # generate comparison plots from results/
├── run_benchmark.sh          # end-to-end: launch + bench + stop
├── stop.sh                   # kill all vllm serve processes
└── results/
    ├── kvcached/             # per-concurrency JSON results
    ├── baseline/             # per-concurrency JSON results
    ├── summary.csv           # combined metrics table
    ├── ttft_vs_concurrency.png
    └── e2e_vs_concurrency.png
```
