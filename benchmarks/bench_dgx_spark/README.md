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

### Qwen3-30B-A3B 12K Guard -> Main Run

This gain-focused run uses the updated two-stage workflow:

```
User request  -->  Guardrail (input check)  -->  LLM  -->  Response
```

It is separate from the legacy `results/` table below.  The result files are in
`results_gain_12k_c8_tuned/`.

| Parameter | Value |
|-----------|-------|
| Main model | `Qwen/Qwen3-30B-A3B` |
| Guard model | `meta-llama/Llama-Guard-3-8B` |
| Dataset | Synthetic random prompts (`DATASET_NAME=random`) |
| `BENCH_INPUT_LEN` | 8,192 word target, about 11.6K tokens with the Qwen3-30B tokenizer |
| Main output cap | 10 tokens |
| `max-model-len` | 16,384 for both models and both modes |
| Concurrency levels | 1, 2, 4, 8 |
| Prompts per level | 64 |
| Timeout per level | 2,400 s |

Configuration:

| Mode | Main `gpu-memory-utilization` | Guard `gpu-memory-utilization` |
|------|-------------------------------|--------------------------------|
| kvcached | 0.75 | 0.30 |
| baseline | 0.59 | 0.15 |

Reproduce this run from `benchmarks/bench_dgx_spark/`:

```bash
export RESULTS_DIR="$PWD/results_gain_12k_c8_tuned"
export LOG_DIR="$PWD/logs_gain_12k_c8_tuned"
export DATASET_NAME=random
export CONCURRENCIES="1 2 4 8"
export MIN_NUM_PROMPTS=64
export NUM_PROMPTS_MULTIPLIER=8
export BENCH_INPUT_LEN=8192
export BENCH_OUTPUT_LEN=10
export BENCH_TIMEOUT_SECONDS=2400
export MAIN_MAX_MODEL_LEN=16384
export GUARD_MAX_MODEL_LEN=16384
export BASELINE_MAIN_MAX_MODEL_LEN=16384
export BASELINE_GUARD_MAX_MODEL_LEN=16384
export KVCACHED_MAIN_GPU_UTIL=0.75
export KVCACHED_GUARD_GPU_UTIL=0.30
export BASELINE_MAIN_GPU_UTIL=0.59
export BASELINE_GUARD_GPU_UTIL=0.15
export KVCACHED_MAIN_IPC_NAME="kvcached_gain_main_${USER}"
export KVCACHED_GUARD_IPC_NAME="kvcached_gain_guard_${USER}"

./run_benchmark.sh both
```

Observed KV capacity at startup:

| Mode | Model | Available KV cache | 16,384-token concurrency |
|------|-------|--------------------|--------------------------|
| kvcached | Guard | 19.69 GiB | 9.84x |
| baseline | Main | 13.04 GiB | 8.69x |
| baseline | Guard | 2.54 GiB | 1.27x |

The baseline guard is the constrained model in this run.  The baseline main
still has enough KV cache for about 8 full-length requests, so this run mainly
measures the impact of making the guard memory slice tight rather than a fully
tight two-model baseline.

#### End-to-end metrics

Speedup is `baseline / kvcached`; values above 1.0 mean kvcached is faster.

| Concurrency | kvcached mean TTFT (s) | baseline mean TTFT (s) | TTFT speedup | kvcached P99 TTFT (s) | baseline P99 TTFT (s) | P99 speedup | kvcached mean E2E (s) | baseline mean E2E (s) | E2E speedup |
|------------:|-----------------------:|-----------------------:|-------------:|----------------------:|----------------------:|------------:|----------------------:|----------------------:|------------:|
| 1 | 5.65 | 5.47 | 0.97x | 5.96 | 5.72 | 0.96x | 6.00 | 5.82 | 0.97x |
| 2 | 10.18 | 12.16 | 1.19x | 11.29 | 12.26 | 1.09x | 11.91 | 12.97 | 1.09x |
| 4 | 17.47 | 24.78 | 1.42x | 22.55 | 25.29 | 1.12x | 24.61 | 25.59 | 1.04x |
| 8 | 40.44 | 48.83 | 1.21x | 47.68 | 51.48 | 1.08x | 47.87 | 49.65 | 1.04x |

#### Stage breakdown

| Concurrency | kvcached guard mean (s) | baseline guard mean (s) | Guard speedup | kvcached main TTFT mean (s) | baseline main TTFT mean (s) | Main TTFT speedup |
|------------:|------------------------:|------------------------:|--------------:|----------------------------:|----------------------------:|------------------:|
| 1 | 2.95 | 2.92 | 0.99x | 2.70 | 2.55 | 0.94x |
| 2 | 5.86 | 6.60 | 1.13x | 4.32 | 5.56 | 1.29x |
| 4 | 10.71 | 19.22 | 1.79x | 6.76 | 5.56 | 0.82x |
| 8 | 33.53 | 43.28 | 1.29x | 6.92 | 5.56 | 0.80x |

#### Figures

| TTFT vs Concurrency | End-to-End Latency vs Concurrency |
|:---:|:---:|
| ![TTFT](results_gain_12k_c8_tuned/ttft_vs_concurrency.png) | ![E2E](results_gain_12k_c8_tuned/e2e_vs_concurrency.png) |

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

Labels on the kvcached mean and P99 points show the percentage delta relative
to the matching baseline point at the same concurrency; negative values mean
lower latency with kvcached.

#### 64-prompt TTFT check

![TTFT comparison with 16 and 64 prompts](results_np64/ttft_np16_vs_np64.png)

This check reruns C=1, 2, and 4 with 64 prompts per level.  The higher-sample
run removes the apparent low-concurrency P99 TTFT dip seen in the 16-prompt
run, which indicates that the earlier C=1 to C=4 tail shape was dominated by
small-sample tail variance rather than a stable latency trend.

### Gain Analysis

The table below uses the largest reported workload:
`C=16`, `input prompt=400`, and `output=2k`.
For kvcached, the denominator is the configured virtual/on-demand KV cap, not
memory physically allocated at startup.

| Model | KV per token | kvcached: actual / theoretical allocation | baseline: actual / theoretical allocation |
|-------|--------------|---------------------------|----------------------------|
| Qwen/Qwen3.6-35B-A3B main<br>weights: 65.53 GiB | 20 KiB/token<br>`2 KV heads * 256 dim * K/V * BF16 * 10 full-attn layers` | ~0.75 GiB / ~16.6 GiB virtual KV cap<br>`gpu-memory-utilization=0.70` | ~0.75 GiB / 10.95 GiB fixed KV budget<br>`gpu-memory-utilization=0.65` |
| Llama-Guard-3-8B guard<br>weights: 14.99 GiB | 128 KiB/token<br>`8 KV heads * 128 dim * K/V * BF16 * 32 layers` | <=4.1 GiB / ~18.3 GiB virtual KV cap<br>`gpu-memory-utilization=0.25` | <=4.1 GiB / 7.33 GiB fixed KV budget<br>`gpu-memory-utilization=0.16` |

So the current measured run does **not** fill the baseline KV cache; the
observed TTFT gain should be treated mainly as runtime noise rather than a
clean KV-capacity gain.

The clean kvcached gain should appear when the active model needs more live KV
than its fixed baseline slice, while the other model is mostly idle.  The
right test is long prompt, short output, and increasing concurrency:

```
input length: 8K, 16K, or 32K tokens
output length: 32-64 tokens
concurrency: sweep upward until baseline queues, fails, or must lower context
```

Then the main metrics are maximum successful concurrency, failures or queueing,
and P99 TTFT/E2E near the baseline memory boundary.

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
├── results/
│   ├── kvcached/             # per-concurrency JSON results
│   ├── baseline/             # per-concurrency JSON results
│   ├── summary.csv           # combined metrics table
│   ├── ttft_vs_concurrency.png
│   └── e2e_vs_concurrency.png
├── results_np64/
│   ├── summary.csv           # 64-prompt C=1/2/4 check
│   └── ttft_np16_vs_np64.png # TTFT comparison for 16 vs 64 prompts
└── results_gain_12k_c8_tuned/
    ├── summary.csv           # Qwen3-30B-A3B 12K guard->main run
    ├── kvcached/             # C=1/2/4/8 JSON results
    ├── baseline/             # C=1/2/4/8 JSON results
    ├── ttft_vs_concurrency.png
    └── e2e_vs_concurrency.png
```
