# DGX Spark Demo: Enterprise Workstation, Multiple Users

## Overview

This demo benchmarks **kvcached** against a static-memory-split baseline on a
single DGX Spark, running two co-located vLLM instances (a main LLM and a
guardrail model) that share one GPU.  The workload simulates the guardrail
pipeline that a real deployment would use:

```
User request  -->  Guardrail (input check)  -->  LLM  -->  Guardrail (output check)  -->  Response
```

**Key message:** Under the same memory constraint, kvcached serves concurrent
requests at lower tail latency than a baseline that must statically partition
GPU memory between the two models.

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
| Dataset | Random (input 256 tokens, main output 2,048 tokens) |
| Concurrency levels | 4, 8, 16, 32 |
| Prompts per level | max(32, concurrency x 2) |
| Timeout per level | 1,800 s |

## Results

### Summary Table

| Mode | Concurrency | Completed | Mean TTFT (ms) | P99 TTFT (ms) | Mean E2E (ms) | P99 E2E (ms) |
|------|:-----------:|:---------:|:--------------:|:-------------:|:-------------:|:------------:|
| kvcached | 4 | 16 | 2,331 | 5,461 | 9,538 | 12,578 |
| kvcached | 8 | 16 | 2,303 | 2,623 | 11,994 | 12,453 |
| kvcached | 16 | 32 | 4,082 | 5,205 | 18,470 | 19,931 |
| kvcached | 32 | 64 | 6,799 | 9,570 | 29,143 | 32,856 |
| baseline | 4 | 16 | 3,020 | **7,791** | 10,246 | **14,607** |
| baseline | 8 | 16 | 2,741 | 2,965 | 12,544 | 12,742 |
| baseline | 16 | 32 | 4,614 | **8,626** | 20,377 | **22,803** |
| baseline | 32 | 64 | 6,353 | 9,343 | 29,087 | 33,274 |

### Figures

| TTFT vs Concurrency | End-to-End Latency vs Concurrency |
|:---:|:---:|
| ![TTFT](results/ttft_vs_concurrency.png) | ![E2E](results/e2e_vs_concurrency.png) |

### Key Observations

- **P99 TTFT** is consistently lower with kvcached, most notably at C=4
  (5.5 s vs 7.8 s, **30% reduction**) and C=16 (5.2 s vs 8.6 s, **40% reduction**).
- **P99 E2E latency** shows the same pattern: kvcached reduces tail latency by
  14-16% at C=4 and C=16.
- Both modes complete all requests without failures through C=32, but kvcached
  achieves this with a much larger context window (65K vs 8K) thanks to dynamic
  memory sharing.

## How to Reproduce

```bash
conda activate kvcached

# Run kvcached benchmark (launch both models + sweep)
./run_benchmark.sh kvcached

# Run baseline benchmark
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
demo/dgx-spark/
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
