# DGX Spark Demo

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
| Main LLM, Experiment 1 | `Qwen/Qwen3.6-35B-A3B` | 35B total / 3B active (MoE) | ~67 GB |
| Main LLM, Experiment 2 | `Qwen/Qwen3-30B-A3B` | 30B total / 3B active (MoE) | ~57 GB |
| Guardrail | `meta-llama/Llama-Guard-3-8B` | 8B | ~15 GB |

## Results

| Experiment | Result dir | Workflow | Main model | Input / output | Concurrency |
|------------|------------|----------|------------|----------------|-------------|
| Experiment 1 | `results/` | Guard -> Main -> Guard | `Qwen/Qwen3.6-35B-A3B` | ~400 prompt tokens / 2,048 output tokens | 1, 2, 4, 8, 16 |
| Experiment 2 | `results_gain_12k_c8_tuned/` | Guard -> Main | `Qwen/Qwen3-30B-A3B` | ~11.6K prompt tokens / 10 output tokens | 1, 2, 4, 8 |

### Experiment 2: Qwen3-30B-A3B 12K Guard -> Main

```
User request  -->  Guardrail (input check)  -->  LLM  -->  Response
```

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

Short reproduce command from `benchmarks/bench_dgx_spark/`:

```bash
RESULTS_DIR="$PWD/results_gain_12k_c8_tuned" \
LOG_DIR="$PWD/logs_gain_12k_c8_tuned" \
DATASET_NAME=random CONCURRENCIES="1 2 4 8" \
MIN_NUM_PROMPTS=64 NUM_PROMPTS_MULTIPLIER=8 \
BENCH_INPUT_LEN=8192 BENCH_OUTPUT_LEN=10 BENCH_TIMEOUT_SECONDS=2400 \
MAIN_MAX_MODEL_LEN=16384 GUARD_MAX_MODEL_LEN=16384 \
BASELINE_MAIN_MAX_MODEL_LEN=16384 BASELINE_GUARD_MAX_MODEL_LEN=16384 \
KVCACHED_MAIN_GPU_UTIL=0.75 KVCACHED_GUARD_GPU_UTIL=0.30 \
BASELINE_MAIN_GPU_UTIL=0.59 BASELINE_GUARD_GPU_UTIL=0.15 \
./run_benchmark.sh both
```

Observed KV capacity at startup:

| Mode | Model | Available KV cache | 16,384-token concurrency |
|------|-------|--------------------|--------------------------|
| kvcached | Guard | 19.69 GiB | 9.84x |
| baseline | Main | 13.04 GiB | 8.69x |
| baseline | Guard | 2.54 GiB | 1.27x |

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

### Experiment 1: Qwen3.6-35B-A3B Guard -> Main -> Guard

```
User request  -->  Guardrail (input check)  -->  LLM  -->  Guardrail (output check)  -->  Response
```

Important configuration:

| Parameter | Value |
|-----------|-------|
| Main model | `Qwen/Qwen3.6-35B-A3B` |
| Guard model | `meta-llama/Llama-Guard-3-8B` |
| Dataset | Synthetic random prompts (`DATASET_NAME=random`) |
| Input length | `random-input-len=256`, about 400 prompt tokens |
| Main output cap | 2,048 tokens |
| `max-model-len` | kvcached main 65,536; baseline main 8,192; guard 8,192 |
| Concurrency levels | 1, 2, 4, 8, 16 |
| Prompts per level | 16 for C=1/2/4/8; 32 for C=16 |

Memory split:

| Mode | Main `gpu-memory-utilization` | Guard `gpu-memory-utilization` |
|------|-------------------------------|--------------------------------|
| kvcached | 0.70 | 0.25 |
| baseline | 0.65 | 0.16 |

Short reproduce command from `benchmarks/bench_dgx_spark/`:

```bash
DATASET_NAME=random CONCURRENCIES="1 2 4 8 16" \
MIN_NUM_PROMPTS=16 NUM_PROMPTS_MULTIPLIER=2 \
BENCH_INPUT_LEN=256 BENCH_OUTPUT_LEN=2048 \
./run_benchmark.sh both
```

#### Summary Table

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

#### Figures

| TTFT vs Concurrency | End-to-End Latency vs Concurrency |
|:---:|:---:|
| ![TTFT](results/ttft_vs_concurrency.png) | ![E2E](results/e2e_vs_concurrency.png) |

#### 64-prompt TTFT check

![TTFT comparison with 16 and 64 prompts](results_np64/ttft_np16_vs_np64.png)

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
