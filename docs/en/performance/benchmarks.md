# Benchmarks

## Multi-LLM Serving Performance

kvcached achieves **2-28x TTFT reduction** compared to static allocation when serving multiple LLMs on a single GPU under workloads with intermittent peaks.

### Test Setup

- **Hardware**: A100-80G GPU
- **Models**: 3x Llama-3.1-8B instances
- **Workload**: Intermittent peak traffic pattern
- **Metrics**: TTFT (Time-to-First-Token) mean and P99

### Key Results (from Prism, OSDI 2026)

| Metric | Static Allocation | With kvcached | Improvement |
|--------|------------------|---------------|-------------|
| TTFT (mean) | Baseline | Up to 28x reduction | — |
| TTFT (P99) | Baseline | Up to 2x reduction | — |
| SLO attainment (TTFT) | Baseline | Up to 3.3x higher | — |
| Cost efficiency | Baseline | Up to 2x cost reduction | — |

## Running Benchmarks

### Simple Benchmark

```bash
cd benchmarks/simple_bench
./start_server.sh [sglang|vllm] --venv-path $VENV_PATH --model meta-llama/Llama-3.2-1B-Instruct
# Wait until ready
./start_client.sh [sglang|vllm] --venv-path $VENV_PATH --model meta-llama/Llama-3.2-1B-Instruct
```

### Latency Benefit Benchmark

```bash
cd benchmarks/bench_latency_benefit
# Follow the README for multi-model latency comparison
```

## Reproducing Results

The benchmark scripts automatically set `ENABLE_KVCACHED=true`. Refer to each script in `benchmarks/` for detailed instructions.
