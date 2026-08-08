# Benchmarking Guide

This guide explains how to benchmark kvcached performance and compare it with baseline vLLM/SGLang configurations.

## Overview

kvcached introduces dynamic GPU memory management for KV caches, which has different performance characteristics than static allocation. This guide covers:

- Setting up benchmark environments
- Running performance comparisons
- Interpreting results
- Common benchmarking scenarios

## Key Metrics

When benchmarking kvcached, focus on these metrics:

| Metric | Description | Impact |
|--------|-------------|--------|
| **Throughput** | Requests/second or tokens/second | Overall system capacity |
| **TTFT** | Time to First Token | User-perceived latency |
| **ITL** | Inter-Token Latency | Streaming response smoothness |
| **Memory Utilization** | GPU memory efficiency | Multi-model capacity |
| **E2E Latency** | End-to-end request time | Total request duration |

## Benchmark Types

### 1. Overhead Benchmark

Measures kvcached's overhead vs. native engines at various request rates.

**Location**: `benchmarks/bench_kvcached_overhead/`

```bash
# Start server with kvcached
cd benchmarks/bench_kvcached_overhead
export ENABLE_KVCACHED=true
./start_server.sh

# In another terminal, run client
./start_client.sh
```

**Expected Results**:
- At low request rates (8-16 req/s): Minimal overhead (<5%)
- At high request rates (32+ req/s): May see higher overhead due to dynamic allocation

### 2. Latency Benefit Benchmark

Measures latency improvements from kvcached's dynamic memory management.

**Location**: `benchmarks/bench_latency_benefit/`

```bash
cd benchmarks/bench_latency_benefit

# Edit bench-config.yaml to set your models
./run_benchmark.sh
```

### 3. GSM8K Accuracy Benchmark

Validates that kvcached doesn't affect model accuracy.

**Location**: `benchmarks/gsm8k/`

```bash
# Start vLLM server
vllm serve Qwen/Qwen2.5-Math-1.5B --port 12346

# Run benchmark
cd benchmarks/gsm8k
python bench_vllm.py \
  --model Qwen/Qwen2.5-Math-1.5B \
  --port 12346 \
  --num-questions 100 \
  --num-shots 5 \
  --parallel 8
```

### 4. Multi-Model Benchmark

Measures memory efficiency when running multiple models.

```bash
# Start multiple models with kvcached
export ENABLE_KVCACHED=true
export KVCACHED_GPU_UTILIZATION=0.95

# Model A
vllm serve model-a --port 8001 &

# Model B
vllm serve model-b --port 8002 &

# Monitor memory
kvctl kvtop
```

## Setting Up Benchmarks

### Environment Preparation

1. **Install dependencies**:
   ```bash
   pip install kvcached vllm sglang
   ```

2. **Verify GPU access**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.device_count())"
   ```

3. **Check kvcached installation**:
   ```bash
   python -c "from kvcached import _C; print('kvcached OK')"
   ```

### Configuration

Create a benchmark configuration file:

```yaml
# bench-config.yaml
models:
  - name: llama-7b
    path: meta-llama/Llama-2-7b-hf
    port: 8001

  - name: qwen-7b
    path: Qwen/Qwen2-7B-Instruct
    port: 8002

benchmark:
  request_rates: [1, 2, 4, 8, 16, 32]
  num_requests: 1000
  input_len: 512
  output_len: 128

kvcached:
  gpu_utilization: 0.95
  page_size_mb: 2
```

## Running Comparisons

### A/B Testing Procedure

1. **Baseline (without kvcached)**:
   ```bash
   unset ENABLE_KVCACHED
   vllm serve model --port 8001
   # Run benchmark, save results as baseline.json
   ```

2. **With kvcached**:
   ```bash
   export ENABLE_KVCACHED=true
   export KVCACHED_AUTOPATCH=1
   vllm serve model --port 8001
   # Run benchmark, save results as kvcached.json
   ```

3. **Compare results**:
   ```python
   import json

   baseline = json.load(open('baseline.json'))
   kvcached = json.load(open('kvcached.json'))

   print(f"Throughput: {kvcached['throughput'] / baseline['throughput']:.2%}")
   print(f"TTFT: {kvcached['ttft_p50'] / baseline['ttft_p50']:.2%}")
   ```

### Simple Benchmark Script

```python
#!/usr/bin/env python3
"""Simple kvcached benchmark script."""

import asyncio
import time
import httpx

async def benchmark(url: str, num_requests: int = 100):
    """Run a simple throughput benchmark."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        start = time.time()

        async def send_request():
            response = await client.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "max_tokens": 50,
                }
            )
            return response.json()

        tasks = [send_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start
        print(f"Completed {num_requests} requests in {elapsed:.2f}s")
        print(f"Throughput: {num_requests / elapsed:.2f} req/s")

if __name__ == "__main__":
    asyncio.run(benchmark("http://localhost:8000"))
```

## Interpreting Results

### When kvcached Helps

kvcached shows benefits when:
- Running multiple models on a single GPU
- Memory pressure is high
- Dynamic workloads with varying sequence lengths
- Need to maximize GPU memory efficiency

### When kvcached Has Overhead

kvcached may add overhead when:
- Single model, fixed workload
- Very high request rates (>100 req/s)
- Short sequences with minimal memory variation

### Performance Expectations

| Scenario | Expected Overhead | Expected Benefit |
|----------|-------------------|------------------|
| Single model, low load | 1-3% | None |
| Single model, high load | 3-8% | Memory flexibility |
| Multi-model | 5-10% | 2-3x more models |
| Long sequences | 2-5% | Better memory utilization |

## Monitoring During Benchmarks

### Real-time Memory Monitoring

```bash
# Watch kvcached memory usage
kvctl watch

# Detailed per-segment view
kvctl kvtop
```

### GPU Metrics

```bash
# NVIDIA SMI monitoring
watch -n 1 nvidia-smi

# Detailed memory breakdown
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv -l 1
```

## Troubleshooting Benchmarks

### Inconsistent Results

- **Warm up**: Run 100+ requests before measuring
- **Cool down**: Wait 10s between runs
- **Isolation**: Stop other GPU processes

### Memory Issues

```bash
# Check shared memory
df -h /dev/shm

# List kvcached segments
ls -la /dev/shm/kvcached*

# Clean up stale segments
kvctl delete <segment_name>
```

### Debugging

```bash
# Enable debug logging
export KVCACHED_LOG_LEVEL=DEBUG

# Check for errors
grep -i error /var/log/kvcached.log
```

## Reproducibility

For reproducible benchmarks:

1. **Fix random seeds**:
   ```bash
   export PYTHONHASHSEED=42
   ```

2. **Disable dynamic features**:
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   ```

3. **Document environment**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import vllm; print(vllm.__version__)"
   nvidia-smi --query-gpu=name,driver_version --format=csv
   ```

## Contributing Benchmark Results

When sharing benchmark results, include:

1. Hardware specs (GPU model, memory, CPU)
2. Software versions (PyTorch, vLLM/SGLang, kvcached)
3. Configuration (model, batch size, sequence length)
4. Raw data files
5. Reproduction steps

Submit results via GitHub issue or PR to help improve kvcached.
