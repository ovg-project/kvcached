# Quick Start

This guide will get you running **two LLM models on a single GPU** with elastic memory sharing in under 5 minutes.

## Step 1: Enable kvcached

Set the environment variables to activate kvcached:

```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
```

These two variables tell the serving engine to use kvcached's elastic memory management instead of static allocation.

## Step 2: Start Two Model Servers

Open **two separate terminals** and start one model server in each.

=== "vLLM"

    **Terminal 1:**
    ```bash
    export ENABLE_KVCACHED=true
    export KVCACHED_AUTOPATCH=1
    vllm serve meta-llama/Llama-3.2-1B-Instruct \
      --no-enable-prefix-caching \
      --port 12346
    ```

    **Terminal 2:**
    ```bash
    export ENABLE_KVCACHED=true
    export KVCACHED_AUTOPATCH=1
    vllm serve Qwen/Qwen3-0.6B \
      --no-enable-prefix-caching \
      --port 12347
    ```

=== "SGLang"

    **Terminal 1:**
    ```bash
    export ENABLE_KVCACHED=true
    export KVCACHED_AUTOPATCH=1
    python -m sglang.launch_server \
      --model meta-llama/Llama-3.2-1B-Instruct \
      --disable-radix-cache \
      --port 30000
    ```

    **Terminal 2:**
    ```bash
    export ENABLE_KVCACHED=true
    export KVCACHED_AUTOPATCH=1
    python -m sglang.launch_server \
      --model Qwen/Qwen3-0.6B \
      --disable-radix-cache \
      --port 30001
    ```

## Step 3: Send Requests

Once both servers are ready, send requests to verify they are both operational:

```bash
# Request to Model 1
curl -s -X POST http://127.0.0.1:12346/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-1B-Instruct","prompt":"Explain how LLM works.","max_tokens":128}'

# Request to Model 2
curl -s -X POST http://127.0.0.1:12347/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"What is GPU memory?","max_tokens":128}'
```

## What's Happening?

Without kvcached, each model would statically reserve all available GPU memory, making it impossible to run both on the same GPU. With kvcached:

1. Each engine reserves a large **virtual** address space (cheap, no physical memory used)
2. Physical GPU memory is allocated **on demand** as requests arrive
3. When a model is idle, its physical memory can be reclaimed and used by the other model
4. The total physical memory usage stays within GPU capacity

!!! tip "No `--gpu-memory-utilization` needed"
    When kvcached is enabled, there is **no need** to set memory utilization limits. kvcached automatically manages memory allocation based on actual demand.

## Monitor Memory Usage

Use `kvtop` to visualize real-time GPU memory usage:

```bash
kvtop
```

This shows a live view of how each model's KV cache grows and shrinks with load.

## Next Steps

- [Multi-Model Serving](../user-guide/multi-model.md) — Advanced multi-model configurations
- [Memory Control CLI](../user-guide/memory-control.md) — Manage memory limits with `kvctl`
- [Architecture](../core-concepts/architecture.md) — Understand how the virtual memory system works
