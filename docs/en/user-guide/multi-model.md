# Multi-Model Serving

This guide covers deploying multiple LLM models on a single GPU (or GPU group) with kvcached elastic memory sharing.

## Overview

kvcached enables multiple LLMs to share GPU memory elastically. Instead of statically partitioning memory, each model allocates physical memory on demand and releases it when idle.

## Basic Setup

### Two Models on One GPU

```bash
# Terminal 1: Start first model
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
vllm serve meta-llama/Llama-3.2-1B-Instruct --no-enable-prefix-caching --port 12346

# Terminal 2: Start second model
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
vllm serve Qwen/Qwen3-0.6B --no-enable-prefix-caching --port 12347
```

### Using the Provided Script

```bash
cd examples/01_simple_two_models

bash start_two_models.sh \
  --engine-a vllm --engine-b vllm \
  --model-a meta-llama/Llama-3.2-1B-Instruct --port-a 12346 \
  --model-b Qwen/Qwen3-0.6B --port-b 12347 \
  --venv-vllm-path ${VENV_PATH}
```

## Mixed Engine Deployment

You can mix vLLM and SGLang on the same GPU:

```bash
# Terminal 1: vLLM
export ENABLE_KVCACHED=true && export KVCACHED_AUTOPATCH=1
vllm serve meta-llama/Llama-3.2-1B-Instruct --no-enable-prefix-caching --port 12346

# Terminal 2: SGLang
export ENABLE_KVCACHED=true && export KVCACHED_AUTOPATCH=1
python -m sglang.launch_server --model Qwen/Qwen3-0.6B --disable-radix-cache --port 30000
```

## Controller-Based Deployment

For production multi-model setups, use the [Controller](router-sleep.md) for unified routing and management.

## How Memory Is Shared

Both models share the same physical GPU memory pool transparently:

1. Model A starts → reserves virtual space, allocates minimal physical pages
2. Model B starts → reserves its own virtual space, allocates minimal physical pages  
3. Requests arrive for Model A → its physical allocation grows
4. Model A goes idle → physical pages are released
5. Requests arrive for Model B → it uses the freed physical memory

The total physical memory used by both models never exceeds GPU capacity.

## Further Reading

- [Router & Sleep Management](router-sleep.md) — Production deployment with the controller
- [Memory Control CLI](memory-control.md) — Monitor and manage memory limits
- [Quick Start](../getting-started/quick-start.md) — Basic two-model setup
