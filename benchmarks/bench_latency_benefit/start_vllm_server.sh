#!/bin/bash
set -ex

# Set environment variables to enable kvcached
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Model configuration
MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8000

# Start vLLM server
vllm serve "$MODEL" \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.5 \
    --port="$PORT" \
    --tensor-parallel-size=1 \
    --max-num-batched-tokens 16384
