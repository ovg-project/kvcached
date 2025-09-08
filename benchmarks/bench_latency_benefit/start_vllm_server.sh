#!/bin/bash
set -x

# Set environment variables to enable kvcached
export ENABLE_KVCACHED=false
export KVCACHED_IPC_NAME=VLLM
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Model configuration
MODEL="meta-llama/Llama-3.2-1B"  # Use smaller model for testing
PORT=8000

# Start vLLM server
vllm serve "$MODEL" \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.5 \
    --port="$PORT" \
    --tensor-parallel-size=1
