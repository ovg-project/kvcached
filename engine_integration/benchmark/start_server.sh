#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)

DEFAULT_MODEL=meta-llama/Llama-3.2-1B
DEFAULT_VLLM_PORT=12346
DEFAULT_SGL_PORT=30000

op=$1
port_arg=$2
model_arg=$3

MODEL=${model_arg:-$DEFAULT_MODEL}
VLLM_PORT=${port_arg:-$DEFAULT_VLLM_PORT}
SGL_PORT=${port_arg:-$DEFAULT_SGL_PORT}

source "$SCRIPT_DIR/env_detect.sh"

# Detect if the first visible GPU is an NVIDIA L4. 
GPU_NAME=$(command -v nvidia-smi >/dev/null 2>&1 && \
           nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || echo "")
if [[ "$GPU_NAME" == *"L4"* ]]; then
    IS_L4=true
else
    IS_L4=false
fi

if [ "$op" == "vllm" ]; then
    if [ "$IN_DOCKER" = false ]; then
        source "$ENGINE_DIR/vllm-v0.9.2/.venv/bin/activate"
    fi
    export VLLM_USE_V1=1
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    export ENABLE_KVCACHED=true
    export KVCACHED_IPC_NAME=VLLM

    VLLM_L4_ARGS=""
    if [ "$IS_L4" = true ]; then
        VLLM_L4_ARGS="--enforce-eager"
    fi
    vllm serve "$MODEL" \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.5 \
    --port="$VLLM_PORT" \
    $VLLM_L4_ARGS
elif [ "$op" == "sgl" -o "$op" == "sglang" ]; then
    if [ "$IN_DOCKER" = false ]; then
        source "$ENGINE_DIR/sglang-v0.4.9/.venv/bin/activate"
    fi
    export ENABLE_KVCACHED=true
    export KVCACHED_IPC_NAME=SGLANG

    SGL_L4_ARGS=""
    if [ "$IS_L4" = true ]; then
        export TORCHINDUCTOR_DISABLE=1
        export TORCHDYNAMO_DISABLE=1
        SGL_L4_ARGS="--attention-backend torch_native"
    fi
    python -m sglang.launch_server --model "$MODEL" \
    --disable-radix-cache \
    --trust-remote-code \
    --mem-fraction-static 0.5 \
    --port "$SGL_PORT" \
    $SGL_L4_ARGS
else
    echo "Invalid option: $op"
    exit 1
fi
