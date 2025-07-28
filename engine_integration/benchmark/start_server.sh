#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)

DEFAULT_MODEL=meta-llama/Llama-3.1-8B
DEFAULT_VLLM_PORT=12346
DEFAULT_SGL_PORT=30000

op=$1
port_arg=$2
model_arg=$3

MODEL=${model_arg:-$DEFAULT_MODEL}
VLLM_PORT=${port_arg:-$DEFAULT_VLLM_PORT}
SGL_PORT=${port_arg:-$DEFAULT_SGL_PORT}

source "$SCRIPT_DIR/env_detect.sh"

if [ "$op" == "vllm" ]; then
    if [ "$IN_DOCKER" = false ]; then
        source "$ENGINE_DIR/vllm-v0.9.2/.venv/bin/activate"
    fi
    export VLLM_USE_V1=1
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    export ENABLE_KVCACHED=true
    export KVCACHED_IPC_NAME=VLLM
    vllm serve "$MODEL" \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.5 \
    --port="$VLLM_PORT"
elif [ "$op" == "sgl" -o "$op" == "sglang" ]; then
    if [ "$IN_DOCKER" = false ]; then
        source "$ENGINE_DIR/sglang-v0.4.9/.venv/bin/activate"
    fi
    export ENABLE_KVCACHED=true
    export KVCACHED_IPC_NAME=SGLANG
    python -m sglang.launch_server --model "$MODEL" \
    --disable-radix-cache \
    --trust-remote-code \
    --mem-fraction-static 0.5 \
    --port "$SGL_PORT"
else
    echo "Invalid option: $op"
    exit 1
fi
