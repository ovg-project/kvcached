#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)

MODEL=meta-llama/Llama-3.1-8B
VLLM_PORT=12346
SGL_PORT=30000

TP_SIZE=4

op=$1

if [ "$op" == "vllm" ]; then
    source "$ENGINE_DIR/vllm-v0.8.4/.venv/bin/activate"
    export PYTHONPATH="$KVCACHED_DIR:$PYTHONPATH"
    export VLLM_USE_V1=1
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    export ENABLE_KVCACHED=true
    vllm serve "$MODEL" --disable-log-requests --no-enable-prefix-caching --port="$VLLM_PORT" --tensor-parallel-size="$TP_SIZE"
elif [ "$op" == "sgl" -o "$op" == "sglang" ]; then
    source "$ENGINE_DIR/sglang-v0.4.6.post2/.venv/bin/activate"
    export PYTHONPATH="$KVCACHED_DIR:$PYTHONPATH"
    export ENABLE_KVCACHED=true
    python -m sglang.launch_server --model "$MODEL" --disable-radix-cache --disable-overlap-schedule --trust-remote-code --port "$SGL_PORT" --tensor-parallel-size="$TP_SIZE"
else
    echo "Invalid option: $op"
    exit 1
fi
