#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)


DEFAULT_MODEL=meta-llama/Llama-3.2-1B
DEFAULT_VLLM_PORT=12346
DEFAULT_SGL_PORT=30000
DEFAULT_MODE="dev"

NUM_PROMPTS=1000
REQUEST_RATE=10

op=$1
port=$2
model=$3
mode=$4

MODEL=${model:-$DEFAULT_MODEL}
VLLM_PORT=${port:-$DEFAULT_VLLM_PORT}
SGL_PORT=${port:-$DEFAULT_SGL_PORT}
MODE=${mode:-$DEFAULT_MODE}

PYTHON=${PYTHON:-python3}

source "$SCRIPT_DIR/env_detect.sh"

check_and_download_sharegpt() {
    pushd $SCRIPT_DIR
    if [ ! -f "ShareGPT_V3_unfiltered_cleaned_split.json" ]; then
        wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    fi
    popd
}

if [ "$op" == "vllm" ]; then
    check_and_download_sharegpt
    if [ "$MODE" = "dev" ]; then
        source "$ENGINE_DIR/vllm-kvcached-venv/bin/activate"
    fi
    vllm bench serve \
      --model $MODEL \
      --dataset-name sharegpt \
      --dataset-path $SCRIPT_DIR/ShareGPT_V3_unfiltered_cleaned_split.json \
      --request-rate $REQUEST_RATE \
      --num-prompts $NUM_PROMPTS \
      --port $VLLM_PORT
    if [ "$MODE" = "dev" ]; then
        deactivate
    fi
elif [ "$op" == "sgl" -o "$op" == "sglang" ]; then
    check_and_download_sharegpt
    if [ "$MODE" = "dev" ]; then
        source "$ENGINE_DIR/sglang-kvcached-venv/bin/activate"
    fi

    $PYTHON -m sglang.bench_serving --backend sglang-oai \
        --model $MODEL \
        --dataset-name sharegpt \
        --dataset-path $SCRIPT_DIR/ShareGPT_V3_unfiltered_cleaned_split.json \
        --request-rate $REQUEST_RATE \
        --num-prompts $NUM_PROMPTS \
        --port $SGL_PORT
    if [ "$MODE" = "dev" ]; then
        deactivate
    fi
else
    echo "Invalid option: $op"
    exit 1
fi
