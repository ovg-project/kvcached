#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)


DEFAULT_MODEL=meta-llama/Llama-3.2-1B
DEFAULT_VLLM_PORT=12346
DEFAULT_SGL_PORT=30000

NUM_PROMPTS=1000
REQUEST_RATE=10

op=$1
port_arg=$2
model_arg=$3

MODEL=${model_arg:-$DEFAULT_MODEL}
VLLM_PORT=${port_arg:-$DEFAULT_VLLM_PORT}
SGL_PORT=${port_arg:-$DEFAULT_SGL_PORT}

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
    if [ "$IN_DOCKER" = false ]; then
        source "$ENGINE_DIR/vllm-v0.9.2/.venv/bin/activate"
    fi
    python -m pip install -q pandas datasets
    if [ "$IN_DOCKER" = false ]; then
        pushd $ENGINE_DIR/vllm-v0.9.2/benchmarks
    else
        pushd /workspace/vllm-v0.9.2/benchmarks
    fi
    python benchmark_serving.py --backend=vllm \
      --model $MODEL \
      --dataset-name sharegpt \
      --dataset-path $SCRIPT_DIR/ShareGPT_V3_unfiltered_cleaned_split.json \
      --request-rate $REQUEST_RATE \
      --num-prompts $NUM_PROMPTS \
      --port $VLLM_PORT
    if [ "$IN_DOCKER" = false ]; then
        deactivate
    fi
    popd
elif [ "$op" == "sgl" -o "$op" == "sglang" ]; then
    check_and_download_sharegpt
    if [ "$IN_DOCKER" = false ]; then
        source "$ENGINE_DIR/sglang-v0.4.9/.venv/bin/activate"
    fi

    python -m sglang.bench_serving --backend sglang-oai \
        --model $MODEL \
        --dataset-name sharegpt \
        --dataset-path $SCRIPT_DIR/ShareGPT_V3_unfiltered_cleaned_split.json \
        --request-rate $REQUEST_RATE \
        --num-prompts $NUM_PROMPTS \
        --port $SGL_PORT
    if [ "$IN_DOCKER" = false ]; then
        deactivate
    fi
else
    echo "Invalid option: $op"
    exit 1
fi
