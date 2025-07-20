#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)


MODEL=meta-llama/Llama-3.1-8B
VLLM_PORT=12346
SGL_PORT=30000

NUM_PROMPTS=1000
REQUEST_RATE=10

op=$1


check_and_download_sharegpt() {
    pushd $SCRIPT_DIR
    if [ ! -f "ShareGPT_V3_unfiltered_cleaned_split.json" ]; then
        wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    fi
    popd
}

if [ "$op" == "vllm" ]; then
    check_and_download_sharegpt
    source "$ENGINE_DIR/vllm-v0.9.2/.venv/bin/activate"
    uv pip install pandas datasets
    pushd $ENGINE_DIR/vllm-v0.9.2/benchmarks
    python benchmark_serving.py --backend=vllm \
      --model $MODEL \
      --dataset-name sharegpt \
      --dataset-path $SCRIPT_DIR/ShareGPT_V3_unfiltered_cleaned_split.json \
      --request-rate $REQUEST_RATE \
      --num-prompts $NUM_PROMPTS \
      --port $VLLM_PORT
    deactivate
    popd
elif [ "$op" == "sgl" -o "$op" == "sglang" ]; then
    check_and_download_sharegpt
    source "$ENGINE_DIR/sglang-v0.4.6.post2/.venv/bin/activate"
    pushd $SCRIPT_DIR
    python -m sglang.bench_serving --backend sglang-oai \
      --model $MODEL \
      --dataset-name sharegpt \
      --dataset-path $SCRIPT_DIR/ShareGPT_V3_unfiltered_cleaned_split.json \
      --request-rate $REQUEST_RATE \
      --num-prompts $NUM_PROMPTS \
      --port $SGL_PORT
    deactivate
    popd
else
    echo "Invalid option: $op"
    exit 1
fi
