#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/../../engine_integration" && pwd)


MODEL=meta-llama/Llama-3.2-3B
VLLM_PORT=12346
SGL_PORT=30000


REQUEST_RATE=32
NUM_PROMPTS=$((REQUEST_RATE * 60))
INPUT_LENGTH=1913
OUTPUT_LENGTH=162

op=$1


if [ "$op" == "vllm" ]; then
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
    source "$ENGINE_DIR/sglang-v0.4.6.post2/.venv/bin/activate"
    pushd $SCRIPT_DIR
    python -m sglang.bench_serving --backend sglang-oai \
      --model $MODEL \
      --port $SGL_PORT \
      --dataset-name random \
      --request-rate $REQUEST_RATE \
      --num-prompts $NUM_PROMPTS \
      --random-input-len $INPUT_LENGTH \
      --random-output-len $OUTPUT_LENGTH
    deactivate
    popd
else
    echo "Invalid option: $op"
    exit 1
fi
