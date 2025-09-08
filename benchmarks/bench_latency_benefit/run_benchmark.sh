#!/bin/bash
set -x

# Set environment variables
export ENABLE_KVCACHED=true
export KVCACHED_IPC_NAME=VLLM

# Add vLLM benchmarks and kvcached to Python path
export PYTHONPATH="../../engine_integration/vllm-v0.9.2/benchmarks:../../:../../benchmarks:$PYTHONPATH"

# Benchmark parameters
PROMPT_LEN=1024 # Input length
COMPLETION_LEN=128  # Output length
NUM_PROMPTS=100  # Use fewer prompts for quick testing
REQUEST_RATE=20  # Lower request rate
BACKEND="vllm"

# Generate filenames based on prompt length
DATASET_PATH="datasets/custom_dataset_${PROMPT_LEN}.jsonl"

mkdir -p datasets results
# Create dataset with PROMPT_LEN character prompts
echo "Creating dataset with ${PROMPT_LEN} character prompts..."
python create_custom_dataset.py \
    --output "$DATASET_PATH" \
    --num-samples "$NUM_PROMPTS" \
    --prompt-length "$PROMPT_LEN" \
    --completion-length "$COMPLETION_LEN" \
    --seed 42

# Run benchmark through kvcached router

MODEL_1="meta-llama/Llama-3.2-1B"
MODEL_NAME_1=$(echo "$MODEL_1" | tr '/' '-')
RESULT_FILE="results/${BACKEND}-${MODEL_NAME_1}-qps_${REQUEST_RATE}-prompt_${PROMPT_LEN}-completion_${COMPLETION_LEN}.json"
python bench_kvcached_vllm.py \
    --backend "$BACKEND" \
    --model "$MODEL_1" \
    --dataset-name mycustom \
    --dataset-path "$DATASET_PATH" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --custom-output-len "$COMPLETION_LEN" \
    --host "localhost" \
    --port 12346 \
    --endpoint "/v1/completions" \
    --save-result \
    --result-filename "$RESULT_FILE"

MODEL_2="Qwen/Qwen3-0.6B"
MODEL_NAME_2=$(echo "$MODEL_2" | tr '/' '-')
RESULT_FILE="results/${BACKEND}-${MODEL_NAME_2}-qps_${REQUEST_RATE}-prompt_${PROMPT_LEN}-completion_${COMPLETION_LEN}.json"
python bench_kvcached_vllm.py \
    --backend "$BACKEND" \
    --model "$MODEL_2" \
    --dataset-name mycustom \
    --dataset-path "$DATASET_PATH" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --custom-output-len "$COMPLETION_LEN" \
    --host "localhost" \
    --port 30000 \
    --endpoint "/v1/completions" \
    --save-result \
    --result-filename "$RESULT_FILE"