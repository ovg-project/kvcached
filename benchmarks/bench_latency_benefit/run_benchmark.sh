#!/bin/bash
set -ex

# Set environment variables
export ENABLE_KVCACHED=true
export KVCACHED_IPC_NAME=VLLM

# Add vLLM benchmarks and kvcached to Python path
export PYTHONPATH="../../engine_integration/vllm-v0.9.2/benchmarks:../../:../../benchmarks:$PYTHONPATH"

# Benchmark parameters
PROMPT_LEN=1024 # Input length
COMPLETION_LEN=128  # Output length
NUM_PROMPTS=400  # Use fewer prompts for quick testing
REQUEST_RATE=20  # Request rate (RPS)
BURSTINESS=1.0  # Use Poisson process (1.0 = Poisson, <1.0 = more bursty, >1.0 = more uniform)
BACKEND="vllm"
MODEL_2_DELAY=3  # Delay in seconds before starting Model 2

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

# Run benchmark through kvcached router with unified time baseline

# Record unified start time
UNIFIED_START_TIME=$(date +%s.%N)
echo "Unified benchmark start time: $UNIFIED_START_TIME"

# Model 1
MODEL_1="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME_1=$(echo "$MODEL_1" | tr '/' '-')
RESULT_FILE_1="results/metrics/${BACKEND}-${MODEL_NAME_1}-qps_${REQUEST_RATE}-prompt_${PROMPT_LEN}-completion_${COMPLETION_LEN}-1.json"

echo "Starting benchmark for $MODEL_1..."
python bench_kvcached_vllm.py \
    --backend "$BACKEND" \
    --model "$MODEL_1" \
    --dataset-name mycustom \
    --dataset-path "$DATASET_PATH" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --burstiness 1.0 \
    --custom-output-len "$COMPLETION_LEN" \
    --host "localhost" \
    --port 12346 \
    --endpoint "/v1/completions" \
    --save-result \
    --result-filename "$RESULT_FILE_1" \
    --metadata "unified_start_time=$UNIFIED_START_TIME" &

# Start Model 1 in background and get its PID
MODEL_1_PID=$!

# Optional delay before starting Model 2
if [ "$MODEL_2_DELAY" -gt 0 ]; then
    echo "Waiting ${MODEL_2_DELAY} seconds before starting Model 2..."
    sleep $MODEL_2_DELAY
fi

# Model 2 (run in parallel)
MODEL_2="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME_2=$(echo "$MODEL_2" | tr '/' '-')
RESULT_FILE_2="results/metrics/${BACKEND}-${MODEL_NAME_2}-qps_${REQUEST_RATE}-prompt_${PROMPT_LEN}-completion_${COMPLETION_LEN}-2.json"

echo "Starting benchmark for $MODEL_2..."
# python bench_kvcached_vllm.py \
#     --backend "$BACKEND" \
#     --model "$MODEL_2" \
#     --dataset-name mycustom \
#     --dataset-path "$DATASET_PATH" \
#     --num-prompts "$NUM_PROMPTS" \
#     --request-rate "$REQUEST_RATE" \
#     --burstiness 10.0 \
#     --custom-output-len "$COMPLETION_LEN" \
#     --host "localhost" \
#     --port 30000 \
#     --endpoint "/v1/completions" \
#     --save-result \
#     --result-filename "$RESULT_FILE_2" \
#     --metadata "unified_start_time=$UNIFIED_START_TIME" &

# Start Model 2 in background and get its PID
MODEL_2_PID=$!

# Wait for both benchmarks to complete
echo "Waiting for both benchmarks to complete..."
wait $MODEL_1_PID
MODEL_1_EXIT_CODE=$?

wait $MODEL_2_PID
MODEL_2_EXIT_CODE=$?

echo "Model 1 benchmark exit code: $MODEL_1_EXIT_CODE"
echo "Model 2 benchmark exit code: $MODEL_2_EXIT_CODE"

echo "Both benchmarks completed!"
echo "Results saved to:"
echo "  - $RESULT_FILE_1"
echo "  - $RESULT_FILE_2"