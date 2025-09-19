#!/bin/bash
set -ex

# Set environment variables
export KVCACHED_IPC_NAME=VLLM

# Add vLLM benchmarks and kvcached to Python path
export PYTHONPATH="../../engine_integration/vllm-v0.9.2/benchmarks:../../:../../benchmarks:$PYTHONPATH"

# Benchmark parameters
PROMPT_LEN=256
COMPLETION_LEN=$2
BACKEND="vllm"


# Ramp-up-down parameters
RAMP_START_RPS=0          # Starting request rate
RAMP_PEAK_RPS=$1          # Peak request rate (middle)
RAMP_END_RPS=1            # Ending request rate
RAMP_INCREMENT=1          # RPS increment/decrement per second



# Calculate total number of requests based on ramp pattern
RAMP_UP_DURATION=$(( (RAMP_PEAK_RPS - RAMP_START_RPS) / RAMP_INCREMENT ))
RAMP_DOWN_DURATION=$(( (RAMP_PEAK_RPS - RAMP_END_RPS) / RAMP_INCREMENT ))
TOTAL_DURATION=$((RAMP_UP_DURATION + RAMP_DOWN_DURATION))

MODEL_DELAY=$((RAMP_UP_DURATION/4 + RAMP_UP_DURATION*2))  # Delay in seconds before starting next model

# # Calculate total requests: sum of all RPS values across all seconds
# TOTAL_REQUESTS=$((RAMP_PEAK_RPS * RAMP_PEAK_RPS / 2))
# for (( sec=1; sec<=RAMP_UP_DURATION; sec++ )); do
#     RPS=$((RAMP_MIN_RPS + sec * RAMP_INCREMENT))
#     TOTAL_REQUESTS=$((TOTAL_REQUESTS + RPS))
# done
# for (( sec=1; sec<=RAMP_DOWN_DURATION; sec++ )); do
#     RPS=$((RAMP_PEAK_RPS - sec * RAMP_INCREMENT))
#     TOTAL_REQUESTS=$((TOTAL_REQUESTS + RPS))
# done

NUM_PROMPTS=$((RAMP_PEAK_RPS * RAMP_PEAK_RPS))
echo "Calculated NUM_PROMPTS: $NUM_PROMPTS (based on ramp pattern: ${TOTAL_DURATION}s duration)"

mkdir -p results results/metrics

# Define models and their configurations
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct:12346"
    "meta-llama/Llama-3.1-8B-Instruct:30000"
    "meta-llama/Llama-3.1-8B-Instruct:40000"
)
NUM_MODELS=${#MODELS[@]}

# Record unified start time
UNIFIED_START_TIME=$(date +%s.%N)
echo "Unified benchmark start time: $UNIFIED_START_TIME"

# Model 1
MODEL_1="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME_1=$(echo "$MODEL_1" | tr '/' '-')
RESULT_FILE_1="results/metrics/${BACKEND}-${MODEL_NAME_1}-qps_${REQUEST_RATE}-prompt_${PROMPT_LEN}-completion_${COMPLETION_LEN}-burstiness-${BURSTINESS}-1.json"

echo "Starting benchmark for $MODEL_1..."
python bench_kvcached_vllm.py \
    --backend "$BACKEND" \
    --model "$MODEL_1" \
    --dataset-name random \
    --random-input-len "$PROMPT_LEN" \
    --random-output-len "$COMPLETION_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --burstiness "$BURSTINESS" \
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
RESULT_FILE_2="results/metrics/${BACKEND}-${MODEL_NAME_2}-qps_${REQUEST_RATE}-prompt_${PROMPT_LEN}-completion_${COMPLETION_LEN}-burstiness-${BURSTINESS}-2.json"

echo "Starting benchmark for $MODEL_2..."
python bench_kvcached_vllm.py \
    --backend "$BACKEND" \
    --model "$MODEL_2" \
    --dataset-name random \
    --random-input-len "$PROMPT_LEN" \
    --random-output-len "$COMPLETION_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --burstiness "$BURSTINESS" \
    --host "localhost" \
    --port 30000 \
    --endpoint "/v1/completions" \
    --save-result \
    --result-filename "$RESULT_FILE_2" \
    --metadata "unified_start_time=$UNIFIED_START_TIME" &

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

echo "All benchmarks completed!"
echo "Results saved to:"
for result_file in "${RESULT_FILES[@]}"; do
    echo "  - $result_file"
done

# Summary of exit codes
echo "Exit code summary:"
for i in "${!EXIT_CODES[@]}"; do
    echo "  Model $((i + 1)): ${EXIT_CODES[$i]}"
done