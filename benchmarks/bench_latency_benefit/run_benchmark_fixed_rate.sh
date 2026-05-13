#!/bin/bash
set -ex

# Usage: ./run_benchmark_fixed_rate.sh <FIXED_RPS> <COMPLETION_LEN> [NUM_PROMPTS] [DURATION] [MODEL_DELAY] [BURSTINESS] [PROMPT_LEN]
# Example: ./run_benchmark_fixed_rate.sh 12 256 720 60 30 10.0 4096
#
# Parameters:
#   FIXED_RPS      - Fixed request rate (requests per second)
#   COMPLETION_LEN - Completion length
#   NUM_PROMPTS    - Total number of prompts (optional, will calculate from FIXED_RPS * DURATION if not provided)
#   DURATION       - Duration in seconds (default: 30)
#   MODEL_DELAY    - Delay between models in seconds (default: DURATION)
#   BURSTINESS     - Higher values = more uniform timing (default: 10000.0)
#                    Use high values like 10-100 for near-constant intervals
#   PROMPT_LEN     - Prompt length (default: 4096)

# Set environment variables
export KVCACHED_IPC_NAME=VLLM

# Add vLLM benchmarks and kvcached to Python path
export PYTHONPATH="../../engine_integration/vllm-v0.9.2/benchmarks:../../:../../benchmarks:$PYTHONPATH"

# Benchmark parameters
PROMPT_LEN=${7:-4096}
COMPLETION_LEN=$2
BACKEND="vllm"
# Fixed request rate parameters
FIXED_RPS=$1               # Fixed request rate (requests per second)
DURATION=${4:-0}          # Duration in seconds (default: 30s)
BURSTINESS=${6:-10000.0}   # Higher burstiness for more uniform requests (default: 10000.0)

# Calculate total number of requests
if [ -n "$3" ]; then
    NUM_PROMPTS=$3
    echo "Using provided NUM_PROMPTS: $NUM_PROMPTS"
else
    NUM_PROMPTS=$((FIXED_RPS * DURATION))
    echo "Calculated NUM_PROMPTS: $NUM_PROMPTS (fixed rate: ${FIXED_RPS} RPS for ${DURATION}s)"
fi

mkdir -p results results/metrics

# Define models and their configurations
MODELS=(
    "Qwen/Qwen2-7B-Instruct:12346"
    "Qwen/Qwen2-7B-Instruct:30000"
    "Qwen/Qwen2-7B-Instruct:40000"
)
NUM_MODELS=${#MODELS[@]}

# Record unified start time
UNIFIED_START_TIME=$(date +%s.%N)
echo "Unified benchmark start time: $UNIFIED_START_TIME"

# Model delay (can be adjusted if needed)
MODEL_DELAY=${5:-$DURATION}       # Delay in seconds before starting next model (default: DURATION)

# Arrays to store PIDs and result files
PIDS=()
RESULT_FILES=()

# Run benchmarks for each model
for i in "${!MODELS[@]}"; do
    # Parse model and port
    MODEL=$(echo "${MODELS[$i]}" | cut -d':' -f1)
    PORT=$(echo "${MODELS[$i]}" | cut -d':' -f2)

    # Generate model name and result file
    MODEL_NAME=$(echo "$MODEL" | tr '/' '-')
    MODEL_INDEX=$((i + 1))

    # Generate result file name for fixed rate strategy
    RESULT_FILE="results/metrics/${BACKEND}-${MODEL_NAME}-fixed-rate-${FIXED_RPS}rps-duration-${DURATION}s-burstiness-${BURSTINESS}-prompt_${PROMPT_LEN}-completion_${COMPLETION_LEN}-${MODEL_INDEX}-delay-${MODEL_DELAY}-model-num-${NUM_MODELS}-num-prompt-${NUM_PROMPTS}.json"

    # Add delay before starting next model (except for the first one)
    if [ $i -gt 0 ] && [ "$MODEL_DELAY" -gt 0 ]; then
        echo "Waiting ${MODEL_DELAY} seconds before starting Model ${MODEL_INDEX}..."
        sleep $MODEL_DELAY
    fi

    echo "Starting benchmark for $MODEL (Model ${MODEL_INDEX}) on port $PORT..."

    # Use fixed rate strategy
    echo "Using fixed rate strategy: ${FIXED_RPS} RPS for ${DURATION} seconds (burstiness: ${BURSTINESS})"

    python bench_kvcached_vllm.py \
        --backend "$BACKEND" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$PROMPT_LEN" \
        --random-output-len "$COMPLETION_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --host "localhost" \
        --port "$PORT" \
        --endpoint "/v1/completions" \
        --save-result \
        --result-filename "$RESULT_FILE" \
        --metadata "unified_start_time=$UNIFIED_START_TIME" \
        --request-rate "$FIXED_RPS" \
        --burstiness "$BURSTINESS" &

    # Store PID and result file
    PIDS+=($!)
    RESULT_FILES+=("$RESULT_FILE")

    echo "Started Model ${MODEL_INDEX} with PID ${PIDS[$i]}"
done

# Wait for all benchmarks to complete
echo "Waiting for all benchmarks to complete..."
EXIT_CODES=()

for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    EXIT_CODES+=($EXIT_CODE)
    echo "Model $((i + 1)) benchmark exit code: $EXIT_CODE"
done

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