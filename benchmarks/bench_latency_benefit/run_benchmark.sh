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
    
    # Generate result file name for ramp-up-down strategy
    RESULT_FILE="results/metrics/${BACKEND}-${MODEL_NAME}-ramp-up-down-${RAMP_START_RPS}to${RAMP_PEAK_RPS}to${RAMP_END_RPS}-inc${RAMP_INCREMENT}-prompt_${PROMPT_LEN}-completion_${COMPLETION_LEN}-${MODEL_INDEX}-delay-${MODEL_DELAY}-model-num-${NUM_MODELS}.json"
    
    # Add delay before starting next model (except for the first one)
    if [ $i -gt 0 ] && [ "$MODEL_DELAY" -gt 0 ]; then
        echo "Waiting ${MODEL_DELAY} seconds before starting Model ${MODEL_INDEX}..."
        sleep $MODEL_DELAY
    fi
    
    echo "Starting benchmark for $MODEL (Model ${MODEL_INDEX}) on port $PORT..."
    
    NUM_PROMPTS=$((NUM_PROMPTS + (NUM_MODELS - i) * MODEL_DELAY))
    # Use ramp-up-down strategy
    echo "Using ramp-up-down strategy: ${RAMP_START_RPS} -> ${RAMP_PEAK_RPS} -> ${RAMP_END_RPS} RPS (increment: ±${RAMP_INCREMENT} RPS/sec)"
    
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
        --ramp-up-strategy ramp-up-down \
        --ramp-start-rps "$RAMP_START_RPS" \
        --ramp-end-rps "$RAMP_END_RPS" \
        --ramp-peak-rps "$RAMP_PEAK_RPS" \
        --ramp-increment "$RAMP_INCREMENT" &
    
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