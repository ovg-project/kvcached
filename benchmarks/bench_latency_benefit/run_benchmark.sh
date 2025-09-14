#!/bin/bash
set -ex

# Set environment variables
export ENABLE_KVCACHED=true
export KVCACHED_IPC_NAME=VLLM

# Add vLLM benchmarks and kvcached to Python path
export PYTHONPATH="../../engine_integration/vllm-v0.9.2/benchmarks:../../:../../benchmarks:$PYTHONPATH"

# Benchmark parameters
PROMPT_LEN=256 
COMPLETION_LEN=400  
NUM_PROMPTS=500 
REQUEST_RATE=10  
BURSTINESS=1.0  # Use Poisson process (1.0 = Poisson, <1.0 = more bursty, >1.0 = more uniform)
BACKEND="vllm"
MODEL_DELAY=50  # Delay in seconds before starting next model

mkdir -p results results/metrics

# Define models and their configurations
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct:12346"
    "meta-llama/Llama-3.1-8B-Instruct:30000"
    "meta-llama/Llama-3.1-8B-Instruct:40000"
)

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
    RESULT_FILE="results/metrics/${BACKEND}-${MODEL_NAME}-qps_${REQUEST_RATE}-prompt_${PROMPT_LEN}-completion_${COMPLETION_LEN}-burstiness-${BURSTINESS}-${MODEL_INDEX}-delay-${MODEL_DELAY}.json"
    
    # Add delay before starting next model (except for the first one)
    if [ $i -gt 0 ] && [ "$MODEL_DELAY" -gt 0 ]; then
        echo "Waiting ${MODEL_DELAY} seconds before starting Model ${MODEL_INDEX}..."
        sleep $MODEL_DELAY
    fi
    
    echo "Starting benchmark for $MODEL (Model ${MODEL_INDEX}) on port $PORT..."
    python bench_kvcached_vllm.py \
        --backend "$BACKEND" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$PROMPT_LEN" \
        --random-output-len "$COMPLETION_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$REQUEST_RATE" \
        --burstiness "$BURSTINESS" \
        --host "localhost" \
        --port "$PORT" \
        --endpoint "/v1/completions" \
        --save-result \
        --result-filename "$RESULT_FILE" \
        --metadata "unified_start_time=$UNIFIED_START_TIME" &
    
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