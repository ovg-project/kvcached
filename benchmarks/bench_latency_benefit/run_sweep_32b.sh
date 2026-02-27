#!/bin/bash
# Sweep constant RPS from START_RPS to END_RPS for 2x Qwen2.5-32B
# Both instances run simultaneously at each RPS point.
#
# Usage: bash run_sweep_32b.sh <tag> [start_rps] [end_rps] [step]
# Example (kvcached):  bash run_sweep_32b.sh kvcached-32b 12 20 2
# Example (baseline):  bash run_sweep_32b.sh baseline-32b 12 20 2
set -e

PYTHON="${PYTHON:-/home/sz81/kvcached/py12-vllm-0.14.0-pip/bin/python}"
export PYTHONPATH="../../:../../benchmarks:$PYTHONPATH"

RUN_TAG="${1:-kvcached-32b}"
START_RPS="${2:-1}"
END_RPS="${3:-20}"
STEP="${4:-1}"

PROMPT_LEN=256
COMPLETION_LEN=128
BACKEND="vllm"
# Number of prompts per instance per RPS point: ~90 seconds of load
# Adjust down if 32B throughput is lower than expected
DURATION_S=90
# Stagger between instance starts — gives instance 1's initial test time
# to complete before instance 2 fires its own (avoids GPU contention at startup)
INSTANCE_STAGGER_S=60

MODELS=(
    "Qwen/Qwen2.5-32B-Instruct:12346"
    "Qwen/Qwen2.5-32B-Instruct:30000"
)
NUM_MODELS=${#MODELS[@]}

mkdir -p results/sweep

echo "=== 32B RPS sweep: ${START_RPS} to ${END_RPS} step ${STEP} tag=${RUN_TAG} ==="

for RPS in $(seq "$START_RPS" "$STEP" "$END_RPS"); do
    NUM_PROMPTS=$(( RPS * DURATION_S ))
    echo ""
    echo "--- RPS=${RPS}  num_prompts=${NUM_PROMPTS} per instance ---"

    PIDS=()
    RESULT_FILES=()

    for i in "${!MODELS[@]}"; do
        MODEL=$(echo "${MODELS[$i]}" | cut -d':' -f1)
        PORT=$(echo "${MODELS[$i]}" | cut -d':' -f2)
        MODEL_NAME=$(echo "$MODEL" | tr '/' '-')
        INST=$((i + 1))
        RESULT_FILE="results/sweep/${RUN_TAG}-${BACKEND}-${MODEL_NAME}-rps${RPS}-prompt${PROMPT_LEN}-completion${COMPLETION_LEN}-inst${INST}.json"

        "$PYTHON" bench_kvcached_vllm.py \
            --backend "$BACKEND" \
            --model "$MODEL" \
            --dataset-name random \
            --random-input-len "$PROMPT_LEN" \
            --random-output-len "$COMPLETION_LEN" \
            --num-prompts "$NUM_PROMPTS" \
            --request-rate "$RPS" \
            --host "localhost" \
            --port "$PORT" \
            --endpoint "/v1/completions" \
            --save-result \
            --result-filename "$RESULT_FILE" \
            --metadata "rps=${RPS},tag=${RUN_TAG},num_models=${NUM_MODELS}" &

        PIDS+=($!)
        RESULT_FILES+=("$RESULT_FILE")
        echo "  Started instance ${INST} (port ${PORT}) PID=${PIDS[$i]}"
    done

    echo "  Waiting for RPS=${RPS} to complete..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" || { echo "  WARNING: instance $((i+1)) failed"; FAILED=1; }
    done

    if [ "$FAILED" -eq 0 ]; then
        echo "  RPS=${RPS} done. Results:"
        for f in "${RESULT_FILES[@]}"; do
            # Print mean_ttft_ms from the JSON
            "$PYTHON" -c "
import json, sys
try:
    d = json.load(open('$f'))
    print(f'    {\"$f\".split(\"/\")[-1]}: mean_ttft={d.get(\"mean_ttft_ms\",\"N/A\"):.1f}ms  p99_ttft={d.get(\"p99_ttft_ms\",\"N/A\"):.1f}ms  throughput={d.get(\"request_throughput\",\"N/A\"):.2f}req/s')
except Exception as e:
    print(f'    (could not parse: {e})')
" 2>/dev/null || true
        done
    fi
done

echo ""
echo "=== Sweep complete. All results in results/sweep/ ==="
