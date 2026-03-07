#!/bin/bash
# Staggered-peak sweep for 3x Qwen2.5-14B-Instruct.
#
# Each instance gets a ramp-up-down traffic burst, staggered in time so that
# while one instance is at peak load, the others are idle. This lets kvcached
# demonstrate elastic KV sharing: the busy instance can borrow physical pages
# from the idle instances' pool, while the baseline is hard-capped per-instance.
#
# Usage: bash run_sweep_staggered_14b.sh [tag] [results_dir]
# Example (kvcached):  bash run_sweep_staggered_14b.sh kvcached-14b results/sweep14b-kvcached
# Example (baseline):  bash run_sweep_staggered_14b.sh baseline-14b results/sweep14b-baseline

set -e

PYTHON="${PYTHON:-/home/amd-rice-collab/home/kvcached/py12-vllm-0.14.0-pip/bin/python}"
export PYTHONPATH="../../:../../benchmarks:$PYTHONPATH"

RUN_TAG="${1:-kvcached-14b}"
RESULTS_DIR="${2:-results/sweep14b-kvcached}"

PROMPT_LEN=256
COMPLETION_LEN=512
BACKEND="vllm"
RAMP_START_RPS=0
RAMP_END_RPS=1
RAMP_INCREMENT=1
PEAK_START_RPS=25
PEAK_END_RPS=40

MODELS=(
    "Qwen/Qwen2.5-14B-Instruct:12346"
    "Qwen/Qwen2.5-14B-Instruct:30000"
    "Qwen/Qwen2.5-14B-Instruct:40000"
)
NUM_MODELS=${#MODELS[@]}

mkdir -p "$RESULTS_DIR"

echo "=== Staggered 14B sweep: peaks ${PEAK_START_RPS}-${PEAK_END_RPS}, tag=${RUN_TAG}, results=${RESULTS_DIR} ==="
echo "    Instances: ${NUM_MODELS}, prompt=${PROMPT_LEN}, completion=${COMPLETION_LEN}"

for RAMP_PEAK_RPS in $(seq $PEAK_START_RPS 1 $PEAK_END_RPS); do
    RAMP_UP_DURATION=$(( (RAMP_PEAK_RPS - RAMP_START_RPS) / RAMP_INCREMENT ))
    RAMP_DOWN_DURATION=$(( (RAMP_PEAK_RPS - RAMP_END_RPS) / RAMP_INCREMENT ))
    MODEL_DELAY=$(( RAMP_UP_DURATION / 4 + RAMP_UP_DURATION * 2 ))
    BASE_NUM_PROMPTS=$(( RAMP_PEAK_RPS * RAMP_PEAK_RPS ))

    echo ""
    echo "--- Peak RPS=${RAMP_PEAK_RPS}  ramp_up=${RAMP_UP_DURATION}s  model_delay=${MODEL_DELAY}s ---"

    PIDS=()
    RESULT_FILES=()
    INST_START_TIMES=()
    LAUNCH_STAGGER=20
    INIT_BUFFER_EXTRA=30
    LAST_LAUNCH_OFFSET=$(( (NUM_MODELS - 1) * LAUNCH_STAGGER ))
    INIT_BUFFER=$(( LAST_LAUNCH_OFFSET + INIT_BUFFER_EXTRA ))
    SWEEP_START=$(date +%s%3N)
    FIRST_START_AT=$(echo "scale=3; ($SWEEP_START + $INIT_BUFFER * 1000) / 1000" | bc)

    echo "  Launch stagger: ${LAUNCH_STAGGER}s  Init buffer: ${INIT_BUFFER}s  First send at: t=+${INIT_BUFFER}s"

    for i in "${!MODELS[@]}"; do
        MODEL=$(echo "${MODELS[$i]}" | cut -d':' -f1)
        PORT=$(echo "${MODELS[$i]}" | cut -d':' -f2)
        INST=$((i + 1))
        RESULT_FILE="${RESULTS_DIR}/${RUN_TAG}-${BACKEND}-peak${RAMP_PEAK_RPS}-prompt${PROMPT_LEN}-completion${COMPLETION_LEN}-inst${INST}.json"

        if [ $i -gt 0 ]; then
            sleep "$LAUNCH_STAGGER"
        fi

        START_AT=$(echo "scale=3; $FIRST_START_AT + $i * $MODEL_DELAY" | bc)
        INST_START_TIMES[$i]=$(date +%s)
        INST_NUM_PROMPTS=$BASE_NUM_PROMPTS

        "$PYTHON" bench_kvcached_vllm.py \
            --backend "$BACKEND" \
            --model "$MODEL" \
            --dataset-name random \
            --random-input-len "$PROMPT_LEN" \
            --random-output-len "$COMPLETION_LEN" \
            --num-prompts "$INST_NUM_PROMPTS" \
            --host "localhost" \
            --port "$PORT" \
            --endpoint "/v1/completions" \
            --save-result \
            --result-filename "$RESULT_FILE" \
            --metadata "peak_rps=${RAMP_PEAK_RPS},tag=${RUN_TAG},num_models=${NUM_MODELS}" \
            --ramp-up-strategy ramp-up-down \
            --ramp-start-rps "$RAMP_START_RPS" \
            --ramp-end-rps "$RAMP_END_RPS" \
            --ramp-peak-rps "$RAMP_PEAK_RPS" \
            --ramp-increment "$RAMP_INCREMENT" \
            --start-at "$START_AT" &

        PIDS+=($!)
        RESULT_FILES+=("$RESULT_FILE")
        echo "  Launched instance ${INST} (port ${PORT}) PID=${PIDS[$i]}, start_at=+$(echo "scale=1; $i * $MODEL_DELAY + $INIT_BUFFER" | bc)s"
    done

    echo "  Waiting for all instances to finish..."
    FAILED=0
    INST_END_TIMES=()
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" || { echo "  WARNING: instance $((i+1)) failed"; FAILED=1; }
        INST_END_TIMES[$i]=$(date +%s)
    done

    echo ""
    echo "  --- Instance timing (wall clock) ---"
    for i in "${!INST_START_TIMES[@]}"; do
        INST=$((i + 1))
        START=${INST_START_TIMES[$i]}
        END=${INST_END_TIMES[$i]}
        DURATION=$(( END - START ))
        OFFSET=$(( START - SWEEP_START / 1000 ))
        echo "  Instance ${INST}: duration=${DURATION}s"
    done

    if [ "$FAILED" -eq 0 ]; then
        echo "  Peak=${RAMP_PEAK_RPS} done. Results:"
        for f in "${RESULT_FILES[@]}"; do
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
echo "=== Sweep complete. All results in ${RESULTS_DIR}/ ==="
