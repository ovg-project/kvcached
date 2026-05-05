#!/bin/bash
# Sweep e2e configs for kvcached overhead investigation.
# Each config: start server with specified env, run warmup + 3 seeds, parse, kill.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULTS_DIR="$SCRIPT_DIR/sweep_results"
LOG_DIR="$SCRIPT_DIR/sweep_logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

VENV_PY="/home/xingqi/miniforge3/envs/kvcached/bin/python"
VLLM="/home/xingqi/miniforge3/envs/kvcached/bin/vllm"

MODEL="Qwen/Qwen3-0.6B"
PORT=12347
GPU_MEM_UTIL=0.5
MAX_MODEL_LEN=2048

WARMUP_PROMPTS=100
NUM_PROMPTS=500
INPUT_LEN=512
OUTPUT_LEN=128
SEEDS=(42 99 7)

cleanup_server() {
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    # Wait for port to be free
    for i in $(seq 1 30); do
        if ! ss -ltn 2>/dev/null | grep -q ":$PORT "; then
            return 0
        fi
        sleep 1
    done
    return 1
}

wait_for_server() {
    local log=$1
    for i in $(seq 1 240); do
        if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        if grep -q "Error\|Traceback" "$log" 2>/dev/null; then
            echo "Server crashed; see $log"
            return 1
        fi
        sleep 2
    done
    return 1
}

# Args: label, request_rate, env_string ("KEY1=VAL1 KEY2=VAL2"), extra_server_args
run_config() {
    local label=$1
    local request_rate=$2
    local env_str=$3
    local extra=$4

    echo
    echo "=========================================="
    echo "Config: $label   rate=$request_rate"
    echo "Env: $env_str"
    echo "=========================================="

    cleanup_server

    local server_log="$LOG_DIR/${label}.server.log"
    : > "$server_log"

    # shellcheck disable=SC2086
    env $env_str \
        "$VLLM" serve "$MODEL" \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        $extra \
        > "$server_log" 2>&1 &
    local server_pid=$!

    if ! wait_for_server "$server_log"; then
        echo "  FAILED: server didn't come up"
        kill -9 $server_pid 2>/dev/null || true
        cleanup_server
        return 1
    fi

    echo "  Server up (pid=$server_pid). Running warmup..."
    "$VLLM" bench serve \
        --backend vllm \
        --model "$MODEL" \
        --port "$PORT" \
        --dataset-name random \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --num-prompts $WARMUP_PROMPTS \
        --request-rate "$request_rate" \
        --seed 1 \
        > "$LOG_DIR/${label}.warmup.log" 2>&1 || true

    for seed in "${SEEDS[@]}"; do
        echo "  Run seed=$seed..."
        local out="$RESULTS_DIR/${label}.seed${seed}.json"
        "$VLLM" bench serve \
            --backend vllm \
            --model "$MODEL" \
            --port "$PORT" \
            --dataset-name random \
            --random-input-len $INPUT_LEN \
            --random-output-len $OUTPUT_LEN \
            --num-prompts $NUM_PROMPTS \
            --request-rate "$request_rate" \
            --seed "$seed" \
            --save-result \
            --result-dir "$RESULTS_DIR" \
            --result-filename "${label}.seed${seed}.json" \
            > "$LOG_DIR/${label}.seed${seed}.log" 2>&1
    done

    echo "  Done. Tearing down server."
    kill -INT $server_pid 2>/dev/null || true
    sleep 3
    cleanup_server
}

# ---------------- The actual sweep ----------------

# Config A: vanilla vLLM (no kvcached)
run_config "A_vanilla_inf"  inf  ""  ""

# Config B: kvcached default (fix branch)
run_config "B_kvcached_default_inf"  inf  \
    "ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1"  ""

# Config C: kvcached, CONTIGUOUS_LAYOUT=false
run_config "C_layout_false_inf"  inf  \
    "ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_CONTIGUOUS_LAYOUT=false"  ""

# Config D: kvcached, MAX_RESERVED_PAGES=200
run_config "D_reserved200_inf"  inf  \
    "ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_MIN_RESERVED_PAGES=50 KVCACHED_MAX_RESERVED_PAGES=200"  ""

# Config E: kvcached, both knobs
run_config "E_both_inf"  inf  \
    "ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_CONTIGUOUS_LAYOUT=false KVCACHED_MIN_RESERVED_PAGES=50 KVCACHED_MAX_RESERVED_PAGES=200"  ""

# Config F: vanilla, request-rate=16
run_config "F_vanilla_r16"  16  ""  ""

# Config G: kvcached default, request-rate=16
run_config "G_kvcached_default_r16"  16  \
    "ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1"  ""

# Config H: kvcached best, request-rate=16
run_config "H_best_r16"  16  \
    "ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_CONTIGUOUS_LAYOUT=false KVCACHED_MIN_RESERVED_PAGES=50 KVCACHED_MAX_RESERVED_PAGES=200"  ""

echo
echo "=========================================="
echo "All configs done. Results in $RESULTS_DIR"
echo "=========================================="
