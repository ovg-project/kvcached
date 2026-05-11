#!/bin/bash
# Re-run C (CONTIGUOUS_LAYOUT=false) and H (best r16) after the
# kv_cache_manager.py:107 fix.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULTS_DIR="$SCRIPT_DIR/sweep_results"
LOG_DIR="$SCRIPT_DIR/sweep_logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

VLLM="/home/xingqi/miniforge3/envs/kvcached/bin/vllm"
MODEL="Qwen/Qwen3-0.6B"
PORT=12347
SEEDS=(42 99 7)

cleanup_server() {
    pkill -INT -f "vllm serve Qwen" 2>/dev/null || true
    pkill -INT -f "VLLM::EngineCore" 2>/dev/null || true
    for i in $(seq 1 30); do
        if ! ss -lnt 2>/dev/null | awk '$4 ~ /:'"$PORT"'$/' | grep -q .; then
            return 0
        fi
        sleep 1
    done
    pkill -KILL -f "vllm serve" 2>/dev/null || true
    pkill -KILL -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 2
    return 0
}

wait_for_ready() {
    local log=$1
    for i in $(seq 1 240); do
        if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        if grep -qE "EngineCore failed|Traceback|RuntimeError|FATAL|Engine core initialization failed|invalid argument" "$log" 2>/dev/null; then
            return 1
        fi
        sleep 2
    done
    return 1
}

run_config() {
    local label=$1
    local request_rate=$2
    local env_str=$3

    echo
    echo "=========================================="
    echo "[$(date +%H:%M:%S)] Config: $label   rate=$request_rate"
    echo "Env: $env_str"
    echo "=========================================="

    cleanup_server

    local server_log="$LOG_DIR/${label}.server.log"
    : > "$server_log"

    # shellcheck disable=SC2086
    env $env_str \
        "$VLLM" serve "$MODEL" \
        --port "$PORT" \
        --gpu-memory-utilization 0.5 \
        --max-model-len 2048 \
        > "$server_log" 2>&1 &
    local server_pid=$!

    if ! wait_for_ready "$server_log"; then
        echo "  FAILED: server didn't come up; tail:"
        tail -25 "$server_log"
        kill -9 $server_pid 2>/dev/null || true
        cleanup_server
        return 1
    fi

    echo "  [$(date +%H:%M:%S)] Server up. Warmup..."
    "$VLLM" bench serve \
        --backend vllm --model "$MODEL" --port "$PORT" \
        --dataset-name random --random-input-len 512 --random-output-len 128 \
        --num-prompts 100 --request-rate "$request_rate" --seed 1 \
        > "$LOG_DIR/${label}.warmup.log" 2>&1 || true

    for seed in "${SEEDS[@]}"; do
        echo "  [$(date +%H:%M:%S)] seed=$seed"
        "$VLLM" bench serve \
            --backend vllm --model "$MODEL" --port "$PORT" \
            --dataset-name random --random-input-len 512 --random-output-len 128 \
            --num-prompts 500 --request-rate "$request_rate" --seed "$seed" \
            --save-result --result-dir "$RESULTS_DIR" \
            --result-filename "${label}.seed${seed}.json" \
            > "$LOG_DIR/${label}.seed${seed}.log" 2>&1
        grep -E "Request throughput|Mean TTFT|Mean TPOT|P99 TTFT|P99 TPOT|Successful requests" \
            "$LOG_DIR/${label}.seed${seed}.log" | sed 's/^/    /'
    done

    echo "  [$(date +%H:%M:%S)] Tearing down..."
    kill -INT $server_pid 2>/dev/null || true
    sleep 3
    cleanup_server
}

# Re-run C with the fix
run_config "C_layout_false_inf" inf \
    "ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_CONTIGUOUS_LAYOUT=false"

# Re-run H with the fix
run_config "H_best_r16" 16 \
    "ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_CONTIGUOUS_LAYOUT=false KVCACHED_MIN_RESERVED_PAGES=50 KVCACHED_MAX_RESERVED_PAGES=200"

echo
echo "=========================================="
echo "[$(date +%H:%M:%S)] Re-run done."
echo "=========================================="
