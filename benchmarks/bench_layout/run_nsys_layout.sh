#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Launch vllm-serve under nsys with KVCACHED_CONTIGUOUS_LAYOUT in {true,false}
# and drive it with a small bench burst. Captures GPU kernel timeline so we can
# diff per-kernel time between the two layouts.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR="$SCRIPT_DIR/nsys_runs"
LOG_DIR="$SCRIPT_DIR/nsys_logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

VLLM="/home/xingqi/miniforge3/envs/kvcached/bin/vllm"
MODEL="Qwen/Qwen3-0.6B"
PORT=12348
NUM_PROMPTS=${NUM_PROMPTS:-100}
RATE=${RATE:-inf}
CAPTURE_DURATION=${CAPTURE_DURATION:-90}  # seconds
WARMUP_PROMPTS=${WARMUP_PROMPTS:-30}

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
}

wait_for_ready() {
    local log=$1
    for i in $(seq 1 240); do
        if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        if grep -qE "EngineCore failed|Traceback|RuntimeError|FATAL" "$log" 2>/dev/null; then
            return 1
        fi
        sleep 2
    done
    return 1
}

run_one() {
    local label=$1
    local env_str=$2

    echo
    echo "=========================================="
    echo "[$(date +%H:%M:%S)] nsys profile: $label"
    echo "Env: $env_str"
    echo "=========================================="

    cleanup_server

    local server_log="$LOG_DIR/${label}.server.log"
    : > "$server_log"

    # Launch vllm serve under nsys. We don't constrain duration here -- we'll
    # rely on cudaProfilerApi-style capture range below for the steady-state
    # decode window. Until we add explicit hooks, just let nsys profile the
    # full lifetime and use --capture-range=none + --duration on the bench
    # window.  Simpler: profile the full run, mark the bench start/end with
    # NVTX ranges so we can filter post-hoc.
    #
    # Using --sample=none keeps overhead modest and CPU samples out.
    # --trace=cuda,nvtx,osrt captures kernel launches + NVTX + a bit of OS.

    # shellcheck disable=SC2086
    env $env_str \
        nsys profile \
            --output="$OUT_DIR/${label}" \
            --force-overwrite=true \
            --trace=cuda,nvtx \
            --sample=none \
            --cpuctxsw=none \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --gpu-metrics-device=none \
            "$VLLM" serve "$MODEL" \
            --port "$PORT" \
            --gpu-memory-utilization 0.5 \
            --max-model-len 2048 \
            --profiler-config '{"profiler":"cuda","ignore_frontend":true}' \
            > "$server_log" 2>&1 &
    local server_pid=$!

    if ! wait_for_ready "$server_log"; then
        echo "  FAILED: server didn't come up; tail of log:"
        tail -30 "$server_log"
        kill -9 $server_pid 2>/dev/null || true
        cleanup_server
        return 1
    fi
    echo "  [$(date +%H:%M:%S)] Server ready"

    # Warmup (untimed, fills caches / triggers compile)
    if [[ "$WARMUP_PROMPTS" -gt 0 ]]; then
        echo "  [$(date +%H:%M:%S)] Warmup ${WARMUP_PROMPTS} prompts..."
        "$VLLM" bench serve --backend vllm --model "$MODEL" --port "$PORT" \
            --dataset-name random --random-input-len 512 --random-output-len 128 \
            --num-prompts "$WARMUP_PROMPTS" --request-rate "$RATE" --seed 1 \
            > "$LOG_DIR/${label}.warmup.log" 2>&1 || true
    fi

    # Start the profiler window via the cudaProfilerStart side-channel.
    # Easiest path: hit vllm's /start_profile endpoint, run the bench, then
    # hit /stop_profile. vllm's start_profile uses torch.profiler, which calls
    # cudaProfilerStart underneath. This matches nsys --capture-range=cudaProfilerApi.
    #
    # If /start_profile is not available (vllm too old), we fall back to a
    # full-trace capture without --capture-range.

    if curl -fsS -X POST "http://localhost:$PORT/start_profile" >/dev/null 2>&1; then
        echo "  [$(date +%H:%M:%S)] /start_profile hit. Running bench..."

        "$VLLM" bench serve --backend vllm --model "$MODEL" --port "$PORT" \
            --dataset-name random --random-input-len 512 --random-output-len 128 \
            --num-prompts "$NUM_PROMPTS" --request-rate "$RATE" --seed 42 \
            > "$LOG_DIR/${label}.bench.log" 2>&1
        local bench_rc=$?

        curl -fsS -X POST "http://localhost:$PORT/stop_profile" >/dev/null 2>&1 || true
        echo "  [$(date +%H:%M:%S)] /stop_profile hit. bench rc=$bench_rc"
    else
        echo "  /start_profile not available; profiler window not bounded. Tearing down."
    fi

    # Extract bench summary
    grep -E "Request throughput|Mean TTFT|Mean TPOT|Total token throughput|Successful requests" \
        "$LOG_DIR/${label}.bench.log" | sed 's/^/    /'

    kill -INT $server_pid 2>/dev/null || true
    sleep 5
    cleanup_server
    sleep 3
}

# Layout = true (default)
run_one "layout_true" \
    "ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1"

# Layout = false
run_one "layout_false" \
    "ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_CONTIGUOUS_LAYOUT=false"

echo
echo "=========================================="
echo "Traces in $OUT_DIR"
ls -la "$OUT_DIR" | tail -10
echo "=========================================="
