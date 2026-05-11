#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# After nsys identifies the regressing kernel(s), this captures detailed
# memory metrics via ncu. Set KERNEL_REGEX to a regex matching the kernel name
# (e.g. "flash_attn|fa_fwd|fa3").

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR="$SCRIPT_DIR/ncu_runs"
LOG_DIR="$SCRIPT_DIR/ncu_logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

VLLM="/home/xingqi/miniforge3/envs/kvcached/bin/vllm"
MODEL="Qwen/Qwen3-0.6B"
PORT=12349
KERNEL_REGEX="${KERNEL_REGEX:-flash_fwd_splitkv_kernel}"
LAUNCH_SKIP="${LAUNCH_SKIP:-400}"
LAUNCH_COUNT="${LAUNCH_COUNT:-8}"

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
    echo "[$(date +%H:%M:%S)] ncu: $label"
    echo "Env: $env_str"
    echo "Kernel regex: $KERNEL_REGEX"
    echo "Skip: $LAUNCH_SKIP, Count: $LAUNCH_COUNT"
    echo "=========================================="

    cleanup_server

    local server_log="$LOG_DIR/${label}.server.log"
    : > "$server_log"

    # shellcheck disable=SC2086
    env $env_str \
        ncu \
            --target-processes all \
            --export "$OUT_DIR/${label}" \
            --force-overwrite \
            --kernel-name "regex:$KERNEL_REGEX" \
            --launch-skip "$LAUNCH_SKIP" \
            --launch-count "$LAUNCH_COUNT" \
            --section MemoryWorkloadAnalysis \
            --section MemoryWorkloadAnalysis_Tables \
            --section LaunchStats \
            --section Occupancy \
            --section SourceCounters \
            --section SpeedOfLight \
            --section SpeedOfLight_RooflineChart \
            --section WarpStateStats \
            "$VLLM" serve "$MODEL" \
            --port "$PORT" \
            --gpu-memory-utilization 0.5 \
            --max-model-len 2048 \
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

    # Just push 50 prompts; ncu replays each kernel many times so this will
    # take ~5-15 minutes. We don't need throughput here, only kernel metrics.
    "$VLLM" bench serve --backend vllm --model "$MODEL" --port "$PORT" \
        --dataset-name random --random-input-len 512 --random-output-len 128 \
        --num-prompts 50 --request-rate inf --seed 42 \
        > "$LOG_DIR/${label}.bench.log" 2>&1 &
    local bench_pid=$!
    echo "  [$(date +%H:%M:%S)] Bench started (pid $bench_pid)"
    # Wait until launch-count kernels collected then kill bench
    wait "$bench_pid" 2>/dev/null || true

    kill -INT $server_pid 2>/dev/null || true
    sleep 5
    cleanup_server
    sleep 3
}

run_one "layout_true" \
    "ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1"

run_one "layout_false" \
    "ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_CONTIGUOUS_LAYOUT=false"

echo
echo "=========================================="
echo "Reports in $OUT_DIR"
ls -la "$OUT_DIR" | tail -10
echo "=========================================="
