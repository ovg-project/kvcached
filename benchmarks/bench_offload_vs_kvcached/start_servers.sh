#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Start two vLLM servers sharing one GPU under one of three KV-memory modes:
#   static   - vanilla vLLM, static GPU memory split (0.45 each)
#   offload  - static split + native CPU offloading (OffloadingConnector)
#   kvcached - elastic GPU memory sharing via kvcached
#
# Usage: ./start_servers.sh <static|offload|kvcached> [logdir]
set -e

MODE=${1:?usage: $0 <static|offload|kvcached> [logdir]}
LOGDIR=${2:-logs/$MODE}
mkdir -p "$LOGDIR"

ENV_BIN=/home/exouser/miniconda3/envs/kvc-bench/bin
MODEL="Qwen/Qwen3-4B"
PORT_A=8100
PORT_B=8200

# CPU RAM per instance for offloaded KV blocks. Sized to hold the full
# benchmark working set (~12 GiB) while keeping 2x pinned allocations + vLLM
# process memory within system RAM (115 GiB; 40 GiB each triggered the OOM
# killer).
CPU_BYTES=$((16 * 1024 * 1024 * 1024))
OFFLOAD_CFG="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$CPU_BYTES}}"

COMMON_ARGS=(
    --max-model-len 8192
    --max-num-seqs 64
    --block-size 16
)

start_one() {
    local port=$1 name=$2; shift 2
    echo "starting instance $name on port $port (mode=$MODE)"
    if [ "$MODE" = "kvcached" ]; then
        # MAX_CACHED_TOKENS=0: evict prefix-cache blocks on free so empty
        # pages return to the shared pool immediately. With the default
        # (16000), cached blocks fragment across 2MB pages and pin several
        # GiB on an idle instance, starving the other instance's burst.
        # This is conservative for kvcached: it gives up cross-request
        # prefix reuse that the static/offload baselines keep.
        ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_IPC_NAME=VLLM \
        KVCACHED_MAX_CACHED_TOKENS=0 \
        "$ENV_BIN/vllm" serve "$MODEL" --port "$port" "${COMMON_ARGS[@]}" "$@" \
            > "$LOGDIR/server-$name.log" 2>&1 &
    else
        "$ENV_BIN/vllm" serve "$MODEL" --port "$port" "${COMMON_ARGS[@]}" "$@" \
            > "$LOGDIR/server-$name.log" 2>&1 &
    fi
    echo $! > "$LOGDIR/server-$name.pid"
}

wait_health() {
    local port=$1 name=$2
    for i in $(seq 1 180); do
        if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
            echo "instance $name healthy"
            return 0
        fi
        if ! kill -0 "$(cat "$LOGDIR/server-$name.pid")" 2>/dev/null; then
            echo "instance $name died; tail of log:"
            tail -20 "$LOGDIR/server-$name.log"
            return 1
        fi
        sleep 5
    done
    echo "instance $name failed to become healthy in time"
    return 1
}

case "$MODE" in
  static)
    start_one $PORT_A A --gpu-memory-utilization 0.45
    wait_health $PORT_A A
    start_one $PORT_B B --gpu-memory-utilization 0.45
    wait_health $PORT_B B
    ;;
  offload)
    start_one $PORT_A A --gpu-memory-utilization 0.45 --kv-transfer-config "$OFFLOAD_CFG"
    wait_health $PORT_A A
    start_one $PORT_B B --gpu-memory-utilization 0.45 --kv-transfer-config "$OFFLOAD_CFG"
    wait_health $PORT_B B
    ;;
  kvcached)
    start_one $PORT_A A
    wait_health $PORT_A A
    start_one $PORT_B B
    wait_health $PORT_B B
    ;;
  *)
    echo "unknown mode: $MODE"; exit 1;;
esac

echo "both servers up (mode=$MODE)"
grep -h "GPU KV cache size" "$LOGDIR"/server-*.log || true
