#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Phase 2 co-tenant: launch the guardrail model (Qwen2-VL-7B-Instruct) sharing
# the GPU with gpt-oss-120b.
#   ./launch_guard.sh --mode baseline --gpu-util 0.16
#   ./launch_guard.sh --mode kvcached
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODE=baseline
GPU_UTIL_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode) MODE="$2"; shift 2 ;;
    --gpu-util) GPU_UTIL_OVERRIDE="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ "$MODE" == "kvcached" ]]; then
  UTIL="${GPU_UTIL_OVERRIDE:-$KVCACHED_GUARD_GPU_UTIL}"
else
  UTIL="${GPU_UTIL_OVERRIDE:-$BASELINE_GUARD_GPU_UTIL}"
fi
LOG="${LOG_DIR}/serve_guard_${MODE}.log"; rm -f "$LOG"

COMMON=(
  --served-model-name "$GUARD_MODEL"
  --max-model-len "$GUARD_MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --block-size "$BLOCK_SIZE"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --enforce-eager
  --no-enable-prefix-caching
  --gpu-memory-utilization "$UTIL"
  --port "$GUARD_PORT"
)

if [[ "$MODE" == "kvcached" ]]; then
  export ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1
  export KVCACHED_IPC_NAME="$KVCACHED_GUARD_IPC"
  export KVCACHED_CONTIGUOUS_LAYOUT=true
  export VLLM_USE_V1=1
else
  unset ENABLE_KVCACHED KVCACHED_AUTOPATCH KVCACHED_IPC_NAME 2>/dev/null || true
  export VLLM_USE_V1=1
fi

echo "launching guard $GUARD_MODEL (mode=$MODE, util=$UTIL)"
nohup vllm serve "$GUARD_MODEL" "${COMMON[@]}" > "$LOG" 2>&1 &
PID=$!; disown
echo "$PID" > "${LOG_DIR}/serve_guard.pid"

for i in $(seq 1 "$WAIT_HEALTH_ATTEMPTS"); do
  curl -sf "http://localhost:${GUARD_PORT}/health" >/dev/null 2>&1 && break
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: guard exited early - tail of $LOG:" >&2; tail -60 "$LOG" >&2; exit 1
  fi
  sleep "$WAIT_HEALTH_INTERVAL"
done
curl -sf "http://localhost:${GUARD_PORT}/health" >/dev/null 2>&1 \
  && echo "guard ready (mode=$MODE)" \
  || { echo "ERROR: guard never healthy" >&2; tail -60 "$LOG" >&2; exit 1; }
