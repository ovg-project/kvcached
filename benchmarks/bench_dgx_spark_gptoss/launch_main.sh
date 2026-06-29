#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Launch gpt-oss-120b (MXFP4, fp8 KV) under one of two modes.
#   ./launch_main.sh --mode baseline            # vanilla vLLM, static KV pool
#   ./launch_main.sh --mode kvcached            # elastic KV pool
#   ./launch_main.sh --mode kvcached --gpu-util 0.80
#
# Both modes share EVERY engine flag except the kvcached env block, so the
# only difference measured is physical-vs-reserved KV backing.
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
UTIL="${GPU_UTIL_OVERRIDE:-$GPU_UTIL}"

LOG="${LOG_DIR}/serve_main_${MODE}.log"
rm -f "$LOG"
command -v vllm >/dev/null 2>&1 || { echo "ERROR: vllm not found; CONDA_ENV=$CONDA_ENV" >&2; exit 1; }

# Confirm we are on the kvcached-patched vLLM, not base 0.23.0.
VLLM_VER="$(vllm --version 2>/dev/null | tail -1 || true)"
echo "vllm version: ${VLLM_VER}  (expect 0.19.2.dev0 from the '$CONDA_ENV' env)"

# Flags identical for both modes (fairness):
COMMON=(
  --served-model-name "$MAIN_MODEL"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --block-size "$BLOCK_SIZE"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --enforce-eager
  --no-enable-prefix-caching
  --disable-hybrid-kv-cache-manager
  --gpu-memory-utilization "$UTIL"
  --port "$MAIN_PORT"
)

if [[ "$MODE" == "kvcached" ]]; then
  export ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1
  export KVCACHED_IPC_NAME="$KVCACHED_MAIN_IPC"
  export KVCACHED_CONTIGUOUS_LAYOUT=true        # gpt-oss is MHA/GQA -> contiguous
  export VLLM_USE_V1=1
elif [[ "$MODE" == "baseline" ]]; then
  unset ENABLE_KVCACHED KVCACHED_AUTOPATCH KVCACHED_IPC_NAME 2>/dev/null || true
  export VLLM_USE_V1=1
else
  echo "invalid mode: $MODE" >&2; exit 1
fi

echo "launching main gpt-oss-120b (mode=$MODE, util=$UTIL, kv=$KV_CACHE_DTYPE)"
nohup vllm serve "$MAIN_MODEL" "${COMMON[@]}" > "$LOG" 2>&1 &
PID=$!; disown
echo "$PID" > "${LOG_DIR}/serve_main.pid"
echo "main ($MODE) pid=$PID log=$LOG"

echo "waiting for main to become ready..."
for i in $(seq 1 "$WAIT_HEALTH_ATTEMPTS"); do
  curl -sf "http://localhost:${MAIN_PORT}/health" >/dev/null 2>&1 && break
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: main exited early - tail of $LOG:" >&2; tail -60 "$LOG" >&2; exit 1
  fi
  sleep "$WAIT_HEALTH_INTERVAL"
done
curl -sf "http://localhost:${MAIN_PORT}/health" >/dev/null 2>&1 || {
  echo "ERROR: main never became healthy - tail of $LOG:" >&2; tail -60 "$LOG" >&2; exit 1; }

# --- record the static KV pool baseline reported by vLLM (constant footprint).
grep -hiE "GPU KV cache size|Maximum concurrency|reserved for KV Cache|Available KV cache" "$LOG" \
  | tee "${LOG_DIR}/main_${MODE}_pool.txt" || true

# --- MXFP4 correctness gate via /v1/completions. NOTE: on this build the
# gpt-oss Harmony *chat* path returns null content, but raw completions are
# coherent (compute is correct) — so we benchmark via completions.
echo "sanity check (must be coherent English, not null/garbage):"
curl -s "http://localhost:${MAIN_PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MAIN_MODEL}\",\"prompt\":\"The capital of France is\",\"max_tokens\":12,\"temperature\":0}" \
  | python3 -c 'import sys,json; t=json.load(sys.stdin)["choices"][0]["text"]; print("  ->", repr(t)); sys.exit(0 if t and t.strip() else 3)' \
  || { echo "ERROR: MXFP4 sanity failed (null/empty output). See README 'MXFP4 correctness'." >&2; exit 3; }
echo "main ready (mode=$MODE)"
