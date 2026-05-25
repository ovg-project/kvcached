#!/bin/bash
# Launch the main LLM for the DGX Spark workflow demo.
#
# Usage:
#   ./launch_main.sh                     # kvcached mode (default)
#   ./launch_main.sh --mode baseline     # baseline with static memory split
#   ./launch_main.sh --mode baseline --gpu-util 0.65
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODE=kvcached
GPU_UTIL="$BASELINE_MAIN_GPU_UTIL"

while [[ $# -gt 0 ]]; do
  case $1 in
    --mode) MODE="$2"; shift 2 ;;
    --gpu-util) GPU_UTIL="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

LOG="${LOG_DIR}/serve_main.log"
rm -f "$LOG"

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm not found; activate the ${CONDA_ENV} environment or set CONDA_ENV" >&2
  exit 1
fi

if [[ "$MODE" == "kvcached" ]]; then
  export ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1
  export KVCACHED_IPC_NAME="${KVCACHED_MAIN_IPC_NAME:-kvcached_main}"
  export KVCACHED_CONTIGUOUS_LAYOUT=false
  export KVCACHED_PAGE_PREALLOC_ENABLED="${KVCACHED_PAGE_PREALLOC_ENABLED:-false}"
  export KVCACHED_PAGE_SIZE_MB
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  export VLLM_USE_V1=1
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  nohup vllm serve "$MAIN_MODEL_PATH" \
    --enforce-eager --no-enable-prefix-caching \
    --served-model-name "$MAIN_MODEL" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":40960}}' \
    --max-model-len "$MAIN_MAX_MODEL_LEN" --port "$MAIN_PORT" \
    --gpu-memory-utilization "$KVCACHED_MAIN_GPU_UTIL" > "$LOG" 2>&1 &
elif [[ "$MODE" == "baseline" ]]; then
  unset ENABLE_KVCACHED KVCACHED_AUTOPATCH KVCACHED_IPC_NAME 2>/dev/null || true
  export VLLM_USE_V1=1
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  nohup vllm serve "$MAIN_MODEL_PATH" \
    --enforce-eager --no-enable-prefix-caching \
    --served-model-name "$MAIN_MODEL" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len "$BASELINE_MAIN_MAX_MODEL_LEN" --port "$MAIN_PORT" \
    --gpu-memory-utilization "$GPU_UTIL" > "$LOG" 2>&1 &
else
  echo "invalid mode: $MODE (use kvcached or baseline)" >&2; exit 1
fi

PID=$!
disown
echo "main ($MODE) launched - pid=$PID log=$LOG"

echo "waiting for main to become ready..."
for i in $(seq 1 "$WAIT_HEALTH_ATTEMPTS"); do
  curl -sf "http://localhost:${MAIN_PORT}/health" >/dev/null 2>&1 && break
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: main process exited early - check $LOG" >&2
    tail -80 "$LOG" >&2 || true
    exit 1
  fi
  sleep "$WAIT_HEALTH_INTERVAL"
done

if curl -sf "http://localhost:${MAIN_PORT}/health" >/dev/null 2>&1; then
  echo "main ready"
  curl -s "http://localhost:${MAIN_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MAIN_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":20,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | python3 -c 'import sys,json; print("sanity:", repr(json.load(sys.stdin)["choices"][0]["message"]["content"]))'
else
  echo "ERROR: main failed to start - check $LOG" >&2
  tail -80 "$LOG" >&2 || true
  exit 1
fi
