#!/bin/bash
# Run a concurrency sweep against main on :12346 using vllm bench serve.
# Arg 1 = phase name (subdir under /workspace/kvcached/_p4_results/).
set -uo pipefail
PHASE=${1:-kvcached}
RESULTS_DIR=/workspace/kvcached/_p4_results/$PHASE
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"
exec >>"$LOG" 2>&1
echo "===== $(date -Is) BENCH START phase=$PHASE ====="

# Wait for main to be ready (max ~10 min)
for i in $(seq 1 120); do
  curl -sf http://localhost:12346/health >/dev/null 2>&1 && { echo "main ready @ ${i}*5s"; break; }
  sleep 5
done
curl -sf http://localhost:12346/health >/dev/null 2>&1 || { echo "MAIN NEVER READY"; touch "$RESULTS_DIR/failed.marker"; exit 1; }

CONCS=(4 8 16 32 64 128)  # drop C=1 — too slow at output 2048, low-concurrency is uninteresting
for C in "${CONCS[@]}"; do
  NP=$((C * 2 < 32 ? 32 : C * 2))
  echo "--- $(date -Is) C=$C NP=$NP ---"
  timeout 1200 vllm bench serve \
    --backend openai-chat \
    --base-url http://localhost:12346 \
    --endpoint /v1/chat/completions \
    --model Qwen/Qwen3-30B-A3B \
    --dataset-name random \
    --random-input-len 256 --random-output-len 2048 \
    --num-prompts "$NP" \
    --max-concurrency "$C" \
    --save-result \
    --result-dir "$RESULTS_DIR" \
    --result-filename "c${C}.json" 2>&1 | tail -30
  echo "--- done C=$C ---"
done
touch "$RESULTS_DIR/done.marker"
echo "===== $(date -Is) BENCH DONE phase=$PHASE ====="
