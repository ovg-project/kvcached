#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Phase 2: Guard -> Main concurrency sweep for ONE mode, sampling combined
# physical KV footprint (main + guard). Servers must already be up.
#   ./bench_workflow.sh baseline
#   ./bench_workflow.sh kvcached
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODE=${1:?usage: $0 <baseline|kvcached>}
OUT_DIR="${RESULTS_DIR}_phase2/${MODE}"
mkdir -p "$OUT_DIR"
ensure_sharegpt

# kvcached: sample BOTH segments (filter "kvcached" or empty=all). baseline: none.
IPC_FILTER=""
[[ "$MODE" == "kvcached" ]] && IPC_FILTER=""   # empty -> kv_monitor reports nothing for baseline; for kvcached set below
[[ "$MODE" == "kvcached" ]] && IPC_FILTER="$KVCACHED_MAIN_IPC"  # main is the big one; switch to "" to sum both

read -r -a CONCS <<< "$CONCURRENCIES"
for C in "${CONCS[@]}"; do
  NP=$(( C * NUM_PROMPTS_MULTIPLIER < MIN_NUM_PROMPTS ? MIN_NUM_PROMPTS : C * NUM_PROMPTS_MULTIPLIER ))
  echo "=== phase2 mode=$MODE C=$C NP=$NP ==="
  python3 "$SCRIPT_DIR/kv_monitor.py" --name-filter "$IPC_FILTER" \
    --metrics-url "http://localhost:${MAIN_PORT}/metrics" --interval 0.2 --gpu \
    --out "${OUT_DIR}/c${C}_mem.json" &
  MON=$!
  timeout 1800 python3 "$SCRIPT_DIR/workflow_benchmark.py" \
    --main-base-url "http://localhost:${MAIN_PORT}" \
    --guard-base-url "http://localhost:${GUARD_PORT}" \
    --main-model "$MAIN_MODEL" --guard-model "$GUARD_MODEL" \
    --dataset-name "$DATASET_NAME" --dataset-path "$DATASET_PATH" \
    --main-output-len "$MAIN_OUTPUT_LEN" --random-input-len "$RANDOM_INPUT_LEN" \
    --num-prompts "$NP" --max-concurrency "$C" --seed "$SEED" \
    --phase "$MODE" --result-file "${OUT_DIR}/c${C}.json" \
    || echo "WARN: workflow bench failed for C=$C"
  kill -TERM "$MON" 2>/dev/null || true; wait "$MON" 2>/dev/null || true
  echo "--- done C=$C ---"
done
echo "phase2 $MODE results in $OUT_DIR"
