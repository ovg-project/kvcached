#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Phase 1: single-model ShareGPT concurrency sweep for ONE mode, with a
# physical-KV-footprint sampler running alongside each concurrency point.
#
#   ./bench_single.sh baseline
#   ./bench_single.sh kvcached
#
# Results: results/<mode>/cN.json (perf, from vllm bench serve)
#          results/<mode>/cN_mem.json (peak physical KV + saturation guards)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODE=${1:?usage: $0 <baseline|kvcached>}
OUT_DIR="${RESULTS_DIR}/${MODE}"
mkdir -p "$OUT_DIR"
ensure_sharegpt

IPC_FILTER=""
[[ "$MODE" == "kvcached" ]] && IPC_FILTER="$KVCACHED_MAIN_IPC"

read -r -a CONCS <<< "$CONCURRENCIES"
for C in "${CONCS[@]}"; do
  NP=$(( C * NUM_PROMPTS_MULTIPLIER < MIN_NUM_PROMPTS ? MIN_NUM_PROMPTS : C * NUM_PROMPTS_MULTIPLIER ))
  echo "=== mode=$MODE C=$C NP=$NP ==="

  # start footprint sampler
  python3 "$SCRIPT_DIR/kv_monitor.py" \
    --name-filter "$IPC_FILTER" \
    --metrics-url "http://localhost:${MAIN_PORT}/metrics" \
    --interval 0.2 --gpu \
    --out "${OUT_DIR}/c${C}_mem.json" &
  MON=$!

  EXTRA=()
  [[ -n "$SHAREGPT_OUTPUT_LEN" ]] && EXTRA+=(--sharegpt-output-len "$SHAREGPT_OUTPUT_LEN")

  # gpt-oss Harmony chat path returns null on this build; raw completions are
  # coherent, so we drive the model via /v1/completions for the benchmark.
  vllm bench serve \
    --backend openai \
    --base-url "http://localhost:${MAIN_PORT}" \
    --endpoint /v1/completions \
    --model "$MAIN_MODEL" \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --num-prompts "$NP" \
    --max-concurrency "$C" \
    --seed "$SEED" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90 \
    --ignore-eos \
    --save-result \
    --result-dir "$OUT_DIR" \
    --result-filename "c${C}.json" \
    "${EXTRA[@]}" || echo "WARN: bench failed for C=$C"

  kill -TERM "$MON" 2>/dev/null || true
  wait "$MON" 2>/dev/null || true
  echo "--- done C=$C ---"
done
echo "phase1 $MODE results in $OUT_DIR"
