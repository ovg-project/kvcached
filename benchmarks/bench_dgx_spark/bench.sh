#!/bin/bash
# Run a concurrency sweep for Guardrail -> LLM -> Guardrail.
#
# Usage:
#   ./bench.sh                # results saved to results/kvcached/
#   ./bench.sh baseline       # results saved to results/baseline/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

PHASE=${1:-kvcached}
OUT_DIR="${RESULTS_DIR}/${PHASE}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/bench.log"
exec >> "$LOG" 2>&1

echo "===== $(date -Is) bench start phase=$PHASE ====="

for i in $(seq 1 120); do
  curl -sf "http://localhost:${MAIN_PORT}/health" >/dev/null 2>&1 && { echo "main ready @ ${i}*5s"; break; }
  sleep 5
done
curl -sf "http://localhost:${MAIN_PORT}/health" >/dev/null 2>&1 || { echo "main never ready"; exit 1; }

for i in $(seq 1 120); do
  curl -sf "http://localhost:${GUARD_PORT}/health" >/dev/null 2>&1 && { echo "guard ready @ ${i}*5s"; break; }
  sleep 5
done
curl -sf "http://localhost:${GUARD_PORT}/health" >/dev/null 2>&1 || { echo "guard never ready"; exit 1; }

if [[ "$DATASET_NAME" == "sharegpt" && ! -f "$DATASET_PATH" ]]; then
  if [[ "$DOWNLOAD_SHAREGPT" == "1" ]]; then
    echo "downloading ShareGPT dataset to $DATASET_PATH"
    curl -L --fail --retry 3 -o "$DATASET_PATH" "$SHAREGPT_URL"
  else
    echo "ShareGPT dataset missing: $DATASET_PATH" >&2
    exit 1
  fi
fi

rm -f "$OUT_DIR"/c*.json "$OUT_DIR"/summary.json "$OUT_DIR"/done.marker "$OUT_DIR"/failed.marker

read -r -a CONCS <<< "$CONCURRENCIES"
for C in "${CONCS[@]}"; do
  NP=$(( C * NUM_PROMPTS_MULTIPLIER < MIN_NUM_PROMPTS ? MIN_NUM_PROMPTS : C * NUM_PROMPTS_MULTIPLIER ))
  echo "--- $(date -Is) C=$C NP=$NP ---"
  if ! timeout "$BENCH_TIMEOUT_SECONDS" python3 "$SCRIPT_DIR/workflow_benchmark.py" \
    --main-base-url "http://localhost:${MAIN_PORT}" \
    --guard-base-url "http://localhost:${GUARD_PORT}" \
    --main-model "$MAIN_MODEL" \
    --guard-model "$GUARD_MODEL" \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --random-input-len "$BENCH_INPUT_LEN" \
    --main-output-len "$BENCH_OUTPUT_LEN" \
    --num-prompts "$NP" \
    --max-concurrency "$C" \
    --phase "$PHASE" \
    --result-file "$OUT_DIR/c${C}.json" \
    --disable-thinking; then
    echo "benchmark failed for C=$C"
    touch "$OUT_DIR/failed.marker"
  fi
  python3 - "$OUT_DIR/c${C}.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(f"missing result: {path}")
    raise SystemExit(0)
d = json.loads(path.read_text())
print(
    "completed={completed} failed={failed} mean_ttft_ms={mean_ttft_ms:.2f} "
    "p99_ttft_ms={p99_ttft_ms:.2f} mean_e2e_ms={mean_e2e_ms:.2f} "
    "p99_e2e_ms={p99_e2e_ms:.2f}".format(**d)
)
PY
  echo "--- done C=$C ---"
done

python3 "$SCRIPT_DIR/plot_results.py" --results-dir "$RESULTS_DIR" || true
touch "$OUT_DIR/done.marker"
echo "===== $(date -Is) bench done phase=$PHASE ====="
