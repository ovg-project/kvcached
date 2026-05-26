#!/bin/bash
# End-to-end benchmark: launch servers, run Guardrail -> LLM sweep, stop servers.
#
# Usage:
#   ./run_benchmark.sh                  # run kvcached mode
#   ./run_benchmark.sh baseline         # run baseline mode
#   ./run_benchmark.sh baseline 0.65    # baseline with custom gpu-util for main
#   ./run_benchmark.sh both             # run kvcached, then baseline, then plot
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODE=${1:-kvcached}
GPU_UTIL=${2:-$BASELINE_MAIN_GPU_UTIL}

cleanup() {
  if [[ "${KEEP_SERVERS:-0}" != "1" ]]; then
    "$SCRIPT_DIR/stop.sh" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

run_one() {
  local mode=$1
  local gpu_util=$2

  echo "===== DGX Spark benchmark: mode=$mode ====="

  "$SCRIPT_DIR/stop.sh"
  sleep 2

  if [[ "$mode" == "kvcached" ]]; then
    "$SCRIPT_DIR/launch_guard.sh" --mode kvcached
    "$SCRIPT_DIR/launch_main.sh" --mode kvcached
  elif [[ "$mode" == "baseline" ]]; then
    "$SCRIPT_DIR/launch_main.sh" --mode baseline --gpu-util "$gpu_util"
    "$SCRIPT_DIR/launch_guard.sh" --mode baseline --gpu-util "$BASELINE_GUARD_GPU_UTIL"
  else
    echo "invalid mode: $mode (use kvcached, baseline, or both)" >&2
    exit 1
  fi

  "$SCRIPT_DIR/bench.sh" "$mode"
  "$SCRIPT_DIR/stop.sh"
  echo "===== done: $mode results in ${RESULTS_DIR}/${mode}/ ====="
}

if [[ "$MODE" == "both" ]]; then
  run_one kvcached "$GPU_UTIL"
  run_one baseline "$GPU_UTIL"
  python3 "$SCRIPT_DIR/plot_results.py" --results-dir "$RESULTS_DIR"
else
  run_one "$MODE" "$GPU_UTIL"
fi

echo "===== done - results in ${RESULTS_DIR}/ ====="
