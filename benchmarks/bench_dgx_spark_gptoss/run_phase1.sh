#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Phase 1 end-to-end: for each mode, launch gpt-oss-120b, run the ShareGPT
# concurrency sweep with the footprint sampler, stop. Then plot.
#
#   ./run_phase1.sh            # both modes (baseline then kvcached) + plot
#   ./run_phase1.sh baseline   # one mode only
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

run_one() {
  local mode=$1
  echo "############ Phase 1: mode=$mode ############"
  "$SCRIPT_DIR/stop.sh"; sleep 2
  "$SCRIPT_DIR/launch_main.sh" --mode "$mode" --gpu-util "$GPU_UTIL"
  "$SCRIPT_DIR/bench_single.sh" "$mode"
  "$SCRIPT_DIR/stop.sh"
}

MODE="${1:-both}"
if [[ "$MODE" == "both" ]]; then
  run_one baseline
  run_one kvcached
  python3 "$SCRIPT_DIR/plot_results.py" --results-dir "$RESULTS_DIR" --phase 1 || true
else
  run_one "$MODE"
fi
echo "Phase 1 done. results in $RESULTS_DIR"
