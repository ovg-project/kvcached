#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Phase 2 end-to-end: co-locate gpt-oss-120b (main) + Qwen2-VL-7B (guard).
# baseline statically partitions the GPU; kvcached lets main grow elastically
# into the headroom and still fit the guard -> converts the Phase-1 memory
# saving into throughput/latency.
#   ./run_phase2.sh          # baseline then kvcached
#   ./run_phase2.sh kvcached
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

run_one() {
  local mode=$1
  echo "############ Phase 2: mode=$mode ############"
  "$SCRIPT_DIR/stop.sh"; sleep 2
  if [[ "$mode" == "kvcached" ]]; then
    "$SCRIPT_DIR/launch_guard.sh" --mode kvcached --gpu-util "$KVCACHED_GUARD_GPU_UTIL"
    "$SCRIPT_DIR/launch_main.sh"  --mode kvcached --gpu-util "$KVCACHED_MAIN_GPU_UTIL"
  else
    # baseline: GUARD FIRST so its 15 GiB weights load into clean memory; then
    # main takes the rest. main-first OOMs the guard because main statically
    # nails up its whole KV pool up front.
    "$SCRIPT_DIR/launch_guard.sh" --mode baseline --gpu-util "$BASELINE_GUARD_GPU_UTIL"
    "$SCRIPT_DIR/launch_main.sh"  --mode baseline --gpu-util "$BASELINE_MAIN_GPU_UTIL"
  fi
  "$SCRIPT_DIR/bench_workflow.sh" "$mode"
  "$SCRIPT_DIR/stop.sh"
}

MODE="${1:-both}"
if [[ "$MODE" == "both" ]]; then
  run_one baseline
  run_one kvcached
else
  run_one "$MODE"
fi
echo "Phase 2 done. results in ${RESULTS_DIR}_phase2/"
