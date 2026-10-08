#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Runs inside the weekly H100 sandbox (built from the vLLM image): install
# kvcached from the checkout, download the weights, then run the performance
# comparison and the Hopper correctness check.
#
# usage: sandbox_job.sh <checkout> <results-dir> <perf models...> [-- --skip-correctness]

set -uo pipefail

src=$1
out=$2
shift 2
models=()
correctness=1
for arg in "$@"; do
  case "$arg" in
    --skip-correctness) correctness=0 ;;
    *) models+=("$arg") ;;
  esac
done
mkdir -p "$out"
cd "$src" || exit 10

echo "== install kvcached $(date -u +%T)"
python3 -m pip install -q -r requirements.txt ninja || exit 10
LIBRARY_PATH=/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH} \
  python3 -m pip install -q . --no-build-isolation || exit 10
(cd /tmp && python3 "$src/tools/dev_copy_pth.py" --check) || exit 11
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== download weights $(date -u +%T)"
# The correctness check serves Qwen3.8-27B too.
downloads=("${models[@]}")
[ "$correctness" = 1 ] && downloads+=(qwen38_27b)
python3 - "${downloads[@]}" <<'PY' || exit 12
import sys
import time
from pathlib import Path

sys.path.insert(0, "benchmarks/ci")
from huggingface_hub import snapshot_download
from perf import MODELS
for model in dict.fromkeys(sys.argv[1:]):
    hf_id = MODELS[model]["hf_id"]
    t0 = time.time()
    path = snapshot_download(hf_id, revision=MODELS[model]["revision"])
    size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    dt = time.time() - t0
    print(f"{hf_id}: {size / 2**30:.1f} GiB in {dt:.0f}s ({size / 2**20 / max(dt, 1):.0f} MiB/s)")
PY

rc=0
echo "== performance $(date -u +%T)"
python3 benchmarks/ci/perf.py --out "$out/perf" --models "${models[@]}" || rc=1
if [ "$correctness" = 1 ]; then
  echo "== Hopper correctness $(date -u +%T)"
  python3 tests/e2e/run.py --local --no-install --profile hopper --out "$out/e2e" || rc=1
fi
# GPU state at the end, for runs that fail on the hardware or the driver.
nvidia-smi -q > "$out/nvidia-smi-q.txt" 2>&1 || true
echo "== done $(date -u +%T), status $rc"
exit "$rc"
