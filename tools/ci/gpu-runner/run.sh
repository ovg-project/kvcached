#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# The GPU CI job on the runner machine (setup.sh), from the root of the
# checkout to test: pull the engine images and run the e2e suite. Nightly
# runs the vLLM and the SGLang cases at the same time, each on its own GPU;
# the other profiles run once on every GPU. <results-dir>/<part> gets the
# results of each run (vllm and sglang, or all).
#
# usage: run.sh <profile> <results-dir>

set -euo pipefail

profile=$1
out=$(realpath -m "$2")
hf=/opt/kvcached-ci/hf
mkdir -p "$out"

images=$(python3 - "$profile" <<'PY'
import sys

sys.path.insert(0, "tests")
from e2e.matrix import ENGINES, PROFILES

print(" ".join(ENGINES[e].image for e in PROFILES[sys.argv[1]].engines))
PY
)

# The servers write the results as root, inside their containers; give them
# back to this user, so that the runner can clean up after the job.
own_results() {
  docker run --rm --entrypoint chown -v "$out:/out" "${images%% *}" -R "$(id -u):$(id -g)" /out \
    >/dev/null 2>&1 || true
}
trap own_results EXIT

# Containers of an earlier job that was cancelled.
docker ps -aq --filter name=kvcached-e2e- | xargs -r docker rm -f >/dev/null

pids=()
for image in $images; do
  docker pull -q "$image" &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

run() {
  local part=$1
  shift
  mkdir -p "$out/$part"
  python3 tests/e2e/run.py --profile "$profile" --out "$out/$part" --hf-cache "$hf" "$@" 2>&1 |
    tee "$out/$part/run.log" | sed -u "s/^/[$part] /"
}

rc=0
if [ "$profile" = nightly ] && [ "$(nvidia-smi -L | wc -l)" -ge 2 ]; then
  # Separate ports: the containers share the host's network.
  run vllm --engines vllm --gpus 0 --port 18000 &
  vllm=$!
  run sglang --engines sglang --gpus 1 --port 19000 &
  sglang=$!
  wait "$vllm" || rc=1
  wait "$sglang" || rc=1
else
  run all || rc=1
fi

# GPU state at the end, for runs that fail on the hardware or the driver.
nvidia-smi -q >"$out/nvidia-smi.txt" 2>&1 || true
journalctl -k -b --no-pager 2>/dev/null | tail -n 500 >"$out/kernel.log" || true
exit "$rc"
