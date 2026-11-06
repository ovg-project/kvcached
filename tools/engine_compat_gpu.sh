#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# Run on an isolated, dedicated GPU host. Never mount its home or Docker socket.
set -euo pipefail
SOURCE=$1
OUTPUT=$2
TAG=$3
ENGINE=${ENGINE_COMPAT_ENGINE:-vllm}
if [[ "$ENGINE" == vllm ]]; then default_image="vllm/vllm-openai:$TAG";
elif [[ "$ENGINE" == sglang ]]; then default_image="lmsysorg/sglang:$TAG";
else exit 2; fi
IMAGE=${4:-$default_image}
TOOLS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(\.post[0-9]+)?$ ]] || exit 2
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 2
docker pull "$IMAGE" || exit 2
DIGEST=$(docker image inspect --format '{{.Id}}' "$IMAGE")
HEAD=$(git -C "$SOURCE" rev-parse HEAD)
printf '%s\n' "$DIGEST" > "$OUTPUT/image-digest.txt"
NAME=${COMPAT_CONTAINER_NAME:?Supervisor must supply the container name}
mkdir -p "$OUTPUT/runtime"
analysis=()
if [[ "$ENGINE_COMPAT_PROFILE" == auto ]]; then
  analysis=(-v "${ENGINE_COMPAT_ANALYSIS_DIR:?}:/analysis:ro"
    -e ENGINE_COMPAT_ANALYSIS_DIR=/analysis
    -e "ENGINE_COMPAT_ANALYSIS_DIGEST=${ENGINE_COMPAT_ANALYSIS_DIGEST:?}")
fi
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
set +e
docker run --name "$NAME" --gpus all --shm-size=2g --cap-drop=ALL \
  --security-opt=no-new-privileges --entrypoint /bin/bash \
  -v "$SOURCE:/input:ro" -v "$TOOLS:/checks:ro" -v "$TOOLS/..:/controller:ro" -v "$OUTPUT/runtime:/results" \
  -e EXPECTED_VLLM="${TAG#v}" -e CANDIDATE_SHA="$HEAD" \
  -e EXPECTED_ENGINE="$ENGINE" \
  -e ENGINE_COMPAT_PROFILE="${ENGINE_COMPAT_PROFILE:?}" \
  -e ENGINE_COMPAT_POLICY_DIGEST="${ENGINE_COMPAT_POLICY_DIGEST:?}" \
  -e ENABLE_KVCACHED=false -e KVCACHED_AUTOPATCH=0 \
  -e MAX_JOBS=2 "${analysis[@]}" "$DIGEST" -lc '
    set -euo pipefail
    python3 -c "import os,importlib.metadata as m; assert m.version(os.environ[\"EXPECTED_ENGINE\"]).split(\"+\")[0] == os.environ[\"EXPECTED_VLLM\"]" || exit 2
    mkdir /candidate
    cp -a /input/. /candidate/
    cd /candidate
    python3 -m pip install pytest packaging posix_ipc wrapt || exit 2
    python3 -m pip install --no-build-isolation --no-deps -e . || exit 1
    python3 /controller/tools/engine_compat_profile.py "$ENGINE_COMPAT_PROFILE" \
      --source /candidate --output /results/contracts --gpu
    python3 /controller/tools/engine_compat_profile.py "$ENGINE_COMPAT_PROFILE" \
      --source /candidate --output /results/probe --probe \
      --tag "v$EXPECTED_VLLM" --candidate-sha "$CANDIDATE_SHA"
  '
CODE=$?
set -e
if [[ "$CODE" != 0 && "$CODE" != 1 ]]; then exit 2; fi
exit "$CODE"
