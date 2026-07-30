#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# Entry point for a self-hosted GPU runner.
#
# Profiles:
#   core  - build kvcached and run extension/GPU allocator tests.
#   vllm - run core and one kvcached-backed vLLM correctness request.
#   sglang - run core and one kvcached-backed SGLang correctness request.
#   engines - run both single-GPU engine smoke tests sequentially.
#   nixl  - run core, then the two-GPU vLLM+NIXL P/D smoke test.
#
# CHECK_ONLY=1 validates the runner configuration without requiring a GPU.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
GPU_CI_PROFILE="${GPU_CI_PROFILE:-core}"
GPU_CI_ARTIFACT_DIR="${GPU_CI_ARTIFACT_DIR:-${ROOT_DIR}/gpu-ci-artifacts}"
GPU_CI_INSTALL="${GPU_CI_INSTALL:-1}"
GPU_CI_REPEAT="${GPU_CI_REPEAT:-1}"
GPU_CI_REQUIRE_IDLE="${GPU_CI_REQUIRE_IDLE:-1}"
GPU_CI_SKIP_CORE="${GPU_CI_SKIP_CORE:-0}"
GPU_CI_LOCK_FILE="${GPU_CI_LOCK_FILE:-/tmp/kvcached-gpu-ci.lock}"
CHECK_ONLY="${CHECK_ONLY:-0}"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "${GPU_CI_ARTIFACT_DIR}"

case "${GPU_CI_PROFILE}" in
  core|vllm|sglang|engines|nixl) ;;
  *)
    echo "Unknown GPU_CI_PROFILE: '${GPU_CI_PROFILE}'" >&2
    exit 2
    ;;
esac

if [[ ! "${GPU_CI_REPEAT}" =~ ^[1-9][0-9]*$ ]] ||
   [[ "${GPU_CI_REPEAT}" -gt 10 ]]; then
  echo "GPU_CI_REPEAT must be an integer from 1 to 10" >&2
  exit 2
fi
if [[ "${GPU_CI_SKIP_CORE}" != "0" && "${GPU_CI_SKIP_CORE}" != "1" ]]; then
  echo "GPU_CI_SKIP_CORE must be 0 or 1" >&2
  exit 2
fi

for required_file in \
  pyproject.toml \
  tools/dev_copy_pth.py \
  tools/run_engine_smoke.sh \
  tools/run_vllm_nixl_pd_smoke.sh; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 2
  fi
done

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Python command not found: ${PYTHON}" >&2
  exit 2
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "GPU CI preflight passed: profile=${GPU_CI_PROFILE}, python=${PYTHON}"
  exit 0
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to serialize jobs on the persistent GPU runner" >&2
  exit 2
fi

# GitHub serializes jobs assigned to one runner, but the GPU may also be used by
# a local shell or another runner service. Keep one host-wide lock for the
# complete test transaction.
exec 9>"${GPU_CI_LOCK_FILE}"
if ! flock -n 9; then
  echo "Another GPU CI job owns ${GPU_CI_LOCK_FILE}; retry later" >&2
  exit 75
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required on the self-hosted GPU runner" >&2
  exit 2
fi

finalize() {
  local exit_code=$?
  set +e
  local finished_at
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader \
    >"${GPU_CI_ARTIFACT_DIR}/compute-apps-after.txt" 2>&1
  STARTED_AT="${STARTED_AT}" \
  FINISHED_AT="${finished_at}" \
  EXIT_CODE="${exit_code}" \
  GPU_CI_PROFILE="${GPU_CI_PROFILE}" \
  GPU_CI_REPEAT="${GPU_CI_REPEAT}" \
  GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)" \
    "${PYTHON}" - <<'PY' >"${GPU_CI_ARTIFACT_DIR}/summary.json"
import json
import os

print(json.dumps({
    "exit_code": int(os.environ["EXIT_CODE"]),
    "finished_at": os.environ["FINISHED_AT"],
    "git_commit": os.environ["GIT_COMMIT"],
    "profile": os.environ["GPU_CI_PROFILE"],
    "repeat": int(os.environ["GPU_CI_REPEAT"]),
    "started_at": os.environ["STARTED_AT"],
    "status": "passed" if os.environ["EXIT_CODE"] == "0" else "failed",
}, indent=2, sort_keys=True))
PY
  exit "${exit_code}"
}
trap finalize EXIT

nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader | tee "${GPU_CI_ARTIFACT_DIR}/nvidia-smi.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader \
  >"${GPU_CI_ARTIFACT_DIR}/compute-apps-before.txt" 2>&1

if [[ "${GPU_CI_REQUIRE_IDLE}" == "1" ]] &&
   [[ -s "${GPU_CI_ARTIFACT_DIR}/compute-apps-before.txt" ]]; then
  echo "GPU runner is not idle; refusing to disturb existing compute processes:" >&2
  cat "${GPU_CI_ARTIFACT_DIR}/compute-apps-before.txt" >&2
  exit 2
fi

{
  echo "started_at=${STARTED_AT}"
  echo "hostname=$(hostname)"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo uploaded-worktree)"
  echo "profile=${GPU_CI_PROFILE}"
  echo "repeat=${GPU_CI_REPEAT}"
  echo "python=${PYTHON}"
  command -v "${CC:-cc}" >/dev/null 2>&1 &&
    "${CC:-cc}" --version 2>/dev/null | head -1 || true
  command -v "${CXX:-c++}" >/dev/null 2>&1 &&
    "${CXX:-c++}" --version 2>/dev/null | head -1 || true
  command -v nvcc >/dev/null 2>&1 &&
    nvcc --version 2>/dev/null | tail -1 || true
  df -h /dev/shm 2>/dev/null || true
} | tee "${GPU_CI_ARTIFACT_DIR}/runner-environment.txt"

"${PYTHON}" - <<'PY' | tee "${GPU_CI_ARTIFACT_DIR}/torch-environment.txt"
import sys

import torch

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA on this runner")
for index in range(torch.cuda.device_count()):
    print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")
PY
"${PYTHON}" -m pip freeze \
  >"${GPU_CI_ARTIFACT_DIR}/python-packages.txt"

if [[ "${GPU_CI_INSTALL}" == "1" ]]; then
  "${PYTHON}" -m pip install \
    "packaging>=24.2" \
    "pytest>=8,<9" \
    "setuptools>=77" \
    wheel \
    2>&1 | tee "${GPU_CI_ARTIFACT_DIR}/bootstrap.log"
  "${PYTHON}" -m pip install -e . --no-build-isolation --no-cache-dir \
    2>&1 | tee "${GPU_CI_ARTIFACT_DIR}/install.log"
  "${PYTHON}" tools/dev_copy_pth.py \
    2>&1 | tee "${GPU_CI_ARTIFACT_DIR}/dev-copy-pth.log"
fi

default_pytest_targets=(
  tests/test_kvcache_manager.py
  tests/test_paged_allocator_aliasing.py
  tests/test_alloc_kv_cache_alignment.py
)
if [[ -n "${GPU_CI_PYTEST_TARGETS:-}" ]]; then
  IFS=' ' read -r -a pytest_targets <<< "${GPU_CI_PYTEST_TARGETS}"
else
  pytest_targets=("${default_pytest_targets[@]}")
fi
for iteration in $(seq 1 "${GPU_CI_REPEAT}"); do
  if [[ "${GPU_CI_SKIP_CORE}" != "1" ]]; then
    "${PYTHON}" -m pytest -q "${pytest_targets[@]}" \
      2>&1 | tee "${GPU_CI_ARTIFACT_DIR}/core-pytest-${iteration}.log"
  fi

  if [[ "${GPU_CI_PROFILE}" == "nixl" ]]; then
    if [[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" -lt 2 ]]; then
      echo "The nixl profile requires at least two visible GPUs" >&2
      exit 2
    fi

    GPU_CI_ARTIFACT_DIR="${GPU_CI_ARTIFACT_DIR}" \
      bash tools/run_vllm_nixl_pd_smoke.sh \
      2>&1 | tee "${GPU_CI_ARTIFACT_DIR}/nixl-pd-smoke-${iteration}.log"
  fi

  if [[ "${GPU_CI_PROFILE}" == "vllm" ]] ||
     [[ "${GPU_CI_PROFILE}" == "engines" ]]; then
    ENGINE=vllm \
    MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
    LOG_DIR="${GPU_CI_ARTIFACT_DIR}/vllm-${iteration}" \
      bash tools/run_engine_smoke.sh
  fi

  if [[ "${GPU_CI_PROFILE}" == "sglang" ]] ||
     [[ "${GPU_CI_PROFILE}" == "engines" ]]; then
    ENGINE=sglang \
    MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
    LOG_DIR="${GPU_CI_ARTIFACT_DIR}/sglang-${iteration}" \
      bash tools/run_engine_smoke.sh
  fi

done

echo "GPU CI passed: profile=${GPU_CI_PROFILE}, repeat=${GPU_CI_REPEAT}"
