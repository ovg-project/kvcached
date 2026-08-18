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
#   all   - run core, both engine smoke tests, and the NIXL P/D smoke test.
#
# CHECK_ONLY=1 validates the runner configuration without requiring a GPU.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
VLLM_PYTHON="${VLLM_PYTHON:-${PYTHON}}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${PYTHON}}"
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
  core|vllm|sglang|engines|nixl|all) ;;
  *)
    echo "Unknown GPU_CI_PROFILE: '${GPU_CI_PROFILE}'" >&2
    exit 2
    ;;
esac

EXPECTED_GPU_COUNT=1
if [[ "${GPU_CI_PROFILE}" =~ ^(nixl|all)$ ]]; then
  EXPECTED_GPU_COUNT=2
fi

# A scheduler may provide CUDA_VISIBLE_DEVICES directly. Otherwise require the
# runner/repository configuration to name the physical GPU indices or UUIDs.
KVCACHED_GPU_VISIBLE_DEVICES="${KVCACHED_GPU_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${KVCACHED_GPU_VISIBLE_DEVICES}" ]]; then
  echo "KVCACHED_GPU_VISIBLE_DEVICES (or CUDA_VISIBLE_DEVICES) must select ${EXPECTED_GPU_COUNT} GPU(s)" >&2
  exit 2
fi
if [[ ! "${KVCACHED_GPU_VISIBLE_DEVICES}" =~ ^[^,[:space:]]+(,[^,[:space:]]+)*$ ]]; then
  echo "GPU device selection must be a comma-separated list without spaces" >&2
  exit 2
fi
IFS=',' read -r -a SELECTED_GPU_IDS <<< "${KVCACHED_GPU_VISIBLE_DEVICES}"
if [[ "${#SELECTED_GPU_IDS[@]}" -ne "${EXPECTED_GPU_COUNT}" ]]; then
  echo "Profile '${GPU_CI_PROFILE}' requires exactly ${EXPECTED_GPU_COUNT} selected GPU(s); got ${#SELECTED_GPU_IDS[@]}" >&2
  exit 2
fi
for ((i = 0; i < ${#SELECTED_GPU_IDS[@]}; i++)); do
  for ((j = i + 1; j < ${#SELECTED_GPU_IDS[@]}; j++)); do
    if [[ "${SELECTED_GPU_IDS[i]}" == "${SELECTED_GPU_IDS[j]}" ]]; then
      echo "GPU device selection contains duplicate ID '${SELECTED_GPU_IDS[i]}'" >&2
      exit 2
    fi
  done
done
export KVCACHED_GPU_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${KVCACHED_GPU_VISIBLE_DEVICES}"

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
  tests/manifests/gpu.txt \
  tools/check_test_classification.py \
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
if [[ "${GPU_CI_PROFILE}" =~ ^(vllm|engines|nixl|all)$ ]] &&
   ! command -v "${VLLM_PYTHON}" >/dev/null 2>&1; then
  echo "vLLM Python command not found: ${VLLM_PYTHON}" >&2
  exit 2
fi
if [[ "${GPU_CI_PROFILE}" =~ ^(sglang|engines|all)$ ]] &&
   ! command -v "${SGLANG_PYTHON}" >/dev/null 2>&1; then
  echo "SGLang Python command not found: ${SGLANG_PYTHON}" >&2
  exit 2
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "GPU CI preflight passed: profile=${GPU_CI_PROFILE}, devices=${KVCACHED_GPU_VISIBLE_DEVICES}, python=${PYTHON}"
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
  nvidia-smi --id="${KVCACHED_GPU_VISIBLE_DEVICES}" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader \
    >"${GPU_CI_ARTIFACT_DIR}/compute-apps-after.txt" 2>&1
  STARTED_AT="${STARTED_AT}" \
  FINISHED_AT="${finished_at}" \
  EXIT_CODE="${exit_code}" \
  GPU_CI_PROFILE="${GPU_CI_PROFILE}" \
  GPU_CI_REPEAT="${GPU_CI_REPEAT}" \
  KVCACHED_GPU_VISIBLE_DEVICES="${KVCACHED_GPU_VISIBLE_DEVICES}" \
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
    "selected_devices": os.environ["KVCACHED_GPU_VISIBLE_DEVICES"].split(","),
    "started_at": os.environ["STARTED_AT"],
    "status": "passed" if os.environ["EXIT_CODE"] == "0" else "failed",
}, indent=2, sort_keys=True))
PY
  exit "${exit_code}"
}
trap finalize EXIT

nvidia-smi --id="${KVCACHED_GPU_VISIBLE_DEVICES}" \
  --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader | tee "${GPU_CI_ARTIFACT_DIR}/nvidia-smi.txt"
nvidia-smi --id="${KVCACHED_GPU_VISIBLE_DEVICES}" \
  --query-compute-apps=pid,process_name,used_memory \
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
  echo "selected_physical_devices=${KVCACHED_GPU_VISIBLE_DEVICES}"
  echo "core_python=${PYTHON}"
  echo "vllm_python=${VLLM_PYTHON}"
  echo "sglang_python=${SGLANG_PYTHON}"
  command -v "${CC:-cc}" >/dev/null 2>&1 &&
    "${CC:-cc}" --version 2>/dev/null | head -1 || true
  command -v "${CXX:-c++}" >/dev/null 2>&1 &&
    "${CXX:-c++}" --version 2>/dev/null | head -1 || true
  command -v nvcc >/dev/null 2>&1 &&
    nvcc --version 2>/dev/null | tail -1 || true
  df -h /dev/shm 2>/dev/null || true
} | tee "${GPU_CI_ARTIFACT_DIR}/runner-environment.txt"

EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT}" \
  "${PYTHON}" - <<'PY' | tee "${GPU_CI_ARTIFACT_DIR}/torch-environment.txt"
import os
import sys

import torch

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA on this runner")
expected = int(os.environ["EXPECTED_GPU_COUNT"])
if torch.cuda.device_count() != expected:
    raise SystemExit(
        f"Expected exactly {expected} visible logical GPU(s), "
        f"but PyTorch found {torch.cuda.device_count()}"
    )
for index in range(torch.cuda.device_count()):
    print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")
PY
"${PYTHON}" -m pip freeze \
  >"${GPU_CI_ARTIFACT_DIR}/python-packages.txt"

install_project() {
  local target_python="$1"
  local artifact_prefix="$2"

  "${target_python}" -m pip install \
    "packaging>=24.2" \
    "pytest>=8,<9" \
    "setuptools>=77" \
    wheel \
    2>&1 | tee "${GPU_CI_ARTIFACT_DIR}/${artifact_prefix}-bootstrap.log"
  "${target_python}" -m pip install . --force-reinstall --no-deps \
    --no-build-isolation --no-cache-dir \
    2>&1 | tee "${GPU_CI_ARTIFACT_DIR}/${artifact_prefix}-install.log"
  "${target_python}" tools/dev_copy_pth.py \
    2>&1 | tee "${GPU_CI_ARTIFACT_DIR}/${artifact_prefix}-dev-copy-pth.log"
  "${target_python}" -m pip check \
    2>&1 | tee "${GPU_CI_ARTIFACT_DIR}/${artifact_prefix}-pip-check.log"
}

if [[ "${GPU_CI_INSTALL}" == "1" ]]; then
  install_project "${PYTHON}" core
  if [[ "${GPU_CI_PROFILE}" =~ ^(vllm|engines|nixl|all)$ ]] &&
     [[ "${VLLM_PYTHON}" != "${PYTHON}" ]]; then
    install_project "${VLLM_PYTHON}" vllm
  fi
  if [[ "${GPU_CI_PROFILE}" =~ ^(sglang|engines|all)$ ]] &&
     [[ "${SGLANG_PYTHON}" != "${PYTHON}" ]] &&
     [[ "${SGLANG_PYTHON}" != "${VLLM_PYTHON}" ]]; then
    install_project "${SGLANG_PYTHON}" sglang
  fi
fi

# Read the GPU targets from tests/manifests/gpu.txt rather than repeating them
# here. check_test_classification.py enforces that every test module sits in
# exactly one manifest, so a hardcoded list here would silently stop covering
# new GPU tests as they are added.
# `python -m pytest` puts the working directory first on sys.path. This script
# runs from the repository root, so the source tree would shadow the wheel that
# install_project just put in site-packages -- and the source tree carries no
# compiled extension, so every GPU test dies at collection with
# "No module named 'kvcached.vmm_ops'". Keep the installed package in front.
if [[ "${GPU_CI_INSTALL}" == "1" ]]; then
  export PYTHONSAFEPATH=1
fi

# Whichever kvcached the tests are about to import, say which one and prove the
# extension loads. One clear line here beats a pile of collection errors.
"${PYTHON}" - <<'IMPORTCHECK' | tee "${GPU_CI_ARTIFACT_DIR}/kvcached-import.txt"
import kvcached
import kvcached.vmm_ops as ops

print(f"kvcached={kvcached.__file__}")
print(f"vmm_ops={ops.__file__}")
IMPORTCHECK

default_pytest_targets=()
while IFS= read -r test_path; do
  default_pytest_targets+=("${test_path}")
done < <("${PYTHON}" tools/check_test_classification.py --list-category gpu)
if [[ "${#default_pytest_targets[@]}" -eq 0 ]]; then
  echo "No GPU tests are classified in tests/manifests/gpu.txt" >&2
  exit 2
fi
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

  if [[ "${GPU_CI_PROFILE}" =~ ^(nixl|all)$ ]]; then
    if [[ "$("${VLLM_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" -ne 2 ]]; then
      echo "The nixl profile requires exactly two visible logical GPUs" >&2
      exit 2
    fi

    PATH="$(dirname "$(command -v "${VLLM_PYTHON}")"):${PATH}" \
    PYTHON="${VLLM_PYTHON}" \
    INSTALL_DEPS=0 \
    INSTALL_EDITABLE=0 \
    INSTALL_VLLM=0 \
    PREFILL_GPU="${SELECTED_GPU_IDS[0]}" \
    DECODE_GPU="${SELECTED_GPU_IDS[1]}" \
    LOG_DIR="${GPU_CI_ARTIFACT_DIR}/nixl-${iteration}" \
      bash tools/run_vllm_nixl_pd_smoke.sh \
      2>&1 | tee "${GPU_CI_ARTIFACT_DIR}/nixl-pd-smoke-${iteration}.log"
  fi

  if [[ "${GPU_CI_PROFILE}" =~ ^(vllm|engines|all)$ ]]; then
    ENGINE=vllm \
    PYTHON="${VLLM_PYTHON}" \
    MODEL="${MODEL:-}" \
    LOG_DIR="${GPU_CI_ARTIFACT_DIR}/vllm-${iteration}" \
      bash tools/run_engine_smoke.sh
  fi

  if [[ "${GPU_CI_PROFILE}" =~ ^(sglang|engines|all)$ ]]; then
    ENGINE=sglang \
    PYTHON="${SGLANG_PYTHON}" \
    MODEL="${MODEL:-}" \
    LOG_DIR="${GPU_CI_ARTIFACT_DIR}/sglang-${iteration}" \
      bash tools/run_engine_smoke.sh
  fi

done

echo "GPU CI passed: profile=${GPU_CI_PROFILE}, repeat=${GPU_CI_REPEAT}"
