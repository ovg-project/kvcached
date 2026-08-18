#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# Provision isolated core, vLLM, and SGLang environments for a persistent GPU
# runner. Engine isolation prevents one engine's PyTorch/dependency pins from
# breaking the other engine or the allocator test environment.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_PYTHON="${BASE_PYTHON:-python3.11}"
ENV_ROOT="${ENV_ROOT:-${HOME}/.cache/kvcached/gpu-ci-envs}"
CORE_TORCH_SPEC="${CORE_TORCH_SPEC:-torch==2.8.0}"
VLLM_SPEC="${VLLM_SPEC:-vllm==0.19.0}"
SGLANG_SPEC="${SGLANG_SPEC:-sglang[all]==0.5.10}"
# Which environments to provision. The engine environments each pull their own
# PyTorch, so a host that only runs the core profile can skip them and the
# tens of gigabytes they download.
GPU_CI_ENVS="${GPU_CI_ENVS:-core vllm sglang}"
CHECK_ONLY="${CHECK_ONLY:-0}"

for requested in ${GPU_CI_ENVS}; do
  case "${requested}" in
    core|vllm|sglang) ;;
    *)
      echo "Unknown environment '${requested}' (expected core, vllm, or sglang)" >&2
      exit 2
      ;;
  esac
done

if ! command -v "${BASE_PYTHON}" >/dev/null 2>&1; then
  echo "Base Python command not found: ${BASE_PYTHON}" >&2
  echo "Set BASE_PYTHON to an interpreter this host has; Ubuntu 24.04 ships" >&2
  echo "python3.12 and has no python3.11 package." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required to build the kvcached extension" >&2
  exit 2
fi

compiler="${CXX:-c++}"
if ! command -v "${compiler}" >/dev/null 2>&1; then
  echo "C++ compiler not found: ${compiler}" >&2
  exit 2
fi
compiler_major="$("${compiler}" -dumpversion | cut -d. -f1)"
if [[ ! "${compiler_major}" =~ ^[0-9]+$ ]] || [[ "${compiler_major}" -lt 9 ]]; then
  echo "GCC 9 or newer is required; ${compiler} reports $(${compiler} -dumpversion)" >&2
  exit 2
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "GPU CI bootstrap preflight passed"
  echo "base_python=${BASE_PYTHON}"
  echo "env_root=${ENV_ROOT}"
  echo "envs=${GPU_CI_ENVS}"
  echo "core_torch=${CORE_TORCH_SPEC}"
  echo "vllm=${VLLM_SPEC}"
  echo "sglang=${SGLANG_SPEC}"
  exit 0
fi

mkdir -p "${ENV_ROOT}"

create_env() {
  local name="$1"
  local primary_spec="$2"
  local env_dir="${ENV_ROOT}/${name}"
  local python="${env_dir}/bin/python"

  if [[ ! -x "${python}" ]]; then
    "${BASE_PYTHON}" -m venv "${env_dir}"
  fi

  "${python}" -m pip install --upgrade pip
  "${python}" -m pip install \
    "packaging>=24.2" \
    "pytest>=8,<9" \
    "setuptools>=77" \
    wheel \
    "${primary_spec}"
  "${python}" -m pip install "${ROOT_DIR}" \
    --no-build-isolation --no-cache-dir
  "${python}" "${ROOT_DIR}/tools/dev_copy_pth.py"
  "${python}" -m pip check
}

for requested in ${GPU_CI_ENVS}; do
  case "${requested}" in
    core) create_env core "${CORE_TORCH_SPEC}" ;;
    vllm) create_env vllm "${VLLM_SPEC}" ;;
    sglang) create_env sglang "${SGLANG_SPEC}" ;;
  esac
done

# These interpreters belong to this machine, so they go in the runner's own
# .env rather than in a repository variable: GitHub delivers neither secrets
# nor repository variables to a workflow triggered by a pull request from a
# fork, and reviewing a fork's change before merging it is what the GPU run is
# for. The names below are the ones run_gpu_ci.sh reads.
#
# Report every environment that exists, not only the ones built just now, so a
# host provisioned in stages still gets a complete list.
: >"${ENV_ROOT}/runner.env"
[[ -x "${ENV_ROOT}/core/bin/python" ]] &&
  echo "PYTHON=${ENV_ROOT}/core/bin/python" >>"${ENV_ROOT}/runner.env"
[[ -x "${ENV_ROOT}/vllm/bin/python" ]] &&
  echo "VLLM_PYTHON=${ENV_ROOT}/vllm/bin/python" >>"${ENV_ROOT}/runner.env"
[[ -x "${ENV_ROOT}/sglang/bin/python" ]] &&
  echo "SGLANG_PYTHON=${ENV_ROOT}/sglang/bin/python" >>"${ENV_ROOT}/runner.env"

echo "GPU CI environments are ready in ${ENV_ROOT}"
echo "Append these lines to the runner's .env (usually ~/actions-runner/.env),"
echo "add a CUDA_VISIBLE_DEVICES line naming this host's GPUs, and restart the"
echo "runner service -- .env is read once at startup:"
cat "${ENV_ROOT}/runner.env"
