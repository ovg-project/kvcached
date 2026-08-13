#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# Provider-neutral, self-archiving CPU-offload validation for a CUDA host.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/cpu-offload-vmm-artifacts}"
TOOLCHAIN_DIR="${TOOLCHAIN_DIR:-${HOME}/.cache/kvcached/gcc-11}"
DEVICE="${DEVICE:-cuda:0}"
INSTALL_PACKAGE="${INSTALL_PACKAGE:-1}"
mkdir -p "${ARTIFACT_DIR}"

archive="${ARTIFACT_DIR%/}.tar.gz"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

finalize() {
  local exit_code=$?
  trap - EXIT
  set +e
  nvidia-smi >"${ARTIFACT_DIR}/nvidia-smi-final.txt" 2>&1
  {
    printf 'status=%s\n' "$([[ "${exit_code}" -eq 0 ]] && printf passed || printf failed)"
    printf 'exit_code=%s\n' "${exit_code}"
    printf 'started_at=%s\n' "${started_at}"
    printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'device=%s\n' "${DEVICE}"
  } >"${ARTIFACT_DIR}/run-status.txt"
  (
    cd "${ARTIFACT_DIR}" || exit
    find . -type f ! -name MANIFEST.sha256 -print0 \
      | sort -z \
      | xargs -0 sha256sum > MANIFEST.sha256
  )
  tar -czf "${archive}" -C "$(dirname "${ARTIFACT_DIR}")" \
    "$(basename "${ARTIFACT_DIR}")"
  sha256sum "${archive}" >"${archive}.sha256"
  printf 'Artifacts: %s\nArchive: %s\n' "${ARTIFACT_DIR}" "${archive}"
  cat "${archive}.sha256" 2>/dev/null || true
  exit "${exit_code}"
}
trap finalize EXIT

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required" >&2
  exit 2
fi

compiler="${CXX:-c++}"
compiler_major="$("${compiler}" -dumpversion 2>/dev/null | cut -d. -f1 || true)"
if [[ ! "${compiler_major}" =~ ^[0-9]+$ ]] || [[ "${compiler_major}" -lt 9 ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "GCC 9+ is required and conda is unavailable for automatic provisioning" >&2
    exit 2
  fi
  if [[ ! -x "${TOOLCHAIN_DIR}/bin/x86_64-conda-linux-gnu-g++" ]]; then
    conda create -y -p "${TOOLCHAIN_DIR}" -c conda-forge \
      "gcc_linux-64=11.4" "gxx_linux-64=11.4" \
      2>&1 | tee "${ARTIFACT_DIR}/toolchain-install.log"
  fi
  export CC="${TOOLCHAIN_DIR}/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${TOOLCHAIN_DIR}/bin/x86_64-conda-linux-gnu-g++"
  export CUDAHOSTCXX="${CXX}"
  export PATH="${TOOLCHAIN_DIR}/bin:${PATH}"
fi

nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader | tee "${ARTIFACT_DIR}/nvidia-smi.txt"
{
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'git_branch=%s\n' "$(git branch --show-current)"
  printf 'git_status_begin\n'
  git status --short
  printf 'git_status_end\n'
  printf 'device=%s\n' "${DEVICE}"
  printf 'python=%s\n' "$("${PYTHON}" --version 2>&1)"
  uname -a
  df -h "${ARTIFACT_DIR}"
} | tee "${ARTIFACT_DIR}/environment.txt"
{
  "${CC:-cc}" --version | head -1
  "${CXX:-c++}" --version | head -1
  nvcc --version | tail -1
} | tee "${ARTIFACT_DIR}/compiler.txt"
if [[ "${INSTALL_PACKAGE}" == "1" ]]; then
  "${PYTHON}" -m pip install \
    "packaging>=24.2" "setuptools>=77" wheel \
    2>&1 | tee "${ARTIFACT_DIR}/bootstrap.log"
  "${PYTHON}" -m pip install . --force-reinstall --no-deps \
    --no-build-isolation --no-cache-dir \
    2>&1 | tee "${ARTIFACT_DIR}/install.log"
fi

"${PYTHON}" tools/validate_cpu_offload_vmm.py \
  --device "${DEVICE}" \
  --page-size-mb "${PAGE_SIZE_MB:-2}" \
  --layers "${LAYERS:-8}" \
  --pages "${PAGES:-4}" \
  --cycles "${CYCLES:-5}" \
  --require-reclaimed-bytes \
  --report "${ARTIFACT_DIR}/vmm-roundtrip.json" \
  2>&1 | tee "${ARTIFACT_DIR}/vmm-roundtrip.log"

"${PYTHON}" tools/benchmark_cpu_offload.py \
  --device "${DEVICE}" \
  --page-size-mb "${PAGE_SIZE_MB:-2}" \
  --layers "${BENCH_LAYERS:-32}" \
  --kv-buffers 2 \
  --iterations "${BENCH_ITERATIONS:-100}" \
  --report "${ARTIFACT_DIR}/transfer-benchmark.json" \
  2>&1 | tee "${ARTIFACT_DIR}/transfer-benchmark.log"

echo "CPU offload validation passed"
