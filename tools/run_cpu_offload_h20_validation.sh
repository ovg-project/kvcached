#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# Provider-neutral, self-archiving CPU-offload validation for a CUDA host.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/cpu-offload-vmm-artifacts}"
mkdir -p "${ARTIFACT_DIR}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required" >&2
  exit 2
fi

nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader | tee "${ARTIFACT_DIR}/nvidia-smi.txt"
"${PYTHON}" -m pip install \
  "packaging>=24.2" "setuptools>=77" wheel \
  2>&1 | tee "${ARTIFACT_DIR}/bootstrap.log"
"${PYTHON}" -m pip install . --force-reinstall --no-deps \
  --no-build-isolation --no-cache-dir \
  2>&1 | tee "${ARTIFACT_DIR}/install.log"

"${PYTHON}" tools/validate_cpu_offload_vmm.py \
  --page-size-mb "${PAGE_SIZE_MB:-2}" \
  --layers "${LAYERS:-8}" \
  --pages "${PAGES:-4}" \
  --cycles "${CYCLES:-5}" \
  --report "${ARTIFACT_DIR}/vmm-roundtrip.json" \
  2>&1 | tee "${ARTIFACT_DIR}/vmm-roundtrip.log"

"${PYTHON}" tools/benchmark_cpu_offload.py \
  --page-size-mb "${PAGE_SIZE_MB:-2}" \
  --layers "${BENCH_LAYERS:-32}" \
  --kv-buffers 2 \
  --iterations "${BENCH_ITERATIONS:-100}" \
  --report "${ARTIFACT_DIR}/transfer-benchmark.json" \
  2>&1 | tee "${ARTIFACT_DIR}/transfer-benchmark.log"

(
  cd "${ARTIFACT_DIR}"
  find . -type f ! -name MANIFEST.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > MANIFEST.sha256
)

archive="${ARTIFACT_DIR%/}.tar.gz"
tar -czf "${archive}" -C "$(dirname "${ARTIFACT_DIR}")" \
  "$(basename "${ARTIFACT_DIR}")"
sha256sum "${archive}" > "${archive}.sha256"
echo "CPU offload validation passed"
echo "Artifacts: ${ARTIFACT_DIR}"
echo "Archive: ${archive}"
cat "${archive}.sha256"
