#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# Run the complete two-GPU validation campaign without depending on a specific
# GPU provider. Results are retained even when an individual phase fails.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_REPO_DIR="${VLLM_REPO_DIR:-}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-}"
PYTHON="${PYTHON:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
GPU_CI_REPO_DIR="${GPU_CI_REPO_DIR:-}"
MODEL_MATRIX_REPO_DIR="${MODEL_MATRIX_REPO_DIR:-}"
RUN_NIXL="${RUN_NIXL:-0}"
RUN_MODEL_MATRIX="${RUN_MODEL_MATRIX:-0}"
VLLM_PYTHON="${VLLM_PYTHON:-${PYTHON}}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${PYTHON}}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
BASELINE_PORT="${BASELINE_PORT:-8101}"
OFFLOAD_PORT="${OFFLOAD_PORT:-8100}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
failures=()
baseline_pid=""

if [[ -z "${ARTIFACT_ROOT}" ]]; then
  echo "ARTIFACT_ROOT must name an absolute, persistent-storage directory" >&2
  exit 2
fi
if [[ "${ARTIFACT_ROOT}" != /* ]]; then
  echo "ARTIFACT_ROOT must be an absolute path" >&2
  exit 2
fi
if [[ -z "${VLLM_REPO_DIR}" ]] ||
   [[ ! -x "${VLLM_REPO_DIR}/tools/run_vllm_cpu_offload_smoke.sh" ]]; then
  echo "VLLM_REPO_DIR must point to the zixuan/vllm-cpu-offload checkout" >&2
  exit 2
fi

RUN_DIR="${ARTIFACT_ROOT%/}/kvcached-h20-${RUN_ID}"
archive="${RUN_DIR}.tar.gz"
mkdir -p "${RUN_DIR}/phases"

log() {
  printf '[h20-campaign] %s\n' "$*" | tee -a "${RUN_DIR}/campaign.log"
}

stop_baseline() {
  set +e
  if [[ -n "${baseline_pid}" ]] && kill -0 "${baseline_pid}" >/dev/null 2>&1; then
    kill -TERM -- "-${baseline_pid}" >/dev/null 2>&1 \
      || kill -TERM "${baseline_pid}" >/dev/null 2>&1 \
      || true
    sleep 3
    kill -KILL -- "-${baseline_pid}" >/dev/null 2>&1 \
      || kill -KILL "${baseline_pid}" >/dev/null 2>&1 \
      || true
  fi
  baseline_pid=""
}

finalize() {
  local exit_code=$?
  trap - EXIT
  stop_baseline
  set +e
  nvidia-smi >"${RUN_DIR}/nvidia-smi-final.txt" 2>&1
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader >"${RUN_DIR}/compute-apps-final.txt" 2>&1
  {
    printf 'status=%s\n' "$([[ "${exit_code}" -eq 0 ]] && printf passed || printf failed)"
    printf 'exit_code=%s\n' "${exit_code}"
    printf 'started_at=%s\n' "${started_at}"
    printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'core_commit=%s\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true)"
    printf 'vllm_commit=%s\n' "$(git -C "${VLLM_REPO_DIR}" rev-parse HEAD 2>/dev/null || true)"
    printf 'model=%s\n' "${MODEL}"
    printf 'gpu0=%s\n' "${GPU0}"
    printf 'gpu1=%s\n' "${GPU1}"
    printf 'failed_phases=%s\n' "${failures[*]:-none}"
  } >"${RUN_DIR}/campaign-status.txt"
  (
    cd "${RUN_DIR}" || exit
    find . -type f ! -name MANIFEST.sha256 -print0 \
      | sort -z \
      | xargs -0 sha256sum >MANIFEST.sha256
  )
  tar -czf "${archive}" -C "$(dirname "${RUN_DIR}")" "$(basename "${RUN_DIR}")"
  sha256sum "${archive}" >"${archive}.sha256"
  printf '\nH20 campaign artifacts: %s\nArchive: %s\n' "${RUN_DIR}" "${archive}"
  cat "${archive}.sha256" 2>/dev/null || true
  exit "${exit_code}"
}
trap finalize EXIT

run_phase() {
  local name="$1"
  shift
  local phase_log="${RUN_DIR}/phases/${name}.log"
  local phase_status="${RUN_DIR}/phases/${name}.status"
  local phase_start
  local rc
  phase_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  log "START ${name}"
  set +e
  (set -e; "$@") > >(tee "${phase_log}") 2>&1
  rc=$?
  set -e
  {
    printf 'name=%s\n' "${name}"
    printf 'started_at=%s\n' "${phase_start}"
    printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'exit_code=%s\n' "${rc}"
  } >"${phase_status}"
  if [[ "${rc}" -eq 0 ]]; then
    log "PASS ${name}"
  else
    failures+=("${name}:${rc}")
    log "FAIL ${name} (exit ${rc}); continuing with independent phases"
  fi
}

preflight() {
  command -v nvidia-smi >/dev/null 2>&1
  command -v "${PYTHON}" >/dev/null 2>&1
  command -v "${VLLM_BIN}" >/dev/null 2>&1
  command -v curl >/dev/null 2>&1
  command -v sha256sum >/dev/null 2>&1
  command -v setsid >/dev/null 2>&1
  [[ "${GPU0}" != "${GPU1}" ]]
  [[ "${RUN_NIXL}" =~ ^[01]$ ]]
  [[ "${RUN_MODEL_MATRIX}" =~ ^[01]$ ]]
  if [[ "${RUN_NIXL}" == "1" ]]; then
    [[ -x "${GPU_CI_REPO_DIR}/tools/run_gpu_ci.sh" ]]
  fi
  if [[ "${RUN_MODEL_MATRIX}" == "1" ]]; then
    [[ -x "${MODEL_MATRIX_REPO_DIR}/tools/run_gpu_ci.sh" ]]
  fi
  local gpu_count
  gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
  [[ "${gpu_count}" -ge 2 ]]
  nvidia-smi
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader
  "${PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("PyTorch must see at least two CUDA GPUs")
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_properties(index).total_memory)
PY
  df -h "${ARTIFACT_ROOT}"
  free -h
}

run_core_gpu() {
  local gpu="$1"
  local install="$2"
  local target="${RUN_DIR}/core-gpu${gpu}"
  env \
    ARTIFACT_DIR="${target}" \
    BENCH_ITERATIONS="${BENCH_ITERATIONS:-200}" \
    CYCLES="${CYCLES:-10}" \
    DEVICE="cuda:${gpu}" \
    INSTALL_PACKAGE="${install}" \
    PAGES="${PAGES:-8}" \
    PYTHON="${PYTHON}" \
    bash "${ROOT_DIR}/tools/run_cpu_offload_h20_validation.sh"
}

run_dual_isolation() {
  local dir="${RUN_DIR}/dual-gpu-isolation"
  mkdir -p "${dir}"
  set +e
  "${PYTHON}" "${ROOT_DIR}/tools/validate_cpu_offload_vmm.py" \
    --device "cuda:${GPU0}" --page-size-mb 2 --layers 8 --pages 8 --cycles 10 \
    --require-reclaimed-bytes --report "${dir}/gpu${GPU0}.json" \
    >"${dir}/gpu${GPU0}.log" 2>&1 &
  local pid0=$!
  "${PYTHON}" "${ROOT_DIR}/tools/validate_cpu_offload_vmm.py" \
    --device "cuda:${GPU1}" --page-size-mb 2 --layers 8 --pages 8 --cycles 10 \
    --require-reclaimed-bytes --report "${dir}/gpu${GPU1}.json" \
    >"${dir}/gpu${GPU1}.log" 2>&1 &
  local pid1=$!
  wait "${pid0}"
  local rc0=$?
  wait "${pid1}"
  local rc1=$?
  set -e
  cat "${dir}/gpu${GPU0}.log"
  cat "${dir}/gpu${GPU1}.log"
  [[ "${rc0}" -eq 0 && "${rc1}" -eq 0 ]]
}

run_vllm_baseline_and_offload() {
  local dir="${RUN_DIR}/vllm-end-to-end"
  local baseline_log="${dir}/baseline-server.log"
  mkdir -p "${dir}"
  "${PYTHON}" -m pip install --no-build-isolation -e "${VLLM_REPO_DIR}"

  local baseline_cmd=(
    "${VLLM_BIN}" serve "${MODEL}"
    --host 127.0.0.1
    --port "${BASELINE_PORT}"
    --enable-prefix-caching
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.35}"
    --max-model-len "${MAX_MODEL_LEN:-1024}"
  )
  CUDA_VISIBLE_DEVICES="${GPU1}" ENABLE_KVCACHED=false KVCACHED_AUTOPATCH=0 \
    setsid "${baseline_cmd[@]}" >"${baseline_log}" 2>&1 &
  baseline_pid=$!

  local deadline=$((SECONDS + REQUEST_TIMEOUT))
  until curl -fsS "http://127.0.0.1:${BASELINE_PORT}/v1/models" \
    >"${dir}/baseline-models.json" 2>/dev/null; do
    kill -0 "${baseline_pid}" >/dev/null 2>&1 || {
      tail -200 "${baseline_log}" >&2 || true
      return 1
    }
    [[ "${SECONDS}" -le "${deadline}" ]] || return 1
    sleep 2
  done

  set +e
  CUDA_VISIBLE_DEVICES="${GPU0}" \
  BASELINE_URL="http://127.0.0.1:${BASELINE_PORT}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}" \
  INSTALL_EDITABLE=0 \
  LOG_DIR="${dir}/offload" \
  MODEL="${MODEL}" \
  PORT="${OFFLOAD_PORT}" \
  REQUEST_TIMEOUT="${REQUEST_TIMEOUT}" \
  VLLM_BIN="${VLLM_BIN}" \
    bash "${VLLM_REPO_DIR}/tools/run_vllm_cpu_offload_smoke.sh"
  local rc=$?
  set -e
  stop_baseline
  return "${rc}"
}

run_nixl_regression() {
  env \
    GPU_CI_ARTIFACT_DIR="${RUN_DIR}/nixl-regression" \
    GPU_CI_PROFILE=nixl \
    GPU_CI_REPEAT="${NIXL_REPEAT:-1}" \
    KVCACHED_GPU_VISIBLE_DEVICES="${GPU0},${GPU1}" \
    MODEL="${MODEL}" \
    PYTHON="${PYTHON}" \
    VLLM_PYTHON="${VLLM_PYTHON}" \
    bash "${GPU_CI_REPO_DIR}/tools/run_gpu_ci.sh"
}

run_targeted_model_matrix() {
  env \
    GPU_CI_ARTIFACT_DIR="${RUN_DIR}/model-matrix" \
    GPU_CI_PROFILE=compat \
    KVCACHED_GPU_VISIBLE_DEVICES="${GPU0}" \
    MODEL_COMPAT_ENGINE="${MODEL_COMPAT_ENGINE:-all}" \
    MODEL_COMPAT_LAYOUT="${MODEL_COMPAT_LAYOUT:-all}" \
    MODEL_COMPAT_MODEL="${MODEL_COMPAT_MODEL:-qwen3}" \
    MODEL_COMPAT_MODEL_OVERRIDE="${MODEL_COMPAT_MODEL_OVERRIDE:-}" \
    PYTHON="${PYTHON}" \
    SGLANG_PYTHON="${SGLANG_PYTHON}" \
    VLLM_PYTHON="${VLLM_PYTHON}" \
    bash "${MODEL_MATRIX_REPO_DIR}/tools/run_gpu_ci.sh"
}

{
  printf 'core_repo=%s\n' "${ROOT_DIR}"
  printf 'vllm_repo=%s\n' "${VLLM_REPO_DIR}"
  printf 'core_commit=%s\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD)"
  printf 'vllm_commit=%s\n' "$(git -C "${VLLM_REPO_DIR}" rev-parse HEAD)"
  printf 'model=%s\n' "${MODEL}"
  printf 'gpu0=%s\n' "${GPU0}"
  printf 'gpu1=%s\n' "${GPU1}"
} >"${RUN_DIR}/campaign-config.txt"

run_phase preflight preflight
run_phase core-gpu0 run_core_gpu "${GPU0}" 1
run_phase core-gpu1 run_core_gpu "${GPU1}" 0
run_phase dual-gpu-isolation run_dual_isolation
run_phase vllm-baseline-vs-offload run_vllm_baseline_and_offload
if [[ "${RUN_NIXL}" == "1" ]]; then
  run_phase nixl-regression run_nixl_regression
fi
if [[ "${RUN_MODEL_MATRIX}" == "1" ]]; then
  run_phase targeted-model-matrix run_targeted_model_matrix
fi

"${PYTHON}" -m pip freeze >"${RUN_DIR}/python-packages-final.txt" 2>&1 || true
if [[ "${#failures[@]}" -gt 0 ]]; then
  log "Campaign finished with failed phases: ${failures[*]}"
  exit 1
fi
log "All campaign phases passed"
