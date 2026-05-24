#!/bin/bash
# Shared configuration for the DGX Spark workflow demo.
# Source this file from other scripts; do not run it directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-kvcached}"
if command -v conda >/dev/null 2>&1; then
  _KVCACHED_SHELL_FLAGS="$-"
  set +e +u
  eval "$(conda shell.bash hook 2>/dev/null)"
  conda activate "$CONDA_ENV" 2>/dev/null
  _KVCACHED_CONDA_STATUS=$?
  case "$_KVCACHED_SHELL_FLAGS" in *u*) set -u ;; esac
  case "$_KVCACHED_SHELL_FLAGS" in *e*) set -e ;; esac
  unset _KVCACHED_SHELL_FLAGS
  if [[ "$_KVCACHED_CONDA_STATUS" -ne 0 && -d "${HOME}/miniforge3/envs/${CONDA_ENV}/bin" ]]; then
    export PATH="${HOME}/miniforge3/envs/${CONDA_ENV}/bin:${PATH}"
  fi
  unset _KVCACHED_CONDA_STATUS
elif [[ -d "${HOME}/miniforge3/envs/${CONDA_ENV}/bin" ]]; then
  export PATH="${HOME}/miniforge3/envs/${CONDA_ENV}/bin:${PATH}"
fi

MAIN_MODEL="${MAIN_MODEL:-Qwen/Qwen3.6-35B-A3B}"
GUARD_MODEL="${GUARD_MODEL:-meta-llama/Llama-Guard-3-8B}"
MAIN_PORT="${MAIN_PORT:-12346}"
GUARD_PORT="${GUARD_PORT:-12347}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

hf_cache_snapshot() {
  local model=$1
  local cache_id="${model//\//--}"
  local hub_dir="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
  local snapshots_dir="${hub_dir}/models--${cache_id}/snapshots"
  local snapshot=""
  if [[ -d "$snapshots_dir" ]]; then
    snapshot="$(find "$snapshots_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  fi
  [[ -n "$snapshot" ]] || return 1
  printf '%s\n' "$snapshot"
}

if [[ -z "${MAIN_MODEL_PATH:-}" ]]; then
  if MAIN_MODEL_PATH="$(hf_cache_snapshot "$MAIN_MODEL")"; then
    export MAIN_MODEL_PATH
  else
    MAIN_MODEL_PATH="$MAIN_MODEL"
  fi
fi

if [[ -z "${GUARD_MODEL_PATH:-}" ]]; then
  if GUARD_MODEL_PATH="$(hf_cache_snapshot "$GUARD_MODEL")"; then
    export GUARD_MODEL_PATH
  else
    GUARD_MODEL_PATH="$GUARD_MODEL"
  fi
fi

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
LOG_DIR="${LOG_DIR:-/tmp}"

MAIN_MAX_MODEL_LEN="${MAIN_MAX_MODEL_LEN:-65536}"
GUARD_MAX_MODEL_LEN="${GUARD_MAX_MODEL_LEN:-8192}"
BASELINE_MAIN_MAX_MODEL_LEN="${BASELINE_MAIN_MAX_MODEL_LEN:-8192}"
BASELINE_GUARD_MAX_MODEL_LEN="${BASELINE_GUARD_MAX_MODEL_LEN:-8192}"
KVCACHED_MAIN_GPU_UTIL="${KVCACHED_MAIN_GPU_UTIL:-0.70}"
KVCACHED_GUARD_GPU_UTIL="${KVCACHED_GUARD_GPU_UTIL:-0.25}"
KVCACHED_PAGE_SIZE_MB="${KVCACHED_PAGE_SIZE_MB:-32}"
BASELINE_MAIN_GPU_UTIL="${BASELINE_MAIN_GPU_UTIL:-0.65}"
BASELINE_GUARD_GPU_UTIL="${BASELINE_GUARD_GPU_UTIL:-0.16}"

CONCURRENCIES="${CONCURRENCIES:-4 8 16 32 64 128}"
MIN_NUM_PROMPTS="${MIN_NUM_PROMPTS:-32}"
NUM_PROMPTS_MULTIPLIER="${NUM_PROMPTS_MULTIPLIER:-2}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-256}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-2048}"
BENCH_TIMEOUT_SECONDS="${BENCH_TIMEOUT_SECONDS:-1800}"

DATASET_NAME="${DATASET_NAME:-sharegpt}"
DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json}"
SHAREGPT_URL="${SHAREGPT_URL:-https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json}"
DOWNLOAD_SHAREGPT="${DOWNLOAD_SHAREGPT:-1}"

WAIT_HEALTH_ATTEMPTS="${WAIT_HEALTH_ATTEMPTS:-180}"
WAIT_HEALTH_INTERVAL="${WAIT_HEALTH_INTERVAL:-5}"
