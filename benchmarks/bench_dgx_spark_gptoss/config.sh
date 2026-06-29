#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Shared configuration for the gpt-oss-120b DGX Spark benchmark.
# Source this file from the other scripts; do not run it directly.
#
# Phase 1 (footprint A/B): one model, kvcached vs baseline at the SAME
#   gpu-memory-utilization cap. Goal: identical performance, smaller PHYSICAL
#   KV footprint.
# Phase 2 (co-location): gpt-oss-120b main + a guardrail model on one GPU.
#   Goal: convert the memory kvcached frees into throughput/latency.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- conda env: MUST be the kvcached env (vLLM 0.19.2 + kvcached patches).
# The base `vllm` on PATH is 0.23.0 and is NOT patched -> kvcached would
# silently back the whole pool (zero elasticity, zero saving).
CONDA_ENV="${CONDA_ENV:-kvcached}"
if command -v conda >/dev/null 2>&1; then
  _flags="$-"; set +e +u
  eval "$(conda shell.bash hook 2>/dev/null)"
  conda activate "$CONDA_ENV" 2>/dev/null
  _status=$?
  case "$_flags" in *u*) set -u ;; esac
  case "$_flags" in *e*) set -e ;; esac
  if [[ "$_status" -ne 0 && -d "${HOME}/miniforge3/envs/${CONDA_ENV}/bin" ]]; then
    export PATH="${HOME}/miniforge3/envs/${CONDA_ENV}/bin:${PATH}"
  fi
elif [[ -d "${HOME}/miniforge3/envs/${CONDA_ENV}/bin" ]]; then
  export PATH="${HOME}/miniforge3/envs/${CONDA_ENV}/bin:${PATH}"
fi

# --- models / ports
MAIN_MODEL="${MAIN_MODEL:-openai/gpt-oss-120b}"
# Phase 2 co-tenant. Llama-Guard-3-8B is the canonical guardrail and is fully
# cached; Qwen2-VL-7B is not downloaded and adds vision deps on a tight box.
GUARD_MODEL="${GUARD_MODEL:-meta-llama/Llama-Guard-3-8B}"
MAIN_PORT="${MAIN_PORT:-12346}"
GUARD_PORT="${GUARD_PORT:-12347}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- engine knobs held IDENTICAL across baseline and kvcached (fairness).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"          # user choice: fp8 KV
# Phase 1: the SAME util cap for both modes. This is the whole point — we
# compare physical backing of the same virtual pool, not pool sizes.
# 0.70 on GB10: ~24 GiB of the 122 GiB unified memory is held by the desktop/
# IDE, leaving ~96 GiB free; 0.70*122 = 85 GiB requested fits with headroom.
GPU_UTIL="${GPU_UTIL:-0.70}"

# --- Phase 2 co-location splits (only used by run_phase2.sh).
# baseline must statically partition; kvcached gives main an elastic cap and
# lets the guard coexist in the memory baseline had to wall off.
# Co-location splits on the 121.7 GiB unified box (~24 GiB held by desktop):
# gpt-oss weights ~66 GiB + guard ~15 GiB. baseline must static-split the rest;
# kvcached gives main a big elastic cap and lets the guard coexist in the
# memory baseline had to wall off (kvcached's GPUWorker patch ignores the static
# free-memory check, so main's high cap is virtual, physically backed on demand).
GUARD_MAX_MODEL_LEN="${GUARD_MAX_MODEL_LEN:-4096}"
# NOTE: guard (Llama-Guard-3-8B) weights alone are ~16 GiB = 0.13 of total, so
# guard util MUST exceed that or KV pool computes to <=0 ("No memory for cache
# blocks"). kvcached util is just the virtual cap; physical stays on-demand.
KVCACHED_MAIN_GPU_UTIL="${KVCACHED_MAIN_GPU_UTIL:-0.78}"
KVCACHED_GUARD_GPU_UTIL="${KVCACHED_GUARD_GPU_UTIL:-0.22}"
BASELINE_MAIN_GPU_UTIL="${BASELINE_MAIN_GPU_UTIL:-0.60}"
BASELINE_GUARD_GPU_UTIL="${BASELINE_GUARD_GPU_UTIL:-0.18}"

# --- MXFP4 on GB10 (sm_121): Marlin is the ONLY viable MoE backend here.
# FLASHINFER_TRTLLM/CUTLASS kernels don't support sm_121 (and flashinfer isn't
# importable), and the OAI triton_kernels path requires capability < 11.0.
# Marlin dequantizes FP4->bf16 (correct, portable, a bit slower) — fine for an
# A/B where both modes pay the same cost. Verify coherent output (launch gate).
export VLLM_MXFP4_USE_MARLIN="${VLLM_MXFP4_USE_MARLIN:-1}"
unset VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8 VLLM_USE_FLASHINFER_MXFP4_MOE 2>/dev/null || true
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

# Disable the third-party TriAttention vLLM plugin installed in this env: its
# KV-compression worker hook fatals on long sequences
# (TRIATTN_FATAL_TRITON_SCORING_REQUIRED:stats_path_not_set). VLLM_PLUGINS is an
# allowlist; empty = load none. kvcached is NOT a vLLM plugin (it loads via its
# .pth autopatch), so this does not affect kvcached.
export VLLM_PLUGINS="${VLLM_PLUGINS-}"

# gpt-oss uses the Harmony chat format, which needs the o200k_base tiktoken
# vocab. The auto-download from openaipublic is flaky on this box, so we point
# openai_harmony at a pre-fetched local copy (sha256 446a9538...).
HARMONY_ENC_DIR="${HARMONY_ENC_DIR:-${HOME}/.cache/harmony_encodings}"
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-$HARMONY_ENC_DIR}"

# --- kvcached IPC segment names (so kv_monitor / kvtop can find them).
KVCACHED_MAIN_IPC="${KVCACHED_MAIN_IPC:-gptoss}"
KVCACHED_GUARD_IPC="${KVCACHED_GUARD_IPC:-guard}"

# --- workload (ShareGPT) + concurrency sweep
DATASET_NAME="${DATASET_NAME:-sharegpt}"
DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json}"
SHAREGPT_URL="${SHAREGPT_URL:-https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json}"
DOWNLOAD_SHAREGPT="${DOWNLOAD_SHAREGPT:-1}"
SHAREGPT_OUTPUT_LEN="${SHAREGPT_OUTPUT_LEN:-}"    # empty = use dataset's natural output lengths
# Phase 2 (workflow) knobs. Short ShareGPT (~240 tok) can't saturate a multi-GiB
# KV pool below ~C=200, so to show the co-location speedup we use a long input
# (the realistic "long-context request behind a guardrail" case).
MAIN_OUTPUT_LEN="${MAIN_OUTPUT_LEN:-512}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-6000}"      # words; ~8k+ tokens
CONCURRENCIES="${CONCURRENCIES:-1 4 8 16 32 64}"
NUM_PROMPTS_MULTIPLIER="${NUM_PROMPTS_MULTIPLIER:-8}"
MIN_NUM_PROMPTS="${MIN_NUM_PROMPTS:-256}"
SEED="${SEED:-0}"

# --- dirs / health
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "$LOG_DIR"
WAIT_HEALTH_ATTEMPTS="${WAIT_HEALTH_ATTEMPTS:-240}"
WAIT_HEALTH_INTERVAL="${WAIT_HEALTH_INTERVAL:-5}"

ensure_sharegpt() {
  [[ "$DATASET_NAME" == "sharegpt" ]] || return 0
  [[ -f "$DATASET_PATH" ]] && return 0
  if [[ "$DOWNLOAD_SHAREGPT" == "1" ]]; then
    echo "downloading ShareGPT -> $DATASET_PATH"
    curl -L --fail --retry 3 -o "$DATASET_PATH" "$SHAREGPT_URL"
  else
    echo "ShareGPT dataset missing: $DATASET_PATH" >&2; return 1
  fi
}
