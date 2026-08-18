#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# Start one kvcached-backed engine, issue a correctness request, and stop it.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENGINE="${ENGINE:-vllm}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-}"
PYTHON="${PYTHON:-python}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-360}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
CHECK_ONLY="${CHECK_ONLY:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/gpu-ci-artifacts}"

case "${ENGINE}" in
  vllm)
    PORT="${PORT:-12346}"
    ;;
  sglang)
    PORT="${PORT:-30000}"
    ;;
  *)
    echo "ENGINE must be 'vllm' or 'sglang', got '${ENGINE}'" >&2
    exit 2
    ;;
esac

if [[ ! "${STARTUP_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STARTUP_TIMEOUT must be a positive integer" >&2
  exit 2
fi
if [[ ! "${PORT}" =~ ^[1-9][0-9]*$ ]] || [[ "${PORT}" -gt 65535 ]]; then
  echo "PORT must be an integer from 1 to 65535" >&2
  exit 2
fi
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Python command not found: ${PYTHON}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/${ENGINE}-server.log"
CLIENT_LOG="${LOG_DIR}/${ENGINE}-client.json"
GPU_PIDS_BEFORE="${LOG_DIR}/${ENGINE}-gpu-pids-before.txt"
GPU_PIDS_AFTER="${LOG_DIR}/${ENGINE}-gpu-pids-after.txt"
GPU_PIDS_INTRODUCED="${LOG_DIR}/${ENGINE}-gpu-pids-introduced.txt"

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "Engine smoke preflight passed: engine=${ENGINE}, port=${PORT}"
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required" >&2
  exit 2
fi
GPU_DEVICE_SELECTOR="${KVCACHED_GPU_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${GPU_DEVICE_SELECTOR}" ]]; then
  echo "KVCACHED_GPU_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES must select the smoke-test GPU" >&2
  exit 2
fi
"${PYTHON}" -c "import ${ENGINE}" >/dev/null

SERVER_PID=""
snapshot_gpu_pids() {
  local output_file="$1"
  {
    nvidia-smi --id="${GPU_DEVICE_SELECTOR}" \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null || true
  } | sed -e 's/[[:space:]]//g' -e '/^$/d' | sort -u >"${output_file}"
}

introduced_gpu_pids() {
  snapshot_gpu_pids "${GPU_PIDS_AFTER}"
  comm -13 "${GPU_PIDS_BEFORE}" "${GPU_PIDS_AFTER}" >"${GPU_PIDS_INTRODUCED}"
  [[ ! -s "${GPU_PIDS_INTRODUCED}" ]]
}

cleanup() {
  local exit_code=$?
  set +e
  trap - EXIT INT TERM
  if [[ -n "${SERVER_PID}" ]] && kill -0 -- "-${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null
    for _ in $(seq 1 30); do
      kill -0 -- "-${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true

  for _ in $(seq 1 30); do
    introduced_gpu_pids && break
    sleep 1
  done
  if ! introduced_gpu_pids; then
    echo "Engine smoke left processes on selected GPU(s):" >&2
    cat "${GPU_PIDS_INTRODUCED}" >&2
    exit_code=1
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

snapshot_gpu_pids "${GPU_PIDS_BEFORE}"

export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export VLLM_USE_V1=1
# The default model is a linear-attention hybrid (Qwen3.5 GDN: three
# linear-attention layers per full-attention layer). Its per-block recurrent
# state does not fit the default 2MB page, and kv_cache_manager.py refuses to
# start rather than hand out an empty KV pool. 4MB fits. The contiguous layout
# is supported for this family, so leave it at the default, and do not ask
# either engine to disable its hybrid KV-cache manager -- the attention groups
# have different specs and cannot be unified. See
# examples/08_hybrid_attention_models/README.md.
export KVCACHED_PAGE_SIZE_MB="${KVCACHED_PAGE_SIZE_MB:-4}"
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="${no_proxy}"

if [[ "${ENGINE}" == "vllm" ]]; then
  "${PYTHON}" -c \
    'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
    "${PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --no-enable-prefix-caching \
    --trust-remote-code \
    >"${SERVER_LOG}" 2>&1 &
else
  "${PYTHON}" -c \
    'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
    "${PYTHON}" -m sglang.launch_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --context-length "${MAX_MODEL_LEN}" \
    --disable-radix-cache \
    --trust-remote-code \
    >"${SERVER_LOG}" 2>&1 &
fi
SERVER_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "${ENGINE} exited before becoming ready" >&2
    tail -n 120 "${SERVER_LOG}" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "${ENGINE} did not become ready within ${STARTUP_TIMEOUT}s" >&2
    tail -n 120 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 2
done

ENGINE="${ENGINE}" MODEL="${MODEL}" HOST="${HOST}" PORT="${PORT}" \
CLIENT_LOG="${CLIENT_LOG}" \
  "${PYTHON}" - <<'PY'
import json
import os
import urllib.request

payload = {
    "model": os.environ["MODEL"],
    "prompt": (
        "France is a country in Western Europe. "
        "Question: What is the capital of France? "
        "Answer with only the city name."
    ),
    "temperature": 0,
    "max_tokens": 8,
}
request = urllib.request.Request(
    f"http://{os.environ['HOST']}:{os.environ['PORT']}/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)

with open(os.environ["CLIENT_LOG"], "w", encoding="utf-8") as output:
    json.dump(result, output, indent=2, sort_keys=True)
    output.write("\n")

if "error" in result:
    raise SystemExit(f"engine returned an error: {result['error']}")
try:
    text = result["choices"][0]["text"].strip()
except (KeyError, IndexError, TypeError) as exc:
    raise SystemExit(f"invalid completion response: {result}") from exc
if "paris" not in text.lower():
    raise SystemExit(f"correctness check failed; expected Paris, got {text!r}")
print(f"ENGINE_CLIENT_OK engine={os.environ.get('ENGINE', '')} answer={text!r}")
PY

if grep -E "Traceback|CUDA error|illegal memory access|core dumped|Segmentation fault" \
  "${SERVER_LOG}" >/dev/null 2>&1; then
  echo "Failure signature found in ${ENGINE} server log" >&2
  tail -n 120 "${SERVER_LOG}" >&2
  exit 1
fi

if ! "${PYTHON}" tools/check_engine_activation.py \
  --engine "${ENGINE}" --log "${SERVER_LOG}"; then
  echo "Server answered correctly, but kvcached allocator initialization was not verified" >&2
  tail -n 120 "${SERVER_LOG}" >&2
  exit 1
fi

echo "ENGINE_SMOKE_OK engine=${ENGINE} model=${MODEL}"
