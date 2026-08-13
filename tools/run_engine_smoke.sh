#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# Start one kvcached-backed engine, issue a correctness request, and stop it.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENGINE="${ENGINE:-vllm}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-}"
PYTHON="${PYTHON:-python}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-360}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
LAYOUT="${LAYOUT:-contiguous}"
RESULT_FILE="${RESULT_FILE:-}"
CHECK_ONLY="${CHECK_ONLY:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/gpu-ci-artifacts}"
PHASE="preflight"
RESULT_STATUS=""

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

case "${LAYOUT}" in
  contiguous)
    KVCACHED_CONTIGUOUS_LAYOUT=true
    ;;
  non-contiguous)
    KVCACHED_CONTIGUOUS_LAYOUT=false
    ;;
  *)
    echo "LAYOUT must be 'contiguous' or 'non-contiguous', got '${LAYOUT}'" >&2
    exit 2
    ;;
esac
export KVCACHED_CONTIGUOUS_LAYOUT

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
  echo "Engine smoke preflight passed: engine=${ENGINE}, layout=${LAYOUT}, port=${PORT}"
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
    RESULT_STATUS="crash-at-startup"
  fi
  if [[ -n "${RESULT_FILE}" ]]; then
    local status="${RESULT_STATUS}"
    if [[ -z "${status}" ]]; then
      if [[ "${exit_code}" -eq 0 ]]; then
        status="pass"
      elif [[ "${PHASE}" == "ready" ]]; then
        status="garbled-output"
      else
        status="crash-at-startup"
      fi
    fi
    mkdir -p "$(dirname "${RESULT_FILE}")"
    ENGINE="${ENGINE}" MODEL="${MODEL}" LAYOUT="${LAYOUT}" \
    STATUS="${status}" EXIT_CODE="${exit_code}" PHASE="${PHASE}" \
    RESULT_FILE="${RESULT_FILE}" "${PYTHON}" - <<'PY'
import json
import os

result = {
    "engine": os.environ["ENGINE"],
    "exit_code": int(os.environ["EXIT_CODE"]),
    "layout": os.environ["LAYOUT"],
    "model": os.environ["MODEL"],
    "phase": os.environ["PHASE"],
    "status": os.environ["STATUS"],
}
with open(os.environ["RESULT_FILE"], "w", encoding="utf-8") as output:
    json.dump(result, output, indent=2, sort_keys=True)
    output.write("\n")
PY
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

snapshot_gpu_pids "${GPU_PIDS_BEFORE}"

export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export VLLM_USE_V1=1
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="${no_proxy}"

PHASE="starting"
if [[ "${ENGINE}" == "vllm" ]]; then
  SERVER_COMMAND=(
    "${PYTHON}" -m vllm.entrypoints.openai.api_server
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --no-enable-prefix-caching
  )
else
  SERVER_COMMAND=(
    "${PYTHON}" -m sglang.launch_server
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --context-length "${MAX_MODEL_LEN}" \
    --disable-radix-cache \
    --trust-remote-code
  )
fi

# Start outside the checkout so its unbuilt source tree cannot shadow the
# installed wheel, which contains kvcached's compiled extension modules.
(
  cd "${LOG_DIR}"
  exec "${PYTHON}" -c \
    'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
    "${SERVER_COMMAND[@]}"
) >"${SERVER_LOG}" 2>&1 &
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
PHASE="ready"

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
  RESULT_STATUS="crash-at-startup"
  exit 1
fi

RESULT_STATUS="pass"
echo "ENGINE_SMOKE_OK engine=${ENGINE} model=${MODEL}"
