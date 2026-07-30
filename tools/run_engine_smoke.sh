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

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "Engine smoke preflight passed: engine=${ENGINE}, port=${PORT}"
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 2
fi
"${PYTHON}" -c "import ${ENGINE}" >/dev/null

SERVER_PID=""
cleanup() {
  local exit_code=$?
  set +e
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null
    for _ in $(seq 1 30); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -9 "${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export VLLM_USE_V1=1
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="${no_proxy}"

if [[ "${ENGINE}" == "vllm" ]]; then
  "${PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --no-enable-prefix-caching \
    >"${SERVER_LOG}" 2>&1 &
else
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

if ! grep -i "kvcached" "${SERVER_LOG}" >/dev/null 2>&1; then
  echo "Server answered correctly, but no kvcached activation evidence was logged" >&2
  tail -n 120 "${SERVER_LOG}" >&2
  exit 1
fi

echo "ENGINE_SMOKE_OK engine=${ENGINE} model=${MODEL}"
