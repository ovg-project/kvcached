#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# Temporary end-to-end smoke test for kvcached + vLLM NixlConnector P/D disagg.
#
# Intended use on a GPU box/container. The default install path targets the
# RunPod "PyTorch 2.8.0 / CUDA 12.8" template and pins vLLM to avoid pulling
# CUDA 13 wheels.
#
#   bash tools/run_vllm_nixl_pd_smoke.sh
#
# Useful overrides:
#
#   INSTALL_VLLM=0 \
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct \
#   PREFILL_GPU=0 DECODE_GPU=1 \
#   GPU_MEMORY_UTILIZATION=0.35 \
#   BLOCK_SIZE=128 \
#   CLIENT_ENDPOINT=chat \
#   CLIENT_PROMPT='Question: What is the capital of France? Answer with only the city name.\nAnswer:' \
#   EXPECTED_SUBSTRING=Paris \
#   MIN_REMOTE_BLOCKS=2 \
#   bash tools/run_vllm_nixl_pd_smoke.sh
#
# Leave GPU_MEMORY_UTILIZATION unset to use vLLM's default memory planner.
# Set RUN_BASELINE=0 to skip the first plain vLLM+NIXL comparison pass.
# Set STRICT_EXPECTED_SUBSTRING=0 or CLIENT_ENDPOINT=completions for lower-level
# transport-only debugging where generated text quality is not the pass/fail.
# kvcached+NIXL currently needs KVCACHED_NIXL_CONTIGUOUS_LAYOUT=false because
# vLLM's NixlConnector assumes per-layer K/V block-contiguous regions.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
PREFILL_SIDE_PORT="${PREFILL_SIDE_PORT:-5600}"
DECODE_SIDE_PORT="${DECODE_SIDE_PORT:-5601}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
NUM_REQUESTS="${NUM_REQUESTS:-3}"
MAX_TOKENS="${MAX_TOKENS:-8}"
CLIENT_ENDPOINT="${CLIENT_ENDPOINT:-chat}"
CLIENT_PROMPT_DEFAULT="Context: This is a smoke test for disaggregated prefill and decode. The request should still be a normal factual question with a deterministic answer. France is a country in Western Europe. Its capital city is Paris. This paragraph is intentionally ordinary prose, not technical filler, so the generated answer remains easy to inspect. The extra context gives the prefill worker enough tokens to publish more than one KV transfer block, which avoids a known one-block edge case in this temporary smoke harness. A traveler planning a short visit might read about the Eiffel Tower, the Seine, museums, train stations, cafes, and city neighborhoods, but the factual answer requested below is still just the capital city.\\nQuestion: What is the capital of France? Answer with only the city name.\\nAnswer:"
CLIENT_PROMPT="${CLIENT_PROMPT:-${CLIENT_PROMPT_DEFAULT}}"
EXPECTED_SUBSTRING="${EXPECTED_SUBSTRING:-Paris}"
STRICT_EXPECTED_SUBSTRING="${STRICT_EXPECTED_SUBSTRING:-1}"
MIN_REMOTE_BLOCKS="${MIN_REMOTE_BLOCKS:-2}"
PROMPT_REPETITIONS="${PROMPT_REPETITIONS:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
WATCHDOG_INTERVAL="${WATCHDOG_INTERVAL:-15}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-240}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-1}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
KVCACHED_NIXL_CONTIGUOUS_LAYOUT="${KVCACHED_NIXL_CONTIGUOUS_LAYOUT:-false}"
VLLM_VERSION="${VLLM_VERSION:-0.10.2}"
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
NIXL_LOG_LEVEL="${NIXL_LOG_LEVEL:-DEBUG}"
LOG_DIR="${LOG_DIR:-$(mktemp -d /tmp/kvcached-nixl-pd-smoke.XXXXXX)}"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
else
  GPU_COUNT=1
fi

PREFILL_GPU="${PREFILL_GPU:-0}"
if [[ -z "${DECODE_GPU:-}" ]]; then
  if [[ "${GPU_COUNT}" -ge 2 ]]; then
    DECODE_GPU=1
  else
    DECODE_GPU=0
  fi
fi

PREFILL_LOG="${LOG_DIR}/prefill.log"
DECODE_LOG="${LOG_DIR}/decode.log"
CLIENT_LOG="${LOG_DIR}/client.log"
PREFILL_PID=""
DECODE_PID=""

log() {
  printf '[nixl-smoke] %s\n' "$*" >&2
}

dump_runtime_state() {
  set +e
  printf '\n--- process snapshot ---\n' >&2
  ps -eo pid,ppid,pgid,stat,etime,cmd 2>/dev/null \
    | grep -E 'vllm|EngineCore|api_server|kvcached|python' \
    | grep -v grep >&2 || true

  if command -v nvidia-smi >/dev/null 2>&1; then
    printf '\n--- nvidia-smi compute apps ---\n' >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv >&2 || true
    printf '\n--- nvidia-smi summary ---\n' >&2
    nvidia-smi >&2 || true
  fi

  for port_name in PREFILL_PORT DECODE_PORT; do
    local port="${!port_name}"
    printf '\n--- http diagnostics %s=%s ---\n' "${port_name}" "${port}" >&2
    curl -m 5 -fsS "http://${HOST}:${port}/health" >&2 \
      && printf '\n' >&2 || true
    printf 'load: ' >&2
    curl -m 5 -fsS "http://${HOST}:${port}/load" >&2 \
      && printf '\n' >&2 || true
    printf 'metrics excerpt:\n' >&2
    curl -m 5 -fsS "http://${HOST}:${port}/metrics" 2>/dev/null \
      | grep -E 'vllm:.*(num_requests|gpu_cache|prefix_cache|time_to_first_token|time_per_output|prompt_tokens|generation_tokens|request)' \
      | tail -80 >&2 || true
  done
}

die() {
  printf '[nixl-smoke][FAIL] %s\n' "$*" >&2
  dump_runtime_state
  if [[ -f "${PREFILL_LOG}" ]]; then
    printf '\n--- tail %s ---\n' "${PREFILL_LOG}" >&2
    tail -"${LOG_TAIL_LINES}" "${PREFILL_LOG}" >&2 || true
  fi
  if [[ -f "${DECODE_LOG}" ]]; then
    printf '\n--- tail %s ---\n' "${DECODE_LOG}" >&2
    tail -"${LOG_TAIL_LINES}" "${DECODE_LOG}" >&2 || true
  fi
  if [[ -f "${CLIENT_LOG}" ]]; then
    printf '\n--- %s ---\n' "${CLIENT_LOG}" >&2
    cat "${CLIENT_LOG}" >&2 || true
  fi
  exit 1
}

stop_servers() {
  set +e
  for pid in "${PREFILL_PID}" "${DECODE_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill -TERM "-${pid}" >/dev/null 2>&1 || kill -TERM "${pid}" >/dev/null 2>&1 || true
    fi
  done
  sleep 3
  for pid in "${PREFILL_PID}" "${DECODE_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill -KILL "-${pid}" >/dev/null 2>&1 || kill -KILL "${pid}" >/dev/null 2>&1 || true
    fi
  done
  PREFILL_PID=""
  DECODE_PID=""
}

cleanup() {
  stop_servers
}
trap cleanup EXIT

wait_for_server() {
  local port="$1"
  local name="$2"
  local log_file="$3"
  local deadline=$((SECONDS + REQUEST_TIMEOUT))

  log "Waiting for ${name} on ${HOST}:${port}"
  until curl -fsS "http://${HOST}:${port}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$4" >/dev/null 2>&1; then
      die "${name} process exited before becoming ready"
    fi
    if [[ "${SECONDS}" -gt "${deadline}" ]]; then
      die "Timed out waiting for ${name}; log: ${log_file}"
    fi
    sleep 2
  done
}

check_torch_cuda() {
  log "Checking Torch/CUDA before installing vLLM"
  python - <<'PY' || return 1
import sys

try:
    import torch
except Exception as exc:
    print(f"FAILED_TO_IMPORT_TORCH: {exc!r}", file=sys.stderr)
    raise

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("Torch imports, but torch.cuda.is_available() is false")
if torch.cuda.device_count() < 1:
    raise SystemExit("Torch reports zero CUDA devices")
print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
PY
}

install_vllm_stack() {
  if [[ "${INSTALL_VLLM}" != "1" ]]; then
    return
  fi

  local current_version=""
  current_version="$(python - <<'PY' 2>/dev/null || true
try:
    import vllm
except Exception:
    raise SystemExit(0)
print(getattr(vllm, "__version__", "unknown"))
PY
)"

  if [[ "${current_version}" == "${VLLM_VERSION}" ]] && command -v "${VLLM_BIN}" >/dev/null 2>&1; then
    log "vLLM ${current_version} already installed"
    return
  fi

  if [[ -n "${current_version}" ]]; then
    log "Replacing vLLM ${current_version} with pinned vLLM ${VLLM_VERSION}"
  else
    log "Installing pinned vLLM ${VLLM_VERSION}"
  fi

  python -m pip install \
    "transformers>=4.55.2,<5" \
    "huggingface-hub<1" \
    "vllm==${VLLM_VERSION}" \
    nixl \
    pytest

  hash -r
}

ensure_python_deps() {
  if [[ "${INSTALL_DEPS}" != "1" ]]; then
    return
  fi

  python - <<'PY' >/dev/null 2>&1 || python -m pip install -q pytest
import pytest
PY

  python - <<'PY' >/dev/null 2>&1 || python -m pip install -q nixl
import nixl
PY
}

install_editable() {
  if [[ "${INSTALL_EDITABLE}" != "1" ]]; then
    return
  fi

  log "Installing kvcached from this checkout"
  LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}" \
    python -m pip install -e . --no-build-isolation --no-cache-dir

  log "Installing editable autopatch .pth"
  python tools/dev_copy_pth.py
}

run_unit_tests() {
  if [[ "${RUN_UNIT_TESTS}" != "1" ]]; then
    return
  fi

  log "Running focused Nixl compatibility unit tests"
  python -m pytest tests/test_vllm_nixl_compat.py -q
}

start_vllm() {
  local name="$1"
  local gpu="$2"
  local port="$3"
  local side_port="$4"
  local ipc_name="$5"
  local log_file="$6"
  local enable_kvcached="$7"
  local run_label="$8"
  local autopatch=0
  if [[ "${enable_kvcached}" == "true" ]]; then
    autopatch=1
  fi

  local -a extra_args=()
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${VLLM_EXTRA_ARGS})
  fi
  local -a memory_args=()
  if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
    memory_args=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
  fi
  local -a layout_env=()
  if [[ "${enable_kvcached}" != "true" ]]; then
    layout_env=(VLLM_KV_CACHE_LAYOUT=HND)
  fi

  log "Starting ${run_label} ${name}: gpu=${gpu}, api_port=${port}, nixl_side_port=${side_port}, kvcached=${enable_kvcached}, kvcached_contiguous_layout=${KVCACHED_NIXL_CONTIGUOUS_LAYOUT}"
  setsid env \
    ENABLE_KVCACHED="${enable_kvcached}" \
    KVCACHED_AUTOPATCH="${autopatch}" \
    KVCACHED_CONTIGUOUS_LAYOUT="${KVCACHED_NIXL_CONTIGUOUS_LAYOUT}" \
    KVCACHED_LOG_LEVEL=DEBUG \
    KVCACHED_IPC_NAME="${ipc_name}" \
    VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL}" \
    NIXL_LOG_LEVEL="${NIXL_LOG_LEVEL}" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    VLLM_USE_V1=1 \
    UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}" \
    VLLM_NIXL_SIDE_CHANNEL_HOST="${HOST}" \
    VLLM_NIXL_SIDE_CHANNEL_PORT="${side_port}" \
    "${layout_env[@]}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${VLLM_BIN}" serve "${MODEL}" \
      --host 0.0.0.0 \
      --port "${port}" \
      --enforce-eager \
      --no-enable-prefix-caching \
      --max-model-len "${MAX_MODEL_LEN}" \
      --block-size "${BLOCK_SIZE}" \
      "${memory_args[@]}" \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}' \
      "${extra_args[@]}" \
      >"${log_file}" 2>&1 &

  echo "$!"
}

run_pd_client() {
  local run_label="$1"
  log "Running ${run_label} direct two-step P/D client (${NUM_REQUESTS} requests)"
  MODEL="${MODEL}" \
  RUN_LABEL="${run_label}" \
  HOST="${HOST}" \
  PREFILL_PORT="${PREFILL_PORT}" \
  DECODE_PORT="${DECODE_PORT}" \
  NUM_REQUESTS="${NUM_REQUESTS}" \
  MAX_TOKENS="${MAX_TOKENS}" \
  CLIENT_ENDPOINT="${CLIENT_ENDPOINT}" \
  CLIENT_PROMPT="${CLIENT_PROMPT}" \
  EXPECTED_SUBSTRING="${EXPECTED_SUBSTRING}" \
  STRICT_EXPECTED_SUBSTRING="${STRICT_EXPECTED_SUBSTRING}" \
  MIN_REMOTE_BLOCKS="${MIN_REMOTE_BLOCKS}" \
  PROMPT_REPETITIONS="${PROMPT_REPETITIONS}" \
  REQUEST_TIMEOUT="${REQUEST_TIMEOUT}" \
  WATCHDOG_INTERVAL="${WATCHDOG_INTERVAL}" \
  python - <<'PY' 2>&1 | tee "${CLIENT_LOG}"
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

model = os.environ["MODEL"]
run_label = os.environ["RUN_LABEL"]
host = os.environ["HOST"]
prefill_port = int(os.environ["PREFILL_PORT"])
decode_port = int(os.environ["DECODE_PORT"])
num_requests = int(os.environ["NUM_REQUESTS"])
max_tokens = int(os.environ["MAX_TOKENS"])
client_endpoint = os.environ["CLIENT_ENDPOINT"].strip().lower()
client_prompt = os.environ["CLIENT_PROMPT"].encode("utf-8").decode("unicode_escape")
expected_substring = os.environ["EXPECTED_SUBSTRING"]
strict_expected_substring = os.environ["STRICT_EXPECTED_SUBSTRING"] == "1"
min_remote_blocks = int(os.environ["MIN_REMOTE_BLOCKS"])
prompt_repetitions = int(os.environ["PROMPT_REPETITIONS"])
timeout = int(os.environ["REQUEST_TIMEOUT"])
watchdog_interval = max(1, int(os.environ["WATCHDOG_INTERVAL"]))

prompt = "\n".join([client_prompt] * prompt_repetitions).strip()
total_started = time.time()

endpoint_aliases = {
    "chat": "chat",
    "chat/completions": "chat",
    "chat_completions": "chat",
    "completion": "completions",
    "completions": "completions",
}
client_endpoint = endpoint_aliases.get(client_endpoint, client_endpoint)
if client_endpoint not in {"chat", "completions"}:
    raise SystemExit(
        "CLIENT_ENDPOINT must be one of: chat, chat/completions, "
        "completions"
    )
api_path = (
    "/v1/chat/completions"
    if client_endpoint == "chat"
    else "/v1/completions"
)


def progress(message):
    print(f"[pd-client][{run_label}] {message}", file=sys.stderr, flush=True)


def block_count(block_ids):
    if not block_ids:
        return 0
    if isinstance(block_ids, list) and block_ids and isinstance(block_ids[0], list):
        return sum(len(group) for group in block_ids)
    return len(block_ids)


def block_sample(block_ids):
    if not block_ids:
        return []
    if isinstance(block_ids, list) and block_ids and isinstance(block_ids[0], list):
        return [group[:8] for group in block_ids[:4]]
    return block_ids[:16]


def summarize_kv_transfer(params):
    block_ids = params.get("remote_block_ids") or []
    keys = [
        "do_remote_decode",
        "do_remote_prefill",
        "remote_engine_id",
        "remote_host",
        "remote_port",
        "remote_request_id",
        "remote_num_tokens",
        "remote_dp_rank",
        "remote_dp_size",
        "tp_size",
        "transfer_id",
    ]
    summary = {key: params.get(key) for key in keys if key in params}
    summary["remote_block_count"] = block_count(block_ids)
    summary["remote_block_sample"] = block_sample(block_ids)
    return summary


def http_get(port, path, timeout_sec=3):
    req = urllib.request.Request(f"http://{host}:{port}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return resp.read().decode("utf-8", errors="replace")


def poll_server(port, phase, elapsed):
    progress(f"{phase}: still waiting after {elapsed:.1f}s; polling port {port}")
    for path in ("/health", "/load"):
        try:
            body = http_get(port, path).strip()
            progress(f"{phase}: {path} -> {body[:500]}")
        except Exception as exc:
            progress(f"{phase}: {path} failed: {exc!r}")

    try:
        metrics = http_get(port, "/metrics")
    except Exception as exc:
        progress(f"{phase}: /metrics failed: {exc!r}")
        return

    interesting = []
    needles = (
        "num_requests",
        "gpu_cache",
        "prefix_cache",
        "time_to_first_token",
        "time_per_output",
        "prompt_tokens",
        "generation_tokens",
        "request",
    )
    for line in metrics.splitlines():
        if line.startswith("#"):
            continue
        if "vllm:" in line and any(needle in line for needle in needles):
            interesting.append(line)
    for line in interesting[-30:]:
        progress(f"{phase}: metric {line}")


def post(port, payload, request_id, phase):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}{api_path}",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
            "X-Request-Id": request_id,
        },
    )
    result = {}

    def send():
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result["response"] = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            result["error"] = RuntimeError(
                f"HTTP {exc.code} from port {port}: {body}"
            )
        except Exception as exc:
            result["error"] = exc

    started = time.time()
    thread = threading.Thread(target=send, name=f"{phase}-http", daemon=True)
    thread.start()
    next_poll = started + watchdog_interval
    while thread.is_alive():
        sleep_for = max(0.1, min(1.0, next_poll - time.time()))
        thread.join(sleep_for)
        if thread.is_alive() and time.time() >= next_poll:
            poll_server(port, phase, time.time() - started)
            next_poll += watchdog_interval

    elapsed = time.time() - started
    if "error" in result:
        raise RuntimeError(
            f"{phase} request failed after {elapsed:.1f}s on port {port}: "
            f"{result['error']!r}"
        ) from result["error"]
    progress(f"{phase}: HTTP response received in {elapsed:.3f}s")
    return result["response"]


def make_payload(prompt_text, tokens, kv_transfer_params):
    payload = {
        "model": model,
        "max_tokens": tokens,
        "temperature": 0,
        "stream": False,
        "kv_transfer_params": kv_transfer_params,
    }
    if client_endpoint == "chat":
        payload["messages"] = [{"role": "user", "content": prompt_text}]
    else:
        payload["prompt"] = prompt_text
    return payload


def extract_generated_text(response):
    choices = response.get("choices") or []
    if not choices:
        print(json.dumps(response, indent=2)[:4000], file=sys.stderr)
        raise RuntimeError("Decode response did not include choices")

    choice = choices[0]
    if client_endpoint == "chat":
        message = choice.get("message") or {}
        return message.get("content") or ""
    return choice.get("text") or ""


for idx in range(num_requests):
    request_id = f"kvcached-nixl-smoke-{idx}-{uuid.uuid4()}"
    prompt_text = prompt
    prefill_payload = make_payload(
        prompt_text,
        1,
        {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        },
    )

    started = time.time()
    progress(
        f"request {idx}: prefill start, request_id={request_id}, "
        f"endpoint={client_endpoint}, prompt_chars={len(prompt_text)}, "
        "max_tokens=1"
    )
    prefill_resp = post(prefill_port, prefill_payload, request_id, "prefill")
    kv_transfer_params = prefill_resp.get("kv_transfer_params")
    if not kv_transfer_params:
        print(json.dumps(prefill_resp, indent=2)[:4000], file=sys.stderr)
        raise RuntimeError("Prefill response did not include kv_transfer_params")
    progress(
        "request {}: kv_transfer_params={}".format(
            idx, json.dumps(summarize_kv_transfer(kv_transfer_params), sort_keys=True)
        )
    )
    remote_block_count = block_count(kv_transfer_params.get("remote_block_ids") or [])
    if remote_block_count < min_remote_blocks:
        raise RuntimeError(
            "Prefill produced fewer remote blocks than this smoke expects: "
            f"{remote_block_count} < {min_remote_blocks}. "
            "Use a longer CLIENT_PROMPT or lower MIN_REMOTE_BLOCKS."
        )
    progress(
        "request {}: prefill done, remote_blocks={}".format(
            idx, remote_block_count
        )
    )

    decode_payload = make_payload(prompt_text, max_tokens, kv_transfer_params)
    progress(
        f"request {idx}: decode start, prompt_chars={len(prompt_text)}, "
        f"max_tokens={max_tokens}, timeout={timeout}s"
    )
    decode_resp = post(decode_port, decode_payload, request_id, "decode")
    text = extract_generated_text(decode_resp)
    if expected_substring and expected_substring.lower() not in text.lower():
        message = (
            "Decode response did not contain expected substring "
            f"{expected_substring!r}; text={text[:400]!r}"
        )
        if strict_expected_substring:
            raise RuntimeError(message)
        progress(f"request {idx}: WARNING: {message}")
    elapsed = time.time() - started
    print(
        json.dumps(
            {
                "request": idx,
                "mode": run_label,
                "endpoint": client_endpoint,
                "elapsed_sec": round(elapsed, 3),
                "remote_engine_id": kv_transfer_params.get("remote_engine_id"),
                "remote_request_id": kv_transfer_params.get("remote_request_id"),
                "remote_num_tokens": kv_transfer_params.get("remote_num_tokens"),
                "remote_blocks": remote_block_count,
                "answer": text.strip(),
            },
            sort_keys=True,
        )
    )

print(
    json.dumps(
        {
            "mode": run_label,
            "status": "PD_CLIENT_OK",
            "requests": num_requests,
            "total_elapsed_sec": round(time.time() - total_started, 3),
        },
        sort_keys=True,
    )
)
print("PD_CLIENT_OK")
PY
}

check_logs() {
  local expect_kvcached="$1"
  log "Checking logs for patch evidence and failure signatures"

  if grep -E "Skipping NixlConnector patch: NIXL connector not installed" "${PREFILL_LOG}" "${DECODE_LOG}" >/dev/null 2>&1; then
    die "NIXL connector module was not installed/importable"
  fi

  if [[ "${expect_kvcached}" == "true" ]]; then
    if ! grep -E "Patched NixlConnector for kvcached PD disagg compatibility" "${PREFILL_LOG}" "${DECODE_LOG}" >/dev/null 2>&1; then
      die "Did not find NixlConnector patch success log"
    fi

    if ! grep -E "NixlConnector layout overridden to NHD|NixlConnector num_blocks" "${PREFILL_LOG}" "${DECODE_LOG}" >/dev/null 2>&1; then
      die "Did not find NixlConnector layout or KV registration compatibility log"
    fi
  elif grep -E "Successfully patched vllm|KVCACHED_MEMORY_POOL|NixlConnector num_blocks" "${PREFILL_LOG}" "${DECODE_LOG}" >/dev/null 2>&1; then
    die "Baseline unexpectedly used kvcached/autopatch"
  fi

  if grep -E "set_stride|inconsistent KV block counts|All kv cache tensors must have the same number of blocks|AssertionError|Traceback|NIXL transfer failure" "${PREFILL_LOG}" "${DECODE_LOG}" "${CLIENT_LOG}" >/dev/null 2>&1; then
    die "Found failure signature in logs"
  fi
}

run_case() {
  local run_label="$1"
  local enable_kvcached="$2"

  PREFILL_LOG="${LOG_DIR}/${run_label}.prefill.log"
  DECODE_LOG="${LOG_DIR}/${run_label}.decode.log"
  CLIENT_LOG="${LOG_DIR}/${run_label}.client.log"

  log "===== ${run_label} ====="
  PREFILL_PID="$(start_vllm prefill "${PREFILL_GPU}" "${PREFILL_PORT}" "${PREFILL_SIDE_PORT}" "${run_label}_prefill_$$" "${PREFILL_LOG}" "${enable_kvcached}" "${run_label}")"
  DECODE_PID="$(start_vllm decode "${DECODE_GPU}" "${DECODE_PORT}" "${DECODE_SIDE_PORT}" "${run_label}_decode_$$" "${DECODE_LOG}" "${enable_kvcached}" "${run_label}")"

  wait_for_server "${PREFILL_PORT}" "${run_label} prefill" "${PREFILL_LOG}" "${PREFILL_PID}"
  wait_for_server "${DECODE_PORT}" "${run_label} decode" "${DECODE_LOG}" "${DECODE_PID}"

  if ! run_pd_client "${run_label}"; then
    die "${run_label} direct P/D client failed"
  fi
  check_logs "${enable_kvcached}"

  log "${run_label} PASS"
  log "${run_label} prefill log: ${PREFILL_LOG}"
  log "${run_label} decode log: ${DECODE_LOG}"
  log "${run_label} client log: ${CLIENT_LOG}"
  stop_servers
}

main() {
  mkdir -p "${LOG_DIR}"
  log "Logs: ${LOG_DIR}"
  log "Model: ${MODEL}"
  log "GPU count detected: ${GPU_COUNT}; prefill GPU=${PREFILL_GPU}; decode GPU=${DECODE_GPU}"
  log "Block size: ${BLOCK_SIZE}; max model len: ${MAX_MODEL_LEN}; GPU utilization: ${GPU_MEMORY_UTILIZATION:-vLLM default}"
  log "Client requests: ${NUM_REQUESTS}; max tokens: ${MAX_TOKENS}; prompt repetitions: ${PROMPT_REPETITIONS}"
  log "Client endpoint: ${CLIENT_ENDPOINT}"
  log "Request timeout: ${REQUEST_TIMEOUT}s; watchdog interval: ${WATCHDOG_INTERVAL}s; log tail lines: ${LOG_TAIL_LINES}"
  log "Client prompt: ${CLIENT_PROMPT}"
  log "Expected substring: ${EXPECTED_SUBSTRING:-<none>}"
  log "Strict expected substring: ${STRICT_EXPECTED_SUBSTRING}"
  log "Minimum remote blocks: ${MIN_REMOTE_BLOCKS}"
  log "Pinned vLLM version: ${VLLM_VERSION}; INSTALL_VLLM=${INSTALL_VLLM}"
  log "Run baseline first: ${RUN_BASELINE}"
  log "kvcached NIXL contiguous layout: ${KVCACHED_NIXL_CONTIGUOUS_LAYOUT}"
  log "vLLM logging level: ${VLLM_LOGGING_LEVEL}; NIXL log level: ${NIXL_LOG_LEVEL}"

  command -v curl >/dev/null 2>&1 || die "curl is required"
  command -v setsid >/dev/null 2>&1 || die "setsid is required"

  check_torch_cuda || die "Torch/CUDA is not usable. Use the RunPod PyTorch 2.8.0 CUDA 12.8 template or fix the pod before running this test."
  install_vllm_stack
  ensure_python_deps
  command -v "${VLLM_BIN}" >/dev/null 2>&1 || die "vLLM binary not found after install: ${VLLM_BIN}"
  install_editable
  run_unit_tests

  if [[ "${RUN_BASELINE}" == "1" ]]; then
    run_case without_kvcached false
    printf '\n\n\n\n'
  fi

  run_case with_kvcached true

  log "PASS"
}

main "$@"
