#!/usr/bin/env bash
# =============================================================================
# Debug Script: LMCacheConnectorV1 1P1D PD-disagg harness for kvcached + vLLM
#
# Purpose:
#   Start a minimal 1-prefill / 1-decode vLLM PD-disagg setup with
#   LMCacheConnectorV1 over LMCache/NIXL, route requests through a tiny local
#   proxy, and capture logs for the same request-ID-randomization matrix used
#   for the P2pNcclConnector investigation.
#
# Usage:
#   chmod +x experiments/12_lmcache_connector_v1_debug.sh
#   ./experiments/12_lmcache_connector_v1_debug.sh
#
# Useful overrides:
#   RUN_WITH_KVCACHED=0 ./experiments/12_lmcache_connector_v1_debug.sh
#   DISABLE_REQUEST_ID_RANDOMIZATION=1 ./experiments/12_lmcache_connector_v1_debug.sh
#   TIMEOUT_REQUEST=300 KEEP_ALIVE_ON_FAIL=1 ./experiments/12_lmcache_connector_v1_debug.sh
#
# Logs are saved to experiments/logs_lmcache_v1_debug/<timestamp>/.
# =============================================================================

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-base_model}"
VLLM_VERSION="${VLLM_VERSION:-0.19.0}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_TOKENS="${MAX_TOKENS:-20}"
PROMPT_MODE="${PROMPT_MODE:-long}"
LONG_PROMPT_REPEAT="${LONG_PROMPT_REPEAT:-24}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
PREFILL_GPU_MEM_UTIL="${PREFILL_GPU_MEM_UTIL:-$GPU_MEM_UTIL}"
DECODE_GPU_MEM_UTIL="${DECODE_GPU_MEM_UTIL:-$GPU_MEM_UTIL}"

PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
PREFILL_PORT="${PREFILL_PORT:-7100}"
DECODE_PORT="${DECODE_PORT:-7200}"
PROXY_PORT="${PROXY_PORT:-9100}"
PROXY_BIND_IP="${PROXY_BIND_IP:-0.0.0.0}"

LMCACHE_CONFIG_STYLE="${LMCACHE_CONFIG_STYLE:-pd}"
LMCACHE_PYTHONHASHSEED="${LMCACHE_PYTHONHASHSEED:-123}"
LMCACHE_PREFILL_RPC_PORT="${LMCACHE_PREFILL_RPC_PORT:-producer1}"
LMCACHE_DECODE_RPC_PORT="${LMCACHE_DECODE_RPC_PORT:-consumer1}"
LMCACHE_PD_PROXY_HOST="${LMCACHE_PD_PROXY_HOST:-localhost}"
LMCACHE_PD_PROXY_PORT="${LMCACHE_PD_PROXY_PORT:-7500}"
LMCACHE_PD_PEER_HOST="${LMCACHE_PD_PEER_HOST:-localhost}"
LMCACHE_PD_PEER_INIT_PORT="${LMCACHE_PD_PEER_INIT_PORT:-7300}"
LMCACHE_PD_PEER_ALLOC_PORT="${LMCACHE_PD_PEER_ALLOC_PORT:-7400}"
LMCACHE_NIXL_PEER_HOST="${LMCACHE_NIXL_PEER_HOST:-localhost}"
LMCACHE_NIXL_PEER_PORT="${LMCACHE_NIXL_PEER_PORT:-55555}"
LMCACHE_PREFILL_BUFFER_SIZE="${LMCACHE_PREFILL_BUFFER_SIZE:-1073741824}"
LMCACHE_DECODE_BUFFER_SIZE="${LMCACHE_DECODE_BUFFER_SIZE:-2147483648}"
LMCACHE_BUFFER_DEVICE="${LMCACHE_BUFFER_DEVICE:-cuda}"

TIMEOUT_STARTUP="${TIMEOUT_STARTUP:-300}"
TIMEOUT_REQUEST="${TIMEOUT_REQUEST:-120}"
RUN_WITH_KVCACHED="${RUN_WITH_KVCACHED:-1}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
INSTALL_KVCACHED="${INSTALL_KVCACHED:-1}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
CLEAN_STALE="${CLEAN_STALE:-1}"
KEEP_ALIVE_ON_FAIL="${KEEP_ALIVE_ON_FAIL:-0}"
FORWARD_X_REQUEST_ID="${FORWARD_X_REQUEST_ID:-1}"
DISABLE_REQUEST_ID_RANDOMIZATION="${DISABLE_REQUEST_ID_RANDOMIZATION:-0}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
LOG_ROOT="${LOG_ROOT:-experiments/logs_lmcache_v1_debug}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$REPO_DIR/$LOG_ROOT/$RUN_ID"
LATEST_LINK="$REPO_DIR/$LOG_ROOT/latest"
CONFIG_DIR="$RUN_DIR/configs"

mkdir -p "$RUN_DIR" "$CONFIG_DIR"
ln -sfn "$RUN_DIR" "$LATEST_LINK"

if [ "$RUN_WITH_KVCACHED" = "1" ]; then
    export ENABLE_KVCACHED=true
    export KVCACHED_AUTOPATCH=1
else
    export ENABLE_KVCACHED=false
    export KVCACHED_AUTOPATCH=0
fi
if [ "$DISABLE_REQUEST_ID_RANDOMIZATION" = "1" ]; then
    export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
fi
export KVCACHED_LOG_LEVEL="${KVCACHED_LOG_LEVEL:-INFO}"
export PYTHONHASHSEED="$LMCACHE_PYTHONHASHSEED"
export PYTHONUNBUFFERED=1

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

declare -a CHILD_PIDS=()

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
log_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

kill_port() {
    local port=$1
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti "tcp:$port" | xargs -r kill 2>/dev/null || true
        sleep 1
        lsof -ti "tcp:$port" | xargs -r kill -9 2>/dev/null || true
    fi
}

cleanup_stale() {
    if [ "$CLEAN_STALE" != "1" ]; then
        return 0
    fi
    log_info "Cleaning stale local processes and ports for this harness..."
    kill_port "$PREFILL_PORT"
    kill_port "$DECODE_PORT"
    kill_port "$PROXY_PORT"
    kill_port "$LMCACHE_PD_PROXY_PORT"
    kill_port "$LMCACHE_PD_PEER_INIT_PORT"
    kill_port "$LMCACHE_PD_PEER_ALLOC_PORT"
    kill_port "$LMCACHE_NIXL_PEER_PORT"
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 2
}

summarize_logs() {
    log_info "========== LMCacheConnectorV1 debug summary =========="
    for name in proxy prefill decode; do
        local logfile="$RUN_DIR/$name.log"
        if [ ! -f "$logfile" ]; then
            continue
        fi
        echo "--- $name: $logfile ---"
        grep -E \
            "LMCache|lmcache|NIXL|nixl|Reqid|request_id|Storing KV|Retrieved|prefill_done|decode_done" \
            "$logfile" 2>/dev/null | tail -120 || true
        grep -E \
            "disagg_spec|kv_transfer_params|receiver_init_port|receiver_alloc_port" \
            "$logfile" 2>/dev/null | tail -40 || true
        grep -E \
            "Traceback|ERROR|WARNING|RuntimeError|AssertionError" \
            "$logfile" 2>/dev/null | tail -80 || true
    done

    if grep -q "Storing KV cache" "$RUN_DIR/prefill.log" 2>/dev/null; then
        log_info "Classifier: prefiller reached LMCache store path."
    fi
    if grep -q "Retrieved" "$RUN_DIR/decode.log" 2>/dev/null; then
        log_info "Classifier: decoder reached LMCache retrieve path."
    fi
    if grep -Eq "LMCache hit tokens: [1-9][0-9]*" \
        "$RUN_DIR/decode.log" 2>/dev/null; then
        log_info "Classifier: decoder reported non-zero LMCache hit tokens."
    elif grep -q "LMCache hit tokens: 0" "$RUN_DIR/decode.log" 2>/dev/null; then
        log_info "Classifier: decoder reported zero LMCache hit tokens."
    fi
    if grep -q "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set" \
        "$RUN_DIR/decode.log" "$RUN_DIR/prefill.log" 2>/dev/null; then
        log_info "Classifier: vLLM request ID randomization was disabled."
    fi
}

cleanup() {
    local status=${1:-0}
    if [ "$status" != "0" ] && [ "$KEEP_ALIVE_ON_FAIL" = "1" ]; then
        log_info "KEEP_ALIVE_ON_FAIL=1, leaving child processes running."
        log_info "Logs: $RUN_DIR"
        return 0
    fi

    log_info "Cleaning up child processes..."
    for pid in "${CHILD_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${CHILD_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    pkill -f "VLLM::EngineCore" 2>/dev/null || true
}

on_exit() {
    local status=$?
    if [ "$status" != "0" ]; then
        log_fail "Harness failed with exit status $status"
        summarize_logs || true
    fi
    cleanup "$status"
    exit "$status"
}
trap on_exit EXIT

wait_for_server() {
    local port=$1
    local name=$2
    local logfile=$3
    local elapsed=0

    log_info "Waiting for $name on port $port (timeout ${TIMEOUT_STARTUP}s)..."
    while [ "$elapsed" -lt "$TIMEOUT_STARTUP" ]; do
        if curl -s "http://localhost:$port/health" >/dev/null 2>&1; then
            log_pass "$name is ready (${elapsed}s)"
            return 0
        fi
        if [ -f "$logfile" ] \
            && grep -q "Traceback (most recent call last)" "$logfile" 2>/dev/null; then
            log_fail "$name crashed during startup. See $logfile"
            tail -60 "$logfile"
            return 1
        fi
        if [ -f "$logfile" ] \
            && grep -Eq "error: unrecognized arguments|usage: vllm" \
                "$logfile" 2>/dev/null; then
            log_fail "$name rejected its CLI arguments. See $logfile"
            tail -60 "$logfile"
            return 1
        fi
        if [ $((elapsed % 30)) -eq 0 ] && [ "$elapsed" -gt 0 ]; then
            local last_line
            last_line=$(tail -1 "$logfile" 2>/dev/null | cut -c1-120)
            log_info "  ...${elapsed}s elapsed. Last log: ${last_line:-<empty>}"
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log_fail "$name did not start within ${TIMEOUT_STARTUP}s. See $logfile"
    tail -60 "$logfile"
    return 1
}

check_import() {
    local import_stmt=$1
    python3 - "$import_stmt" <<'PY'
import sys
stmt = sys.argv[1]
exec(stmt)
PY
}

ensure_python_dep() {
    local name=$1
    local import_stmt=$2
    local package=$3
    if check_import "$import_stmt" >/dev/null 2>&1; then
        log_pass "$name is importable"
        return 0
    fi
    if [ "$INSTALL_DEPS" != "1" ]; then
        log_fail "$name is not importable and INSTALL_DEPS=0"
        return 1
    fi
    log_info "Installing $package for $name..."
    pip install -q "$package"
    check_import "$import_stmt"
    log_pass "$name is importable"
}

phase0_setup() {
    log_info "========== PHASE 0: Environment setup =========="
    cleanup_stale

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_fail "nvidia-smi not found. Run this on a GPU node."
        return 1
    fi
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$gpu_count" -lt 2 ]; then
        log_fail "Need at least 2 GPUs, found $gpu_count"
        return 1
    fi
    log_pass "Found $gpu_count GPUs"

    if ! python3 -c "import vllm" 2>/dev/null; then
        log_info "Installing vLLM==$VLLM_VERSION..."
        pip install -q "vllm==$VLLM_VERSION"
    fi
    log_pass "vLLM $(python3 -c 'import vllm; print(vllm.__version__)')"

    ensure_python_dep "aiohttp" "import aiohttp" "aiohttp"
    ensure_python_dep "transformers" "import transformers" "transformers"
    ensure_python_dep "huggingface_hub" "import huggingface_hub" "huggingface_hub"
    ensure_python_dep "lmcache" "import lmcache" "lmcache"

    if ! check_import "from nixl._api import nixl_agent" >/dev/null 2>&1; then
        if ! check_import "from rixl._api import nixl_agent" >/dev/null 2>&1; then
            if [ "$INSTALL_DEPS" = "1" ]; then
                log_info "Installing nixl for LMCache PD transfer..."
                if ! pip install -q nixl; then
                    pip install -q rixl || true
                fi
            fi
            if ! check_import "from nixl._api import nixl_agent" >/dev/null 2>&1 \
                && ! check_import "from rixl._api import nixl_agent" >/dev/null 2>&1; then
                log_fail "NIXL is not importable. Install NIXL, then rerun."
                return 1
            fi
        fi
    fi
    log_pass "NIXL is importable"

    if [ "$SKIP_MODEL_DOWNLOAD" != "1" ]; then
        if ! python3 -c \
            "from transformers import AutoConfig; AutoConfig.from_pretrained('$MODEL')" \
            2>/dev/null; then
            log_info "Downloading model $MODEL..."
            HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download "$MODEL" --quiet
        fi
        log_pass "Model available"
    fi

    if [ "$RUN_WITH_KVCACHED" = "1" ] && [ "$INSTALL_KVCACHED" = "1" ]; then
        log_info "Installing kvcached from this checkout with the normal install path..."
        pip uninstall -y kvcached >/dev/null 2>&1 || true
        CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
            pip install -q "$REPO_DIR" --no-build-isolation
        log_pass "kvcached installed from $REPO_DIR"
    fi

    if [ "$RUN_WITH_KVCACHED" = "1" ]; then
        python3 - <<'PY'
import inspect
from pathlib import Path
import site

import kvcached.integration.vllm.autopatch as autopatch

print(f"kvcached autopatch source: {autopatch.__file__}")
site_dirs = site.getsitepackages()
user_site = site.getusersitepackages()
if user_site:
    site_dirs.append(user_site)
pth_files = [
    Path(site_dir) / "kvcached_autopatch.pth"
    for site_dir in site_dirs
    if (Path(site_dir) / "kvcached_autopatch.pth").exists()
]
assert pth_files, "kvcached_autopatch.pth was not installed into site-packages"
print(f"kvcached autopatch pth: {pth_files[0]}")
assert hasattr(autopatch, "_patch_vllm")
assert inspect.isfunction(autopatch._patch_vllm)
PY
        log_pass "kvcached autopatch source and .pth hook are present"
    else
        log_info "RUN_WITH_KVCACHED=0: running plain vLLM LMCacheConnectorV1."
        log_info "Existing kvcached installs are ignored because ENABLE_KVCACHED=false."
    fi

    if [ "$DISABLE_REQUEST_ID_RANDOMIZATION" = "1" ]; then
        log_info "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1"
    else
        log_info "VLLM request ID randomization is enabled"
    fi
}

write_lmcache_configs() {
    local prefill_config="$CONFIG_DIR/lmcache-prefiller-config.yaml"
    local decode_config="$CONFIG_DIR/lmcache-decoder-config.yaml"

    case "$LMCACHE_CONFIG_STYLE" in
        pd)
            cat > "$prefill_config" <<YAML
local_cpu: True
max_local_cpu_size: 5
max_local_disk_size: 0
enable_pd: True
transfer_channel: "nixl"
pd_role: "sender"
pd_proxy_host: "$LMCACHE_PD_PROXY_HOST"
pd_proxy_port: $LMCACHE_PD_PROXY_PORT
pd_buffer_size: $LMCACHE_PREFILL_BUFFER_SIZE
pd_buffer_device: "$LMCACHE_BUFFER_DEVICE"
YAML
            cat > "$decode_config" <<YAML
local_cpu: False
max_local_cpu_size: 0
max_local_disk_size: 0
enable_pd: True
transfer_channel: "nixl"
pd_role: "receiver"
pd_peer_host: "$LMCACHE_PD_PEER_HOST"
pd_peer_init_port: $LMCACHE_PD_PEER_INIT_PORT
pd_peer_alloc_port: $LMCACHE_PD_PEER_ALLOC_PORT
pd_buffer_size: $LMCACHE_DECODE_BUFFER_SIZE
pd_buffer_device: "$LMCACHE_BUFFER_DEVICE"
nixl_backends: [UCX]
YAML
            ;;
        nixl_legacy)
            cat > "$prefill_config" <<YAML
local_cpu: False
max_local_cpu_size: 0
max_local_disk_size: 0
remote_serde: NULL
enable_nixl: True
nixl_role: "sender"
nixl_peer_host: "$LMCACHE_NIXL_PEER_HOST"
nixl_peer_port: $LMCACHE_NIXL_PEER_PORT
nixl_buffer_size: $LMCACHE_PREFILL_BUFFER_SIZE
nixl_buffer_device: "$LMCACHE_BUFFER_DEVICE"
nixl_enable_gc: True
YAML
            cat > "$decode_config" <<YAML
local_cpu: False
max_local_cpu_size: 0
max_local_disk_size: 0
remote_serde: NULL
enable_nixl: True
nixl_role: "receiver"
nixl_peer_host: "$LMCACHE_NIXL_PEER_HOST"
nixl_peer_port: $LMCACHE_NIXL_PEER_PORT
nixl_buffer_size: $LMCACHE_DECODE_BUFFER_SIZE
nixl_buffer_device: "$LMCACHE_BUFFER_DEVICE"
nixl_enable_gc: True
YAML
            ;;
        *)
            log_fail "Unknown LMCACHE_CONFIG_STYLE=$LMCACHE_CONFIG_STYLE"
            return 1
            ;;
    esac

    log_info "Wrote LMCache config style: $LMCACHE_CONFIG_STYLE"
    log_info "Prefiller config: $prefill_config"
    log_info "Decoder config:   $decode_config"
}

write_proxy() {
    cat > "$RUN_DIR/lmcache_disagg_proxy.py" <<'PY'
import os
import uuid

import aiohttp
from aiohttp import web


async def health(_request):
    return web.json_response({"ok": True})


def _request_id():
    return f"lmcache-disagg-{uuid.uuid4().hex}"


def _port_list(env_name, default):
    raw = os.environ.get(env_name, str(default))
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _attach_prefill_kv_transfer_params(payload, request_id):
    kv_transfer_params = dict(payload.get("kv_transfer_params") or {})
    kv_transfer_params["ret_first_tok"] = True
    kv_transfer_params["disagg_spec"] = {
        "req_id": request_id,
        "receiver_host": os.environ.get("LMCACHE_PD_PEER_HOST", "localhost"),
        "receiver_init_port": _port_list("LMCACHE_PD_PEER_INIT_PORT", 7300),
        "receiver_alloc_port": _port_list("LMCACHE_PD_PEER_ALLOC_PORT", 7400),
    }
    payload["kv_transfer_params"] = kv_transfer_params
    return kv_transfer_params["disagg_spec"]


async def _forward(session, url, payload, request_id):
    headers = {}
    if os.environ.get("FORWARD_X_REQUEST_ID", "1") == "1":
        headers["X-Request-Id"] = request_id
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with session.post(url=url, json=payload, headers=headers) as response:
        body = await response.read()
        content_type = response.headers.get("Content-Type", "application/json")
        return response.status, content_type, body


async def handle_request(request):
    original = await request.json()
    prefill = dict(original)
    prefill["max_tokens"] = 1
    if "max_completion_tokens" in prefill:
        prefill["max_completion_tokens"] = 1
    prefill["stream"] = False

    request_id = _request_id()
    disagg_spec = _attach_prefill_kv_transfer_params(prefill, request_id)
    prefill_url = f"http://{os.environ['PREFILL_HTTP']}{request.path}"
    decode_url = f"http://{os.environ['DECODE_HTTP']}{request.path}"
    print(
        f"handle_request request_id={request_id} "
        f"prefill_url={prefill_url} decode_url={decode_url} "
        f"disagg_spec={disagg_spec}",
        flush=True,
    )

    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        prefill_status, _prefill_type, prefill_body = await _forward(
            session, prefill_url, prefill, request_id
        )
        print(
            f"prefill_done request_id={request_id} status={prefill_status} "
            f"bytes={len(prefill_body)}",
            flush=True,
        )
        if prefill_status >= 400:
            return web.Response(
                body=prefill_body,
                status=prefill_status,
                content_type="application/json",
            )

        decode_status, decode_type, decode_body = await _forward(
            session, decode_url, original, request_id
        )
        print(
            f"decode_done request_id={request_id} status={decode_status} "
            f"bytes={len(decode_body)}",
            flush=True,
        )
        return web.Response(
            body=decode_body,
            status=decode_status,
            content_type=decode_type.split(";")[0],
        )


def main():
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/v1/completions", handle_request)
    app.router.add_post("/v1/chat/completions", handle_request)
    web.run_app(
        app,
        host=os.environ.get("PROXY_BIND_IP", "0.0.0.0"),
        port=int(os.environ.get("PROXY_PORT", "9100")),
    )


if __name__ == "__main__":
    main()
PY
}

make_kv_config() {
    local role=$1
    local rpc_port=$2
    python3 - "$role" "$rpc_port" <<'PY'
import json
import sys

role, rpc_port = sys.argv[1:3]
extra = {
    "discard_partial_chunks": False,
    "lmcache_rpc_port": rpc_port,
}
if role == "kv_consumer":
    extra["skip_last_n_tokens"] = 1

config = {
    "kv_connector": "LMCacheConnectorV1",
    "kv_role": role,
    "kv_connector_extra_config": extra,
}
print(json.dumps(config, separators=(",", ":")))
PY
}

start_proxy() {
    write_proxy
    log_info "Starting LMCache proxy on http://localhost:$PROXY_PORT"
    PROXY_BIND_IP="$PROXY_BIND_IP" \
    PROXY_PORT="$PROXY_PORT" \
    PREFILL_HTTP="127.0.0.1:$PREFILL_PORT" \
    DECODE_HTTP="127.0.0.1:$DECODE_PORT" \
    LMCACHE_PD_PEER_HOST="$LMCACHE_PD_PEER_HOST" \
    LMCACHE_PD_PEER_INIT_PORT="$LMCACHE_PD_PEER_INIT_PORT" \
    LMCACHE_PD_PEER_ALLOC_PORT="$LMCACHE_PD_PEER_ALLOC_PORT" \
    FORWARD_X_REQUEST_ID="$FORWARD_X_REQUEST_ID" \
        python3 "$RUN_DIR/lmcache_disagg_proxy.py" \
        > "$RUN_DIR/proxy.log" 2>&1 &
    CHILD_PIDS+=("$!")
    wait_for_server "$PROXY_PORT" "Proxy" "$RUN_DIR/proxy.log"
}

start_vllm_instance() {
    local name=$1
    local gpu=$2
    local http_port=$3
    local role=$4
    local rpc_port=$5
    local lmcache_config=$6
    local gpu_mem_util=$7
    local logfile="$RUN_DIR/$name.log"
    local kv_config
    kv_config=$(make_kv_config "$role" "$rpc_port")

    log_info "Starting $name on GPU $gpu, HTTP $http_port, role $role"
    echo "$kv_config" > "$RUN_DIR/$name.kv_transfer_config.json"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export ENABLE_KVCACHED
        export KVCACHED_AUTOPATCH
        export KVCACHED_LOG_LEVEL
        export LMCACHE_CONFIG_FILE="$lmcache_config"
        export LMCACHE_USE_EXPERIMENTAL=True
        export VLLM_ENABLE_V1_MULTIPROCESSING=1
        export VLLM_WORKER_MULTIPROC_METHOD=spawn
        export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
        export PYTHONHASHSEED="$LMCACHE_PYTHONHASHSEED"
        if [ "$DISABLE_REQUEST_ID_RANDOMIZATION" = "1" ]; then
            export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
        fi
        exec vllm serve "$MODEL" \
            --host 0.0.0.0 \
            --port "$http_port" \
            --tensor-parallel-size 1 \
            --seed 1024 \
            --served-model-name "$SERVED_MODEL_NAME" \
            --dtype "$DTYPE" \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --trust-remote-code \
            --gpu-memory-utilization "$gpu_mem_util" \
            --enforce-eager \
            --no-enable-prefix-caching \
            --kv-transfer-config "$kv_config" \
            $EXTRA_VLLM_ARGS
    ) > "$logfile" 2>&1 &
    CHILD_PIDS+=("$!")
}

make_payload() {
    local prompt=$1
    python3 - \
        "$SERVED_MODEL_NAME" \
        "$prompt" \
        "$MAX_TOKENS" \
        "$PROMPT_MODE" \
        "$LONG_PROMPT_REPEAT" <<'PY'
import json
import sys

model = sys.argv[1]
prompt = sys.argv[2]
max_tokens = int(sys.argv[3])
prompt_mode = sys.argv[4]
long_prompt_repeat = int(sys.argv[5])

if prompt_mode == "long":
    context_lines = [
        "Reference line "
        f"{idx:02d}: alpha beta gamma delta epsilon zeta eta theta "
        "remains unchanged for cache transfer validation."
        for idx in range(long_prompt_repeat)
    ]
    prompt = (
        "\n".join(context_lines)
        + "\nUse the reference lines only as context. "
        + "Complete this sentence directly: "
        + prompt
    )
elif prompt_mode != "short":
    raise SystemExit(f"unknown PROMPT_MODE={prompt_mode!r}")

print(json.dumps({
    "model": model,
    "prompt": prompt,
    "max_tokens": max_tokens,
    "temperature": 0,
    "stream": False,
}))
PY
}

send_request() {
    local prompt=$1
    local label=$2
    local payload_file="$RUN_DIR/${label}_request.json"
    local response_file="$RUN_DIR/${label}_response.json"
    local status_file="$RUN_DIR/${label}_curl_status.txt"

    make_payload "$prompt" > "$payload_file"
    log_info "Sending $label through proxy (${PROMPT_MODE} prompt): $prompt"

    set +e
    local http_code
    http_code=$(curl -sS \
        --max-time "$TIMEOUT_REQUEST" \
        -o "$response_file" \
        -w "%{http_code}" \
        "http://localhost:$PROXY_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        --data-binary "@$payload_file" 2>"$status_file")
    local curl_status=$?
    set -e

    if [ "$curl_status" -ne 0 ]; then
        log_fail "$label failed or timed out after ${TIMEOUT_REQUEST}s"
        cat "$status_file" || true
        return 1
    fi
    if [ "$http_code" != "200" ]; then
        log_fail "$label returned HTTP $http_code"
        cat "$response_file" || true
        return 1
    fi

    python3 - "$response_file" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
text = data["choices"][0]["text"]
if not text:
    raise SystemExit("empty completion text")
print("Completion:", text[:160].replace("\n", "\\n"))
PY
    log_pass "$label completed"
}

run_harness() {
    log_info "Logs: $RUN_DIR"
    if [ "$RUN_WITH_KVCACHED" = "1" ]; then
        log_info "Mode: vLLM LMCacheConnectorV1 with kvcached enabled"
    else
        log_info "Mode: plain vLLM LMCacheConnectorV1 baseline, kvcached disabled"
    fi
    log_info "Prompt mode: $PROMPT_MODE (LONG_PROMPT_REPEAT=$LONG_PROMPT_REPEAT)"

    write_lmcache_configs

    log_info "========== PHASE 1: Start proxy =========="
    start_proxy

    log_info "========== PHASE 2: Start LMCacheConnectorV1 vLLM instances =========="
    start_vllm_instance \
        "prefill" "$PREFILL_GPU" "$PREFILL_PORT" "kv_producer" \
        "$LMCACHE_PREFILL_RPC_PORT" \
        "$CONFIG_DIR/lmcache-prefiller-config.yaml" \
        "$PREFILL_GPU_MEM_UTIL"
    start_vllm_instance \
        "decode" "$DECODE_GPU" "$DECODE_PORT" "kv_consumer" \
        "$LMCACHE_DECODE_RPC_PORT" \
        "$CONFIG_DIR/lmcache-decoder-config.yaml" \
        "$DECODE_GPU_MEM_UTIL"

    wait_for_server "$PREFILL_PORT" "Prefill" "$RUN_DIR/prefill.log"
    wait_for_server "$DECODE_PORT" "Decode" "$RUN_DIR/decode.log"

    log_info "========== PHASE 3: Send requests =========="
    send_request "The capital of France is" "request_1"
    send_request "Two plus two equals" "request_2"

    summarize_logs
    log_pass "LMCacheConnectorV1 debug harness completed"
    log_info "Logs: $RUN_DIR"
}

phase0_setup
run_harness
