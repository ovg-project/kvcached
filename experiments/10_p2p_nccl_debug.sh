#!/usr/bin/env bash
# =============================================================================
# Debug Script: P2P NCCL hang harness for kvcached + vLLM
#
# Purpose:
#   Start a minimal 1-prefill / 1-decode vLLM PD-disagg setup with
#   P2pNcclConnector, route requests through a tiny local proxy, and capture the
#   kvcached P2P debug breadcrumbs from both sides.
#
# Usage:
#   chmod +x experiments/10_p2p_nccl_debug.sh
#   ./experiments/10_p2p_nccl_debug.sh
#
# Useful overrides:
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct ./experiments/10_p2p_nccl_debug.sh
#   TIMEOUT_REQUEST=300 KVCACHED_P2P_WAIT_LOG_INTERVAL_S=5 ./experiments/10_p2p_nccl_debug.sh
#   KEEP_ALIVE_ON_FAIL=1 ./experiments/10_p2p_nccl_debug.sh
#
# Logs are saved to experiments/logs_p2p_debug/<timestamp>/.
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
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
PREFILL_GPU_MEM_UTIL="${PREFILL_GPU_MEM_UTIL:-$GPU_MEM_UTIL}"
DECODE_GPU_MEM_UTIL="${DECODE_GPU_MEM_UTIL:-$GPU_MEM_UTIL}"

PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
PREFILL_PORT="${PREFILL_PORT:-20001}"
DECODE_PORT="${DECODE_PORT:-20002}"
PROXY_PORT="${PROXY_PORT:-10001}"
PROXY_REGISTER_PORT="${PROXY_REGISTER_PORT:-30001}"
PROXY_BIND_IP="${PROXY_BIND_IP:-0.0.0.0}"
PROXY_IP="${PROXY_IP:-127.0.0.1}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-21001}"
DECODE_KV_PORT="${DECODE_KV_PORT:-22001}"
PREFILL_KV_BUFFER_SIZE="${PREFILL_KV_BUFFER_SIZE:-1e1}"
DECODE_KV_BUFFER_SIZE="${DECODE_KV_BUFFER_SIZE:-8e9}"
P2P_SEND_TYPE="${P2P_SEND_TYPE:-PUT_ASYNC}"
P2P_NCCL_NUM_CHANNELS="${P2P_NCCL_NUM_CHANNELS:-8}"
P2P_MEM_POOL_SIZE_GB="${P2P_MEM_POOL_SIZE_GB:-32}"

TIMEOUT_STARTUP="${TIMEOUT_STARTUP:-240}"
TIMEOUT_REQUEST="${TIMEOUT_REQUEST:-90}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
INSTALL_KVCACHED="${INSTALL_KVCACHED:-1}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
CLEAN_STALE="${CLEAN_STALE:-1}"
KEEP_ALIVE_ON_FAIL="${KEEP_ALIVE_ON_FAIL:-0}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
LOG_ROOT="${LOG_ROOT:-experiments/logs_p2p_debug}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$REPO_DIR/$LOG_ROOT/$RUN_ID"
LATEST_LINK="$REPO_DIR/$LOG_ROOT/latest"

mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$LATEST_LINK"

export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export KVCACHED_LOG_LEVEL="${KVCACHED_LOG_LEVEL:-INFO}"
export KVCACHED_P2P_WAIT_LOG_INTERVAL_S="${KVCACHED_P2P_WAIT_LOG_INTERVAL_S:-5}"
export KVCACHED_P2P_TRACE_TENSORS="${KVCACHED_P2P_TRACE_TENSORS:-true}"
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
    kill_port "$PROXY_REGISTER_PORT"
    kill_port "$PREFILL_KV_PORT"
    kill_port "$DECODE_KV_PORT"
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 2
}

summarize_debug_logs() {
    log_info "========== P2P debug summary =========="
    for name in proxy prefill decode; do
        local logfile="$RUN_DIR/$name.log"
        if [ ! -f "$logfile" ]; then
            continue
        fi
        echo "--- $name: $logfile ---"
        grep -E \
            "kvcached p2p debug|P2pNcclEngine init|handle_request|REGISTER|request_id" \
            "$logfile" 2>/dev/null | tail -80 || true
        grep -E \
            "Traceback|ERROR|WARNING|ncclCommInitRank" \
            "$logfile" 2>/dev/null | tail -80 || tail -80 "$logfile" || true
    done

    local prefill_last decode_last
    prefill_last=$(grep "kvcached p2p debug" "$RUN_DIR/prefill.log" 2>/dev/null | tail -1 || true)
    decode_last=$(grep "kvcached p2p debug" "$RUN_DIR/decode.log" 2>/dev/null | tail -1 || true)
    echo "--- last p2p event per side ---"
    echo "prefill: ${prefill_last:-<none>}"
    echo "decode:  ${decode_last:-<none>}"

    if grep -q "recv_tensor still waiting" "$RUN_DIR/decode.log" 2>/dev/null; then
        log_info "Classifier: decode reached recv_tensor and is waiting for a missing tensor_id."
    fi
    if grep -q "send_sync path" "$RUN_DIR/prefill.log" 2>/dev/null \
        && ! grep -q "nccl send begin" "$RUN_DIR/prefill.log" 2>/dev/null; then
        log_info "Classifier: prefill entered send_sync but did not reach NCCL send."
    fi
    if grep -q "listener received" "$RUN_DIR/decode.log" 2>/dev/null; then
        log_info "Classifier: decode listener received at least one ZMQ command."
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
        summarize_debug_logs || true
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
            tail -40 "$logfile"
            return 1
        fi
        if [ $((elapsed % 30)) -eq 0 ] && [ "$elapsed" -gt 0 ]; then
            local last_line
            last_line=$(tail -1 "$logfile" 2>/dev/null | cut -c1-100)
            log_info "  ...${elapsed}s elapsed. Last log: ${last_line:-<empty>}"
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log_fail "$name did not start within ${TIMEOUT_STARTUP}s. See $logfile"
    tail -40 "$logfile"
    return 1
}

wait_for_registration() {
    local elapsed=0
    log_info "Waiting for proxy to register one prefill and one decode..."
    while [ "$elapsed" -lt "$TIMEOUT_STARTUP" ]; do
        if curl -s "http://localhost:$PROXY_PORT/health" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)
sys.exit(0 if data.get("prefill_count", 0) >= 1 and data.get("decode_count", 0) >= 1 else 1)
'; then
            log_pass "Proxy registered prefill and decode (${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log_fail "Proxy did not register both instances. See $RUN_DIR/proxy.log"
    tail -80 "$RUN_DIR/proxy.log" || true
    return 1
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

    if [ "$INSTALL_DEPS" = "1" ]; then
        python3 - <<'PY' || pip install -q aiohttp msgpack pyzmq transformers huggingface_hub
import aiohttp  # noqa: F401
import msgpack  # noqa: F401
import zmq  # noqa: F401
PY
    fi

    if [ "$SKIP_MODEL_DOWNLOAD" != "1" ]; then
        if ! python3 -c \
            "from transformers import AutoConfig; AutoConfig.from_pretrained('$MODEL')" \
            2>/dev/null; then
            log_info "Downloading model $MODEL..."
            HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download "$MODEL" --quiet
        fi
        log_pass "Model available"
    fi

    if [ "$INSTALL_KVCACHED" = "1" ]; then
        log_info "Installing kvcached from this checkout with the normal install path..."
        pip uninstall -y kvcached >/dev/null 2>&1 || true
        CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
            pip install -q "$REPO_DIR" --no-build-isolation
        log_pass "kvcached installed from $REPO_DIR"
    fi

    python3 - <<'PY'
import inspect
from pathlib import Path
import site

import kvcached.integration.vllm.autopatch as autopatch

source = inspect.getsource(autopatch._patch_p2p_nccl_debug)
assert "P2pNcclConnector" in source
assert "P2pNcclEngine" in source
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
PY
    log_pass "P2P debug autopatch source and .pth hook are present"
}

write_proxy() {
    cat > "$RUN_DIR/p2p_debug_proxy.py" <<'PY'
import os
import threading
import time
import uuid

import aiohttp
import msgpack
import zmq
from aiohttp import web

prefill_instances = {}
decode_instances = {}
instances_lock = threading.Lock()
request_count = 0
PING_TTL_SECONDS = 8


def prune_locked(now=None):
    now = now or time.time()
    for instances in (prefill_instances, decode_instances):
        for key, value in list(instances.items()):
            if value["expires_at"] <= now:
                print(f"EXPIRE HTTP:{key} ZMQ:{value['zmq_address']}", flush=True)
                instances.pop(key, None)


def listen_for_register(bind_ip, register_port):
    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{bind_ip}:{register_port}")
    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)
    print(f"REGISTER listener tcp://{bind_ip}:{register_port}", flush=True)

    while True:
        socks = dict(poller.poll(1000))
        if router_socket not in socks:
            continue
        remote_address, message = router_socket.recv_multipart()
        data = msgpack.loads(message)
        role = data.get("type")
        http_address = data.get("http_address")
        zmq_address = data.get("zmq_address")
        if role not in ("P", "D") or not http_address or not zmq_address:
            print(f"REGISTER unexpected remote={remote_address!r} data={data}", flush=True)
            continue

        with instances_lock:
            target = prefill_instances if role == "P" else decode_instances
            is_new = http_address not in target
            target[http_address] = {
                "zmq_address": zmq_address,
                "expires_at": time.time() + PING_TTL_SECONDS,
            }
            prune_locked()
        if is_new:
            print(
                f"REGISTER role={role} HTTP:{http_address} ZMQ:{zmq_address}",
                flush=True,
            )


def choose_pair():
    global request_count
    with instances_lock:
        prune_locked()
        prefill_list = list(prefill_instances.items())
        decode_list = list(decode_instances.items())
        if not prefill_list or not decode_list:
            raise web.HTTPServiceUnavailable(
                text="proxy has not registered both prefill and decode yet"
            )
        prefill_http, prefill_data = prefill_list[request_count % len(prefill_list)]
        decode_http, decode_data = decode_list[request_count % len(decode_list)]
        request_count += 1
        return (
            prefill_http,
            prefill_data["zmq_address"],
            decode_http,
            decode_data["zmq_address"],
            request_count - 1,
        )


async def health(_request):
    with instances_lock:
        prune_locked()
        return web.json_response(
            {
                "prefill_count": len(prefill_instances),
                "decode_count": len(decode_instances),
                "prefill": prefill_instances,
                "decode": decode_instances,
            }
        )


async def forward(session, url, payload, request_id):
    headers = {"X-Request-Id": request_id}
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

    (
        prefill_http,
        prefill_zmq,
        decode_http,
        decode_zmq,
        count,
    ) = choose_pair()
    request_id = (
        f"___prefill_addr_{prefill_zmq}___decode_addr_{decode_zmq}_"
        f"{uuid.uuid4().hex}"
    )
    print(
        f"handle_request count={count} request_id={request_id} "
        f"prefill_http={prefill_http} decode_http={decode_http}",
        flush=True,
    )

    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        prefill_status, _prefill_type, prefill_body = await forward(
            session,
            f"http://{prefill_http}{request.path}",
            prefill,
            request_id,
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

        decode_status, decode_type, decode_body = await forward(
            session,
            f"http://{decode_http}{request.path}",
            original,
            request_id,
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
    bind_ip = os.environ.get("PROXY_BIND_IP", "0.0.0.0")
    register_port = int(os.environ.get("PROXY_REGISTER_PORT", "30001"))
    http_port = int(os.environ.get("PROXY_PORT", "10001"))
    listener = threading.Thread(
        target=listen_for_register,
        args=(bind_ip, register_port),
        daemon=True,
    )
    listener.start()

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/v1/completions", handle_request)
    app.router.add_post("/v1/chat/completions", handle_request)
    web.run_app(app, host=bind_ip, port=http_port)


if __name__ == "__main__":
    main()
PY
}

make_kv_config() {
    local role=$1
    local kv_port=$2
    local http_port=$3
    local buffer_size=$4
    python3 - "$role" "$kv_port" "$http_port" "$buffer_size" <<'PY'
import json
import os
import sys

role, kv_port, http_port, buffer_size = sys.argv[1:5]
config = {
    "kv_connector": "P2pNcclConnector",
    "kv_role": role,
    "kv_buffer_size": buffer_size,
    "kv_port": str(kv_port),
    "kv_connector_extra_config": {
        "proxy_ip": os.environ["PROXY_IP"],
        "proxy_port": os.environ["PROXY_REGISTER_PORT"],
        "http_port": int(http_port),
        "send_type": os.environ["P2P_SEND_TYPE"],
        "nccl_num_channels": os.environ["P2P_NCCL_NUM_CHANNELS"],
        "mem_pool_size_gb": os.environ["P2P_MEM_POOL_SIZE_GB"],
    },
}
print(json.dumps(config, separators=(",", ":")))
PY
}

start_proxy() {
    write_proxy
    log_info "Starting P2P proxy on http://localhost:$PROXY_PORT"
    PROXY_BIND_IP="$PROXY_BIND_IP" \
    PROXY_REGISTER_PORT="$PROXY_REGISTER_PORT" \
    PROXY_PORT="$PROXY_PORT" \
        python3 "$RUN_DIR/p2p_debug_proxy.py" > "$RUN_DIR/proxy.log" 2>&1 &
    CHILD_PIDS+=("$!")
    wait_for_server "$PROXY_PORT" "Proxy" "$RUN_DIR/proxy.log"
}

start_vllm_instance() {
    local name=$1
    local gpu=$2
    local http_port=$3
    local kv_port=$4
    local role=$5
    local kv_buffer_size=$6
    local gpu_mem_util=$7
    local logfile="$RUN_DIR/$name.log"
    local kv_config
    kv_config=$(make_kv_config "$role" "$kv_port" "$http_port" "$kv_buffer_size")

    log_info "Starting $name on GPU $gpu, HTTP $http_port, KV $kv_port"
    echo "$kv_config" > "$RUN_DIR/$name.kv_transfer_config.json"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export ENABLE_KVCACHED=true
        export KVCACHED_AUTOPATCH=1
        export KVCACHED_LOG_LEVEL
        export KVCACHED_P2P_WAIT_LOG_INTERVAL_S
        export KVCACHED_P2P_TRACE_TENSORS
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
            --kv-transfer-config "$kv_config" \
            $EXTRA_VLLM_ARGS
    ) > "$logfile" 2>&1 &
    CHILD_PIDS+=("$!")
}

make_payload() {
    local prompt=$1
    python3 - "$SERVED_MODEL_NAME" "$prompt" "$MAX_TOKENS" <<'PY'
import json
import sys

model, prompt, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
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
    log_info "Sending $label through proxy: $prompt"

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
    log_info "========== PHASE 1: Start proxy =========="
    start_proxy

    log_info "========== PHASE 2: Start P2P vLLM instances =========="
    start_vllm_instance \
        "prefill" "$PREFILL_GPU" "$PREFILL_PORT" "$PREFILL_KV_PORT" \
        "kv_producer" "$PREFILL_KV_BUFFER_SIZE" "$PREFILL_GPU_MEM_UTIL"
    start_vllm_instance \
        "decode" "$DECODE_GPU" "$DECODE_PORT" "$DECODE_KV_PORT" \
        "kv_consumer" "$DECODE_KV_BUFFER_SIZE" "$DECODE_GPU_MEM_UTIL"

    wait_for_server "$PREFILL_PORT" "Prefill" "$RUN_DIR/prefill.log"
    wait_for_server "$DECODE_PORT" "Decode" "$RUN_DIR/decode.log"
    wait_for_registration

    log_info "========== PHASE 3: Send requests =========="
    send_request "The capital of France is" "request_1"
    send_request "Two plus two equals" "request_2"

    summarize_debug_logs
    log_pass "P2P debug harness completed"
    log_info "Logs: $RUN_DIR"
}

export PROXY_IP
export PROXY_REGISTER_PORT
export P2P_SEND_TYPE
export P2P_NCCL_NUM_CHANNELS
export P2P_MEM_POOL_SIZE_GB

phase0_setup
run_harness
