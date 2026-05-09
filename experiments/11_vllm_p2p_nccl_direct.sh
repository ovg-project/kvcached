#!/usr/bin/env bash
# =============================================================================
# Direct upstream vLLM P2P NCCL experiment
#
# Purpose:
#   Validate P2pNcclConnector using the upstream vLLM example proxy and direct
#   vllm serve processes, with kvcached disabled. This intentionally does not
#   use the kvcached debug proxy from 10_p2p_nccl_debug.sh.
#
# Usage:
#   ./experiments/11_vllm_p2p_nccl_direct.sh
#
# Useful overrides:
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct ./experiments/11_vllm_p2p_nccl_direct.sh
#   TIMEOUT_REQUEST=300 ./experiments/11_vllm_p2p_nccl_direct.sh
#   VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 ./experiments/11_vllm_p2p_nccl_direct.sh
#   DISABLE_REQUEST_ID_RANDOMIZATION=1 ./experiments/11_vllm_p2p_nccl_direct.sh
#   KEEP_ALIVE_ON_FAIL=1 ./experiments/11_vllm_p2p_nccl_direct.sh
#
# Logs are saved to experiments/logs_vllm_p2p_direct/<timestamp>/.
# =============================================================================

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
VLLM_VERSION="${VLLM_VERSION:-0.19.0}"
VLLM_REPO_URL="${VLLM_REPO_URL:-https://github.com/vllm-project/vllm.git}"
VLLM_REPO_DIR="${VLLM_REPO_DIR:-$HOME/vllm-upstream}"
VLLM_REF="${VLLM_REF:-main}"
SYNC_VLLM_REPO="${SYNC_VLLM_REPO:-1}"

DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_TOKENS="${MAX_TOKENS:-20}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
PREFILL_HTTP_PORT="${PREFILL_HTTP_PORT:-20003}"
DECODE_HTTP_PORT="${DECODE_HTTP_PORT:-20005}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-21001}"
DECODE_KV_PORT="${DECODE_KV_PORT:-22001}"
PROXY_HTTP_PORT="${PROXY_HTTP_PORT:-10001}"
PROXY_REGISTER_PORT="${PROXY_REGISTER_PORT:-30001}"
PROXY_CONNECT_IP="${PROXY_CONNECT_IP:-0.0.0.0}"
P2P_SEND_TYPE="${P2P_SEND_TYPE:-PUT_ASYNC}"
P2P_NCCL_NUM_CHANNELS="${P2P_NCCL_NUM_CHANNELS:-16}"
PREFILL_KV_BUFFER_SIZE="${PREFILL_KV_BUFFER_SIZE:-1e1}"
DECODE_KV_BUFFER_SIZE="${DECODE_KV_BUFFER_SIZE:-8e9}"
PREFILL_GPU_MEM_UTIL="${PREFILL_GPU_MEM_UTIL:-0.8}"
DECODE_GPU_MEM_UTIL="${DECODE_GPU_MEM_UTIL:-0.7}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
CLEAN_STALE="${CLEAN_STALE:-1}"
KEEP_ALIVE_ON_FAIL="${KEEP_ALIVE_ON_FAIL:-0}"
DISABLE_REQUEST_ID_RANDOMIZATION="${DISABLE_REQUEST_ID_RANDOMIZATION:-0}"
TIMEOUT_STARTUP="${TIMEOUT_STARTUP:-300}"
TIMEOUT_REQUEST="${TIMEOUT_REQUEST:-300}"
LOG_ROOT="${LOG_ROOT:-experiments/logs_vllm_p2p_direct}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$REPO_DIR/$LOG_ROOT/$RUN_ID"
LATEST_LINK="$REPO_DIR/$LOG_ROOT/latest"

mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$LATEST_LINK"

export ENABLE_KVCACHED=false
export KVCACHED_AUTOPATCH=0
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
if [ "$DISABLE_REQUEST_ID_RANDOMIZATION" = "1" ]; then
    export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
fi

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

    log_info "Cleaning stale local processes and ports for direct vLLM run..."
    kill_port "$PROXY_HTTP_PORT"
    kill_port "$PROXY_REGISTER_PORT"
    kill_port "$PREFILL_HTTP_PORT"
    kill_port "$DECODE_HTTP_PORT"
    kill_port "$PREFILL_KV_PORT"
    kill_port "$DECODE_KV_PORT"
    pkill -f "disagg_proxy_p2p_nccl_xpyd.py" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 2
}

summarize_logs() {
    log_info "========== direct vLLM P2P summary =========="
    for name in proxy prefill decode; do
        local logfile="$RUN_DIR/$name.log"
        if [ ! -f "$logfile" ]; then
            continue
        fi
        echo "--- $name: $logfile ---"
        grep -E \
            "Add \\[HTTP|handle_request|request_id|P2pNcclEngine|ncclCommInitRank|P2pNccl|PUT|GET|ERROR|Error|Traceback|WARNING" \
            "$logfile" 2>/dev/null | tail -120 || tail -80 "$logfile" || true
    done
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
    pkill -f "disagg_proxy_p2p_nccl_xpyd.py" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "VLLM::EngineCore" 2>/dev/null || true
}

on_exit() {
    local status=$?
    if [ "$status" != "0" ]; then
        log_fail "Direct vLLM experiment failed with exit status $status"
        summarize_logs || true
    fi
    cleanup "$status"
    exit "$status"
}
trap on_exit EXIT

wait_for_http() {
    local port=$1
    local name=$2
    local logfile=$3
    local path=${4:-/health}
    local elapsed=0

    log_info "Waiting for $name on port $port (timeout ${TIMEOUT_STARTUP}s)..."
    while [ "$elapsed" -lt "$TIMEOUT_STARTUP" ]; do
        if curl -s "http://localhost:$port$path" >/dev/null 2>&1; then
            log_pass "$name is ready (${elapsed}s)"
            return 0
        fi
        if [ -f "$logfile" ] \
            && grep -q "Traceback (most recent call last)" "$logfile" 2>/dev/null; then
            log_fail "$name crashed during startup. See $logfile"
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
    tail -60 "$logfile" || true
    return 1
}

wait_for_proxy_port() {
    local elapsed=0
    local logfile="$RUN_DIR/proxy.log"

    log_info "Waiting for upstream proxy HTTP port $PROXY_HTTP_PORT..."
    while [ "$elapsed" -lt "$TIMEOUT_STARTUP" ]; do
        if python3 - "$PROXY_HTTP_PORT" <<'PY' >/dev/null 2>&1
import socket
import sys

with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
    pass
PY
        then
            log_pass "Upstream proxy HTTP port is open (${elapsed}s)"
            return 0
        fi
        if [ -f "$logfile" ] \
            && grep -q "Traceback (most recent call last)" "$logfile" 2>/dev/null; then
            log_fail "Proxy crashed during startup. See $logfile"
            tail -60 "$logfile"
            return 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log_fail "Proxy did not open port $PROXY_HTTP_PORT. See $logfile"
    tail -60 "$logfile" || true
    return 1
}

wait_for_registration() {
    local elapsed=0
    local logfile="$RUN_DIR/proxy.log"

    log_info "Waiting for upstream proxy to log prefill/decode registration..."
    while [ "$elapsed" -lt "$TIMEOUT_STARTUP" ]; do
        if grep -q "HTTP:.*:${PREFILL_HTTP_PORT}.*ZMQ" "$logfile" 2>/dev/null \
            && grep -q "HTTP:.*:${DECODE_HTTP_PORT}.*ZMQ" "$logfile" 2>/dev/null; then
            log_pass "Proxy registered prefill and decode (${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    log_fail "Proxy did not log both registrations. See $logfile"
    tail -80 "$logfile" || true
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

    if ! command -v git >/dev/null 2>&1; then
        log_fail "git is required to fetch upstream vLLM examples"
        return 1
    fi

    if ! python3 -c "import vllm" 2>/dev/null; then
        if [ "$INSTALL_VLLM" != "1" ]; then
            log_fail "vLLM is not installed and INSTALL_VLLM=0"
            return 1
        fi
        log_info "Installing vLLM==$VLLM_VERSION..."
        pip install -q "vllm==$VLLM_VERSION"
    fi
    log_pass "vLLM $(python3 -c 'import vllm; print(vllm.__version__)')"

    if [ "$INSTALL_DEPS" = "1" ]; then
        python3 - <<'PY' || pip install -q quart aiohttp msgpack pyzmq pandas datasets transformers huggingface_hub
import aiohttp  # noqa: F401
import msgpack  # noqa: F401
import pandas  # noqa: F401
import quart  # noqa: F401
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

    log_info "kvcached disabled: ENABLE_KVCACHED=$ENABLE_KVCACHED KVCACHED_AUTOPATCH=$KVCACHED_AUTOPATCH"
    log_info "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=${VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:-0}"

    if [ "$PROXY_HTTP_PORT" != "10001" ] || [ "$PROXY_REGISTER_PORT" != "30001" ]; then
        log_fail "The upstream proxy hardcodes HTTP 10001 and register 30001."
        log_fail "Leave PROXY_HTTP_PORT=10001 and PROXY_REGISTER_PORT=30001 for this direct run."
        return 1
    fi
}

sync_upstream_vllm() {
    log_info "========== PHASE 1: Fetch upstream vLLM example =========="
    if [ ! -d "$VLLM_REPO_DIR/.git" ]; then
        log_info "Cloning $VLLM_REPO_URL ($VLLM_REF) into $VLLM_REPO_DIR..."
        git clone --depth 1 --branch "$VLLM_REF" "$VLLM_REPO_URL" "$VLLM_REPO_DIR"
    elif [ "$SYNC_VLLM_REPO" = "1" ]; then
        log_info "Updating existing upstream vLLM checkout at $VLLM_REPO_DIR..."
        git -C "$VLLM_REPO_DIR" fetch --depth 1 origin "$VLLM_REF" \
            || git -C "$VLLM_REPO_DIR" fetch origin "$VLLM_REF"
        git -C "$VLLM_REPO_DIR" checkout -q FETCH_HEAD \
            || git -C "$VLLM_REPO_DIR" checkout -q "$VLLM_REF"
    else
        log_info "Using existing upstream vLLM checkout at $VLLM_REPO_DIR"
    fi

    EXAMPLE_DIR="$VLLM_REPO_DIR/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd"
    PROXY_SCRIPT="$EXAMPLE_DIR/disagg_proxy_p2p_nccl_xpyd.py"
    if [ ! -f "$PROXY_SCRIPT" ]; then
        log_fail "Upstream proxy script not found: $PROXY_SCRIPT"
        return 1
    fi

    git -C "$VLLM_REPO_DIR" rev-parse --short HEAD > "$RUN_DIR/vllm_example_commit.txt"
    log_pass "Using upstream vLLM example commit $(cat "$RUN_DIR/vllm_example_commit.txt")"
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
        "proxy_ip": os.environ["PROXY_CONNECT_IP"],
        "proxy_port": os.environ["PROXY_REGISTER_PORT"],
        "http_port": int(http_port),
        "send_type": os.environ["P2P_SEND_TYPE"],
        "nccl_num_channels": os.environ["P2P_NCCL_NUM_CHANNELS"],
    },
}
print(json.dumps(config, separators=(",", ":")))
PY
}

start_upstream_proxy() {
    log_info "========== PHASE 2: Start upstream vLLM proxy =========="
    log_info "Starting upstream proxy on http://localhost:$PROXY_HTTP_PORT"
    (
        cd "$EXAMPLE_DIR"
        exec python3 "$PROXY_SCRIPT"
    ) > "$RUN_DIR/proxy.log" 2>&1 &
    CHILD_PIDS+=("$!")
    wait_for_proxy_port
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

    log_info "Starting direct vLLM $name on GPU $gpu, HTTP $http_port, KV $kv_port"
    echo "$kv_config" > "$RUN_DIR/$name.kv_transfer_config.json"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export ENABLE_KVCACHED
        export KVCACHED_AUTOPATCH
        export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="${VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:-0}"
        exec vllm serve "$MODEL" \
            --enforce-eager \
            --host 0.0.0.0 \
            --port "$http_port" \
            --tensor-parallel-size 1 \
            --seed 1024 \
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
    python3 - "$MODEL" "$prompt" "$MAX_TOKENS" <<'PY'
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
    log_info "Sending $label through upstream proxy: $prompt"

    set +e
    local http_code
    http_code=$(curl -sS \
        --max-time "$TIMEOUT_REQUEST" \
        -o "$response_file" \
        -w "%{http_code}" \
        "http://localhost:$PROXY_HTTP_PORT/v1/completions" \
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

run_experiment() {
    log_info "Logs: $RUN_DIR"
    log_info "Mode: direct upstream vLLM P2P NCCL, kvcached disabled"
    log_info "Upstream proxy script: $PROXY_SCRIPT"

    start_upstream_proxy

    log_info "========== PHASE 3: Start direct vLLM P2P instances =========="
    start_vllm_instance \
        "prefill" "$PREFILL_GPU" "$PREFILL_HTTP_PORT" "$PREFILL_KV_PORT" \
        "kv_producer" "$PREFILL_KV_BUFFER_SIZE" "$PREFILL_GPU_MEM_UTIL"
    start_vllm_instance \
        "decode" "$DECODE_GPU" "$DECODE_HTTP_PORT" "$DECODE_KV_PORT" \
        "kv_consumer" "$DECODE_KV_BUFFER_SIZE" "$DECODE_GPU_MEM_UTIL"

    wait_for_http "$PREFILL_HTTP_PORT" "Prefill" "$RUN_DIR/prefill.log"
    wait_for_http "$DECODE_HTTP_PORT" "Decode" "$RUN_DIR/decode.log"
    wait_for_registration

    log_info "========== PHASE 4: Send request =========="
    send_request "The capital of France is" "request_1"

    summarize_logs
    log_pass "Direct upstream vLLM P2P NCCL experiment completed"
    log_info "Logs: $RUN_DIR"
}

export PROXY_CONNECT_IP
export PROXY_REGISTER_PORT
export P2P_SEND_TYPE
export P2P_NCCL_NUM_CHANNELS

phase0_setup
sync_upstream_vllm
run_experiment
