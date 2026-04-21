#!/usr/bin/env bash
# =============================================================================
# Test Script: Equivalence harness for kvcached + NixlConnector PD disagg
#
# Purpose:
#   1. Black-box: run the same prompts through NIXL disagg WITHOUT kvcached
#      (Phase A) and WITH kvcached (Phase B), diff the greedy outputs.
#   2. White-box: grep KVCACHED_LAYOUT_DUMP / KVCACHED_BLOCK_SHA lines from
#      Phase B prefill and decode logs and verify tensor metadata +
#      per-block SHAs match across the two GPUs.
#
# Environment: RunPod 2xA100 80GB SXM (or any 2-GPU node with NVLink).
#
# Usage (fresh node):
#   git clone https://github.com/AAbouzeid/kvcached.git
#   cd kvcached && git checkout fix/pd-disagg-nixl-connector
#   chmod +x experiments/09_equivalence.sh
#   ./experiments/09_equivalence.sh
#
# Logs + per-prompt outputs are saved to experiments/logs_eq/.
# =============================================================================

set -euo pipefail

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MAX_MODEL_LEN=1024
GPU_MEM_UTIL=0.8
MAX_TOKENS=20
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=9100
PREFILL_SIDE_CHANNEL=5600
DECODE_SIDE_CHANNEL=5601
LOG_DIR="experiments/logs_eq"
TIMEOUT_STARTUP=180
TIMEOUT_REQUEST=60

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$LOG_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
log_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# Fixed deterministic prompt set. Qwen2.5-1.5B-Instruct under temperature=0
# greedy should produce identical continuations for these across runs with
# the same model/GPU.
PROMPTS=(
    "The capital of France is"
    "List three primary colors:"
    "Two plus two equals"
    "The first president of the United States was"
)

cleanup() {
    log_info "Cleaning up background processes..."
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "VLLM::EngineCore" 2>/dev/null || true
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    sleep 2
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    jobs -p 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

wait_for_server() {
    local port=$1
    local name=$2
    local logfile=$3
    local elapsed=0
    log_info "Waiting for $name on port $port (timeout ${TIMEOUT_STARTUP}s)..."
    while [ $elapsed -lt $TIMEOUT_STARTUP ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            log_pass "$name is ready (${elapsed}s)"
            return 0
        fi
        if [ -f "$logfile" ] && grep -q "Traceback (most recent call last)" "$logfile" 2>/dev/null; then
            log_fail "$name crashed during startup. See $logfile"
            tail -30 "$logfile"
            return 1
        fi
        if [ $((elapsed % 30)) -eq 0 ] && [ $elapsed -gt 0 ]; then
            local last_line
            last_line=$(tail -1 "$logfile" 2>/dev/null | sed 's/^.*] //' | cut -c1-80)
            log_info "  ...${elapsed}s elapsed. Last log: ${last_line:-<empty>}"
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log_fail "$name did not start within ${TIMEOUT_STARTUP}s. See $logfile"
    tail -30 "$logfile"
    return 1
}

start_proxy() {
    local logfile=$1
    python3 -c "
import aiohttp, json, sys
from aiohttp import web

PREFILL='http://localhost:$PREFILL_PORT'
DECODE='http://localhost:$DECODE_PORT'

async def proxy_handler(request):
    body = await request.json()
    prefill_body = {**body, 'max_tokens': 1}
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{PREFILL}/v1/completions', json=prefill_body) as resp:
            await resp.json()
        async with session.post(f'{DECODE}/v1/completions', json=body) as resp:
            result = await resp.json()
    return web.json_response(result)

app = web.Application()
app.router.add_post('/v1/completions', proxy_handler)
web.run_app(app, port=$PROXY_PORT, print=lambda *a: None)
" > "$logfile" 2>&1 &
    sleep 3
}

# Send one prompt through the proxy; write the completion text to a file.
# Fails (non-zero) on empty response, JSON parse error, or HTTP failure.
send_and_capture() {
    local prompt=$1
    local outfile=$2

    local resp
    resp=$(curl -s --max-time $TIMEOUT_REQUEST "http://localhost:$PROXY_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"prompt\": $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$prompt"),
            \"max_tokens\": $MAX_TOKENS,
            \"temperature\": 0
        }")

    echo "$resp" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    t = r['choices'][0]['text']
except Exception as e:
    print('PARSE_ERROR:' + str(e), file=sys.stderr)
    sys.exit(1)
if not t:
    print('EMPTY_RESPONSE', file=sys.stderr)
    sys.exit(1)
with open(sys.argv[1], 'w') as f:
    f.write(t)
" "$outfile" || {
        echo "Raw response: $resp" >&2
        return 1
    }
    return 0
}

# -----------------------------------------------------------------------------
# Phase 0: Environment setup (copied from 06_test_fix1.sh intent; kept minimal)
# -----------------------------------------------------------------------------
phase0_setup() {
    log_info "========== PHASE 0: Environment setup =========="
    pkill -9 -f "vllm" 2>/dev/null || true
    sleep 2

    if ! nvidia-smi > /dev/null 2>&1; then
        log_fail "nvidia-smi not found. Need a GPU node."
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
        log_info "Installing vLLM..."
        pip install -q vllm==0.19.0
    fi
    log_pass "vLLM $(python3 -c 'import vllm; print(vllm.__version__)')"

    if ! python3 -c "from nixl._api import nixl_agent" 2>/dev/null; then
        log_info "Installing NIXL..."
        pip install -q nixl
    fi
    log_pass "NIXL available"

    if ! python3 -c "import aiohttp" 2>/dev/null; then
        pip install -q aiohttp
    fi

    if ! python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$MODEL')" 2>/dev/null; then
        log_info "Downloading model $MODEL..."
        HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download "$MODEL" --quiet
    fi
    log_pass "Model available"

    log_info "Installing kvcached from source..."
    CUDA_HOME=/usr/local/cuda pip install -q "$REPO_DIR" --no-build-isolation
    log_pass "kvcached installed"
}

# -----------------------------------------------------------------------------
# Phase A: baseline without kvcached
#
# Explicitly unset kvcached env so Phase A is a clean baseline even if the
# operator exported any of these in their shell before running the script.
# -----------------------------------------------------------------------------
phaseA_baseline() {
    log_info "========== PHASE A: NIXL disagg WITHOUT kvcached =========="

    ENABLE_KVCACHED=false \
    KVCACHED_AUTOPATCH=false \
    KVCACHED_DUMP_LAYOUT=0 \
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_SIDE_CHANNEL \
    CUDA_VISIBLE_DEVICES=0 \
    vllm serve $MODEL \
        --host 0.0.0.0 --port $PREFILL_PORT \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda"}' \
        > "$LOG_DIR/phaseA_prefill.log" 2>&1 &

    ENABLE_KVCACHED=false \
    KVCACHED_AUTOPATCH=false \
    KVCACHED_DUMP_LAYOUT=0 \
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_SIDE_CHANNEL \
    CUDA_VISIBLE_DEVICES=1 \
    vllm serve $MODEL \
        --host 0.0.0.0 --port $DECODE_PORT \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda"}' \
        > "$LOG_DIR/phaseA_decode.log" 2>&1 &

    wait_for_server $PREFILL_PORT "Phase A prefill" "$LOG_DIR/phaseA_prefill.log" || return 1
    wait_for_server $DECODE_PORT  "Phase A decode"  "$LOG_DIR/phaseA_decode.log"  || return 1

    start_proxy "$LOG_DIR/phaseA_proxy.log"

    local i=0
    for p in "${PROMPTS[@]}"; do
        if send_and_capture "$p" "$LOG_DIR/phaseA_${i}.txt"; then
            log_pass "Phase A prompt $i captured"
        else
            log_fail "Phase A prompt $i failed"
            return 1
        fi
        i=$((i + 1))
    done

    cleanup
    sleep 5
}

# -----------------------------------------------------------------------------
# Phase B: with kvcached + layout dump enabled
# -----------------------------------------------------------------------------
phaseB_kvcached() {
    log_info "========== PHASE B: NIXL disagg WITH kvcached + layout dump =========="

    ENABLE_KVCACHED=true \
    KVCACHED_AUTOPATCH=1 \
    KVCACHED_DUMP_LAYOUT=1 \
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_SIDE_CHANNEL \
    CUDA_VISIBLE_DEVICES=0 \
    vllm serve $MODEL \
        --host 0.0.0.0 --port $PREFILL_PORT \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda"}' \
        > "$LOG_DIR/phaseB_prefill.log" 2>&1 &

    ENABLE_KVCACHED=true \
    KVCACHED_AUTOPATCH=1 \
    KVCACHED_DUMP_LAYOUT=1 \
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_SIDE_CHANNEL \
    CUDA_VISIBLE_DEVICES=1 \
    vllm serve $MODEL \
        --host 0.0.0.0 --port $DECODE_PORT \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda"}' \
        > "$LOG_DIR/phaseB_decode.log" 2>&1 &

    wait_for_server $PREFILL_PORT "Phase B prefill" "$LOG_DIR/phaseB_prefill.log" || return 1
    wait_for_server $DECODE_PORT  "Phase B decode"  "$LOG_DIR/phaseB_decode.log"  || return 1

    start_proxy "$LOG_DIR/phaseB_proxy.log"

    local i=0
    for p in "${PROMPTS[@]}"; do
        if send_and_capture "$p" "$LOG_DIR/phaseB_${i}.txt"; then
            log_pass "Phase B prompt $i captured"
        else
            log_fail "Phase B prompt $i failed"
            return 1
        fi
        i=$((i + 1))
    done

    cleanup
    sleep 5
}

# -----------------------------------------------------------------------------
# Phase C: comparisons
# -----------------------------------------------------------------------------
phaseC_compare() {
    log_info "========== PHASE C: equivalence + layout checks =========="

    local mismatches=0
    local i=0
    for p in "${PROMPTS[@]}"; do
        if diff -q "$LOG_DIR/phaseA_${i}.txt" "$LOG_DIR/phaseB_${i}.txt" > /dev/null; then
            log_pass "Prompt $i: outputs match"
        else
            log_fail "Prompt $i: outputs diverge"
            echo "  A: $(cat "$LOG_DIR/phaseA_${i}.txt")"
            echo "  B: $(cat "$LOG_DIR/phaseB_${i}.txt")"
            mismatches=$((mismatches + 1))
        fi
        i=$((i + 1))
    done

    log_info "--- Layout metadata comparison ---"
    python3 "$SCRIPT_DIR/_compare_layout_dumps.py" \
        "$LOG_DIR/phaseB_prefill.log" \
        "$LOG_DIR/phaseB_decode.log" || mismatches=$((mismatches + 1))

    if [ $mismatches -eq 0 ]; then
        echo ""
        echo "============================================================"
        log_pass "EQUIVALENCE HARNESS PASSED"
        echo "============================================================"
        return 0
    else
        echo ""
        echo "============================================================"
        log_fail "EQUIVALENCE HARNESS FAILED ($mismatches mismatches)"
        echo "============================================================"
        return 1
    fi
}

main() {
    echo "============================================================"
    echo " kvcached + NIXL PD disagg: Equivalence Harness"
    echo " $(date)"
    echo "============================================================"

    phase0_setup || { log_fail "Setup failed"; exit 1; }
    phaseA_baseline || { log_fail "Phase A failed"; exit 1; }
    phaseB_kvcached || { log_fail "Phase B failed"; exit 1; }
    phaseC_compare || exit 1
}

main "$@"
