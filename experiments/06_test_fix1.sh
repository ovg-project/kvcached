#!/usr/bin/env bash
# =============================================================================
# Test Script: Fix 1 — Patch NixlConnector assertion for kvcached PD disagg
#
# Issue:  https://github.com/ovg-project/kvcached/issues/302
# What:   Tests whether patching the NixlConnector assertion (== to >=)
#         allows kvcached to work with PD disaggregation.
#
# Environment: RunPod 2xA100 80GB SXM (or any 2-GPU node with NVLink)
#
# Usage (fresh RunPod — just clone and run):
#   git clone https://github.com/AAbouzeid/kvcached.git
#   cd kvcached && git checkout fix/pd-disagg-nixl-connector
#   chmod +x experiments/06_test_fix1.sh
#   ./experiments/06_test_fix1.sh
#
# The script handles all setup automatically (deps, model download, install).
# Logs are saved to experiments/logs/
# =============================================================================

set -euo pipefail

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MAX_MODEL_LEN=1024
GPU_MEM_UTIL=0.8
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=9100
PREFILL_SIDE_CHANNEL=5600
DECODE_SIDE_CHANNEL=5601
LOG_DIR="experiments/logs"
TIMEOUT_STARTUP=180   # seconds to wait for server to be ready
TIMEOUT_REQUEST=60    # seconds to wait for a request to complete

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
log_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

cleanup() {
    log_info "Cleaning up background processes..."
    # Kill vLLM and its subprocesses (EngineCore, APIServer, etc.)
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
        # Check if process died
        if [ -f "$logfile" ] && grep -q "Traceback (most recent call last)" "$logfile" 2>/dev/null; then
            log_fail "$name crashed during startup. Check $logfile"
            echo "--- Last 30 lines of $logfile ---"
            tail -30 "$logfile"
            return 1
        fi
        # Show progress every 30s so user knows it's not stuck
        if [ $((elapsed % 30)) -eq 0 ] && [ $elapsed -gt 0 ]; then
            local last_line
            last_line=$(tail -1 "$logfile" 2>/dev/null | sed 's/^.*] //' | cut -c1-80)
            log_info "  ...${elapsed}s elapsed. Last log: ${last_line:-<empty>}"
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log_fail "$name did not start within ${TIMEOUT_STARTUP}s. Check $logfile"
    tail -30 "$logfile"
    return 1
}

send_request() {
    local port=$1
    local prompt=$2
    local max_tokens=${3:-30}

    curl -s --max-time $TIMEOUT_REQUEST "http://localhost:$port/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"prompt\": \"$prompt\",
            \"max_tokens\": $max_tokens,
            \"temperature\": 0
        }"
}

check_response() {
    local response=$1
    local test_name=$2

    if echo "$response" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    text = r['choices'][0]['text'].strip()
    if len(text) > 0:
        print(f'Response: {text[:100]}')
        sys.exit(0)
    else:
        print('Empty response')
        sys.exit(1)
except Exception as e:
    print(f'Parse error: {e}')
    print(f'Raw: {sys.stdin.read()[:200] if hasattr(sys.stdin, \"read\") else \"N/A\"}')
    sys.exit(1)
" 2>&1; then
        log_pass "$test_name"
        return 0
    else
        log_fail "$test_name"
        echo "Raw response: $response"
        return 1
    fi
}

# =============================================================================
# PHASE 1: Baseline without kvcached (sanity check)
# =============================================================================
phase1_baseline() {
    log_info "========== PHASE 1: Baseline without kvcached =========="

    # Start prefill instance
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_SIDE_CHANNEL \
    CUDA_VISIBLE_DEVICES=0 \
    vllm serve $MODEL \
        --host 0.0.0.0 --port $PREFILL_PORT \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda"}' \
        > "$LOG_DIR/phase1_prefill.log" 2>&1 &

    # Start decode instance
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_SIDE_CHANNEL \
    CUDA_VISIBLE_DEVICES=1 \
    vllm serve $MODEL \
        --host 0.0.0.0 --port $DECODE_PORT \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda"}' \
        > "$LOG_DIR/phase1_decode.log" 2>&1 &

    wait_for_server $PREFILL_PORT "Prefill" "$LOG_DIR/phase1_prefill.log" || return 1
    wait_for_server $DECODE_PORT "Decode" "$LOG_DIR/phase1_decode.log" || return 1

    # Start proxy
    python3 -c "
import asyncio, aiohttp, json, sys
from aiohttp import web

PREFILL='http://localhost:$PREFILL_PORT'
DECODE='http://localhost:$DECODE_PORT'

async def proxy_handler(request):
    body = await request.json()
    # Send to prefill with max_tokens=1
    prefill_body = {**body, 'max_tokens': 1}
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{PREFILL}/v1/completions', json=prefill_body) as resp:
            await resp.json()
        # Send to decode with original max_tokens
        async with session.post(f'{DECODE}/v1/completions', json=body) as resp:
            result = await resp.json()
    return web.json_response(result)

app = web.Application()
app.router.add_post('/v1/completions', proxy_handler)
web.run_app(app, port=$PROXY_PORT, print=lambda *a: None)
" > "$LOG_DIR/phase1_proxy.log" 2>&1 &

    sleep 3  # Give proxy a moment

    # Test requests
    log_info "Sending test requests via proxy..."
    local resp
    resp=$(send_request $PROXY_PORT "The capital of France is")
    check_response "$resp" "Phase 1: Basic PD disagg baseline" || return 1

    resp=$(send_request $PROXY_PORT "What is 2+2? The answer is")
    check_response "$resp" "Phase 1: Second request baseline" || return 1

    log_pass "Phase 1 complete — baseline PD disagg works"
    cleanup
    sleep 5  # Let GPU memory free up
}

# =============================================================================
# PHASE 2: kvcached + NixlConnector (THE MAIN TEST)
# =============================================================================
phase2_kvcached_nixl() {
    log_info "========== PHASE 2: kvcached + NixlConnector (Fix 1) =========="

    # Start prefill with kvcached
    ENABLE_KVCACHED=true \
    KVCACHED_AUTOPATCH=1 \
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_SIDE_CHANNEL \
    CUDA_VISIBLE_DEVICES=0 \
    vllm serve $MODEL \
        --host 0.0.0.0 --port $PREFILL_PORT \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda"}' \
        > "$LOG_DIR/phase2_prefill.log" 2>&1 &

    # Start decode with kvcached
    ENABLE_KVCACHED=true \
    KVCACHED_AUTOPATCH=1 \
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_SIDE_CHANNEL \
    CUDA_VISIBLE_DEVICES=1 \
    vllm serve $MODEL \
        --host 0.0.0.0 --port $DECODE_PORT \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda"}' \
        > "$LOG_DIR/phase2_decode.log" 2>&1 &

    # Check if kvcached patches were applied (including nixl_connector)
    sleep 10
    log_info "Checking kvcached patch status..."
    if grep -q "nixl_connector" "$LOG_DIR/phase2_prefill.log" 2>/dev/null; then
        log_pass "NixlConnector patch applied on prefill"
    else
        log_info "NixlConnector patch not yet visible in logs (may appear later)"
    fi

    # Check for the old assertion crash
    if grep -q "All kv cache tensors must have the same number of blocks" "$LOG_DIR/phase2_prefill.log" 2>/dev/null; then
        log_fail "ASSERTION STILL FAILING — Fix 1 patch not working"
        tail -30 "$LOG_DIR/phase2_prefill.log"
        cleanup
        return 1
    fi

    # Check for UCX/VMM errors (the big unknown)
    if grep -qi "cuIpcGetMemHandle\|cuPointerSetAttribute.*error\|operation not supported" "$LOG_DIR/phase2_prefill.log" 2>/dev/null; then
        log_fail "UCX/VMM compatibility error detected — VMM memory not supported by transport"
        grep -i "cuIpc\|cuPointer\|operation not supported" "$LOG_DIR/phase2_prefill.log" | head -5
        log_info "Try: UCX_TLS=cuda_copy,tcp (skip cuda_ipc)"
        cleanup
        return 1
    fi

    wait_for_server $PREFILL_PORT "Prefill+kvcached" "$LOG_DIR/phase2_prefill.log" || return 1
    wait_for_server $DECODE_PORT "Decode+kvcached" "$LOG_DIR/phase2_decode.log" || return 1

    # Verify kvcached's num_blocks adjustment message
    if grep -q "num_blocks adjusted" "$LOG_DIR/phase2_prefill.log" 2>/dev/null; then
        log_pass "Fix 1 patch activated — num_blocks adjusted for virtual capacity"
        grep "num_blocks adjusted" "$LOG_DIR/phase2_prefill.log" | head -1
    fi

    # Start proxy
    python3 -c "
import asyncio, aiohttp, json, sys
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
" > "$LOG_DIR/phase2_proxy.log" 2>&1 &

    sleep 3

    # Test requests
    log_info "Sending test requests via proxy (kvcached + NIXL)..."

    local resp
    resp=$(send_request $PROXY_PORT "The capital of France is")
    check_response "$resp" "Phase 2: Basic PD disagg with kvcached" || {
        log_info "--- Prefill log tail ---"
        tail -20 "$LOG_DIR/phase2_prefill.log"
        log_info "--- Decode log tail ---"
        tail -20 "$LOG_DIR/phase2_decode.log"
        cleanup
        return 1
    }

    resp=$(send_request $PROXY_PORT "What is 2+2? The answer is")
    check_response "$resp" "Phase 2: Second request with kvcached" || return 1

    resp=$(send_request $PROXY_PORT "List the first 5 prime numbers:")
    check_response "$resp" "Phase 2: Third request with kvcached" || return 1

    # Multiple rapid requests
    log_info "Sending 5 rapid requests..."
    local pass_count=0
    for i in $(seq 1 5); do
        resp=$(send_request $PROXY_PORT "Count to $i:")
        if echo "$resp" | python3 -c "import sys,json; r=json.load(sys.stdin); sys.exit(0 if r.get('choices') else 1)" 2>/dev/null; then
            pass_count=$((pass_count + 1))
        fi
    done
    if [ $pass_count -eq 5 ]; then
        log_pass "Phase 2: All 5 rapid requests succeeded"
    else
        log_fail "Phase 2: Only $pass_count/5 rapid requests succeeded"
    fi

    log_pass "Phase 2 complete — kvcached + NixlConnector PD disagg WORKS"

    # Dump init timing from logs
    log_info "Init timing info:"
    grep -i "register_kv_caches\|prep_xfer\|register_memory\|num_blocks\|descriptor" \
        "$LOG_DIR/phase2_prefill.log" 2>/dev/null | head -10 || true

    cleanup
    sleep 5
}

# =============================================================================
# PHASE 3: kvcached + NixlConnector with cuda_copy fallback
# (Only runs if Phase 2 fails with UCX/VMM error)
# =============================================================================
phase3_cuda_copy_fallback() {
    log_info "========== PHASE 3: kvcached + cuda_copy transport fallback =========="

    ENABLE_KVCACHED=true \
    KVCACHED_AUTOPATCH=1 \
    UCX_TLS=cuda_copy,tcp \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_SIDE_CHANNEL \
    CUDA_VISIBLE_DEVICES=0 \
    vllm serve $MODEL \
        --host 0.0.0.0 --port $PREFILL_PORT \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda"}' \
        > "$LOG_DIR/phase3_prefill.log" 2>&1 &

    ENABLE_KVCACHED=true \
    KVCACHED_AUTOPATCH=1 \
    UCX_TLS=cuda_copy,tcp \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_SIDE_CHANNEL \
    CUDA_VISIBLE_DEVICES=1 \
    vllm serve $MODEL \
        --host 0.0.0.0 --port $DECODE_PORT \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda"}' \
        > "$LOG_DIR/phase3_decode.log" 2>&1 &

    wait_for_server $PREFILL_PORT "Prefill+kvcached (cuda_copy)" "$LOG_DIR/phase3_prefill.log" || return 1
    wait_for_server $DECODE_PORT "Decode+kvcached (cuda_copy)" "$LOG_DIR/phase3_decode.log" || return 1

    # Quick proxy
    python3 -c "
import asyncio, aiohttp, json
from aiohttp import web
async def h(req):
    b = await req.json()
    async with aiohttp.ClientSession() as s:
        async with s.post('http://localhost:$PREFILL_PORT/v1/completions', json={**b,'max_tokens':1}) as r: await r.json()
        async with s.post('http://localhost:$DECODE_PORT/v1/completions', json=b) as r: result = await r.json()
    return web.json_response(result)
app = web.Application()
app.router.add_post('/v1/completions', h)
web.run_app(app, port=$PROXY_PORT, print=lambda *a: None)
" > "$LOG_DIR/phase3_proxy.log" 2>&1 &

    sleep 3

    local resp
    resp=$(send_request $PROXY_PORT "The capital of France is")
    check_response "$resp" "Phase 3: PD disagg with cuda_copy fallback" || return 1

    log_pass "Phase 3 complete — cuda_copy transport works as fallback"
    cleanup
    sleep 5
}

# =============================================================================
# PHASE 4: Collect diagnostics
# =============================================================================
phase4_diagnostics() {
    log_info "========== PHASE 4: Diagnostics =========="

    echo "GPU info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true

    echo ""
    echo "CUDA version:"
    nvcc --version 2>/dev/null | grep "release" || true

    echo ""
    echo "vLLM version:"
    python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || true

    echo ""
    echo "kvcached version:"
    python3 -c "import kvcached; print(kvcached.__version__)" 2>/dev/null || true

    echo ""
    echo "UCX info:"
    ucx_info -v 2>/dev/null | head -3 || echo "ucx_info not found"

    echo ""
    echo "NIXL available:"
    python3 -c "from nixl._api import nixl_agent; print('yes')" 2>/dev/null || echo "no"

    echo ""
    echo "NVLink topology:"
    nvidia-smi topo -m 2>/dev/null || true

    log_pass "Diagnostics collected"
}

# =============================================================================
# PHASE 0: Auto-setup (install deps, model, kvcached)
# =============================================================================
phase0_setup() {
    log_info "========== PHASE 0: Environment Setup =========="

    # Kill any leftover GPU processes
    log_info "Killing leftover GPU processes..."
    pkill -9 -f "vllm" 2>/dev/null || true
    sleep 2

    # Check GPUs are available
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

    # Install vLLM if not present
    if ! python3 -c "import vllm" 2>/dev/null; then
        log_info "Installing vLLM (takes ~5 min)..."
        pip install -q vllm==0.19.0
    fi
    log_pass "vLLM $(python3 -c 'import vllm; print(vllm.__version__)')"

    # Install NIXL if not present
    if ! python3 -c "from nixl._api import nixl_agent" 2>/dev/null; then
        log_info "Installing NIXL..."
        pip install -q nixl
    fi
    log_pass "NIXL available"

    # Install aiohttp for proxy
    if ! python3 -c "import aiohttp" 2>/dev/null; then
        log_info "Installing aiohttp..."
        pip install -q aiohttp
    fi

    # Download model if not cached
    if ! python3 -c "
from transformers import AutoConfig
AutoConfig.from_pretrained('$MODEL')
" 2>/dev/null; then
        log_info "Downloading model $MODEL..."
        HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download "$MODEL" --quiet
    fi
    log_pass "Model $MODEL available"

    # Install kvcached from source (normal install, not editable)
    log_info "Installing kvcached from source..."
    CUDA_HOME=/usr/local/cuda pip install -q "$REPO_DIR" --no-build-isolation
    log_pass "kvcached $(python3 -c 'import kvcached; print(getattr(kvcached, "__version__", "installed"))')"

    # Verify patches activate
    local patch_output
    patch_output=$(ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 python3 -c "import vllm" 2>&1 | head -1)
    if echo "$patch_output" | grep -q "Applying.*patches"; then
        log_pass "kvcached patches activate on vllm import"
    else
        log_fail "kvcached patches NOT activating. Got: $patch_output"
        return 1
    fi

    log_pass "Phase 0 complete — environment ready"
}

# =============================================================================
# MAIN
# =============================================================================
main() {
    echo "============================================================"
    echo " kvcached + PD Disaggregation: Fix 1 Test Suite"
    echo " $(date)"
    echo "============================================================"
    echo ""

    # Phase 0 — auto-setup
    if ! phase0_setup; then
        log_fail "Setup failed. Check errors above."
        exit 1
    fi
    echo ""

    # Phase 4 — get environment info
    phase4_diagnostics 2>&1 | tee "$LOG_DIR/diagnostics.log"
    echo ""

    # Phase 1 — baseline
    if phase1_baseline; then
        echo ""
    else
        log_fail "Baseline failed — environment issue, not kvcached. Fix before continuing."
        exit 1
    fi

    # Phase 2 — the main test
    if phase2_kvcached_nixl; then
        echo ""
        echo "============================================================"
        log_pass "FIX 1 WORKS. kvcached + NixlConnector PD disagg is functional."
        echo "============================================================"
    else
        echo ""
        log_fail "Phase 2 failed. Checking failure mode..."

        # Check if it was a UCX/VMM issue
        if grep -qi "cuIpcGetMemHandle\|cuPointerSetAttribute.*error\|operation not supported" \
            "$LOG_DIR/phase2_prefill.log" "$LOG_DIR/phase2_decode.log" 2>/dev/null; then
            log_info "UCX/VMM incompatibility detected. Trying cuda_copy fallback (Phase 3)..."
            echo ""
            if phase3_cuda_copy_fallback; then
                echo ""
                echo "============================================================"
                log_pass "FIX 1 WORKS with cuda_copy transport (not cuda_ipc)."
                echo "  Set UCX_TLS=cuda_copy,tcp to use kvcached + NIXL PD disagg."
                echo "============================================================"
            else
                log_fail "cuda_copy fallback also failed. Deeper investigation needed."
                log_info "Check logs in $LOG_DIR/"
            fi
        else
            log_info "Not a UCX/VMM issue. Check logs for the actual error:"
            log_info "  $LOG_DIR/phase2_prefill.log"
            log_info "  $LOG_DIR/phase2_decode.log"
            # Show relevant errors
            echo "--- Errors from prefill log ---"
            grep -i "error\|exception\|assert\|fail\|crash" "$LOG_DIR/phase2_prefill.log" 2>/dev/null | tail -10 || true
            echo "--- Errors from decode log ---"
            grep -i "error\|exception\|assert\|fail\|crash" "$LOG_DIR/phase2_decode.log" 2>/dev/null | tail -10 || true
        fi
    fi

    echo ""
    log_info "All logs saved to $LOG_DIR/"
    echo "  ls -la $LOG_DIR/"
}

main "$@"
