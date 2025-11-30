#!/bin/bash
set -x

# Defaults
DEFAULT_MODEL="openai/gpt-oss-20b"
DEFAULT_PORT_A=12346
DEFAULT_PORT_B=12347

model_a=""
model_b=""
port_a=""
port_b=""
venv_vllm_path=""

usage() {
    cat <<EOF
Usage: $0 [--model-a MODEL] [--model-b MODEL] \
          [--port-a PORT] [--port-b PORT] \
          [--venv-vllm-path PATH]

Defaults:
  model-a=$DEFAULT_MODEL, model-b=$DEFAULT_MODEL
  port-a=$DEFAULT_PORT_A, port-b=$DEFAULT_PORT_B
  tensor-parallel-size=1 (fixed)

Example:
  $0 --venv-vllm-path ../../engine_integration/vllm-v0.9.2/.venv
EOF
}

# GNU getopt parsing
TEMP=$(getopt \
    --options h \
    --longoptions model-a:,model-b:,port-a:,port-b:,venv-vllm-path:,help \
    --name "$0" -- "$@")

if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        --model-a) model_a="$2"; shift 2;;
        --model-b) model_b="$2"; shift 2;;
        --port-a) port_a="$2"; shift 2;;
        --port-b) port_b="$2"; shift 2;;
        --venv-vllm-path) venv_vllm_path="$2"; shift 2;;
        --help|-h) usage; exit 0;;
        --) shift; break;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

# Apply defaults
MODEL_A=${model_a:-$DEFAULT_MODEL}
MODEL_B=${model_b:-$DEFAULT_MODEL}
PORT_A=${port_a:-$DEFAULT_PORT_A}
PORT_B=${port_b:-$DEFAULT_PORT_B}

# Validate venv if provided
if [[ -n "$venv_vllm_path" && ! -f "$venv_vllm_path/bin/activate" ]]; then
    echo "Error: --venv-vllm-path '$venv_vllm_path' is invalid (activate not found)" >&2
    exit 1
fi

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" || true
        fi
    done
}
trap cleanup EXIT

run_vllm() {
    local model="$1"
    local port="$2"
    local venv="$3"
    local ipc_name="$4"

    if [[ -n "$venv" ]]; then source "$venv/bin/activate"; fi
    export ENABLE_KVCACHED=true
    export KVCACHED_AUTOPATCH=1
    export KVCACHED_IPC_NAME="$ipc_name"
    export VLLM_USE_V1=1
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    vllm serve "$model" \
      --disable-log-requests \
      --no-enable-prefix-caching \
      --port "$port" \
      --tensor-parallel-size 1 \
      --disable-hybrid-kv-cache-manager \
      --enable-sleep-mode &
    PIDS+=("$!")
    echo "Started vLLM (model=$model) on port $port, IPC=$ipc_name, pid=${PIDS[-1]}"
    if [[ -n "$venv" ]]; then deactivate; fi
}

# Start both models with different kvcached segments
run_vllm "$MODEL_A" "$PORT_A" "$venv_vllm_path" "kvcached_instance_a"
run_vllm "$MODEL_B" "$PORT_B" "$venv_vllm_path" "kvcached_instance_b"

wait -n || true
wait || true
