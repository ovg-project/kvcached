#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Defaults
DEFAULT_ENGINE_A="vllm"      # vllm | sglang
DEFAULT_ENGINE_B="vllm"      # vllm | sglang
DEFAULT_MODEL_A="meta-llama/Llama-3.2-1B"
DEFAULT_MODEL_B="Qwen/Qwen3-0.6B"
DEFAULT_PORT_A=12346
DEFAULT_PORT_B=12347
DEFAULT_TP_A=1
DEFAULT_TP_B=1

engine_a=""
engine_b=""
model_a=""
model_b=""
port_a=""
port_b=""
tp_a=""
tp_b=""
venv_vllm_path=""   # optional vLLM venv
venv_sgl_path=""    # optional sglang venv

usage() {
    cat <<EOF
Usage: $0 [--engine-a vllm|sglang] [--engine-b vllm|sglang] \
          [--model-a MODEL] [--model-b MODEL] \
          [--port-a PORT] [--port-b PORT] \
          [--tp-a N] [--tp-b N] \
          [--venv-vllm-path PATH] [--venv-sgl-path PATH]

Engines:
  vllm or sglang (choose per model)

Defaults:
  engine-a=$DEFAULT_ENGINE_A, engine-b=$DEFAULT_ENGINE_B
  model-a=$DEFAULT_MODEL_A, model-b=$DEFAULT_MODEL_B
  port-a=$DEFAULT_PORT_A, port-b=$DEFAULT_PORT_B

Examples:
  # Two vLLM instances sharing kvcached
  $0 --engine-a vllm --engine-b vllm \
     --venv-vllm-path ../../engine_integration/vllm-v0.9.2/.venv

  # Cross-engine (isolated by default)
  $0 --engine-a vllm --engine-b sglang \
     --venv-vllm-path ../../engine_integration/vllm-v0.9.2/.venv \
     --venv-sgl-path  ../../engine_integration/sglang-v0.4.9/.venv
EOF
}

# GNU getopt parsing
TEMP=$(getopt \
    --options h \
    --longoptions engine-a:,engine-b:,model-a:,model-b:,port-a:,port-b:,tp-a:,tp-b:,venv-vllm-path:,venv-sgl-path:,help \
    --name "$0" -- "$@")

if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        --engine-a) engine_a="$2"; shift 2;;
        --engine-b) engine_b="$2"; shift 2;;
        --model-a) model_a="$2"; shift 2;;
        --model-b) model_b="$2"; shift 2;;
        --port-a) port_a="$2"; shift 2;;
        --port-b) port_b="$2"; shift 2;;
        --tp-a) tp_a="$2"; shift 2;;
        --tp-b) tp_b="$2"; shift 2;;
        --venv-vllm-path) venv_vllm_path="$2"; shift 2;;
        --venv-sgl-path) venv_sgl_path="$2"; shift 2;;
        --help|-h) usage; exit 0;;
        --) shift; break;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

# Apply defaults
ENGINE_A=${engine_a:-$DEFAULT_ENGINE_A}
ENGINE_B=${engine_b:-$DEFAULT_ENGINE_B}
MODEL_A=${model_a:-$DEFAULT_MODEL_A}
MODEL_B=${model_b:-$DEFAULT_MODEL_B}
PORT_A=${port_a:-$DEFAULT_PORT_A}
PORT_B=${port_b:-$DEFAULT_PORT_B}
TP_A=${tp_a:-$DEFAULT_TP_A}
TP_B=${tp_b:-$DEFAULT_TP_B}

# Validate engine values
validate_engine() {
    local e="$1"
    if [[ "$e" != "vllm" && "$e" != "sglang" ]]; then
        echo "Error: engine must be 'vllm' or 'sglang' (got '$e')" >&2
        exit 1
    fi
}
validate_engine "$ENGINE_A"
validate_engine "$ENGINE_B"

# Validate venvs if provided
if [[ -n "$venv_vllm_path" && ! -f "$venv_vllm_path/bin/activate" ]]; then
    echo "Error: --venv-vllm-path '$venv_vllm_path' is invalid (activate not found)" >&2
    exit 1
fi
if [[ -n "$venv_sgl_path" && ! -f "$venv_sgl_path/bin/activate" ]]; then
    echo "Error: --venv-sgl-path '$venv_sgl_path' is invalid (activate not found)" >&2
    exit 1
fi

PYTHON=${PYTHON:-python3}

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
    local tp="$3"
    local venv="$4"

    if [[ -n "$venv" ]]; then source "$venv/bin/activate"; fi
    export ENABLE_KVCACHED=true
    export KVCACHED_IPC_NAME=VLLM
    export VLLM_USE_V1=1
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    vllm serve "$model" \
      --disable-log-requests \
      --no-enable-prefix-caching \
      --port "$port" \
      --tensor-parallel-size "$tp" \
      --enable-sleep-mode &
    PIDS+=("$!")
    echo "Started vLLM (model=$model) on port $port, pid=${PIDS[-1]}"
    if [[ -n "$venv" ]]; then deactivate; fi
}

run_sgl() {
    local model="$1"
    local port="$2"
    local tp="$3"
    local venv="$4"

    if [[ -n "$venv" ]]; then source "$venv/bin/activate"; fi
    export ENABLE_KVCACHED=true
    export KVCACHED_IPC_NAME=SGLANG
    $PYTHON -m sglang.launch_server --model "$model" \
      --disable-radix-cache \
      --trust-remote-code \
      --port "$port" \
      --tp "$tp" &
    PIDS+=("$!")
    echo "Started sglang (model=$model) on port $port, pid=${PIDS[-1]}"
    if [[ -n "$venv" ]]; then deactivate; fi
}

# Start A
if [[ "$ENGINE_A" == "vllm" ]]; then
    run_vllm "$MODEL_A" "$PORT_A" "$TP_A" "$venv_vllm_path"
else
    run_sgl "$MODEL_A" "$PORT_A" "$TP_A" "$venv_sgl_path"
fi

# Start B
if [[ "$ENGINE_B" == "vllm" ]]; then
    run_vllm "$MODEL_B" "$PORT_B" "$TP_B" "$venv_vllm_path"
else
    run_sgl "$MODEL_B" "$PORT_B" "$TP_B" "$venv_sgl_path"
fi

wait -n || true
wait || true


