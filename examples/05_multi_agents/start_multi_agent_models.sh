#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Defaults for multi-agent system
DEFAULT_RESEARCH_ENGINE="vllm"
DEFAULT_WRITING_ENGINE="vllm"
DEFAULT_RESEARCH_MODEL="meta-llama/Llama-3.2-3B"
DEFAULT_WRITING_MODEL="Qwen/Qwen3-4B"
DEFAULT_RESEARCH_PORT=12346
DEFAULT_WRITING_PORT=12347
DEFAULT_RESEARCH_TP=1
DEFAULT_WRITING_TP=1

research_engine=""
writing_engine=""
research_model=""
writing_model=""
research_port=""
writing_port=""
research_tp=""
writing_tp=""
venv_vllm_path=""   # optional vLLM venv
venv_sgl_path=""    # optional sglang venv

usage() {
    cat <<EOF
Multi-Agent Model Server Startup

Usage: $0 [--research-engine vllm|sglang] [--writing-engine vllm|sglang] \
          [--research-model MODEL] [--writing-model MODEL] \
          [--research-port PORT] [--writing-port PORT] \
          [--research-tp N] [--writing-tp N] \
          [--venv-vllm-path PATH] [--venv-sgl-path PATH]

This script starts two specialized models for the multi-agent system:
- Research Agent: Specializes in analysis and information gathering
- Writing Agent: Specializes in content creation and summarization

Engines:
  vllm or sglang (choose per model)

Defaults:
  research-engine=$DEFAULT_RESEARCH_ENGINE, writing-engine=$DEFAULT_WRITING_ENGINE
  research-model=$DEFAULT_RESEARCH_MODEL
  writing-model=$DEFAULT_WRITING_MODEL
  research-port=$DEFAULT_RESEARCH_PORT, writing-port=$DEFAULT_WRITING_PORT

Examples:
  # Start with default models (recommended for multi-agent demo)
  $0

  # Use custom models optimized for different tasks
  $0 --research-model meta-llama/Llama-3.2-3B --writing-model Qwen/Qwen3-8B

  # Mixed engines with custom venv paths
  $0 --research-engine vllm --writing-engine sglang \
     --venv-vllm-path ../../engine_integration/vllm-v0.9.2/.venv \
     --venv-sgl-path ../../engine_integration/sglang-v0.4.9/.venv

After starting the models, run the multi-agent system with:
  python3 multi_agent_system.py --research-port $DEFAULT_RESEARCH_PORT --writing-port $DEFAULT_WRITING_PORT
EOF
}

# GNU getopt parsing
TEMP=$(getopt \
    --options h \
    --longoptions research-engine:,writing-engine:,research-model:,writing-model:,research-port:,writing-port:,research-tp:,writing-tp:,venv-vllm-path:,venv-sgl-path:,help \
    --name "$0" -- "$@")

if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        --research-engine) research_engine="$2"; shift 2;;
        --writing-engine) writing_engine="$2"; shift 2;;
        --research-model) research_model="$2"; shift 2;;
        --writing-model) writing_model="$2"; shift 2;;
        --research-port) research_port="$2"; shift 2;;
        --writing-port) writing_port="$2"; shift 2;;
        --research-tp) research_tp="$2"; shift 2;;
        --writing-tp) writing_tp="$2"; shift 2;;
        --venv-vllm-path) venv_vllm_path="$2"; shift 2;;
        --venv-sgl-path) venv_sgl_path="$2"; shift 2;;
        --help|-h) usage; exit 0;;
        --) shift; break;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

# Apply defaults
RESEARCH_ENGINE=${research_engine:-$DEFAULT_RESEARCH_ENGINE}
WRITING_ENGINE=${writing_engine:-$DEFAULT_WRITING_ENGINE}
RESEARCH_MODEL=${research_model:-$DEFAULT_RESEARCH_MODEL}
WRITING_MODEL=${writing_model:-$DEFAULT_WRITING_MODEL}
RESEARCH_PORT=${research_port:-$DEFAULT_RESEARCH_PORT}
WRITING_PORT=${writing_port:-$DEFAULT_WRITING_PORT}
RESEARCH_TP=${research_tp:-$DEFAULT_RESEARCH_TP}
WRITING_TP=${writing_tp:-$DEFAULT_WRITING_TP}

# Validate engines
for engine in "$RESEARCH_ENGINE" "$WRITING_ENGINE"; do
    if [[ "$engine" != "vllm" && "$engine" != "sglang" ]]; then
        echo "Error: Invalid engine '$engine'. Must be 'vllm' or 'sglang'"
        exit 1
    fi
done

echo "========================================"
echo "Multi-Agent System Model Startup"
echo "========================================"
echo "Research Agent:"
echo "  Engine: $RESEARCH_ENGINE"
echo "  Model:  $RESEARCH_MODEL"
echo "  Port:   $RESEARCH_PORT"
echo ""
echo "Writing Agent:"
echo "  Engine: $WRITING_ENGINE"
echo "  Model:  $WRITING_MODEL"
echo "  Port:   $WRITING_PORT"
echo "========================================"

PIDS=()
cleanup() {
    echo ""
    echo "Shutting down model servers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Stopping process $pid"
            kill "$pid"
        fi
    done
    echo "Cleanup complete"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Function to run vLLM
run_vllm() {
    local model="$1"
    local port="$2"
    local tp="$3"
    local venv_path="$4"
    local agent_name="$5"

    echo "Starting vLLM server for $agent_name..."

    local PYTHON="python3"
    if [[ -n "$venv_path" ]]; then
        PYTHON="$venv_path/bin/python3"
    fi

    export ENABLE_KVCACHED=true
    export KVCACHED_IPC_NAME=VLLM
    export VLLM_USE_V1=1
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN

    $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$model" \
      --disable-log-requests \
      --disable-log-stats \
      --port "$port" \
      --tensor-parallel-size "$tp" &

    PIDS+=($!)
    echo "$agent_name vLLM server started (model=$model, port=$port, pid=${PIDS[-1]})"
}

# Function to run SGLang
run_sgl() {
    local model="$1"
    local port="$2"
    local tp="$3"
    local venv_path="$4"
    local agent_name="$5"

    echo "Starting SGLang server for $agent_name..."

    local PYTHON="python3"
    if [[ -n "$venv_path" ]]; then
        PYTHON="$venv_path/bin/python3"
    fi

    export ENABLE_KVCACHED=true
    export KVCACHED_IPC_NAME=SGLANG

    $PYTHON -m sglang.launch_server --model "$model" \
      --disable-radix-cache \
      --trust-remote-code \
      --port "$port" \
      --tp-size "$tp" &

    PIDS+=($!)
    echo "$agent_name SGLang server started (model=$model, port=$port, pid=${PIDS[-1]})"
}

# Start Research Model
if [[ "$RESEARCH_ENGINE" == "vllm" ]]; then
    run_vllm "$RESEARCH_MODEL" "$RESEARCH_PORT" "$RESEARCH_TP" "$venv_vllm_path" "Research Agent"
else
    run_sgl "$RESEARCH_MODEL" "$RESEARCH_PORT" "$RESEARCH_TP" "$venv_sgl_path" "Research Agent"
fi

# Start Writing Model
if [[ "$WRITING_ENGINE" == "vllm" ]]; then
    run_vllm "$WRITING_MODEL" "$WRITING_PORT" "$WRITING_TP" "$venv_vllm_path" "Writing Agent"
else
    run_sgl "$WRITING_MODEL" "$WRITING_PORT" "$WRITING_TP" "$venv_sgl_path" "Writing Agent"
fi

wait -n || true
wait || true
