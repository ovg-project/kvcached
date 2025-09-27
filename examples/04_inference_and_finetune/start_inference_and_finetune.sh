#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

# Default values
DEFAULT_LLM_ENGINE="vllm"
DEFAULT_LLM_MODEL="meta-llama/Llama-3.2-1B"
DEFAULT_PORT_VLLM=12346
DEFAULT_PORT_SGL=30000
DEFAULT_LLM_TP_SIZE=1

DEFAULT_FINETUNE_CONFIG="llama3_lora_sft.yaml"
DEFAULT_FINETUNE_GPUS="0"

# CLI variables
llm_engine=""
llm_model=""
llm_port=""
llm_venv_path=""
llm_tp_size=""

llama_factory_venv_path=""
finetune_config=""
finetune_gpus=""

usage() {
    cat <<EOF
Usage: $0 [--llm-engine ENGINE] [--llm-model MODEL] [--llm-port PORT] [--llm-venv-path PATH] [--llm-tp TP_SIZE] [--llama-factory-venv-path PATH] [--finetune-config CONFIG] [--finetune-gpus GPUS]

Options:
  # For LLM server
  --llm-engine              LLM engine (vllm | sglang) (default: $DEFAULT_LLM_ENGINE)
  --llm-model               Model identifier (default: $DEFAULT_LLM_MODEL)
  --llm-port                Port for LLM server (default: vllm=$DEFAULT_PORT_VLLM, sglang=$DEFAULT_PORT_SGL)
  --llm-venv-path           Path to virtual environment for LLM engine (optional)
  --llm-tp-size             Tensor parallel size (default: $DEFAULT_LLM_TP_SIZE)
  # For finetuning
  --llama-factory-venv-path Path to LLaMA Factory virtual environment (default: ./llama-factory-venv)
  --finetune-config         Finetuning configuration file (default: $DEFAULT_FINETUNE_CONFIG)
  --finetune-gpus           GPU IDs for finetuning (default: $DEFAULT_FINETUNE_GPUS)
  -h, --help                Show this help and exit

This script runs both an LLM server and LLaMA Factory finetuning concurrently,
sharing GPU memory through kvcached.

Example:
  $0 --llm-engine vllm --llm-model meta-llama/Llama-3.2-1B
  $0 --llm-engine sglang --finetune-config llama3_lora_sft.yaml --finetune-gpus "0,1"
EOF
}

# GNU getopt parsing
TEMP=$(getopt \
    --options h \
    --longoptions llm-engine:,llm-model:,llm-port:,llm-venv-path:,llm-tp-size:,llama-factory-venv-path:,finetune-config:,finetune-gpus:,help \
    --name "$0" -- "$@")

if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        --llm-engine)
            llm_engine="$2"; shift 2 ;;
        --llm-model)
            llm_model="$2"; shift 2 ;;
        --llm-port)
            llm_port="$2"; shift 2 ;;
        --llm-venv-path)
            llm_venv_path="$2"; shift 2 ;;
        --llm-tp-size)
            llm_tp_size="$2"; shift 2 ;;
        --llama-factory-venv-path)
            llama_factory_venv_path="$2"; shift 2 ;;
        --finetune-config)
            finetune_config="$2"; shift 2 ;;
        --finetune-gpus)
            finetune_gpus="$2"; shift 2 ;;
        --help|-h)
            usage; exit 0 ;;
        --)
            shift; break ;;
        *)
            echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# Apply defaults
LLM_ENGINE=${llm_engine:-$DEFAULT_LLM_ENGINE}
LLM_MODEL=${llm_model:-$DEFAULT_LLM_MODEL}
LLM_TP_SIZE=${llm_tp_size:-$DEFAULT_LLM_TP_SIZE}

# Apply finetuning defaults
LLAMA_FACTORY_VENV_PATH=${llama_factory_venv_path:-"$SCRIPT_DIR/llama-factory-venv"}
FINETUNE_CONFIG=${finetune_config:-$DEFAULT_FINETUNE_CONFIG}
FINETUNE_GPUS=${finetune_gpus:-$DEFAULT_FINETUNE_GPUS}

if [[ -n "$llm_port" ]]; then
    LLM_PORT="$llm_port"
else
    if [[ "$LLM_ENGINE" == "vllm" ]]; then
        LLM_PORT=$DEFAULT_PORT_VLLM
    else
        LLM_PORT=$DEFAULT_PORT_SGL
    fi
fi

# Validate engine
if [[ "$LLM_ENGINE" != "vllm" && "$LLM_ENGINE" != "sglang" ]]; then
    echo "Error: engine must be 'vllm' or 'sglang'" >&2
    usage; exit 1
fi

# Validate venv_path if provided
if [[ -n "$llm_venv_path" ]]; then
    if [[ ! -f "$llm_venv_path/bin/activate" ]]; then
        echo "Error: --llm-venv-path '$llm_venv_path' is invalid (activate script not found)" >&2
        exit 1
    fi
fi

# Validate LLaMA Factory venv path
if [[ ! -f "$LLAMA_FACTORY_VENV_PATH/bin/activate" ]]; then
    echo "Error: LLaMA Factory venv not found at '$LLAMA_FACTORY_VENV_PATH'" >&2
    echo "Please run setup.sh first to create the virtual environment" >&2
    exit 1
fi

# Validate finetuning config file
if [[ ! -f "$SCRIPT_DIR/$FINETUNE_CONFIG" ]]; then
    echo "Error: Finetuning config file '$SCRIPT_DIR/$FINETUNE_CONFIG' not found" >&2
    exit 1
fi

PIDS=()
cleanup() {
    echo "Cleaning up processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
            echo "Killing process $pid"
            kill "$pid" || true
        fi
    done
    wait
}
trap cleanup EXIT INT TERM

echo "Starting LLM inference and finetuning with kvcached integration..."
echo "LLM Engine: $LLM_ENGINE"
echo "LLM Model: $LLM_MODEL"
echo "LLM Port: $LLM_PORT"
echo "LLM Tensor Parallel Size: $LLM_TP_SIZE"
echo "Finetuning Config: $FINETUNE_CONFIG"
echo "Finetuning GPUs: $FINETUNE_GPUS"

# Start LLM server in background
echo "Starting LLM server..."
LLM_ARGS="$LLM_ENGINE --model $LLM_MODEL --port $LLM_PORT --tp $LLM_TP_SIZE"
if [[ -n "$llm_venv_path" ]]; then
    LLM_ARGS="$LLM_ARGS --venv-path $llm_venv_path"
fi

$SCRIPT_DIR/start_llm_server.sh $LLM_ARGS &
LLM_PID=$!
PIDS+=("$LLM_PID")
echo "LLM server started with PID: $LLM_PID"

# Wait a bit for LLM server to initialize
echo "Waiting for LLM server to initialize..."
if [[ "$LLM_ENGINE" == "vllm" ]]; then
    while ! grep -q "Application startup complete" "$SCRIPT_DIR/vllm.log"; do
        sleep 1
    done
else
    while ! grep -q "The server is fired up and ready to roll" "$SCRIPT_DIR/sglang.log"; do
        sleep 1
    done
fi

# Check if LLM server is still running
if ! kill -0 "$LLM_PID" >/dev/null 2>&1; then
    echo "Error: LLM server failed to start"
    exit 1
fi

# Start finetuning process
echo "Starting finetuning process..."

# Activate LLaMA Factory environment and start finetuning
(
    source "$LLAMA_FACTORY_VENV_PATH/bin/activate"
    export CUDA_VISIBLE_DEVICES=$FINETUNE_GPUS
    export DISABLE_VERSION_CHECK=1
    export ENABLE_KVCACHED=true
    export KVCACHED_IPC_NAME=LLAMAFACTORY

    # Clean up previous runs
    rm -rf ${SCRIPT_DIR}/llama_factory_saves

    cd "$SCRIPT_DIR"
    llamafactory-cli train "$FINETUNE_CONFIG" 2>&1 | tee finetuning.log
    deactivate
) &

FINETUNE_PID=$!
PIDS+=("$FINETUNE_PID")
echo "Finetuning process started with PID: $FINETUNE_PID"

echo "Both processes are running. Press Ctrl+C to stop."
echo "LLM server is available at: http://localhost:$LLM_PORT"
echo "You can test the LLM server with: $SCRIPT_DIR/start_llm_client.sh $LLM_ENGINE --port $LLM_PORT"
echo "Finetuning logs are available in: $SCRIPT_DIR/finetuning.log"

# Wait for either process to finish
wait -n
echo "One of the processes has finished. Cleaning up..."
