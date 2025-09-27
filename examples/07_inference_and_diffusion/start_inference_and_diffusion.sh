#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Default values
DEFAULT_LLM_ENGINE="vllm"
DEFAULT_LLM_MODEL="meta-llama/Llama-3.2-1B"
DEFAULT_PORT_VLLM=12346
DEFAULT_PORT_SGL=30000
DEFAULT_LLM_TP_SIZE=1

DEFAULT_DIFF_MODEL="stabilityai/stable-diffusion-3.5-medium"
DEFAULT_DIFF_NUM_INFERENCE_STEPS=20
DEFAULT_DIFF_SAVE_IMAGES=""

# CLI variables
llm_engine=""
llm_model=""
llm_port=""
llm_venv_path=""
llm_tp_size=""

diffusers_venv_path=""
diff_model=""
diff_num_inference_steps=""
diff_save_images=""

usage() {
    cat <<EOF
Usage: $0 [--llm-engine ENGINE] [--llm-model MODEL] [--llm-port PORT] [--llm-venv-path PATH] [--llm-tp TP_SIZE] [--diff-model MODEL] [--diff-num-inference-steps N] [--diff-save-images]

Options:
  # For LLM server
  --llm-engine       LLM engine (vllm | sglang) (default: $DEFAULT_LLM_ENGINE)
  --llm-model        Model identifier (default: $DEFAULT_LLM_MODEL)
  --llm-port         Port for LLM server (default: vllm=$DEFAULT_PORT_VLLM, sglang=$DEFAULT_PORT_SGL)
  --llm-venv-path    Path to virtual environment for LLM engine (optional)
  --llm-tp-size      Tensor parallel size (default: $DEFAULT_LLM_TP_SIZE)
  # For diffusion
  --diff-model                Diffusion model (default: $DEFAULT_DIFF_MODEL)
  --diff-num-inference-steps  Number of diffusion inference steps (default: $DEFAULT_DIFF_NUM_INFERENCE_STEPS)
  --diff-save-images          Save generated diffusion images
  -h, --help     Show this help and exit

This script runs both a diffusion model and an LLM server concurrently,
sharing GPU memory through kvcached.

Example:
  $0 --llm-engine vllm --llm-model meta-llama/Llama-3.2-1B
  $0 --llm-engine sglang --diff-model stabilityai/stable-diffusion-3.5-medium --diff-num-inference-steps 20 --diff-save-images
EOF
}

# GNU getopt parsing
TEMP=$(getopt \
    --options h \
    --longoptions llm-engine:,llm-model:,llm-port:,llm-venv-path:,llm-tp-size:,diffusers-venv-path:,diff-model:,diff-num-inference-steps:,diff-save-images,help \
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
        --diffusers-venv-path)
            diffusers_venv_path="$2"; shift 2 ;;
        --diff-model)
            diff_model="$2"; shift 2 ;;
        --diff-num-inference-steps)
            diff_num_inference_steps="$2"; shift 2 ;;
        --diff-save-images)
            diff_save_images="--save-images"; shift ;;
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

# Apply diffusion defaults
DIFF_MODEL=${diff_model:-$DEFAULT_DIFF_MODEL}
DIFF_NUM_INFERENCE_STEPS=${diff_num_inference_steps:-$DEFAULT_DIFF_NUM_INFERENCE_STEPS}
DIFF_SAVE_IMAGES="$diff_save_images"
DIFFUSERS_VENV_PATH="$diffusers_venv_path"

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

echo "Starting diffusion and LLM with kvcached integration..."
echo "Engine: $LLM_ENGINE"
echo "Model: $LLM_MODEL"
echo "Port: $LLM_PORT"
echo "Tensor Parallel Size: $LLM_TP_SIZE"

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
sleep 10

# Check if LLM server is still running
if ! kill -0 "$LLM_PID" >/dev/null 2>&1; then
    echo "Error: LLM server failed to start"
    exit 1
fi

# Start diffusion process
echo "Starting diffusion process..."
DIFFUSION_ARGS="--model $DIFF_MODEL --num-inference-steps $DIFF_NUM_INFERENCE_STEPS $DIFF_SAVE_IMAGES"
if [[ -n "$DIFFUSERS_VENV_PATH" ]]; then
    DIFFUSION_ARGS="$DIFFUSION_ARGS --venv-path $DIFFUSERS_VENV_PATH"
fi

$SCRIPT_DIR/start_diffusion.sh $DIFFUSION_ARGS &

DIFFUSION_PID=$!
PIDS+=("$DIFFUSION_PID")
echo "Diffusion process started with PID: $DIFFUSION_PID"

echo "Both processes are running. Press Ctrl+C to stop."
echo "LLM server is available at: http://localhost:$LLM_PORT"
echo "You can test the LLM server with: $SCRIPT_DIR/start_llm_client.sh $LLM_ENGINE --port $LLM_PORT"

# Wait for either process to finish
wait -n
echo "One of the processes has finished. Cleaning up..."
