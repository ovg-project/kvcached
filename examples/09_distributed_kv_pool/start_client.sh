#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/../../engine_integration" && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)

# -----------------------------------------------------------------------------
# Defaults & Globals
# -----------------------------------------------------------------------------
DEFAULT_MODEL="/workspace/models/Qwen3-32B"
DEFAULT_PORT_VLLM=12346
DEFAULT_PORT_SGL=12348

NUM_PROMPTS=1000
REQUEST_RATE=10

# CLI variables
engine=""        # positional
port=""
model=""
venv_path=""

usage() {
    cat <<EOF
Usage: $0 <engine> [--venv-path PATH] [--port PORT] [--model MODEL_ID]

Positional arguments:
  engine         Target engine (vllm | sglang) [required]
Options:
  --venv-path    Path to a virtual environment to activate (optional)
  --port         Server port (default: vllm=$DEFAULT_PORT_VLLM, sglang=$DEFAULT_PORT_SGL)
  --model        Model identifier (default: $DEFAULT_MODEL)
  -h, --help     Show this help and exit

Example:
  $0 vllm --venv-path ../../engine_integration/vllm-pip-venv --model meta-llama/Llama-3.2-1B
EOF
}

# Parse long options via getopt
TEMP=$(getopt \
    --options h \
    --longoptions port:,model:,venv-path:,help \
    --name "$0" -- "$@")

if [[ $? -ne 0 ]]; then exit 1; fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        --port) port="$2"; shift 2;;
        --model) model="$2"; shift 2;;
        --venv-path) venv_path="$2"; shift 2;;
        --help|-h) usage; exit 0;;
        --) shift; break;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

# Positional engine arg
if [[ $# -lt 1 ]]; then echo "Error: engine positional argument required" >&2; usage; exit 1; fi
engine="$1"; shift

# Validate engine
if [[ "$engine" != "vllm" && "$engine" != "sglang" ]]; then
    echo "Error: engine must be 'vllm' or 'sglang'" >&2; usage; exit 1
fi

# Validate venv_path if supplied
if [[ -n "$venv_path" ]]; then
    if [[ ! -f "$venv_path/bin/activate" ]]; then
        echo "Error: --venv-path '$venv_path' is invalid (activate script not found)" >&2; exit 1
    fi
fi

# Apply defaults
MODEL=${model:-$DEFAULT_MODEL}
if [[ -n "$port" ]]; then
    ENGINE_PORT=$port
else
    if [[ "$engine" == "vllm" ]]; then
        ENGINE_PORT=$DEFAULT_PORT_VLLM;
    else
        ENGINE_PORT=$DEFAULT_PORT_SGL;
    fi
fi
if [[ "$engine" == "vllm" ]]; then
    VLLM_PORT=$ENGINE_PORT;
else
    SGL_PORT=$ENGINE_PORT;
fi

PYTHON=${PYTHON:-python3}

source "$SCRIPT_DIR/env_detect.sh"

check_and_download_sharegpt() {
    pushd $SCRIPT_DIR
    if [ ! -f "ShareGPT_V3_unfiltered_cleaned_split.json" ]; then
        wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    fi
    popd
}

if [ "$engine" == "vllm" ]; then
    check_and_download_sharegpt
    if [[ -n "$venv_path" ]]; then source "$venv_path/bin/activate"; fi
    vllm bench serve \
      --model $MODEL \
      --dataset-name sharegpt \
      --dataset-path $SCRIPT_DIR/ShareGPT_V3_unfiltered_cleaned_split.json \
      --request-rate $REQUEST_RATE \
      --num-prompts $NUM_PROMPTS \
      --port $VLLM_PORT
    if [[ -n "$venv_path" ]]; then deactivate; fi
elif [ "$engine" == "sgl" -o "$engine" == "sglang" ]; then
    check_and_download_sharegpt
    if [[ -n "$venv_path" ]]; then source "$venv_path/bin/activate"; fi

    $PYTHON -m sglang.bench_serving --backend sglang-oai \
        --model $MODEL \
        --dataset-name sharegpt \
        --dataset-path $SCRIPT_DIR/ShareGPT_V3_unfiltered_cleaned_split.json \
        --request-rate $REQUEST_RATE \
        --num-prompts $NUM_PROMPTS \
        --port $SGL_PORT
    if [[ -n "$venv_path" ]]; then deactivate; fi
else
    echo "Invalid engine: $engine"
    exit 1
fi
