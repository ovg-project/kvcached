#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/../../engine_integration" && pwd)

# Default values
DEFAULT_MODEL="meta-llama/Llama-3.2-1B"
DEFAULT_PORT_VLLM=12346
DEFAULT_PORT_SGL=30000
DEFAULT_TP_SIZE=1

# CLI args
engine=""      # positional: vllm | sglang
port=""        # if omitted, falls back to engine-specific defaults
model=""       # if omitted, falls back to DEFAULT_MODEL
venv_path=""   # optional
tp_size=$DEFAULT_TP_SIZE

usage() {
    cat <<EOF
Usage: $0 <engine> [--venv-path PATH] [--port PORT] [--model MODEL_ID] [--tp TP_SIZE]

Positional arguments:
  engine         Target engine (vllm | sglang) [required]
Options:
  --venv-path    Path to an existing virtual environment to activate (optional)
  --port         Port to run the engine on (default: vllm=$DEFAULT_PORT_VLLM, sglang=$DEFAULT_PORT_SGL)
  --model        Model identifier (default: $DEFAULT_MODEL)
  --tp           Tensor parallel size (default: $DEFAULT_TP_SIZE)
  -h, --help     Show this help and exit

Example:
  $0 vllm --venv-path ../../engine_integration/vllm-kvcached-venv --model meta-llama/Llama-3.2-1B
  $0 vllm --model meta-llama/Llama-3.2-1B
EOF
}

# GNU getopt parsing
TEMP=$(getopt \
    --options h \
    --longoptions port:,model:,venv-path:,tp:,help \
    --name "$0" -- "$@")

if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        --port)
            port="$2"; shift 2 ;;
        --model)
            model="$2"; shift 2 ;;
        --venv-path)
            venv_path="$2"; shift 2 ;;
        --tp)
            tp_size="$2"; shift 2 ;;
        --help|-h)
            usage; exit 0 ;;
        --)
            shift; break ;;
        *)
            echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# Remaining arguments after option parsing are treated as positional.
if [[ $# -lt 1 ]]; then
    echo "Error: engine (vllm|sglang) positional argument is required" >&2
    usage; exit 1
fi
engine="$1"; shift

# Validate engine positional arg
if [[ "$engine" != "vllm" && "$engine" != "sglang" ]]; then
    echo "Error: engine must be 'vllm' or 'sglang'" >&2
    usage; exit 1
fi

# Apply defaults
MODEL=${model:-$DEFAULT_MODEL}
if [[ -n "$port" ]]; then
    ENGINE_PORT="$port"
else
    if [[ "$engine" == "vllm" ]]; then
        ENGINE_PORT=$DEFAULT_PORT_VLLM
    else
        ENGINE_PORT=$DEFAULT_PORT_SGL
    fi
fi
VENV_PATH="$venv_path"
TP_SIZE="$tp_size"

# Validate VENV_PATH if provided
if [[ -n "$VENV_PATH" ]]; then
    if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
        echo "Error: --venv-path '$VENV_PATH' is invalid (expected '$VENV_PATH/bin/activate' to exist)" >&2
        exit 1
    fi
fi

# Expose port variables expected later in the script
if [[ "$engine" == "vllm" ]]; then
    VLLM_PORT="$ENGINE_PORT"
else
    SGL_PORT="$ENGINE_PORT"
fi

PYTHON=${PYTHON:-python3}

# Detect if the first visible GPU is an NVIDIA L4.
GPU_NAME=$(command -v nvidia-smi >/dev/null 2>&1 && \
           nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || echo "")
if [[ "$GPU_NAME" == *"L4"* ]]; then
    IS_L4=true
else
    IS_L4=false
fi

rm -f $SCRIPT_DIR/$engine.log

if [[ "$engine" == "vllm" ]]; then
    # Activate virtual environment if provided
    if [[ -n "$VENV_PATH" ]]; then source "$VENV_PATH/bin/activate"; fi
    export VLLM_USE_V1=1
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    export ENABLE_KVCACHED=true
    export KVCACHED_IPC_NAME=VLLM

    VLLM_L4_ARGS=""
    if [[ "$IS_L4" = true ]]; then
        VLLM_L4_ARGS="--enforce-eager"
    fi
    vllm serve "$MODEL" \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --port="$VLLM_PORT" \
    --tensor-parallel-size="$TP_SIZE" \
    $VLLM_L4_ARGS 2>&1 | tee $SCRIPT_DIR/$engine.log
    if [[ -n "$VENV_PATH" ]]; then deactivate; fi
elif [[ "$engine" == "sgl" || "$engine" == "sglang" ]]; then
    # Activate virtual environment if provided
    if [[ -n "$VENV_PATH" ]]; then source "$VENV_PATH/bin/activate"; fi
    export ENABLE_KVCACHED=true
    export KVCACHED_IPC_NAME=SGLANG

    SGL_L4_ARGS=""
    if [[ "$IS_L4" = true ]]; then
        export TORCHINDUCTOR_DISABLE=1
        export TORCHDYNAMO_DISABLE=1
        SGL_L4_ARGS="--attention-backend torch_native"
    fi
    $PYTHON -m sglang.launch_server --model "$MODEL" \
    --disable-radix-cache \
    --trust-remote-code \
    --port "$SGL_PORT" \
    --tp "$TP_SIZE" \
    $SGL_L4_ARGS 2>&1 | tee $SCRIPT_DIR/$engine.log
    if [[ -n "$VENV_PATH" ]]; then deactivate; fi
else
    echo "Invalid engine: $engine"
    exit 1
fi
