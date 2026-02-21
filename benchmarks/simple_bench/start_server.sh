#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/../../engine_integration" && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)

# Default values
DEFAULT_MODEL="meta-llama/Llama-3.2-1B"
DEFAULT_PORT_VLLM=12346
DEFAULT_PORT_SGL=30000
DEFAULT_TP_SIZE=1
DEFAULT_PP_SIZE=1
DEFAULT_DP_SIZE=1
DEFAULT_PCP_SIZE=1
DEFAULT_EP_SIZE=1

# CLI args (set via getopt) plus one positional 'engine'
engine=""      # positional: vllm | sglang
port=""        # if omitted, falls back to engine-specific defaults
model=""       # if omitted, falls back to DEFAULT_MODEL
venv_path=""   # optional
tp_size=$DEFAULT_TP_SIZE
pp_size=$DEFAULT_PP_SIZE
dp_size=$DEFAULT_DP_SIZE
pcp_size=$DEFAULT_PCP_SIZE
ep_size=$DEFAULT_EP_SIZE

usage() {
    cat <<EOF
Usage: $0 <engine> [--venv-path PATH] [--port PORT] [--model MODEL_ID] [--tp TP_SIZE] [--pp PP_SIZE] [--dp DP_SIZE] [--pcp PCP_SIZE] [--ep EP_SIZE]

Positional arguments:
  engine         Target engine (vllm | sglang) [required]
Options:
  --venv-path    Path to an existing virtual environment to activate (optional)
  --port         Port to run the engine on (default: vllm=$DEFAULT_PORT_VLLM, sglang=$DEFAULT_PORT_SGL)
  --model        Model identifier (default: $DEFAULT_MODEL)
  --tp           Tensor parallel size (default: $DEFAULT_TP_SIZE)
  --pp           Pipeline parallel size (default: $DEFAULT_PP_SIZE)
  --dp           Data parallel size (default: $DEFAULT_DP_SIZE, sglang only)
  --pcp          Prefill context parallel size (default: $DEFAULT_PCP_SIZE, vllm only)
  --ep           Expert parallel size (default: $DEFAULT_EP_SIZE, vllm only)
  -h, --help     Show this help and exit

Example:
  $0 vllm --venv-path ../../engine_integration/vllm-pip-venv --model meta-llama/Llama-3.2-1B
EOF
}

# -----------------------------------------------------------------------------
# GNU getopt parsing
# -----------------------------------------------------------------------------
TEMP=$(getopt \
    --options h \
    --longoptions port:,model:,venv-path:,tp:,pp:,dp:,pcp:,ep:,help \
    --name "$0" -- "$@")

if [[ $? -ne 0 ]]; then
    exit 1  # getopt already printed an error
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
        --pp)
            pp_size="$2"; shift 2 ;;
        --dp)
            dp_size="$2"; shift 2 ;;
        --pcp)
            pcp_size="$2"; shift 2 ;;
        --ep)
            ep_size="$2"; shift 2 ;;
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
PP_SIZE="$pp_size"
DP_SIZE="$dp_size"
PCP_SIZE="$pcp_size"
EP_SIZE="$ep_size"

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

source "$SCRIPT_DIR/env_detect.sh"

# Detect if the first visible GPU is an NVIDIA L4.
GPU_NAME=$(command -v nvidia-smi >/dev/null 2>&1 && \
           nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || echo "")
if [[ "$GPU_NAME" == *"L4"* ]]; then
    IS_L4=true
else
    IS_L4=false
fi

if [ "$engine" == "vllm" ]; then
    # Activate virtual environment if provided
    if [[ -n "$VENV_PATH" ]]; then source "$VENV_PATH/bin/activate"; fi
    export VLLM_USE_V1=${VLLM_USE_V1:-1}
    
    # Capture backend preferences from env or defaults, but do NOT export them to avoid warnings.
    # We will pass them as CLI arguments.
    ATTN_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
    ALL2ALL_BACKEND=${VLLM_ALL2ALL_BACKEND:-deepep_low_latency}

    export ENABLE_KVCACHED=true
    export KVCACHED_AUTOPATCH=1
    
    VLLM_L4_ARGS=""
    if [ "$IS_L4" = true ]; then
        VLLM_L4_ARGS="--enforce-eager"
    fi

    # Handle EP configuration
    VLLM_EP_ARGS=""
    if [[ "$EP_SIZE" -gt 1 ]]; then
        VLLM_EP_ARGS="--enable-expert-parallel"
        
        # If user requested EP > 1 but DP is 1, infer DP size from EP and TP.
        # Formula: EP_SIZE = TP_SIZE * DP_SIZE
        if [[ "$DP_SIZE" -eq 1 ]]; then
            if [[ $(( EP_SIZE % TP_SIZE )) -ne 0 ]]; then
                echo "Error: EP_SIZE ($EP_SIZE) must be divisible by TP_SIZE ($TP_SIZE) to infer DP_SIZE."
                exit 1
            fi
            DP_SIZE=$(( EP_SIZE / TP_SIZE ))
            echo "Auto-configuring: Set DP_SIZE=$DP_SIZE to satisfy EP_SIZE=$EP_SIZE (TP=$TP_SIZE)"
        fi

        # Pass All2All backend via CLI
        VLLM_EP_ARGS="$VLLM_EP_ARGS --all2all-backend $ALL2ALL_BACKEND"
    fi

    # Pass Data Parallelism argument if > 1 (supported by vLLM for EP/MoE contexts)
    # Even if EP=1, if user set DP>1 explicitly, we should pass it.
    if [[ "$DP_SIZE" -gt 1 ]]; then
        VLLM_EP_ARGS="$VLLM_EP_ARGS --data-parallel-size $DP_SIZE"
    fi

    vllm serve "$MODEL" \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --port="$VLLM_PORT" \
    --tensor-parallel-size="$TP_SIZE" \
    --pipeline-parallel-size="$PP_SIZE" \
    --prefill-context-parallel-size="$PCP_SIZE" \
    --attention-backend "$ATTN_BACKEND" \
    $VLLM_EP_ARGS \
    $VLLM_L4_ARGS
    if [[ -n "$VENV_PATH" ]]; then deactivate; fi
elif [ "$engine" == "sgl" -o "$engine" == "sglang" ]; then
    # Activate virtual environment if provided
    if [[ -n "$VENV_PATH" ]]; then source "$VENV_PATH/bin/activate"; fi
    export ENABLE_KVCACHED=true
    export KVCACHED_AUTOPATCH=1

    SGL_L4_ARGS=""
    if [ "$IS_L4" = true ]; then
        export TORCHINDUCTOR_DISABLE=1
        export TORCHDYNAMO_DISABLE=1
        SGL_L4_ARGS="--attention-backend torch_native"
    fi
    $PYTHON -m sglang.launch_server --model "$MODEL" \
    --disable-radix-cache \
    --trust-remote-code \
    --port "$SGL_PORT" \
    --tp "$TP_SIZE" \
    --dp "$DP_SIZE" \
    --pipeline-parallel-size "$PP_SIZE" \
    $SGL_L4_ARGS
    if [[ -n "$VENV_PATH" ]]; then deactivate; fi
else
    echo "Invalid engine: $engine"
    exit 1
fi
