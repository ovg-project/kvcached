#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

# Default values
DEFAULT_VENV_PATH="$SCRIPT_DIR/diffusers-venv"
DEFAULT_DATASET_PATH="datasets/vidprom.txt"
DEFAULT_MODEL="stabilityai/stable-diffusion-3.5-medium"
DEFAULT_NUM_INFERENCE_STEPS=20
DEFAULT_SAVE_IMAGES=""

# CLI variables
venv_path=""
model=""
num_inference_steps=""
save_images=""

usage() {
    cat <<EOF
Usage: $0 [--venv-path PATH] [--model MODEL] [--num-inference-steps N] [--save-images]

Options:
  --venv-path             Path to an existing virtual environment to activate (default: $SCRIPT_DIR/diffusers-venv)
  --model                 Diffusion model (default: $DEFAULT_MODEL)
  --num-inference-steps   Number of diffusion inference steps (default: $DEFAULT_NUM_INFERENCE_STEPS)
  --save-images           Save generated diffusion images
  -h, --help              Show this help and exit

Example:
  $0 --venv-path ./my-diffusers-venv --model stabilityai/stable-diffusion-3.5-medium
  $0 --num-inference-steps 30 --save-images
EOF
}

# GNU getopt parsing
TEMP=$(getopt \
    --options h \
    --longoptions venv-path:,model:,num-inference-steps:,save-images,help \
    --name "$0" -- "$@")

if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        --venv-path)
            venv_path="$2"; shift 2 ;;
        --model)
            model="$2"; shift 2 ;;
        --num-inference-steps)
            num_inference_steps="$2"; shift 2 ;;
        --save-images)
            save_images="--save_images"; shift ;;
        --help|-h)
            usage; exit 0 ;;
        --)
            shift; break ;;
        *)
            echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# Apply defaults
DIFFUSERS_VENV_PATH=${venv_path:-"$DEFAULT_VENV_PATH"}
MODEL=${model:-$DEFAULT_MODEL}
NUM_INFERENCE_STEPS=${num_inference_steps:-$DEFAULT_NUM_INFERENCE_STEPS}
SAVE_IMAGES="$save_images"

# Validate VENV_PATH if provided
if [[ -n "$DIFFUSERS_VENV_PATH" ]]; then
    if [[ ! -f "$DIFFUSERS_VENV_PATH/bin/activate" ]]; then
        echo "Error: --venv-path '$DIFFUSERS_VENV_PATH' is invalid (expected '$DIFFUSERS_VENV_PATH/bin/activate' to exist)" >&2
        exit 1
    fi
fi

# Activate virtual environment if it exists
if [[ -f "$DIFFUSERS_VENV_PATH/bin/activate" ]]; then
    source "$DIFFUSERS_VENV_PATH/bin/activate"
    echo "Activated virtual environment: $DIFFUSERS_VENV_PATH"
fi

python diffusion_serving.py --dataset_path "$DEFAULT_DATASET_PATH" --model_path "$MODEL" --num_inference_steps "$NUM_INFERENCE_STEPS" $SAVE_IMAGES 2>&1 | tee diffusion.log

# Deactivate virtual environment if it was activated
if [[ -f "$DIFFUSERS_VENV_PATH/bin/activate" ]]; then
    deactivate
fi
