#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
DIFFUSERS_VENV_NAME="$SCRIPT_DIR/diffusers-venv"

check_and_install_uv() {
    echo "Checking uv..."
    if ! command -v uv &> /dev/null; then
        echo "uv not found, installing..."
        curl -fsSL https://astral.sh/uv/install.sh | sh
    fi
    echo "uv installed"
}

setup_diffusers_venv() {
    echo "Setting up uv venv..."
    pushd $SCRIPT_DIR
    uv venv --python 3.11 --seed $DIFFUSERS_VENV_NAME
    source $DIFFUSERS_VENV_NAME/bin/activate
    echo "uv venv setup complete"

    echo "Setting up diffusion..."
    uv pip install diffusers sentencepiece transformers accelerate protobuf
    echo "diffusion setup complete"

    deactivate
    popd
}

check_and_install_uv
setup_diffusers_venv

echo "Setup complete"
