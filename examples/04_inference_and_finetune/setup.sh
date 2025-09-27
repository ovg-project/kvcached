#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
LLAMA_FACTORY_VENV_NAME="$SCRIPT_DIR/llama-factory-venv"

check_and_install_uv() {
    echo "Checking uv..."
    if ! command -v uv &> /dev/null; then
        echo "uv not found, installing..."
        curl -fsSL https://astral.sh/uv/install.sh | sh
    fi
    echo "uv installed"
}

setup_llama_factory() {
    echo "Setting up uv venv..."
    pushd $SCRIPT_DIR
    uv venv --python 3.11 --seed $LLAMA_FACTORY_VENV_NAME
    source $LLAMA_FACTORY_VENV_NAME/bin/activate
    echo "uv venv setup complete"

    echo "Setting up llama factory..."
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    pushd LLaMA-Factory
    uv pip install -e ".[torch,metrics]" --no-build-isolation
    deactivate
    popd
    popd
    echo "llama-factory setup complete"
}

check_and_install_uv
setup_llama_factory

echo "Setup complete"
