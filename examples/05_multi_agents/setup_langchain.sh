#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
LANGCHAIN_VENV_NAME="$SCRIPT_DIR/langchain-venv"

check_and_install_uv() {
    echo "Checking uv..."
    if ! command -v uv &> /dev/null; then
        echo "uv not found, installing..."
        curl -fsSL https://astral.sh/uv/install.sh | sh
    fi
    echo "uv installed"
}

setup_langchain_venv() {
    echo "Setting up uv venv for LangChain..."
    pushd $SCRIPT_DIR
    uv venv --python 3.11 --seed $LANGCHAIN_VENV_NAME
    source $LANGCHAIN_VENV_NAME/bin/activate
    echo "uv venv setup complete"

    echo "Installing LangChain and dependencies..."
    uv pip install langchain langchain-openai langchain-community langchain-core
    echo "LangChain installation complete"

    deactivate
    popd
}

check_and_install_uv
setup_langchain_venv

echo "Setup complete"
echo ""
echo "To activate the LangChain environment:"
echo "  source $LANGCHAIN_VENV_NAME/bin/activate"
echo ""
echo "To run the multi-agent system with LangChain:"
echo "  source $LANGCHAIN_VENV_NAME/bin/activate"
echo "  python3 multi_agent_system.py --topic 'your topic here'"
