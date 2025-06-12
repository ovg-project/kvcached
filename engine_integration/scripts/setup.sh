#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)

check_uv() {
    if ! command -v uv &> /dev/null; then
        echo "Error: uv is not installed"
        echo "Please install uv first, e.g., 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
        exit 1
    fi
}

setup_vllm() {
    pushd "$ENGINE_DIR"

    git clone -b v0.8.4 https://github.com/vllm-project/vllm.git vllm-v0.8.4
    cd vllm-v0.8.4

    uv venv --python=python3.11
    source .venv/bin/activate
    uv pip install --upgrade pip

    VLLM_USE_PRECOMPILED=1 uv pip install --editable .
    git apply "$SCRIPT_DIR/kvcached-vllm-v0.8.4.patch"

    pushd "$KVCACHED_DIR"
    uv pip install -e . --no-build-isolation
    popd

    deactivate
    popd
}

setup_sgl() {
    pushd "$ENGINE_DIR"

    git clone -b v0.4.6.post2 https://github.com/sgl-project/sglang.git sglang-v0.4.6.post2
    cd sglang-v0.4.6.post2

    uv venv --python=python3.11
    source .venv/bin/activate
    uv pip install --upgrade pip

    uv pip install -e "python[all]"
    git apply "$SCRIPT_DIR/kvcached-sglang-v0.4.6.post2.patch"

    pushd "$KVCACHED_DIR"
    uv pip install -e . --no-build-isolation
    popd


    deactivate
    popd
}

# Check for uv before proceeding
check_uv

setup_vllm
setup_sgl
