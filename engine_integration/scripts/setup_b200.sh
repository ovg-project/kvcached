#!/bin/bash
set -euo pipefail

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

setup_python_venv() {
    uv venv $1 --python=python3.12
    local venv_dir=$1
    source $venv_dir/bin/activate
    uv pip install --upgrade pip
}

install_vllm_nightly() {
    uv pip install torch==2.8.0 torchvision==0.23.0
    uv pip install -U vllm \
      --torch-backend=auto \
      --extra-index-url https://wheels.vllm.ai/nightly
}

install_sglang_nightly() {
    uv pip install torch==2.8.0 torchvision==0.23.0
    uv pip install "sglang[all]"
}

install_kvcached() {
    pushd $KVCACHED_DIR
    uv pip install -e . --no-build-isolation --no-cache-dir
    python $KVCACHED_DIR/tools/dev_copy_pth.py
    popd
}


engine=${1:-sglang}

check_uv

case $engine in
    vllm)
        setup_python_venv vllm-kvcached-venv
        install_vllm_nightly
        install_kvcached ;;
    sglang)
        setup_python_venv sglang-kvcached-venv
        install_sglang_nightly
        install_kvcached ;;
    *)
        echo "Error: Unknown engine '$engine' (expected 'vllm' or 'sglang')" >&2
        exit 1 ;;
esac
