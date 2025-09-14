#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)
DEV_MODE=true  # Set to false to use the released kvcached package

check_uv() {
    if ! command -v uv &> /dev/null; then
        echo "Error: uv is not installed"
        echo "Please install uv first, e.g., 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
        exit 1
    fi
}

install_requirements() {
    pushd "$KVCACHED_DIR"
    uv pip install -r requirements.txt
    popd
}

# Build kvcached wheel inside current venv so C++ is
# compiled against this venv's PyTorch version.  Wheel goes to a tmp dir.
build_and_install_kvcached() {
    local src_dir="$KVCACHED_DIR"
    local tmp_dir
    tmp_dir=$(mktemp -d)
    echo "Building kvcached wheel in $tmp_dir against torch $(python -c 'import torch,sys;print(torch.__version__)')"
    # Build wheel using standard pip because `uv pip wheel` is not yet supported
    pip wheel "$src_dir" -w "$tmp_dir" --no-build-isolation --no-cache-dir
    uv pip install "$tmp_dir"/kvcached-*.whl --no-cache-dir
    rm -rf "$tmp_dir"
}


# Hybrid editable install: Create a "proxy" package in site-packages that
# contains the compiled binary for the current venv and points to the
# Python source files in the workspace.
install_kvcached_editable() {
    # 1. Compile & install the wheel. This places a complete, working package
    #    with the correct C++ binary (.so file) into site-packages.
    build_and_install_kvcached

    # 2. Get necessary paths.
    local site_packages
    site_packages=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    local installed_pkg_dir="$site_packages/kvcached"

    # 3. Save the compiled .so file(s) to a temporary location.
    local tmp_so_dir
    tmp_so_dir=$(mktemp -d)
    # The wheel *should* always contain vmm_ops.so, but be tolerant just in case.
    find "$installed_pkg_dir" -name 'vmm_ops*.so' -exec mv {} "$tmp_so_dir/" \; || true

    # 4. Uninstall the wheel's Python files to prevent shadowing.
    uv pip uninstall kvcached

    # 5. Re-create the package directory in site-packages and move the .so file back.
    #    This directory will now only contain the compiled extension.
    mkdir -p "$installed_pkg_dir"
    mv "$tmp_so_dir"/*.so "$installed_pkg_dir/"
    rm -rf "$tmp_so_dir"

    # 6. Create a proxy __init__.py that extends its path to include the source.
    #    This is more robust than a .pth file as it's part of the package itself.
    echo "Creating a proxy __init__.py in $installed_pkg_dir"
    cat > "$installed_pkg_dir/__init__.py" <<EOF
import os
import sys

# Add the source directory to this package's search path.
# This makes the editable source files available for import.
__path__.insert(0, os.path.abspath(os.path.join("$KVCACHED_DIR", "kvcached")))
EOF

    # 7. Ensure the site-packages copy of kvcached takes precedence over the
    #    repository path ("" / CWD).  We do this via a .pth file executed by
    #    the site module at startup; it reorders sys.path so that the
    #    directory that *contains* this .pth is placed at index 0.
    local pth_file="$site_packages/00_kvcached_prepend.pth"
    echo "Creating $pth_file to prepend site-packages on sys.path"
    printf '%s\n' 'import sys,sysconfig; p=sysconfig.get_paths().get("purelib"); sys.path.insert(0,p) if p and p in sys.path else None' > "$pth_file"

    # Manually add CLI entrypoints for kvtop and kvctl
    bin_dir="$(python -c 'import sysconfig; print(sysconfig.get_paths()["scripts"])')"

    cat > "$bin_dir/kvtop" <<EOF
#!/usr/bin/env bash
exec python -m kvcached.cli.kvtop "\$@"
EOF

    cat > "$bin_dir/kvctl" <<EOF
#!/usr/bin/env bash
exec python -m kvcached.cli.kvctl "\$@"
EOF

    chmod +x "$bin_dir/kvtop" "$bin_dir/kvctl"

    # 8. Remove any stray compiled extensions from the source tree itself to
    #    avoid confusion when switching between virtual-envs.
    find "$KVCACHED_DIR/kvcached" -maxdepth 1 -name 'vmm_ops*.so' -exec rm -f {} + || true
}

setup_vllm() {
    pushd "$ENGINE_DIR"

    git clone -b v0.9.2 https://github.com/vllm-project/vllm.git vllm-v0.9.2
    cd vllm-v0.9.2

    uv venv --python=python3.11
    source .venv/bin/activate
    uv pip install --upgrade pip

    # Install requirements for kvcached first to avoid overwriting vLLM's requirements
    install_requirements
    # vLLM-v0.9.2 requires transformers>=4.51.1 but not too new.
    uv pip install transformers==4.51.1

    # use specific version of precompiled wheel
    pip download "vllm==0.9.2" --no-deps -d /tmp
    export VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/vllm-0.9.2-cp38-abi3-manylinux1_x86_64.whl
    uv pip install --editable .
    git apply "$SCRIPT_DIR/kvcached-vllm-v0.9.2.patch"

    # Install kvcached after installing VLLM to find the correct torch version
    if [ "$DEV_MODE" = true ]; then
        install_kvcached_editable
    else
        uv pip install kvcached --no-build-isolation --no-cache-dir
    fi

    deactivate
    popd
}

setup_sglang() {
    pushd "$ENGINE_DIR"

    git clone -b v0.4.9 https://github.com/sgl-project/sglang.git sglang-v0.4.9
    cd sglang-v0.4.9

    uv venv --python=python3.11
    source .venv/bin/activate
    uv pip install --upgrade pip

    # Install requirements for kvcached first to avoid overwriting sglang's requirements
    install_requirements

    uv pip install -e "python[all]"
    git apply "$SCRIPT_DIR/kvcached-sglang-v0.4.9.patch"

    # Install kvcached after install sglang to find the correct torch version
    if [ "$DEV_MODE" = true ]; then
        install_kvcached_editable
    else
        uv pip install kvcached --no-build-isolation --no-cache-dir
    fi


    deactivate
    popd
}

setup_ollama() {
    pushd "$ENGINE_DIR"

    # Install prerequisites for Linux
    echo "Installing Ollama build prerequisites..."

    # Install Go if not present
    if ! command -v go &> /dev/null; then
        echo "Installing Go..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y golang-go
        elif command -v yum &> /dev/null; then
            sudo yum install -y golang
        else
            echo "Please install Go manually: https://golang.org/doc/install"
            exit 1
        fi
    fi

    # Install CMake if not present
    if ! command -v cmake &> /dev/null; then
        echo "Installing CMake..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get install -y cmake
        elif command -v yum &> /dev/null; then
            sudo yum install -y cmake
        else
            echo "Please install CMake manually"
            exit 1
        fi
    fi

    # Optional: Install CUDA SDK for NVIDIA GPU support
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected. Installing CUDA SDK..."
        # Note: CUDA installation is complex and may require manual setup
        # This is a placeholder for CUDA installation
        echo "⚠️  Please install CUDA SDK manually if needed: https://developer.nvidia.com/cuda-downloads"
    fi

    # Clone Ollama source code
    if [ ! -d "ollama-v0.11.8" ]; then
        echo "Cloning Ollama source code..."
        git clone -b v0.11.8 https://github.com/ollama/ollama.git ollama-v0.11.8
    else
        echo "Ollama source code already exists, updating..."
        cd ollama-v0.11.8
        git pull
        cd ..
    fi

    # Apply kvcached patch to Ollama (this creates the C bridge files)
    echo "Applying kvcached patch to Ollama..."
    cd ollama-v0.11.8
    if [ -f "../scripts/kvcached-ollama-v0.11.8.patch" ]; then
        git apply "../scripts/kvcached-ollama-v0.11.8.patch"
        echo "✓ Patch applied successfully - C bridge files created"
    else
        echo "❌ Patch file not found at ../scripts/kvcached-ollama-v0.11.8.patch"
        exit 1
    fi

    # Build kvcached C bridge library (created by the patch)
    echo "Building kvcached C bridge library..."
    # Get Python include and library paths dynamically
    PYTHON_INCLUDES=$(python3-config --includes)
    PYTHON_LDFLAGS=$(python3-config --ldflags)

    gcc -shared -fPIC -o kvcached_bridge/libkvcached_bridge.so kvcached_bridge/kvcached_bridge.c \
        $PYTHON_INCLUDES \
        $PYTHON_LDFLAGS
    echo "✓ Built kvcached bridge library successfully"

    # Build Ollama with CMake (following official guide)
    echo "Building Ollama with CMake..."

    # Configure and build the project
    cmake -B build
    cmake --build build

    cd ..

    # Install Python dependencies for kvcached
    uv venv --python=python3.11
    source .venv/bin/activate
    uv pip install --upgrade pip

    # Install requirements for kvcached first
    install_requirements

    # Install kvcached
    if [ "$DEV_MODE" = true ]; then
        install_kvcached_editable
    else
        uv pip install kvcached --no-build-isolation --no-cache-dir
    fi

    deactivate

    echo ""
    echo "✅ Ollama + kvcached integration built successfully!"
    echo "📖 See README-kvcached.md for usage instructions"
    echo ""
    echo "🚀 Usage (following Ollama developer guide):"
    echo "   cd ollama-v0.11.8"
    echo "   LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \\"
    echo "   PYTHONPATH=/path/to/kvcached:$PYTHONPATH \\"
    echo "   go run . serve"
    echo ""
    echo "   # In another terminal:"
    echo "   LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \\"
    echo "   PYTHONPATH=/path/to/kvcached:$PYTHONPATH \\"
    echo "   go run . run gemma3"
    popd
}

op=${1:-}

if [ -z "$op" ]; then
    echo "Usage: $0 <vllm|sglang|ollama|all>"
    exit 1
fi

# Check for uv before proceeding
check_uv

case "$op" in
    "vllm")
        setup_vllm
        ;;
    "sglang")
        setup_sglang
        ;;
    "ollama")
        setup_ollama
        ;;
    "all")
        setup_vllm
        setup_sglang
        setup_ollama
        ;;
    *)
        echo "Error: Unknown option '$op'"
        echo "Usage: $0 <vllm|sglang|ollama|all>"
        exit 1
        ;;
esac