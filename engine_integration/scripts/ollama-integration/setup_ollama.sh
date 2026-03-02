#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
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

setup_ollama() {
    pushd "$ENGINE_DIR"

    # Install prerequisites for Ollama build
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

    # Clone Ollama source code
    if [ ! -d "ollama-v0.11.8" ]; then
        echo "Cloning Ollama source code..."
        git clone -b v0.11.8 https://github.com/ollama/ollama.git ollama-v0.11.8
        cd ollama-v0.11.8
    else
        echo "Ollama source code already exists, updating..."
        cd ollama-v0.11.8
        git pull
    fi

    # Install Python dependencies for kvcached
    uv venv --python=python3.11
    source .venv/bin/activate
    uv pip install --upgrade pip

    # Install requirements for kvcached first
    install_requirements

    # Apply kvcached patch to Ollama (this creates the C bridge files)
    echo "Applying kvcached patch to Ollama..."
    git apply "$SCRIPT_DIR/../kvcached-ollama-v0.11.8.patch"
    echo "Patch applied successfully - C bridge files created"

    # Build kvcached C bridge library (created by the patch)
    echo "Building kvcached C bridge library..."

    gcc -shared -fPIC -o kvcached_bridge/libkvcached_bridge.so kvcached_bridge/kvcached_bridge.c \
        $(python3-config --cflags --ldflags) -lpython3.11 -lpthread

    echo "Built kvcached bridge library successfully"

    # Build Ollama with CMake (following official guide)
    echo "Building Ollama with CMake..."

    # Configure and build the project
    cmake -B build
    cmake --build build

    # Install kvcached
    if [ "$DEV_MODE" = true ]; then
        install_kvcached_editable
    else
        uv pip install kvcached --no-build-isolation --no-cache-dir
    fi

    deactivate

    echo ""
    echo "Ollama + kvcached integration built successfully!"
    echo "Usage (following Ollama developer guide):"
    echo "   cd ollama-v0.11.8"
    echo "   LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \\"
    echo "   PYTHONPATH=/path/to/kvcached:$PYTHONPATH \\"
    echo "   ENABLE_KVCACHED=true \\"
    echo "   go run . serve"
    echo ""
    echo "   # In another terminal:"
    echo "   LD_LIBRARY_PATH=./kvcached_bridge:$LD_LIBRARY_PATH \\"
    echo "   PYTHONPATH=/path/to/kvcached:$PYTHONPATH \\"
    echo "   ENABLE_KVCACHED=true \\"
    echo "   go run . run gemma3"
    echo ""
    echo "Environment variables:"
    echo "   ENABLE_KVCACHED=true/false  # Enable or disable kvcached integration (default: false)"
    popd
}

# Main execution
check_uv
setup_ollama
