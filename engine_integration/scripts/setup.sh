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

    # 9. Copy the autopatch.pth file to the site-packages directory
    if [[ "$method" == "pip" ]]; then
        PYTHON=${PYTHON:-python3}
        $PYTHON "$KVCACHED_DIR/tools/dev_copy_pth.py"
    fi
}

install_kvcached_after_engine() {
    # Install kvcached after installing engines to find the correct torch version
    if [ "$DEV_MODE" = true ]; then
        install_kvcached_editable
    else
        uv pip install kvcached --no-build-isolation --no-cache-dir
    fi
}

setup_python_venv() {
    uv venv $1 --python=python3.11
    local venv_dir=$1
    source $venv_dir/bin/activate
    uv pip install --upgrade pip
}

setup_vllm_pip() {
    # $1: version (default 0.9.2)
    local vllm_ver=${1:-0.9.2}

    pushd "$ENGINE_DIR"

    setup_python_venv vllm-kvcached-venv

    # Install requirements for kvcached first to avoid overwriting vLLM's requirements
    install_requirements
    # vLLM-v0.9.2 requires transformers>=4.51.1 but not too new.
    if [ "$vllm_ver" == "0.9.2" ]; then
        uv pip install transformers==4.51.1
    fi
    uv pip install "vllm==${vllm_ver}"

    install_kvcached_after_engine

    deactivate
    popd
}

setup_sglang_pip() {
    # $1: version (default 0.4.9)
    local sglang_ver=${1:-0.4.9}

    pushd "$ENGINE_DIR"

    setup_python_venv sglang-kvcached-venv

    # Install requirements for kvcached first to avoid overwriting sglang's requirements
    install_requirements

    uv pip install torch==2.7.0
    uv pip install "sglang[all]==${sglang_ver}"

    install_kvcached_after_engine

    deactivate
    popd
}

setup_vllm_from_source() {
    # $1: version (default 0.9.2)
    local vllm_ver=${1:-0.9.2}

    pushd "$ENGINE_DIR"

    local repo_dir="vllm-v${vllm_ver}"
    git clone -b "v${vllm_ver}" https://github.com/vllm-project/vllm.git "$repo_dir"
    cd "$repo_dir"

    setup_python_venv .venv

    # Install requirements for kvcached first to avoid overwriting vLLM's requirements
    install_requirements
    if [ "$vllm_ver" == "0.9.2" ]; then
        uv pip install transformers==4.51.1
    fi

    # use specific version of precompiled wheel (best effort)
    pip download "vllm==${vllm_ver}" --no-deps -d /tmp || true
    export VLLM_PRECOMPILED_WHEEL_LOCATION="/tmp/vllm-${vllm_ver}-cp38-abi3-manylinux1_x86_64.whl"
    uv pip install --editable .

    # Apply patch if present for this version
    if [ -f "$SCRIPT_DIR/kvcached-vllm-v${vllm_ver}.patch" ]; then
        git apply "$SCRIPT_DIR/kvcached-vllm-v${vllm_ver}.patch"
    fi

    install_kvcached_after_engine

    deactivate
    popd
}

setup_sglang_from_source() {
    # $1: version (default 0.4.6.post2)
    local sglang_ver=${1:-0.4.6.post2}

    pushd "$ENGINE_DIR"

    local repo_dir="sglang-v${sglang_ver}"
    git clone -b "v${sglang_ver}" https://github.com/sgl-project/sglang.git "$repo_dir"
    cd "$repo_dir"

    setup_python_venv .venv

    # Install requirements for kvcached first to avoid overwriting sglang's requirements
    install_requirements

    uv pip install -e "python[all]"

    # Apply patch if present
    if [ -f "$SCRIPT_DIR/kvcached-sglang-v${sglang_ver}.patch" ]; then
        git apply "$SCRIPT_DIR/kvcached-sglang-v${sglang_ver}.patch"
    fi

    install_kvcached_after_engine

    deactivate
    popd
}

# Dispatch helper wrappers that pick defaults when VERSION is not provided
setup_vllm() {
    local _version=${version:-"0.9.2"}
    # Validate supported versions
    if [[ "$method" == "pip" ]]; then
        if [[ "$_version" != "0.9.2" ]]; then
            echo "Error: vLLM pip installation supports only version 0.9.2 (requested $_version)" >&2
            exit 1
        fi
    else  # source
        if [[ "$_version" != "0.9.2" && "$_version" != "0.8.4" ]]; then
            echo "Error: vLLM source installation supports only versions 0.9.2 and 0.8.4 (requested $_version)" >&2
            exit 1
        fi
    fi
    if [[ "$method" == "source" ]]; then
        setup_vllm_from_source "$_version"
    else
        setup_vllm_pip "$_version"
    fi
}

setup_sglang() {
    local _default_ver
    if [[ "$method" == "source" ]]; then
        _default_ver="0.4.6.post2"
    else
        _default_ver="0.4.9"
    fi
    local _version=${version:-"${_default_ver}"}

    # Validate supported versions
    if [[ "$method" == "pip" ]]; then
        if [[ "$_version" != "0.4.9" ]]; then
            echo "Error: sglang pip installation supports only version 0.4.9 (requested $_version)" >&2
            exit 1
        fi
    else  # source
        if [[ "$_version" != "0.4.9" && "$_version" != "0.4.6.post2" ]]; then
            echo "Error: sglang source installation supports only versions 0.4.9 and 0.4.6.post2 (requested $_version)" >&2
            exit 1
        fi
    fi

    if [[ "$method" == "source" ]]; then
        setup_sglang_from_source "$_version"
    else
        setup_sglang_pip "$_version"
    fi
}

# -----------------------------------------------------------------------------
# Usage helper
# -----------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 <vllm|sglang> [pip|source] [VERSION] [dev|prod]

Arguments:
  <engine>   Target engine to set up (vllm, sglang)
  [method]   Installation method: pip (default) or source
  [VERSION]  Specific version to install. Supported versions:
               - vllm   : pip -> 0.9.2 | source -> 0.9.2, 0.8.4
               - sglang : pip -> 0.4.9 | source -> 0.4.9, 0.4.6.post2

  [dev|prod] Choose whether to install kvcached from source (dev) or from PyPI (prod). Defaults to dev.

Examples:
  $0 vllm                       # vLLM 0.9.2 (pip) with dev kvcached install
  $0 vllm pip 0.9.2 prod        # vLLM 0.9.2 from PyPI and kvcached from PyPI
  $0 sglang source 0.4.6.post2  # sglang 0.4.6.post2 built from source, dev kvcached
EOF
}

###############################################################################
# CLI argument parsing
# Usage: ./setup.sh <vllm|sglang> [pip|source] [VERSION]
#   <engine>  : The engine to prepare (vllm, sglang)
#   [method]  : Installation method; "pip" (default) or "source"
#   [VERSION] : Engine version (defaults depend on engine/method)
#
# Examples:
#   ./setup.sh vllm                 # Install vLLM 0.9.2 from PyPI (default)
#   ./setup.sh vllm source 0.9.2    # Install vLLM v0.9.2 from source
#   ./setup.sh sglang pip 0.4.9     # Install sglang 0.4.9 from PyPI
###############################################################################

engine=${1:-}
method=${2:-pip}   # pip (default) | source
version=${3:-}
# Fourth optional parameter: dev|prod (default dev)
dev_flag=${4:-dev}

# Help flags
if [[ "$engine" == "-h" || "$engine" == "--help" || "$engine" == "help" ]]; then
    usage
    exit 0
fi

if [[ -z "$engine" ]]; then
    usage
    exit 1
fi

# Ensure dev_flag is valid
if [[ "$dev_flag" != "dev" && "$dev_flag" != "prod" ]]; then
    echo "Error: Unknown kvcached installation method '$dev_flag' (expected 'dev' or 'prod')" >&2
    usage
    exit 1
fi

# Check for uv before proceeding
check_uv

case "$engine" in
    "vllm")
        setup_vllm
        ;;
    "sglang")
        setup_sglang
        ;;
    *)
        echo "Error: Unknown engine '$engine' (expected 'vllm', 'sglang')" >&2
        usage
        exit 1
        ;;
esac