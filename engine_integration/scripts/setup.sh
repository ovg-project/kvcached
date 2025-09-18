#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)

# -----------------------------------------------------------------------------
# Pretty printing helpers (colors, bold) – disable if stdout is not a TTY
# -----------------------------------------------------------------------------
if [[ -t 1 ]]; then
    BOLD=$(tput bold)
    RESET=$(tput sgr0)
    CYAN=$(tput setaf 6)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    RED=$(tput setaf 1)
else
    BOLD=""
    RESET=""
    CYAN=""
    GREEN=""
    YELLOW=""
    RED=""
fi

# Helper to print errors in red
error() {
    echo "${BOLD}${RED}Error:${RESET} $*" >&2
}

check_uv() {
    if ! command -v uv &> /dev/null; then
        error "uv is not installed"
        error "Please install uv first, e.g., 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
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
    if [[ "$kc_method" == "source" ]]; then
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
    # $1: version (default 0.10.1)
    local vllm_ver=${1:-0.10.1}

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
    if [ -f "$ENGINE_DIR/patches/kvcached-vllm-v${vllm_ver}.patch" ]; then
        git apply "$ENGINE_DIR/patches/kvcached-vllm-v${vllm_ver}.patch"
    else
        error "patch for vLLM-v${vllm_ver} not found"
        exit 1
    fi

    install_kvcached_after_engine

    deactivate
    popd
}

setup_sglang_from_source() {
    # $1: version (default 0.4.9)
    local sglang_ver=${1:-0.4.9}

    pushd "$ENGINE_DIR"

    local repo_dir="sglang-v${sglang_ver}"
    git clone -b "v${sglang_ver}" https://github.com/sgl-project/sglang.git "$repo_dir"
    cd "$repo_dir"

    setup_python_venv .venv

    # Install requirements for kvcached first to avoid overwriting sglang's requirements
    install_requirements

    uv pip install -e "python[all]"

    # Apply patch if present
    if [ -f "$ENGINE_DIR/patches/kvcached-sglang-v${sglang_ver}.patch" ]; then
        git apply "$ENGINE_DIR/patches/kvcached-sglang-v${sglang_ver}.patch"
    else
        error "patch for sglang-v${sglang_ver} not found"
        exit 1
    fi

    install_kvcached_after_engine

    deactivate
    popd
}

# Dispatch helper wrappers that pick defaults when VERSION is not provided
setup_vllm() {
    local _default_ver="0.10.1"
    if [[ "$method" == "source" ]]; then
        _default_ver="0.9.2"
    fi
    local _version=${version:-"$_default_ver"}
    # Validate supported versions
    if [[ "$method" == "pip" ]]; then
        if [[ "$_version" != "0.10.1" ]]; then
            error "vLLM pip installation supports only version 0.10.1 (requested $_version)"
            exit 1
        fi
    else  # source
        if [[ "$_version" != "0.9.2" && "$_version" != "0.8.4" ]]; then
            error "vLLM source installation supports only versions 0.9.2 and 0.8.4 (requested $_version)"
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
    local _version=${version:-"0.4.9"}

    # Validate supported versions
    if [[ "$method" == "pip" ]]; then
        if [[ "$_version" != "0.4.9" ]]; then
            error "sglang pip installation supports only version 0.4.9 (requested $_version)"
            exit 1
        fi
    else  # source
        if [[ "$_version" != "0.4.9" && "$_version" != "0.4.6.post2" ]]; then
            error "sglang source installation supports only versions 0.4.9 and 0.4.6.post2 (requested $_version)"
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
${BOLD}${CYAN}Usage:${RESET} $0 --engine <vllm|sglang> [--engine-method pip|source] [--engine-version VERSION] [--kvcached-method source|pip]

${BOLD}${CYAN}Arguments:${RESET}
  ${BOLD}--engine${RESET}            Target engine to set up (vllm, sglang) [required]
  ${BOLD}--engine-method${RESET}     Engine installation method: pip (default) or source
  ${BOLD}--engine-version${RESET}    Specific engine version to install. Supported versions:
        - vllm   : pip -> 0.10.1 | source -> 0.9.2, 0.8.4
        - sglang : pip -> 0.4.9  | source -> 0.4.9, 0.4.6.post2
  ${BOLD}--kvcached-method${RESET}   source (install kvcached from source) or pip (install from PyPI). Default: source

${BOLD}${CYAN}Examples:${RESET}
  $0 --engine vllm                                                   # vLLM 0.10.1 (pip) + source kvcached (default)
  $0 --engine vllm --engine-method source --engine-version 0.9.2     # vLLM 0.9.2 (source) + kvcached (source)
  $0 --engine sglang --engine-method source --engine-version 0.4.9   # sglang 0.4.9 (source) + kvcached (source)
EOF
}

###############################################################################
# CLI argument parsing via GNU getopt
###############################################################################

# Parse options (long opts + -h)
TEMP=$(getopt \
    --options h \
    --longoptions engine:,engine-method:,engine-version:,kvcached-method:,help \
    --name "$0" -- "$@")

if [[ $? -ne 0 ]]; then
    # getopt already printed an error
    exit 1
fi

# Note: the eval/set magic is required to handle quoted values with spaces.
eval set -- "$TEMP"

# Defaults
engine=""
method="pip"             # engine installation method (pip|source)
version=""
# kvcached installation method (source|pip)
kc_method="source"

while true; do
    case "$1" in
        --engine)
            engine="$2"; shift 2 ;;
        --engine-method)
            method="$2"; shift 2 ;;
        --engine-version)
            version="$2"; shift 2 ;;
        --kvcached-method)
            kc_method="$2"; shift 2 ;;
        --help)
            usage; exit 0 ;;
        -h)
            usage; exit 0 ;;
        --)
            shift; break ;;
        *)
            echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# Validate required engine option
if [[ -z "$engine" ]]; then
    error "--engine is required"
    usage
    exit 1
fi

# Validate method
if [[ "$method" != "pip" && "$method" != "source" ]]; then
    error "Unknown --engine-method '$method' (expected 'pip' or 'source')"
    usage
    exit 1
fi

# Validate mode
if [[ "$kc_method" != "source" && "$kc_method" != "pip" ]]; then
    error "Unknown --kvcached-method '$kc_method' (expected 'source' or 'pip')"
    usage
    exit 1
fi

# Pre-flight summary
echo "[setup.sh] Engine=$engine | Engine-Method=$method | Engine-Version=${version:-default} | kvcached-Method=$kc_method"

# Check for uv before proceeding
check_uv

case "$engine" in
    vllm)
        setup_vllm ;;
    sglang)
        setup_sglang ;;
    *)
        error "Unknown engine '$engine' (expected 'vllm' or 'sglang')"
        usage
        exit 1 ;;
esac