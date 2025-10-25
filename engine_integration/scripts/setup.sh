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

install_kvcached_from_source() {
    pushd $KVCACHED_DIR
    uv pip install -e . --no-build-isolation --no-cache-dir
    python tools/dev_copy_pth.py
    popd
}


setup_python_venv() {
    uv venv $1 --python=python3.11
    local venv_dir=$1
    source $venv_dir/bin/activate
    uv pip install --upgrade pip
}

setup_vllm_pip() {
    local vllm_ver=${1}

    pushd "$ENGINE_DIR"

    setup_python_venv vllm-pip-venv

    # Install requirements for kvcached first to avoid overwriting vLLM's requirements
    install_requirements
    # vLLM-v0.9.2 requires transformers>=4.51.1 but not too new.
    if [ "$vllm_ver" == "0.9.2" ]; then
        uv pip install transformers==4.51.1
    fi
    uv pip install "vllm==${vllm_ver}"

    install_kvcached_from_source


    deactivate
    popd
}

setup_sglang_pip() {
    local sglang_ver=${1}

    pushd "$ENGINE_DIR"

    setup_python_venv sglang-pip-venv

    # Install requirements for kvcached first to avoid overwriting sglang's requirements
    install_requirements

    if [ "$sglang_ver" == "0.5.3" ]; then
        uv pip install transformers==4.57.0
    fi

    # uv pip install torch==2.7.0
    uv pip install "sglang[all]==${sglang_ver}" --prerelease=allow

    install_kvcached_from_source

    deactivate
    popd
}

setup_vllm_from_source() {
    local vllm_ver=${1}

    pushd "$ENGINE_DIR"
    setup_python_venv vllm-source-venv

    local repo_dir="vllm-v${vllm_ver}"
    git clone -b "v${vllm_ver}" https://github.com/vllm-project/vllm.git "$repo_dir"
    cd "$repo_dir"

    # Install requirements for kvcached first to avoid overwriting vLLM's requirements
    install_requirements
    if [ "$vllm_ver" == "0.9.2" ]; then
        uv pip install transformers==4.51.1
    fi

    # use specific version of precompiled wheel (best effort)
    pip download "vllm==${vllm_ver}" --no-deps -d /tmp || true
    export VLLM_PRECOMPILED_WHEEL_LOCATION="/tmp/vllm-${vllm_ver}-cp38-abi3-manylinux1_x86_64.whl"
    uv pip install --editable .

    install_kvcached_from_source

    deactivate
    popd
}

setup_sglang_from_source() {
    local sglang_ver=${1}

    pushd "$ENGINE_DIR"
    setup_python_venv sglang-source-venv

    local repo_dir="sglang-v${sglang_ver}"
    git clone -b "v${sglang_ver}" https://github.com/sgl-project/sglang.git "$repo_dir"
    cd "$repo_dir"

    # Install requirements for kvcached first to avoid overwriting sglang's requirements
    install_requirements

    if [ "$sglang_ver" == "0.5.3" ]; then
        uv pip install transformers==4.57.0
    fi

    uv pip install -e "python[all]"  --prerelease=allow

    install_kvcached_from_source

    deactivate
    popd
}

# Dispatch helper wrappers that pick defaults when VERSION is not provided
setup_vllm() {
    local _default_ver="0.11.0"
    local _version=${version:-"$_default_ver"}

    if [[ "$method" == "source" ]]; then
        setup_vllm_from_source "$_version"
    else
        setup_vllm_pip "$_version"
    fi
}

setup_sglang() {
    local _version=${version:-"0.5.3"}

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
${BOLD}${CYAN}Usage:${RESET} $0 --engine <vllm|sglang> [--method pip|source] [--version VERSION]

${BOLD}${CYAN}Arguments:${RESET}
  ${BOLD}--engine${RESET}            Target engine to set up (vllm, sglang) [required]
  ${BOLD}--method${RESET}            Engine installation method: pip (default) or source
  ${BOLD}--version${RESET}           Specific engine version to install. Default versions:
        - vllm   : 0.11.0
        - sglang : 0.5.3

${BOLD}${CYAN}Examples:${RESET}
  $0 --engine vllm                                     # vLLM 0.11.0 (pip) + kvcached (source)
  $0 --engine vllm --method source --version 0.11.0    # vLLM 0.11.0 (source) + kvcached (source)
  $0 --engine sglang --method source --version 0.5.3   # sglang 0.5.3 (source) + kvcached (source)
EOF
}

###############################################################################
# CLI argument parsing via GNU getopt
###############################################################################

# Parse options (long opts + -h)
TEMP=$(getopt \
    --options h \
    --longoptions engine:,method:,version:,help \
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

while true; do
    case "$1" in
        --engine)
            engine="$2"; shift 2 ;;
        --method)
            method="$2"; shift 2 ;;
        --version)
            version="$2"; shift 2 ;;
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
    error "Unknown --method '$method' (expected 'pip' or 'source')"
    usage
    exit 1
fi

# Pre-flight summary
echo "[setup.sh] Engine=$engine | Engine-Method=$method | Engine-Version=${version:-default}"

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