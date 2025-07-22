#!/bin/bash

# Some code is borrowed from vLLM. Thanks!

CI=${1:-0}
PYTHON_VERSION=${2:-local}

if [[ "$CI" -eq 1 ]]; then
    set -e
fi

if [[ "$PYTHON_VERSION" == "local" ]]; then
    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
fi

EXCLUDE_PATTERN='engine_integration/.*'

run_mypy() {
    local target=$1; shift || true
    # Default to the current directory (full repo) if no target is specified.
    if [[ -z "$target" ]]; then
        target="."
    fi

    echo "Running mypy on $target"

    if [[ "$CI" -eq 1 ]]; then
        # In CI, run mypy with full strictness.
        mypy --python-version "${PYTHON_VERSION}" --namespace-packages --exclude "${EXCLUDE_PATTERN}" "$@" "$target"
    else
        # Local runs are a bit more lenient and skip heavy import following.
        mypy --follow-imports skip --python-version "${PYTHON_VERSION}" --namespace-packages --exclude "${EXCLUDE_PATTERN}" "$@" "$target"
    fi
}

# run_mypy tests
# run_mypy benchmarks
run_mypy kvcached
# run_mypy kvcached/cli
# run_mypy kvcached/integration
