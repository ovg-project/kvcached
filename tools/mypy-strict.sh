#!/bin/bash

# Run mypy with the same strictness as CI
# Usage: ./tools/mypy-strict.sh [python_version]
# Example: ./tools/mypy-strict.sh 3.9

set -e

PYTHON_VERSION=${1:-local}

if [[ "$PYTHON_VERSION" == "local" ]]; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
fi

EXCLUDE_PATTERN='engine_integration/.*'

echo "Running strict mypy (CI mode) with Python ${PYTHON_VERSION}"

mypy --python-version "${PYTHON_VERSION}" --namespace-packages --exclude "${EXCLUDE_PATTERN}" kvcached
mypy --python-version "${PYTHON_VERSION}" --namespace-packages --exclude "${EXCLUDE_PATTERN}" tests
