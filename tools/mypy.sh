#!/bin/bash

# Run mypy in either local (changed files) mode or full-project CI mode.
# Usage:
#   tools/mypy.sh <ci_mode> <python_version> [files...]
#     ci_mode        0 = local pre-commit (checks only passed files)
#                    1 = CI/manual stage (checks the entire repo)
#     python_version Informational label (display only)
#     files...       File list forwarded by pre-commit (only in local mode)

set -euo pipefail

MODE="$1"        # 0 or 1
PY_VERSION="$2"  # "local", "3.9", "3.10", ...
shift 2 || true

# Ensure stub packages are installed automatically when missing.
COMMON_ARGS=(--install-types --non-interactive --show-error-codes)

if [[ "$MODE" == "1" ]]; then
  echo "[mypy] Running full type-check for CI on Python ${PY_VERSION} …"
  mypy "${COMMON_ARGS[@]}" --strict .
else
  echo "[mypy] Running type-check on changed files (Python ${PY_VERSION}) …"
  if [[ $# -gt 0 ]]; then
    mypy "${COMMON_ARGS[@]}" "$@"
  else
    mypy "${COMMON_ARGS[@]}" .
  fi
fi 