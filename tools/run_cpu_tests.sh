#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# Run the dependency-isolated tests classified for hosted CPU CI.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

tests=()
while IFS= read -r test_path; do
  tests+=("${test_path}")
done < <("${PYTHON}" tools/check_test_classification.py --list-category cpu)

"${PYTHON}" -m pytest "${tests[@]}" "$@"
