#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
test_build_dir="$(mktemp -d "${TMPDIR:-/tmp}/kvcached-cpp-tests.XXXXXX")"
trap 'rm -rf "${test_build_dir}"' EXIT

"${PYTHON:-python3}" "${script_dir}/test_physical_capacity_device.py" "${test_build_dir}"

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
  -I"${repo_root}/csrc/inc" \
  "${script_dir}/test_math_utils.cpp" \
  -o "${test_build_dir}/test_math_utils"

"${test_build_dir}/test_math_utils"

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -pthread -DKVCACHED_BUDGET_TESTING \
  -I"${repo_root}/csrc/inc" \
  "${repo_root}/csrc/gpu_budget.cpp" "${script_dir}/test_gpu_budget.cpp" \
  -o "${test_build_dir}/test_gpu_budget"
"${test_build_dir}/test_gpu_budget"
