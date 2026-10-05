#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
test_build_dir="$(mktemp -d "${TMPDIR:-/tmp}/kvcached-cpp-tests.XXXXXX")"
trap 'rm -rf "${test_build_dir}"' EXIT

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
  -I"${repo_root}/csrc/inc" \
  "${script_dir}/test_math_utils.cpp" \
  -o "${test_build_dir}/test_math_utils"

"${test_build_dir}/test_math_utils"
