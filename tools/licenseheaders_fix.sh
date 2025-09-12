#!/usr/bin/env bash
# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##
## SPDX-FileCopyrightText: Copyright contributors to the kvcached project
## SPDX-License-Identifier: Apache-2.0
##
set -euo pipefail

# 1) Insert/update headers using the template and settings (licenseheaders is provided by pre-commit venv)
licenseheaders -t .license-header.txt -s licenseheaders.settings.json -f "$@"

# 2) Remove a single blank line after the first two lines (no extra spacing)
for file in "$@"; do
  [ -f "$file" ] || continue
  awk '
    NR==1 { l1=$0; next }
    NR==2 { l2=$0; next }
    NR==3 {
      if ($0 ~ /^[[:space:]]*$/) { print l1; print l2; next }
      else { print l1; print l2; print; next }
    }
    { print }
    END {
      if (NR==1) print l1;
      else if (NR==2) { print l1; print l2 }
    }
  ' "$file" > "$file.__tmp__" && mv "$file.__tmp__" "$file"
done


