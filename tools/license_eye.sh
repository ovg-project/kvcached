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

# tools/license_eye.sh
set -euo pipefail
VERSION=${LICENSE_EYE_VERSION:-v0.7.0}
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
BIN="$ROOT/tools/license-eye"
if [[ ! -x "$BIN" ]]; then
  VERSION_NO_V="${VERSION#v}"
  url="https://github.com/apache/skywalking-eyes/releases/download/${VERSION}/skywalking-license-eye-${VERSION_NO_V}-bin.tgz"
  mkdir -p "$ROOT/tools"
  # Prefer an already-extracted binary in tools/ if present
  candidate_bin=""
  base_dir="$ROOT/tools/skywalking-license-eye-${VERSION_NO_V}-bin/bin"
  for osdir in linux darwin windows; do
    if [[ -x "$base_dir/$osdir/license-eye" ]]; then
      candidate_bin="$base_dir/$osdir/license-eye"
      break
    elif [[ -x "$base_dir/$osdir/license-eye.exe" ]]; then
      candidate_bin="$base_dir/$osdir/license-eye.exe"
      break
    fi
  done
  if [[ -n "$candidate_bin" ]]; then
    BIN="$candidate_bin"
  else
    # Extract into temporary directory and search for binary
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT
    curl -fSL "$url" | tar -zxf - -C "$tmpdir"
    found_bin="$(find "$tmpdir" -type f \( -name 'license-eye' -o -name 'license-eye-*' -o -name 'license-eye.*' \) -print0 | xargs -0 file | awk -F: '/executable|ELF/ {print $1; exit}')"
    if [[ -z "$found_bin" ]]; then
      echo "license-eye binary not found after extracting ${url}. Contents:" >&2
      find "$tmpdir" -maxdepth 3 -print >&2 || true
      exit 2
    fi
    mv -f "$found_bin" "$BIN"
    chmod +x "$BIN"
  fi
fi
exec "$BIN" header fix