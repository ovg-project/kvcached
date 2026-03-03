#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
TOOLS_DIR="$REPO_ROOT/tools"
mkdir -p "$TOOLS_DIR"

# Paths and lock for concurrency-safe install
LOCK_FILE="$TOOLS_DIR/.addlicense.lock"
ADDLICENSE_CACHED="$TOOLS_DIR/addlicense"

# Prefer cached binary in tools/, then PATH
if [[ -x "$ADDLICENSE_CACHED" ]]; then
  ADDLICENSE_BIN="$ADDLICENSE_CACHED"
elif command -v addlicense >/dev/null 2>&1; then
  ADDLICENSE_BIN="addlicense"
else
  # Acquire lock and perform double-checked, atomic install to avoid ETXTBUSY
  exec 9>"$LOCK_FILE"
  if ! flock -w 60 9; then
    echo "Failed to acquire addlicense install lock: $LOCK_FILE" >&2
    exit 1
  fi

  if [[ ! -x "$ADDLICENSE_CACHED" ]]; then
    if [[ "$(uname -s)" == "Linux" ]]; then
      tmpdir="$(mktemp -d "$TOOLS_DIR/addlicense.dl.XXXXXX")"
      trap 'rm -rf "$tmpdir"' EXIT
      curl -sSL https://github.com/google/addlicense/releases/download/v1.1.1/addlicense_1.1.1_Linux_x86_64.tar.gz | tar -zx -C "$tmpdir" addlicense
      chmod +x "$tmpdir/addlicense"
      mv -f "$tmpdir/addlicense" "$ADDLICENSE_CACHED"
      rm -rf "$tmpdir"
      trap - EXIT
    else
      echo "addlicense not found. Please install it: https://github.com/google/addlicense" >&2
      exit 127
    fi
  fi
  ADDLICENSE_BIN="$ADDLICENSE_CACHED"
fi

exec "$ADDLICENSE_BIN" -f "$REPO_ROOT/.license-header.txt" "$@"


