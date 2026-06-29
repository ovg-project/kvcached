#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
# Stop all vllm servers + any stray kv_monitor; clean kvcached shm segments.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

pkill -f "kv_monitor.py" 2>/dev/null || true
for pidf in "${LOG_DIR}"/serve_*.pid; do
  [[ -f "$pidf" ]] && kill "$(cat "$pidf")" 2>/dev/null || true
done
pkill -f "vllm serve" 2>/dev/null || true
sleep 3
pkill -9 -f "vllm serve" 2>/dev/null || true

# kvcached leaves persistent shm segments; remove ours so the next run starts clean.
for ipc in "$KVCACHED_MAIN_IPC" "$KVCACHED_GUARD_IPC"; do
  rm -f "/dev/shm/${ipc}" 2>/dev/null || true
done
rm -f /dev/shm/kvcached_vLLM_* 2>/dev/null || true
echo "stopped."
