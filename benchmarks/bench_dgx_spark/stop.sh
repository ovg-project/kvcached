#!/bin/bash
# Stop vLLM serve processes used by the DGX Spark demo.
set -euo pipefail

echo "stopping vllm serve processes..."
MAIN_IPC_NAME="${KVCACHED_MAIN_IPC_NAME:-kvcached_main}"
GUARD_IPC_NAME="${KVCACHED_GUARD_IPC_NAME:-kvcached_guard}"
mapfile -t PIDS < <(pgrep -f "[v]llm serve" || true)
if [[ "${#PIDS[@]}" -gt 0 ]]; then
  for pid in "${PIDS[@]}"; do
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
  done

  for _ in $(seq 1 30); do
    mapfile -t LIVE < <(pgrep -f "[v]llm serve" || true)
    [[ "${#LIVE[@]}" -eq 0 ]] && break
    sleep 1
  done

  mapfile -t LIVE < <(pgrep -f "[v]llm serve" || true)
  if [[ "${#LIVE[@]}" -gt 0 ]]; then
    for pid in "${LIVE[@]}"; do
      pkill -KILL -P "$pid" 2>/dev/null || true
      kill -KILL "$pid" 2>/dev/null || true
    done
  fi
  echo "stopped ${#PIDS[@]} vllm serve process(es)"
else
  echo "no vllm serve processes found"
fi

rm -rf "/tmp/kvcached-tp-${MAIN_IPC_NAME}-"* "/tmp/kvcached-tp-${GUARD_IPC_NAME}-"* 2>/dev/null || true
rm -f "/dev/shm/${MAIN_IPC_NAME}" "/dev/shm/${GUARD_IPC_NAME}" 2>/dev/null || true
