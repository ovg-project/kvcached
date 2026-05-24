#!/bin/bash
# Stop vLLM serve processes used by the DGX Spark demo.
set -euo pipefail

echo "stopping vllm serve processes..."
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

rm -rf /tmp/kvcached-tp-kvcached_main-* /tmp/kvcached-tp-kvcached_guard-* 2>/dev/null || true
rm -f /dev/shm/kvcached_main /dev/shm/kvcached_guard 2>/dev/null || true
