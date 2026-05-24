#!/bin/bash
# Baseline main: kvcached OFF + static gpu_memory_utilization split.
# Smaller context (8K) than the kvcached run because the bench uses ~2.3K tokens
# per request; no YaRN / tool-calling needed (no Hermes here).
cd /workspace/kvcached
unset ENABLE_KVCACHED KVCACHED_AUTOPATCH KVCACHED_IPC_NAME
nohup vllm serve Qwen/Qwen3-30B-A3B \
  --enforce-eager --no-enable-prefix-caching \
  --max-model-len 8192 --port 12346 \
  --gpu-memory-utilization 0.65 > /tmp/serve_main_baseline.log 2>&1 &
disown
echo "main BASELINE launched pid $!"
