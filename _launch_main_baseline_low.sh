#!/bin/bash
# Baseline main with LOW gpu_memory_utilization to leave room for guard.
# util=0.50 → budget ~61 GiB → weights 57 GiB → ~2 GiB KV cache (~10 concurrent slots).
cd /workspace/kvcached
unset ENABLE_KVCACHED KVCACHED_AUTOPATCH KVCACHED_IPC_NAME
nohup vllm serve Qwen/Qwen3-30B-A3B \
  --enforce-eager --no-enable-prefix-caching \
  --max-model-len 8192 --port 12346 \
  --gpu-memory-utilization 0.50 > /tmp/serve_main_baseline.log 2>&1 &
disown
echo "main BASELINE (low util) launched pid $!"
