#!/bin/bash
# Baseline guard: kvcached OFF + small static util (just enough for weights+headroom).
cd /workspace/kvcached
unset ENABLE_KVCACHED KVCACHED_AUTOPATCH KVCACHED_IPC_NAME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup vllm serve meta-llama/Llama-Guard-3-8B \
  --enforce-eager --no-enable-prefix-caching \
  --max-model-len 8192 --port 12347 \
  --gpu-memory-utilization 0.18 > /tmp/serve_guard_baseline.log 2>&1 &
disown
echo "guard BASELINE launched pid $!"
