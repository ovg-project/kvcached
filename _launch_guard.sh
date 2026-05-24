#!/bin/bash
cd /workspace/kvcached
export ENABLE_KVCACHED=true KVCACHED_IPC_NAME=kvcached_guard
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup vllm serve meta-llama/Llama-Guard-3-8B \
  --enforce-eager --no-enable-prefix-caching \
  --max-model-len 8192 --port 12347 > /tmp/serve_guard.log 2>&1 &
disown
echo "guard launched pid $!"
