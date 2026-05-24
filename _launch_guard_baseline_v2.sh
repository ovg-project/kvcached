#!/bin/bash
# Baseline guard (launched AFTER main is healthy).
# gpu_memory_utilization=0.85 because vLLM's KV budget = util*total - peak,
# and peak includes the main model's pre-allocated memory (~83 GiB).
# 0.85*128 - (83+15+overhead) ≈ 10 GiB KV cache for guard.
cd /workspace/kvcached
unset ENABLE_KVCACHED KVCACHED_AUTOPATCH KVCACHED_IPC_NAME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup vllm serve meta-llama/Llama-Guard-3-8B \
  --enforce-eager --no-enable-prefix-caching \
  --max-model-len 8192 --port 12347 \
  --gpu-memory-utilization 0.85 > /tmp/serve_guard_baseline.log 2>&1 &
disown
echo "guard BASELINE v2 launched pid $!"
