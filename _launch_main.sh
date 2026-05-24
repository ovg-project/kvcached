#!/bin/bash
cd /workspace/kvcached
export ENABLE_KVCACHED=true KVCACHED_IPC_NAME=kvcached_main
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
nohup vllm serve Qwen/Qwen3-30B-A3B \
  --enforce-eager --no-enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":40960}}' \
  --max-model-len 65536 --port 12346 > /tmp/serve_main.log 2>&1 &
echo "main launched pid $!"
sleep 25
if grep -qiE "unrecognized arguments|ValidationError|Value error|usage: vllm" /tmp/serve_main.log; then
  echo "FAIL_CONFIG:"; grep -iE "unrecognized|error|usage" /tmp/serve_main.log | grep -ivE "Unknown vLLM" | tail -3; exit 1
fi
echo "config accepted, loading..."
for i in $(seq 1 130); do curl -sf http://localhost:12346/health >/dev/null 2>&1 && { echo "MAIN_READY @ ${i}*5s"; break; }; sleep 5; done
if curl -sf http://localhost:12346/health >/dev/null 2>&1; then
  echo "MAIN_OK; sanity check:"
  curl -s http://localhost:12346/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-30B-A3B","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":20,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c 'import sys,json; print("SANITY:", repr(json.load(sys.stdin)["choices"][0]["message"]["content"]))'
else echo "MAIN_DOWN"; fi
