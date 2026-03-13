# Concurrently running two (or more) models with kvcached

This example shows the minimal, end-to-end setup to colocate two models on the same GPU using kvcached. The two models can be served by either two vLLM engines, two SGLang engines, or a combination of one vLLM and one SGLang engine. The same procedure can be easily extended to support more than two models, following similar steps.

## Prerequisites
- A working vLLM/SGLang installation with kvcached.
- GPU with enough memory for the selected two models.

## Quickstart

### Start engine servers

For vLLM:

```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
vllm serve "${MODEL}" \
  --disable-log-requests \
  --no-enable-prefix-caching \
  --port "${PORT}"
```

For SGLang:

```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
python3 -m sglang.launch_server --model "${MODEL}" \
  --disable-radix-cache \
  --trust-remote-code \
  --port "${PORT}"
```

You might want to start the two engine servers in different terminals.

### Testing by sending requests

```bash
export PORT=12346
export MODEL="meta-llama/Llama-3.2-1B"
export PROMPT="Explain how LLM works."
curl -s -X POST http://127.0.0.1:${PORT}/v1/completions \
  -H "Content-Type: application/json" \
  --data-binary @<(printf '{"model":"%s","prompt":"%s","max_tokens":128,"top_p":1,"seed":0}' "$MODEL" "$PROMPT")
```

### Using provided scripts

(1) Start two engine servers:

```
bash start_two_models.sh \
  --engine-a vllm --engine-b vllm \
  --model-a meta-llama/Llama-3.2-1B --port-a 12346 \
  --model-b Qwen/Qwen3-0.6B        --port-b 12347 \
  --venv-vllm-path ${VENV_PATH}
```

(2) In a separate terminal, send simple requests:

```
bash send_requests.sh --port-a 12346 --port-b 12347
```

You should see responses printed for each server. With default settings, both servers share the same kvcached segment, demonstrating concurrent running.
