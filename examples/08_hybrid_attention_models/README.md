# Concurrently running two models with kvcached

This example shows the minimal, end-to-end setup to colocate two models on the same GPU using kvcached. Both models are served by vLLM engines and share GPU memory elastically through kvcached.

## Prerequisites
- A working vLLM installation with kvcached.
- GPU with enough memory for the selected two models.

## Quickstart

### Start two vLLM servers

```bash
bash start_two_models.sh [--venv-vllm-path ${VENV_PATH}]
```

By default, this starts two instances of `gpt-oss-20b` on ports 12346 and 12347. You can customize the models and ports:

```bash
bash start_two_models.sh \
  --model-a gpt-oss-20b --port-a 12346 \
  --model-b gpt-oss-20b --port-b 12347 \
  --venv-vllm-path ${VENV_PATH}
```

### Testing by sending requests

In a separate terminal, send requests to both servers:

```bash
bash send_requests.sh --port-a 12346 --port-b 12347
```

You can also send requests manually:

```bash
export PORT=12346
export MODEL="gpt-oss-20b"
export PROMPT="Explain how LLM works."
curl -s -X POST http://127.0.0.1:${PORT}/v1/completions \
  -H "Content-Type: application/json" \
  --data-binary @<(printf '{"model":"%s","prompt":"%s","max_tokens":128,"top_p":1,"seed":0}' "$MODEL" "$PROMPT")
```

With default settings, both servers share the same kvcached segment, demonstrating concurrent running and elastic memory sharing.
