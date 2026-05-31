# Concurrently running two models with kvcached

This example shows the minimal, end-to-end setup to colocate two models on the same GPU using kvcached. Both models are served by vLLM engines and share GPU memory elastically through kvcached.

## Two flavors of hybrid models

vLLM uses the term "hybrid" for two very different things, and kvcached needs different settings for each:

| Flavor | Examples | vLLM flag | kvcached env |
|---|---|---|---|
| **Attention-only hybrid** (full attention + sliding window, all groups unify to `FullAttentionSpec`) | GPT-OSS | `--disable-hybrid-kv-cache-manager` | (default; `KVCACHED_CONTIGUOUS_LAYOUT=true`) |
| **Linear-attention hybrid** (full attention + Mamba/SSM, groups have different specs and cannot be unified) | Jamba, Bamba, NemotronH, Zamba2, Plamo2 | **do NOT pass** `--disable-hybrid-kv-cache-manager` | `KVCACHED_CONTIGUOUS_LAYOUT=false` |

The `start_two_models.sh` script defaults to GPT-OSS (attention-only). For Jamba/Bamba and other Mamba-hybrid models, drop `--disable-hybrid-kv-cache-manager` from the `vllm serve` command and export `KVCACHED_CONTIGUOUS_LAYOUT=false` before launching.

## Prerequisites
- A working vLLM installation with kvcached.
- GPU with enough memory for the selected two models.

## Quickstart

### Start two vLLM servers

```bash
bash start_two_models.sh [--venv-vllm-path ${VENV_PATH}]
```

By default, this starts two instances of `openai/gpt-oss-20b` on ports 12346 and 12347. You can customize the models and ports:

```bash
bash start_two_models.sh \
  --model-a openai/gpt-oss-20b --port-a 12346 \
  --model-b openai/gpt-oss-20b --port-b 12347 \
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
export MODEL="openai/gpt-oss-20b"
export PROMPT="Explain how LLM works."
curl -s -X POST http://127.0.0.1:${PORT}/v1/completions \
  -H "Content-Type: application/json" \
  --data-binary @<(printf '{"model":"%s","prompt":"%s","max_tokens":128,"top_p":1,"seed":0}' "$MODEL" "$PROMPT")
```

## SGLang support

GPT-OSS (`openai/gpt-oss-20b`) is also supported via SGLang. Although GPT-OSS is a hybrid attention model (alternating sliding-window and full-attention layers), SGLang manages this entirely at the attention kernel level — each layer passes its own `sliding_window_size` to `RadixAttention`. The KV pool itself remains a single standard `MHATokenToKVPool`, which kvcached replaces with `ElasticMHATokenToKVPool`. No special configuration is needed.

```bash
python -m sglang.launch_server \
--model openai/gpt-oss-20b \
--disable-radix-cache \
--port 30001 \
--page-size 1
```
