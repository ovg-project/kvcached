# Concurrently running two models with kvcached

This example shows the minimal, end-to-end setup to colocate two models on the same GPU using kvcached. Both models are served by vLLM engines and share GPU memory elastically through kvcached.

## Two flavors of hybrid models

vLLM uses the term "hybrid" for two very different things, and kvcached needs different settings for each:

| Flavor | Examples | vLLM flag | kvcached env |
|---|---|---|---|
| **Attention-only hybrid, uniform geometry** (full + sliding window, all attention groups share the same block geometry) | GPT-OSS | `--disable-hybrid-kv-cache-manager` **optional** (see note) | (default; `KVCACHED_CONTIGUOUS_LAYOUT=true`) |
| **Attention-only hybrid, heterogeneous geometry** (sliding vs full layers have different KV dims / block_size, same block_mem_size) | Gemma 3 / Gemma 4 | **do NOT pass** `--disable-hybrid-kv-cache-manager` | (default; either layout³) |
| **Linear-attention hybrid** (full attention + Mamba/SSM, groups have different specs and cannot be unified) | Qwen3-Next / Qwen3.5-3.6 GDN, Jamba, Bamba, NemotronH, Zamba2, Plamo2 | **do NOT pass** `--disable-hybrid-kv-cache-manager` | `KVCACHED_CONTIGUOUS_LAYOUT=false`² |

> **Note on GPT-OSS / `--disable-hybrid-kv-cache-manager`:** kvcached now supports
> multiple attention KV-cache groups directly, so GPT-OSS runs correctly **with the
> hybrid KV-cache manager enabled** (flag omitted) — verified byte-for-byte identical
> to the disabled-manager output on vLLM 0.24. The flag is therefore no longer
> required for GPT-OSS. In our measurement the flag makes ~no difference to the
> static KV-cache capacity (GPT-OSS's attention groups are homogeneous, so vLLM's
> hybrid manager and the disabled path allocate the same per-block memory); its
> benefit is runtime block recycling for long-context/high-concurrency workloads.
> `start_two_models.sh` still passes it for GPT-OSS for backward compatibility.

The `start_two_models.sh` script defaults to GPT-OSS (attention-only). For Jamba/Bamba and other Mamba-hybrid models, drop `--disable-hybrid-kv-cache-manager` from the `vllm serve` command and export `KVCACHED_CONTIGUOUS_LAYOUT=false` before launching. For Gemma 3/4 (heterogeneous attention geometry), likewise do not disable the hybrid manager; the layout no longer needs overriding.

> ³ **Contiguous layout for heterogeneous attention geometry is now supported.** It was previously blocked by a startup guard that predated any measurement. On vLLM 0.24 / `google/gemma-4-12B-it` (48 layers: 40 sliding `head_dim=256`×8 KV heads + 8 full `head_dim=512`×1 KV head, both 65536 B/block) the two layouts are token-for-token identical to each other and to the no-kvcached baseline. See the PR that removed the guard for the full matrix.

> ² **Contiguous layout for linear-attention hybrids is newly supported** (code + CPU unit tests landed; see [`docs/HYBRID_LINEAR_CONTIGUOUS_LAYOUT_PLAN.md`](../../docs/HYBRID_LINEAR_CONTIGUOUS_LAYOUT_PLAN.md)). GPU token-parity validated on vLLM 0.22.1 (tiny GDN hybrids, ratio=5, 1- and 2-slot pools): token-for-token identical to the no-kvcached baseline for both layouts. `contiguous + kernel_block_size != block_size` (ratio>1) is supported via kernel-block-granular per-block views.

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
