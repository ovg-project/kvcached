# Engine Options

## vLLM Options

Recommended flags when using kvcached with vLLM:

```bash
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1

vllm serve <model> \
  --no-enable-prefix-caching \    # Or keep prefix caching (default)
  --port <port> \
  --tensor-parallel-size <tp>
```

!!! note
    Do **not** set `--gpu-memory-utilization` when using kvcached. Memory is managed dynamically.

## SGLang Options

```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1

python -m sglang.launch_server \
  --model <model> \
  --disable-radix-cache \    # Or keep RadixCache (default)
  --port <port> \
  --tp <tp>
```

## Hybrid Attention Models

For hybrid models (GPT-OSS, Jamba, Bamba):

| Model Type | vLLM Flag | kvcached Env |
|-----------|-----------|--------------|
| Attention-only hybrid (GPT-OSS) | `--disable-hybrid-kv-cache-manager` | Default (contiguous) |
| Linear-attention hybrid (Jamba) | Do NOT pass above flag | `KVCACHED_CONTIGUOUS_LAYOUT=false` |
