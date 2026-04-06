# Prefix Caching

Prefix caching avoids redundant computation by reusing KV caches from shared prefixes (e.g., system prompts) across requests. kvcached supports **automatic prefix caching (APC)** for vLLM and **RadixCache** for SGLang while keeping memory elastic.

## Elastic memory with prefix caching

Without prefix caching, kvcached's memory is purely elastic, as physical memory is allocated on demand for active requests and freed immediately after. With prefix caching enabled, a portion of memory is retained for reusable token prefixes:

```
GPU Memory (per model)
┌─────────────────────────────────────┐
│  Model Weights          (fixed)     │
├─────────────────────────────────────┤
│  Active KV Cache      (elastic)     │ ← grows/shrinks with live requests
├─────────────────────────────────────┤
│  Cached Prefixes      (bounded)     │ ← reusable across requests, up to bound
├─────────────────────────────────────┤
│  Free                               │ ← available to other models/workloads
└─────────────────────────────────────┘
```

The **memory bound** (`KVCACHED_MAX_CACHED_TOKENS`, default `16000`; `0` means unlimited) caps how many tokens of cached prefixes a model may keep. When the cache exceeds this bound, older prefixes are evicted. This prevents prefix caching from consuming all free memory, which would undermine the elastic sharing between co-located models that kvcached is designed for.

## Usage

Prefix caching is enabled by default when kvcached is active. No additional flags are needed.

To control the cached token budget:

```bash
export KVCACHED_MAX_CACHED_TOKENS=16000   # default; set to 0 for unlimited
```

To disable prefix caching:

```bash
# vLLM
vllm serve <model> --no-enable-prefix-caching

# SGLang
python -m sglang.launch_server --model <model> --disable-radix-cache
```
