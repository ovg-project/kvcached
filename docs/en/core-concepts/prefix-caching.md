# Prefix Caching

## Overview

Prefix caching avoids redundant computation by reusing KV caches from shared prefixes (e.g., system prompts) across requests. kvcached supports **automatic prefix caching (APC)** for vLLM and **RadixCache** for SGLang while maintaining memory elasticity.

## Elastic Memory with Prefix Caching

Without prefix caching, kvcached's memory is purely elastic—physical memory is allocated on demand and freed immediately after. With prefix caching enabled, a portion of memory is retained for reusable token prefixes:

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

## Memory Bound Configuration

The `KVCACHED_MAX_CACHED_TOKENS` environment variable controls the prefix cache budget:

| Value | Behavior |
|-------|----------|
| `-1` | **Unlimited**: closest to vanilla prefix-cache behavior, at the cost of memory elasticity |
| `0` | **Disabled**: cached prefixes are evicted immediately (no cross-request reuse) |
| `N > 0` | **Bounded**: cap cached prefixes at N tokens (default: `16000`) |

```bash
export KVCACHED_MAX_CACHED_TOKENS=16000   # default
```

## Usage

Prefix caching is **enabled by default** when kvcached is active. No additional flags are needed.

To disable prefix caching entirely (skip the caching path):

=== "vLLM"

    ```bash
    vllm serve <model> --no-enable-prefix-caching
    ```

=== "SGLang"

    ```bash
    python -m sglang.launch_server --model <model> --disable-radix-cache
    ```

## How It Works with Elasticity

The bounded prefix cache ensures that prefix caching does not consume all free memory, which would undermine elastic sharing:

1. When the cache exceeds the token bound, older prefixes are evicted
2. Evicted prefix pages are unmapped and returned to the physical memory pool
3. Other co-located models can then use that freed memory

This maintains the balance between prefix reuse (reducing computation) and memory elasticity (enabling multi-model sharing).

## Further Reading

- [GPU Virtual Memory](virtual-memory.md) — How physical pages are managed
- [Environment Variables](../configuration/environment.md) — Full configuration reference
