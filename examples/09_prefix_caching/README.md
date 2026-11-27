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

The **memory bound** (`KVCACHED_MAX_CACHED_TOKENS`, default `16000`) caps how many tokens of cached prefixes a model may keep. When the cache exceeds this bound, older prefixes are evicted. This prevents prefix caching from consuming all free memory, which would undermine the elastic sharing between co-located models that kvcached is designed for.

Sentinel values:

- `KVCACHED_MAX_CACHED_TOKENS=-1` — **unlimited**: closest to vanilla vLLM/SGLang prefix-cache behavior, at the cost of memory elasticity.
- `KVCACHED_MAX_CACHED_TOKENS=0` — **disabled at the kvcached layer**: the framework's prefix-cache module still runs, but every cached prefix is evicted as soon as it becomes evictable, so there is no cross-request reuse. To fully turn off prefix caching (skip the caching path entirely), use the framework flags below instead.
- `KVCACHED_MAX_CACHED_TOKENS=N` (`N>0`) — cap cached prefixes at `N` tokens.

A memory limit set with `kvctl limit` or `set_instance_memory_limit` takes precedence: lowering it evicts cached prefixes until the instance fits, whatever the bound above allows (except SGLang with tensor or pipeline parallelism).

## Usage

Prefix caching is enabled by default when kvcached is active. No additional flags are needed.

To control the cached token budget:

```bash
export KVCACHED_MAX_CACHED_TOKENS=16000   # default; -1 = unlimited, 0 = disabled
```

To disable prefix caching:

```bash
# vLLM
vllm serve <model> --no-enable-prefix-caching

# SGLang
python -m sglang.launch_server --model <model> --disable-radix-cache
```

## Update the bound at runtime

`KVCACHED_MAX_CACHED_TOKENS` only sets the initial bound. A controller can
publish a new bound to a running instance from another process:

```python
from kvcached.control import set_prefix_cache_retention_limit

result = set_prefix_cache_retention_limit(4000, instance_name="my-model")
```

`instance_name` is the engine's `KVCACHED_IPC_NAME`, and the engine must finish
initialization first. Limits use the same values as the environment variable:
`-1` for unlimited, `0` to drop cached prefixes as soon as they become
evictable, or a token count.

The engine applies the bound on its engine or scheduler thread, including
while idle, and evicts cached prefixes above it. SGLang tensor-parallel ranks
apply the update together. With pipeline parallelism, publication may still be
accepted, but the scheduler keeps the existing bound.

The returned `status` has the same meaning as for `set_instance_memory_limit`;
see [Memory control](../02_memory_control/README.md#embedding-revisioned-limits-in-a-production-controller).

## Read the bound in the engine

Inside the engine process, read the bound currently in effect:

```python
from kvcached.control import get_prefix_cache_retention_limit

limit_tokens = get_prefix_cache_retention_limit()
```
