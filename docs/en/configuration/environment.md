# Environment Variables

Complete reference of all environment variables used by kvcached.

## Core Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_KVCACHED` | `false` | Enable kvcached memory management. Set to `true` to activate. |
| `KVCACHED_AUTOPATCH` | `0` | Enable automatic engine patching. Set to `1` to activate. |

## Memory Management

| Variable | Default | Description |
|----------|---------|-------------|
| `KVCACHED_MAX_CACHED_TOKENS` | `16000` | Maximum number of tokens to retain in prefix cache. `-1` = unlimited, `0` = disabled. |
| `KVCACHED_CONTIGUOUS_LAYOUT` | `true` (CUDA) / `false` (ROCm) | Use contiguous KV cache layout. Set `false` for AMD ROCm builds. |
| `KVCACHED_IPC_NAME` | auto | IPC segment name for shared memory. Used to isolate different engine instances. Examples: `VLLM`, `SGLANG`. |

## Engine-Specific

### vLLM

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_USE_V1` | — | Use vLLM V1 engine (recommended with kvcached). |
| `VLLM_ATTENTION_BACKEND` | — | Attention backend. `FLASH_ATTN` recommended. |

### SGLang

No additional SGLang-specific variables are required beyond the core kvcached variables.

## Example Configuration

```bash
# Minimal setup
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1

# With prefix caching budget
export KVCACHED_MAX_CACHED_TOKENS=32000

# For AMD ROCm
export KVCACHED_CONTIGUOUS_LAYOUT=false

# For multi-model with distinct IPC names
export KVCACHED_IPC_NAME=MODEL_A
```
