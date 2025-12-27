# kvcached API Reference

This document provides detailed API documentation for kvcached's public interfaces.

## Table of Contents

- [Environment Variables](#environment-variables)
- [CLI Tools](#cli-tools)
- [Python API](#python-api)
- [Integration APIs](#integration-apis)

---

## Environment Variables

### Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_KVCACHED` | `false` | Enable kvcached integration |
| `KVCACHED_AUTOPATCH` | `0` | Enable automatic patching of vLLM/SGLang |
| `KVCACHED_IPC_NAME` | Auto-detected | Custom name for the shared memory segment |
| `KVCACHED_GPU_UTILIZATION` | `0.95` | Maximum GPU memory fraction for KV cache |
| `KVCACHED_PAGE_SIZE_MB` | `2` | Page size in megabytes (must be multiple of 2) |
| `KVCACHED_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Advanced Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `KVCACHED_PAGE_PREALLOC_ENABLED` | `true` | Enable background page preallocation |
| `KVCACHED_MIN_RESERVED_PAGES` | `5` | Minimum pages to keep pre-allocated |
| `KVCACHED_MAX_RESERVED_PAGES` | `10` | Maximum pages to pre-allocate |
| `KVCACHED_CONTIGUOUS_LAYOUT` | `true` | Use contiguous memory layout for KV tensors |
| `KVCACHED_SANITY_CHECK` | `false` | Enable additional runtime safety checks |
| `KVCACHED_LOG_COLOR` | `true` | Enable colored log output |

---

## CLI Tools

### kvctl

The `kvctl` command provides control over kvcached memory segments.

```bash
# Start interactive shell
kvctl shell

# List all IPC segments
kvctl list [--json]

# Set absolute memory limit
kvctl limit <ipc_name> <size>
# Example: kvctl limit kvcached_vLLM_12345 8G

# Set memory limit as percentage of GPU
kvctl limit-percent <ipc_name> <percent>
# Example: kvctl limit-percent kvcached_vLLM_12345 80

# Watch memory usage in real-time
kvctl watch [-n <interval>] [ipc_name...]

# Launch curses-based UI
kvctl kvtop [--refresh <seconds>]

# Delete an IPC segment
kvctl delete <ipc_name>
```

### Size Format

The CLI accepts human-readable size formats:
- `512M` or `512MB` - 512 megabytes
- `2G` or `2GB` - 2 gigabytes
- `1024` - 1024 bytes (raw number)

---

## Python API

### Initialization

```python
# For vLLM integration
from kvcached.integration.vllm import interfaces as kvi

kvi.init_kvcached(
    tp_rank=0,           # Tensor parallel rank
    tp_size=1,           # Tensor parallel world size
    is_worker=False,     # Whether this is a worker process
    device="cuda:0",     # CUDA device string
    async_sched=False,   # Enable async scheduling
)

# For SGLang integration
from kvcached.integration.sglang import interfaces as kvi

kvi.init_kvcached(
    tp_rank=0,
    tp_size=1,
    is_worker=False,
    device="cuda:0",
    async_sched=True,    # Recommended for SGLang
)
```

### Memory Allocation

```python
# Allocate KV cache tensors (vLLM)
kv_tensors = kvi.alloc_kv_cache(
    kvcache_shape,       # Tuple: (2, num_blocks, block_size, heads, head_dim)
    block_size,          # Tokens per block
    dtype,               # torch.float16, torch.bfloat16, etc.
    device,              # "cuda"
    num_layers,          # Number of transformer layers
    attention_type="MHA", # "MHA" or "GQA"
    kv_layout="NHD",     # Layout format
)

# Allocate KV cache tensors (SGLang)
k_tensors, v_tensors = kvi.alloc_kv_cache(
    kvcache_shape,       # Tuple: (num_tokens, heads, head_dim)
    dtype,
    device,
    num_layers,
    page_size=1,         # Tokens per page (default 1 for SGLang)
    attention_type="MHA",
    kv_layout="NHD",
)
```

### KV Cache Manager

```python
# Get a KV cache manager for block-level operations
manager = kvi.get_kv_cache_manager(
    num_blocks,          # Total number of blocks
    block_size,          # Tokens per block
    cell_size,           # Bytes per cell
    num_layers,          # Number of layers
)

# Allocate blocks
block_ids = manager.alloc(num_blocks)

# Free blocks
manager.free(block_ids)

# Check available space
available = manager.available_size()
```

### Shutdown

```python
# Clean up resources
kvi.shutdown_kvcached()
```

---

## Integration APIs

### vLLM Integration

kvcached automatically patches vLLM when `KVCACHED_AUTOPATCH=1`. The following vLLM components are modified:

1. **BlockPool**: Replaced with `ElasticBlockPool` for dynamic allocation
2. **GPUModelRunner**: Patched to use kvcached for KV tensor allocation
3. **Worker**: Modified to skip GPU memory reservation checks

### SGLang Integration

kvcached patches SGLang's memory pool:

1. **MHATokenToKVPool**: Replaced with `ElasticMHATokenToKVPool`
2. **Memory management**: Dynamic allocation instead of static reservation

### Manual Integration

For custom integrations, use the low-level VMM operations:

```python
from kvcached.vmm_ops import (
    create_kv_tensors,
    map_to_kv_tensors,
    unmap_from_kv_tensors,
)

# Create virtual memory tensors
raw_tensors = create_kv_tensors(
    mem_size_per_layer,
    dtype_size,
    device,
    num_layers,
)

# Map physical pages to virtual addresses
map_to_kv_tensors(page_ids, block_ids)

# Unmap pages when done
unmap_from_kv_tensors(page_ids, block_ids)
```

---

## Controller API

The multi-model controller exposes REST endpoints:

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | OpenAI-compatible completions |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/health` | GET | Router health check |
| `/models` | GET | List available models |
| `/health/{model}` | GET | Model-specific health |

### Traffic Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/traffic/stats` | GET | Traffic statistics for all models |
| `/traffic/stats/{model}` | GET | Model-specific traffic stats |
| `/models/idle` | GET | List idle models |
| `/models/active` | GET | List active models |

### Sleep Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sleep/status` | GET | Sleep status of all models |
| `/action/sleep/{model}` | POST | Put model to sleep |
| `/action/wakeup/{model}` | POST | Wake up sleeping model |
| `/sleep/candidates` | GET | Models eligible for sleep |

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: kvcached is not initialized` | Called API before init | Call `init_kvcached()` first |
| `ValueError: Cannot get N free blocks` | Insufficient memory | Reduce requests or increase GPU utilization |
| `FileNotFoundError: /dev/shm/...` | IPC segment missing | Ensure kvcached is initialized |
| `NotImplementedError: Hybrid models...` | Unsupported attention | Update to latest kvcached version |

### Debugging

Enable debug logging for detailed information:

```bash
export KVCACHED_LOG_LEVEL=DEBUG
```

Monitor memory usage:

```bash
kvctl kvtop
```
