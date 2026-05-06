# kvcached KV Cache Management API Reference

## 1. Overview

kvcached uses an autopatch mechanism to **directly inject KV Cache memory management APIs into vLLM and SGLang HTTP services**. Users do not need to deploy additional management services — they can query and manage KV Cache memory directly through the inference framework's own port.

**Supported Engines:** vLLM (>=0.8.4) and SGLang (>=0.4.9)

**Prerequisites:**
```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
# Use vllm serve to launch (do NOT use python -m vllm.entrypoints.openai.api_server)
vllm serve MODEL_NAME --port 8000
```

**Notes:**
- vLLM must be launched with the `vllm serve` command, not `python -m vllm.entrypoints.openai.api_server`
- SGLang can be launched with `python -m sglang.launch_server`
- All APIs behave identically on vLLM and SGLang

---

## 2. API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/kvcache/status` | GET | Query KV Cache memory usage status |
| `/kvcache/limit` | POST | Set absolute KV Cache memory limit |
| `/kvcache/limit_percent` | POST | Set limit as a percentage of total GPU memory |
| `/kvcache/trim` | POST | Force release of free memory (including prefix cache eviction) |
| `/kvcache/safety_floor` | GET | Query safety floor configuration |

---

## 3. Safety Floor Mechanism

All limit-setting APIs have safety floor protection to prevent setting the KV Cache too low, which would make model inference impossible.

### 3.1 Automatic Calculation

The safety floor is automatically calculated based on model parameters using the formula:

```
min_safe_limit = num_layers × num_kv_buffers × page_size × MIN_SAFE_PAGES_PER_LAYER
```

| Parameter | Source | Description |
|-----------|--------|-------------|
| `num_layers` | Auto-detected from model | Number of Transformer layers in the model |
| `num_kv_buffers` | Auto-detected from attention type | MHA/GQA=2 (K+V), MLA=1 |
| `page_size` | Env var `KVCACHED_PAGE_SIZE_MB` | Default 2MB |
| `MIN_SAFE_PAGES_PER_LAYER` | Env var `KVCACHED_MIN_SAFE_PAGES_PER_LAYER` | Default 2 |

**Automatic floor examples for different models:**

| Model | Layers | KV Buffers | Auto Floor |
|-------|--------|------------|------------|
| Qwen3-1.7B | 28 | 2 (MHA) | 28 × 2 × 2MB × 2 = **224 MB** |
| Llama-3.1-8B | 32 | 2 (MHA) | 32 × 2 × 2MB × 2 = **256 MB** |
| Llama-3.1-70B | 80 | 2 (MHA) | 80 × 2 × 2MB × 2 = **640 MB** |
| DeepSeek-V3 | 61 | 1 (MLA) | 61 × 1 × 2MB × 2 = **244 MB** |

### 3.2 Three-Tier Protection

| Tier | Value | Description |
|------|-------|-------------|
| Model-adaptive floor | Auto-calculated | Active by default, based on model parameters |
| Static fallback | 512MB or 2% of GPU (whichever is larger) | Fallback when shared memory is unavailable |
| Hard minimum | 64MB | Cannot go below this even with `force=true` |

### 3.3 Bypassing the Safety Floor

Passing `"force": true` in the request body of `/kvcache/limit`, `/kvcache/limit_percent`, or `/kvcache/trim` can bypass the model-adaptive floor and the static fallback, but is still constrained by the 64MB hard minimum.

---

## 4. API Details

### 4.1 GET /kvcache/status

Queries the current instance's KV Cache memory usage status.

**Request:**
```bash
curl http://localhost:8000/kvcache/status
```

**Response Example:**
```json
{
  "self_ipc_name": "kvcached_vLLM_12345",
  "segments": [
    {
      "ipc_name": "kvcached_vLLM_12345",
      "total_size": 10485760000,
      "total_size_human": "9.77 GB",
      "used_size": 2684354560,
      "used_size_human": "2.50 GB",
      "prealloc_size": 314572800,
      "prealloc_size_human": "300.00 MB",
      "free_size": 7486832640,
      "free_size_human": "6.97 GB",
      "usage_percent": 25.6,
      "physical_occupied": 2998927360,
      "physical_occupied_human": "2.79 GB",
      "min_effective_limit": 2684354560,
      "min_effective_limit_human": "2.50 GB",
      "min_safe_limit": 234881024,
      "min_safe_limit_human": "224.00 MB",
      "safety_floor": 234881024,
      "safety_floor_human": "224.00 MB"
    }
  ],
  "gpu_total_memory": 25252929536,
  "gpu_total_memory_human": "23.52 GB",
  "gpu_free_memory": 14557184000,
  "gpu_free_memory_human": "13.55 GB"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `total_size` | int | Current KV Cache memory upper limit (bytes) |
| `used_size` | int | Memory currently used by inference requests (bytes) |
| `prealloc_size` | int | Memory occupied by reserved pages |
| `free_size` | int | Allocatable space = total - used - prealloc |
| `usage_percent` | float | Usage percentage |
| `physical_occupied` | int | Actual GPU physical memory occupied = used + prealloc |
| `min_effective_limit` | int | Minimum limit value that takes effect immediately (= used_size) |
| `min_safe_limit` | int | Model auto-calculated safety floor (bytes) |
| `safety_floor` | int | Currently active safety floor (combining model-calculated and fallback values) |
| `gpu_total_memory` | int | Total GPU memory |
| `gpu_free_memory` | int | Current free GPU memory |

**Key Metrics Interpretation:**
- `min_effective_limit`: Setting limit >= this value takes effect immediately; < this value enters shrink mode
- `min_safe_limit`: Safety floor — limit cannot go below this (unless force=true)
- `physical_occupied`: Actual GPU physical memory occupied; available space for other instances = gpu_total - sum of all instances' physical_occupied

### 4.2 POST /kvcache/limit

Sets an absolute KV Cache memory limit.

**Request:**
```bash
# Set to 3G
curl -X POST http://localhost:8000/kvcache/limit \
  -H "Content-Type: application/json" \
  -d '{"size": "3G"}'

# Set exact byte count
curl -X POST http://localhost:8000/kvcache/limit \
  -H "Content-Type: application/json" \
  -d '{"size_bytes": 3221225472}'

# Force set (bypass safety floor, minimum 64MB)
curl -X POST http://localhost:8000/kvcache/limit \
  -H "Content-Type: application/json" \
  -d '{"size": "100M", "force": true}'
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `size` | string | One of two | Human-readable size, e.g. `"3G"`, `"512M"`, `"1.5G"` |
| `size_bytes` | int | One of two | Exact byte count |
| `ipc_name` | string | No | Target IPC segment name, defaults to current instance |
| `force` | bool | No | Whether to bypass safety floor (default false) |

**Response Example:**
```json
{
  "ipc_name": "kvcached_vLLM_12345",
  "success": true,
  "previous_total_size": 10485760000,
  "previous_total_size_human": "9.77 GB",
  "new_total_size": 3221225472,
  "new_total_size_human": "3.00 GB",
  "current_used_size": 2684354560,
  "current_used_size_human": "2.50 GB",
  "immediate_reclaim": true,
  "safety_floor": 234881024,
  "safety_floor_human": "224.00 MB",
  "clamped": false,
  "message": "Limit updated 9.77 GB -> 3.00 GB. Free pages will be reclaimed immediately."
}
```

**Effect Behavior:**
- `immediate_reclaim: true`: New limit > used_size, free pages are released immediately
- `immediate_reclaim: false`: New limit < used_size, enters shrink mode, gradually reclaims as requests complete
- `clamped: true`: Requested value was below safety floor and was automatically raised

### 4.3 POST /kvcache/limit_percent

Sets the KV Cache limit as a percentage of total GPU memory.

**Request:**
```bash
# Set to 20% of total GPU memory
curl -X POST http://localhost:8000/kvcache/limit_percent \
  -H "Content-Type: application/json" \
  -d '{"percent": 20}'
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `percent` | float | Yes | Percentage of total GPU memory (0-100) |
| `ipc_name` | string | No | Target IPC segment name |
| `force` | bool | No | Whether to bypass safety floor |

**Response:** Same as `/kvcache/limit`, additionally includes `gpu_total_memory` and `percent` fields.

### 4.4 POST /kvcache/trim

Force release of free memory. Compresses the KV Cache limit down to the current actual usage, releasing all preallocated pages and free pages.

**Request:**
```bash
# Default trim: release down to current usage
curl -X POST http://localhost:8000/kvcache/trim

# Specify target value
curl -X POST http://localhost:8000/kvcache/trim \
  -H "Content-Type: application/json" \
  -d '{"target": "1G"}'

# Force trim to minimum
curl -X POST http://localhost:8000/kvcache/trim \
  -H "Content-Type: application/json" \
  -d '{"target": "0", "force": true}'
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `target` | string | No | Target size, e.g. `"1G"`. If omitted, trims to current used_size |
| `target_bytes` | int | No | Target exact byte count |
| `ipc_name` | string | No | Target IPC segment name |
| `force` | bool | No | Whether to bypass safety floor |

**Response Example:**
```json
{
  "success": true,
  "ipc_name": "kvcached_vLLM_12345",
  "previous_total": 10485760000,
  "previous_total_human": "9.77 GB",
  "previous_used": 2684354560,
  "previous_used_human": "2.50 GB",
  "previous_prealloc": 314572800,
  "previous_prealloc_human": "300.00 MB",
  "new_limit": 2684354560,
  "new_limit_human": "2.50 GB",
  "freed_estimate": 7801405440,
  "freed_estimate_human": "7.27 GB",
  "safety_floor": 234881024,
  "safety_floor_human": "224.00 MB",
  "clamped": false,
  "message": "Trim applied: limit 9.77 GB -> 2.50 GB. Estimated 7.27 GB reclaimable."
}
```

**Note:** After trim, memory is not released instantly but is performed by the PageAllocator in the next alloc/free cycle. Sending an inference request can accelerate this process.

### 4.5 GET /kvcache/safety_floor

Queries detailed configuration information for the current safety floor.

**Request:**
```bash
curl http://localhost:8000/kvcache/safety_floor
```

**Response Example:**
```json
{
  "safety_floor": 234881024,
  "safety_floor_human": "224.00 MB",
  "model_min_safe_limit": 234881024,
  "model_min_safe_limit_human": "224.00 MB",
  "fallback_configured_min_mb": 512,
  "gpu_percent_floor": 505058590,
  "gpu_percent_floor_human": "481.64 MB",
  "hard_minimum": 67108864,
  "hard_minimum_human": "64.00 MB",
  "env_vars": {
    "KVCACHED_MIN_KV_CACHE_MB": "Fallback static floor (default 512)",
    "KVCACHED_MIN_SAFE_PAGES_PER_LAYER": "Pages per layer for model-aware calculation (default 2)"
  },
  "description": "..."
}
```

**Field Descriptions:**

| Field | Description |
|-------|-------------|
| `safety_floor` | Currently active safety floor |
| `model_min_safe_limit` | Floor auto-calculated from model parameters |
| `fallback_configured_min_mb` | Static fallback value (MB) |
| `gpu_percent_floor` | Byte count corresponding to 2% of GPU memory |
| `hard_minimum` | Hard minimum (64MB) |

---

## 5. Typical Usage Scenarios

### 5.1 Multi-Instance Memory Reclamation: Freeing Space for a New Model

```bash
# Scenario: Model A is already running on GPU (port 8001), Model B needs to be deployed

# 1. Check Model A's current memory usage
curl http://localhost:8001/kvcache/status
# Returns: total_size=9.77GB, used_size=2.50GB, min_safe_limit=224MB

# 2. Shrink Model A's KV Cache limit to 3G
curl -X POST http://localhost:8001/kvcache/limit \
  -H "Content-Type: application/json" \
  -d '{"size": "3G"}'

# 3. Send a request to trigger memory reclamation
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "MODEL_A", "prompt": "hi", "max_tokens": 1}'

# 4. Confirm memory has been released
curl http://localhost:8001/kvcache/status
# Returns: total_size=3.00GB

# 5. Now Model B can be launched
```

### 5.2 Maximizing Memory Release

```bash
# Use trim to release all free memory (including prefix cache and preallocated pages)
curl -X POST http://localhost:8001/kvcache/trim

# If you need to release below the safety floor (use with caution)
curl -X POST http://localhost:8001/kvcache/trim \
  -H "Content-Type: application/json" \
  -d '{"target": "100M", "force": true}'
```

### 5.3 Dynamic Expansion

```bash
# Other models have released memory; expand current model's allocation
curl -X POST http://localhost:8001/kvcache/limit \
  -H "Content-Type: application/json" \
  -d '{"size": "8G"}'
```

### 5.4 Proportional Multi-Instance Allocation

```bash
# 3 models sharing a 24G GPU, allocated proportionally
curl -X POST http://localhost:8001/kvcache/limit_percent \
  -H "Content-Type: application/json" -d '{"percent": 30}'

curl -X POST http://localhost:8002/kvcache/limit_percent \
  -H "Content-Type: application/json" -d '{"percent": 40}'

curl -X POST http://localhost:8003/kvcache/limit_percent \
  -H "Content-Type: application/json" -d '{"percent": 30}'
```

---

## 6. Environment Variable Reference

| Environment Variable | Default | Purpose |
|----------------------|---------|---------|
| `ENABLE_KVCACHED` | `false` | Master switch to enable kvcached |
| `KVCACHED_AUTOPATCH` | `0` | Enable automatic patching (including API endpoint injection) |
| `KVCACHED_MIN_KV_CACHE_MB` | `512` | Static safety floor fallback value (MB) |
| `KVCACHED_MIN_SAFE_PAGES_PER_LAYER` | `2` | Model-adaptive floor: minimum physical pages retained per layer |
| `KVCACHED_PAGE_SIZE_MB` | `2` | Physical page size (MB) |

---

## 7. Notes

### 7.1 Common Reasons Limit Settings Don't Take Effect

**Symptom:** A smaller limit was set (e.g. 1G, 2G), but memory usage did not decrease.

**Cause:** The current `used_size` is larger than the set limit. `used_size` includes:
- KV Cache blocks of currently processing inference requests
- Blocks retained by Prefix Cache (cached blocks of completed requests, used to accelerate subsequent requests with the same prefix)

**Resolution:**
1. Check `min_effective_limit` in `/kvcache/status` and set limit >= this value
2. Use `/kvcache/trim` to first release prefix cache and preallocated pages
3. If prefix caching is enabled, sending an inference request can trigger prefix cache eviction

### 7.2 vLLM Launch Method

**Must use `vllm serve` to launch.** When using `python -m vllm.entrypoints.openai.api_server` directly, API endpoints cannot be correctly injected due to Python module loading order issues (returns 404).

### 7.3 Prefix Cache and Memory Reclamation

When prefix caching is enabled (enabled by default in vLLM), KV Cache blocks of completed requests are retained in the cache and are not automatically released back to kvcached. These blocks are considered "used" by kvcached (counted in `used_size`).

Evicting the prefix cache is safe — it does not affect ongoing inference, but subsequent requests with the same prefix will require recomputing the prefill.

To disable prefix caching at launch:
```bash
# vLLM
vllm serve MODEL --no-enable-prefix-caching
# SGLang
python -m sglang.launch_server --model-path MODEL --disable-radix-cache
```
