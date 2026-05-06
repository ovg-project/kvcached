# kvcached KV Cache 管理 API 接口文档

## 1. 概述

kvcached 通过 autopatch 机制，将 KV Cache 显存管理接口**直接注入到 vLLM 和 SGLang 的 HTTP 服务中**。用户无需部署额外的管理服务，直接通过推理框架自身的端口即可查询和管理 KV Cache 显存。

**适用引擎：** vLLM (>=0.8.4) 和 SGLang (>=0.4.9)

**前置条件：**
```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
# 使用 vllm serve 启动（不要用 python -m vllm.entrypoints.openai.api_server）
vllm serve MODEL_NAME --port 8000
```

**注意事项：**
- vLLM 必须使用 `vllm serve` 命令启动，不能用 `python -m vllm.entrypoints.openai.api_server` 直接启动
- SGLang 使用 `python -m sglang.launch_server` 启动即可
- 所有接口在 vLLM 和 SGLang 上行为完全一致

---

## 2. 接口列表

| 端点 | 方法 | 作用 |
|------|------|------|
| `/kvcache/status` | GET | 查询 KV Cache 显存使用状态 |
| `/kvcache/limit` | POST | 设置绝对 KV Cache 显存限制 |
| `/kvcache/limit_percent` | POST | 按 GPU 总显存百分比设置限制 |
| `/kvcache/trim` | POST | 强制释放空闲显存（含 prefix cache 驱逐） |
| `/kvcache/safety_floor` | GET | 查询安全门限配置 |

---

## 3. 安全门限机制

所有设置限制的接口都有安全门限保护，防止将 KV Cache 设置过低导致模型无法推理。

### 3.1 自动计算

安全门限根据模型参数自动计算，公式为：

```
min_safe_limit = num_layers × num_kv_buffers × page_size × MIN_SAFE_PAGES_PER_LAYER
```

| 参数 | 来源 | 说明 |
|------|------|------|
| `num_layers` | 模型自动检测 | 模型的 Transformer 层数 |
| `num_kv_buffers` | 注意力类型自动检测 | MHA/GQA=2（K+V），MLA=1 |
| `page_size` | 环境变量 `KVCACHED_PAGE_SIZE_MB` | 默认 2MB |
| `MIN_SAFE_PAGES_PER_LAYER` | 环境变量 `KVCACHED_MIN_SAFE_PAGES_PER_LAYER` | 默认 2 |

**不同模型的自动门限示例：**

| 模型 | 层数 | KV Buffers | 自动门限 |
|------|------|-----------|----------|
| Qwen3-1.7B | 28 | 2 (MHA) | 28 × 2 × 2MB × 2 = **224 MB** |
| Llama-3.1-8B | 32 | 2 (MHA) | 32 × 2 × 2MB × 2 = **256 MB** |
| Llama-3.1-70B | 80 | 2 (MHA) | 80 × 2 × 2MB × 2 = **640 MB** |
| DeepSeek-V3 | 61 | 1 (MLA) | 61 × 1 × 2MB × 2 = **244 MB** |

### 3.2 三级保护

| 级别 | 值 | 说明 |
|------|------|------|
| 模型自适应门限 | 自动计算 | 默认生效，基于模型参数 |
| 静态兜底值 | 512MB 或 GPU 2%（取较大值） | 共享内存不可用时的兜底 |
| 硬性最低值 | 64MB | 即使 `force=true` 也不能低于此值 |

### 3.3 绕过安全门限

在 `/kvcache/limit`、`/kvcache/limit_percent`、`/kvcache/trim` 的请求 body 中传入 `"force": true` 可以绕过模型自适应门限和静态兜底值，但仍受 64MB 硬性最低值约束。

---

## 4. 接口详细说明

### 4.1 GET /kvcache/status

查询当前实例的 KV Cache 显存使用状态。

**请求：**
```bash
curl http://localhost:8000/kvcache/status
```

**响应示例：**
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

**响应字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_size` | int | 当前 KV Cache 显存上限（字节） |
| `used_size` | int | 正在被推理请求使用的显存（字节） |
| `prealloc_size` | int | 预分配页（reserved pages）占用的显存 |
| `free_size` | int | 可分配空间 = total - used - prealloc |
| `usage_percent` | float | 使用率百分比 |
| `physical_occupied` | int | 实际占用的 GPU 物理显存 = used + prealloc |
| `min_effective_limit` | int | 可立即生效的最小 limit 值（= used_size） |
| `min_safe_limit` | int | 模型自动计算的安全门限（字节） |
| `safety_floor` | int | 当前生效的安全门限（综合模型计算值和兜底值） |
| `gpu_total_memory` | int | GPU 总显存 |
| `gpu_free_memory` | int | GPU 当前空闲显存 |

**关键指标解读：**
- `min_effective_limit`：设置 limit >= 此值可立即生效；< 此值进入 shrink mode
- `min_safe_limit`：安全门限，limit 不能低于此值（除非 force=true）
- `physical_occupied`：实际占用的 GPU 物理显存，其他实例可用空间 = gpu_total - 所有实例的 physical_occupied 之和

### 4.2 POST /kvcache/limit

设置绝对 KV Cache 显存限制。

**请求：**
```bash
# 设置为 3G
curl -X POST http://localhost:8000/kvcache/limit \
  -H "Content-Type: application/json" \
  -d '{"size": "3G"}'

# 设置精确字节数
curl -X POST http://localhost:8000/kvcache/limit \
  -H "Content-Type: application/json" \
  -d '{"size_bytes": 3221225472}'

# 强制设置（绕过安全门限，最低 64MB）
curl -X POST http://localhost:8000/kvcache/limit \
  -H "Content-Type: application/json" \
  -d '{"size": "100M", "force": true}'
```

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `size` | string | 二选一 | 人类可读的大小，如 `"3G"`、`"512M"`、`"1.5G"` |
| `size_bytes` | int | 二选一 | 精确字节数 |
| `ipc_name` | string | 否 | 目标 IPC 段名称，默认为当前实例 |
| `force` | bool | 否 | 是否绕过安全门限（默认 false） |

**响应示例：**
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

**生效行为：**
- `immediate_reclaim: true`：新 limit > used_size，空闲页立即释放
- `immediate_reclaim: false`：新 limit < used_size，进入 shrink mode，等请求完成后逐步回收
- `clamped: true`：请求值低于安全门限，被自动上调

### 4.3 POST /kvcache/limit_percent

按 GPU 总显存百分比设置 KV Cache 限制。

**请求：**
```bash
# 设置为 GPU 总显存的 20%
curl -X POST http://localhost:8000/kvcache/limit_percent \
  -H "Content-Type: application/json" \
  -d '{"percent": 20}'
```

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `percent` | float | 是 | GPU 总显存的百分比（0-100） |
| `ipc_name` | string | 否 | 目标 IPC 段名称 |
| `force` | bool | 否 | 是否绕过安全门限 |

**响应：** 与 `/kvcache/limit` 相同，额外包含 `gpu_total_memory` 和 `percent` 字段。

### 4.4 POST /kvcache/trim

强制释放空闲显存。将 KV Cache 限制压缩到当前实际使用量，释放所有预分配页和空闲页。

**请求：**
```bash
# 默认 trim：释放到当前使用量
curl -X POST http://localhost:8000/kvcache/trim

# 指定目标值
curl -X POST http://localhost:8000/kvcache/trim \
  -H "Content-Type: application/json" \
  -d '{"target": "1G"}'

# 强制 trim 到最低
curl -X POST http://localhost:8000/kvcache/trim \
  -H "Content-Type: application/json" \
  -d '{"target": "0", "force": true}'
```

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `target` | string | 否 | 目标大小，如 `"1G"`。不传则 trim 到当前 used_size |
| `target_bytes` | int | 否 | 目标精确字节数 |
| `ipc_name` | string | 否 | 目标 IPC 段名称 |
| `force` | bool | 否 | 是否绕过安全门限 |

**响应示例：**
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

**注意：** trim 后显存不会瞬间释放，而是在下一次 alloc/free 周期中由 PageAllocator 执行物理页回收。发送一个推理请求可以加速这个过程。

### 4.5 GET /kvcache/safety_floor

查询当前安全门限的详细配置信息。

**请求：**
```bash
curl http://localhost:8000/kvcache/safety_floor
```

**响应示例：**
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

**字段说明：**

| 字段 | 说明 |
|------|------|
| `safety_floor` | 当前生效的安全门限 |
| `model_min_safe_limit` | 模型参数自动计算的门限 |
| `fallback_configured_min_mb` | 静态兜底值（MB） |
| `gpu_percent_floor` | GPU 2% 对应的字节数 |
| `hard_minimum` | 硬性最低值（64MB） |

---

## 5. 典型使用场景

### 5.1 多实例显存回收：为新模型腾空间

```bash
# 场景：GPU 上已有模型 A（端口 8001），需要部署模型 B

# 1. 查看模型 A 当前显存占用
curl http://localhost:8001/kvcache/status
# 返回：total_size=9.77GB, used_size=2.50GB, min_safe_limit=224MB

# 2. 将模型 A 的 KV Cache 限制缩小到 3G
curl -X POST http://localhost:8001/kvcache/limit \
  -H "Content-Type: application/json" \
  -d '{"size": "3G"}'

# 3. 发一个请求触发显存回收
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "MODEL_A", "prompt": "hi", "max_tokens": 1}'

# 4. 确认显存已释放
curl http://localhost:8001/kvcache/status
# 返回：total_size=3.00GB

# 5. 现在可以启动模型 B 了
```

### 5.2 最大化释放显存

```bash
# 使用 trim 释放所有空闲显存（包括 prefix cache 和预分配页）
curl -X POST http://localhost:8001/kvcache/trim

# 如果需要释放到安全门限以下（谨慎使用）
curl -X POST http://localhost:8001/kvcache/trim \
  -H "Content-Type: application/json" \
  -d '{"target": "100M", "force": true}'
```

### 5.3 动态扩容

```bash
# 其他模型释放了显存，给当前模型扩容
curl -X POST http://localhost:8001/kvcache/limit \
  -H "Content-Type: application/json" \
  -d '{"size": "8G"}'
```

### 5.4 按比例分配多实例

```bash
# 3 个模型共享一块 24G GPU，按比例分配
curl -X POST http://localhost:8001/kvcache/limit_percent \
  -H "Content-Type: application/json" -d '{"percent": 30}'

curl -X POST http://localhost:8002/kvcache/limit_percent \
  -H "Content-Type: application/json" -d '{"percent": 40}'

curl -X POST http://localhost:8003/kvcache/limit_percent \
  -H "Content-Type: application/json" -d '{"percent": 30}'
```

---

## 6. 环境变量参考

| 环境变量 | 默认值 | 作用 |
|---------|--------|------|
| `ENABLE_KVCACHED` | `false` | 总开关，启用 kvcached |
| `KVCACHED_AUTOPATCH` | `0` | 启用自动 patch（含 API 端点注入） |
| `KVCACHED_MIN_KV_CACHE_MB` | `512` | 静态安全门限兜底值（MB） |
| `KVCACHED_MIN_SAFE_PAGES_PER_LAYER` | `2` | 模型自适应门限：每层最少保留的物理页数 |
| `KVCACHED_PAGE_SIZE_MB` | `2` | 物理页大小（MB） |

---

## 7. 注意事项

### 7.1 limit 设置不生效的常见原因

**现象：** 设置了较小的 limit（如 1G、2G），但显存占用没有下降。

**原因：** 当前 `used_size` 大于设置的 limit。`used_size` 包含：
- 正在处理的推理请求的 KV Cache block
- Prefix Cache 保留的 block（已完成请求的缓存，用于加速相同前缀的后续请求）

**解决方法：**
1. 查看 `/kvcache/status` 中的 `min_effective_limit`，设置 limit >= 此值
2. 使用 `/kvcache/trim` 先释放 prefix cache 和预分配页
3. 如果开启了 prefix caching，发送推理请求可以触发 prefix cache 驱逐

### 7.2 vLLM 启动方式

**必须使用 `vllm serve` 启动。** 使用 `python -m vllm.entrypoints.openai.api_server` 直接启动时，由于 Python 模块加载顺序问题，API 端点无法正确注入（返回 404）。

### 7.3 Prefix Cache 与显存回收

开启 prefix caching（vLLM 默认开启）时，已完成请求的 KV Cache block 会被保留在缓存中，不会自动释放回 kvcached。这些 block 在 kvcached 看来是「已使用」的（计入 `used_size`）。

驱逐 prefix cache 是安全的——不会影响正在进行的推理，只会导致后续相同前缀的请求需要重新计算 prefill。

如果不需要 prefix caching，可以在启动时关闭：
```bash
# vLLM
vllm serve MODEL --no-enable-prefix-caching
# SGLang
python -m sglang.launch_server --model-path MODEL --disable-radix-cache
```
