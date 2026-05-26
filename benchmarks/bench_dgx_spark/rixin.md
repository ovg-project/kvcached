# kvcached Benchmark on DGX Spark — Full Experiment Report

## TL;DR

在 DGX Spark（128 GB 统一内存）上对比 kvcached 和 baseline 在双模型（Main LLM + Guardrail）共享单 GPU 场景下的表现。跑了 6 组实验，覆盖 3 个模型、3 种 input 长度。

**核心结论**：

1. MoE 模型（KV cache 小）：kvcached ≈ baseline，因为 KV cache 不是瓶颈
2. Dense 模型（KV cache 大）：kvcached **反而更差**，因为 `cudaMemGetInfo` 在统一内存上的行为导致 kvcached 误判可用内存，触发更多 preemption
3. kvcached 的设计假设是离散 GPU（GPU 有独立显存），在 Spark 的统一内存架构上这个假设不成立

## 实验环境

| 组件     | 版本                                  |
| -------- | ------------------------------------- |
| GPU      | NVIDIA GB10, 128 GB 统一 CPU/GPU 内存 |
| vLLM     | 0.19.2.dev0 (from source)             |
| kvcached | 0.1.5                                 |
| Python   | 3.12                                  |
| CUDA     | 13.0                                  |

### Workflow

```
User request → Guardrail (input check) → Main LLM → Response
```

Guard 和 Main 分别跑在不同 port 上，共享同一块 GPU。

### 配置

| 参数                   | kvcached Main | kvcached Guard | baseline Main | baseline Guard |
| ---------------------- | ------------- | -------------- | ------------- | -------------- |
| gpu-memory-utilization | 0.70 *| 0.25*         | 0.65          | 0.16           |
| max-model-len          | 16384         | 16384          | 16384         | 16384          |
| enforce-eager          | yes           | yes            | yes           | yes            |
| prefix-caching         | off           | off            | off           | off            |

\* kvcached 在运行时覆盖这个值，动态分配所有可用物理内存。

kvcached util 总和 0.95 > baseline 0.81，这是 **设计意图**：动态内存共享允许更高的总利用率。

---

## 实验总览

| #    | 名称        | Main Model      | 类型       | KV/token | Input      | Output   | C    | 结果目录                   |
| ---- | ----------- | --------------- | ---------- | -------- | ---------- | -------- | ---- | -------------------------- |
| 1    | v0-random   | Qwen3.6-35B-A3B | MoE hybrid | 20 KB    | ~7835 tok  | 192      | 1–32 | `results/`                 |
| 2    | v0-sharegpt | Qwen3.6-35B-A3B | MoE hybrid | 20 KB    | variable   | variable | 32   | `/tmp/c32_fair_*`          |
| 3    | v2-dense    | Qwen3-32B       | Dense      | 256 KB   | ~3000 tok  | 10       | 1–16 | `results_rixin_v2/`        |
| 4    | v2-moe      | Qwen3-30B-A3B   | MoE        | 96 KB    | ~3000 tok  | 10       | 1–16 | `results_rixin_v2_moe/`    |
| 5    | v2-moe-v1   | Qwen3-30B-A3B   | MoE        | 96 KB    | ~6000 tok  | 10       | 1–16 | `results_rixin_v2_moe_v1/` |
| 6    | v2-moe-v2   | Qwen3-30B-A3B   | MoE        | 96 KB    | ~12000 tok | 10       | 8,16 | `results_rixin_v2_moe_v2/` |

### 模型 KV cache 对比

| Model           | Type                         | Layers | KV heads | head_dim | KV/token   | C=16 × 12k tok |
| --------------- | ---------------------------- | ------ | -------- | -------- | ---------- | -------------- |
| Qwen3.6-35B-A3B | MoE hybrid (10/40 full attn) | 10     | 2        | 128      | **20 KB**  | 3.7 GiB        |
| Qwen3-30B-A3B   | MoE (48 full attn)           | 48     | 4        | 128      | **96 KB**  | 17.6 GiB       |
| Qwen3-32B       | Dense (64 full attn)         | 64     | 8        | 128      | **256 KB** | 47.0 GiB       |

Guard model 固定为 `meta-llama/Llama-Guard-3-8B`（128 KB/token）。

---

## 实验 1: v0-random — Qwen3.6-35B-A3B + Random

max_model_len=8192, input ~7835 tokens, output 192, 64 prompts.

| C    | kvcached TTFT | baseline TTFT | kvcached E2E | baseline E2E |
| ---- | ------------- | ------------- | ------------ | ------------ |
| 1    | 3,880ms       | 3,988ms       | 10,753ms     | 10,883ms     |
| 4    | 11,599ms      | 12,266ms      | 28,509ms     | 28,841ms     |
| 8    | 16,794ms      | 15,354ms      | 49,116ms     | 50,516ms     |
| 16   | 26,786ms      | 26,757ms      | 88,921ms     | 90,427ms     |
| 32   | 78,693ms      | 53,405ms      | 178,785ms    | 182,115ms    |

**结论**：基本持平。MoE 模型 KV cache 极小（20 KB/token），C=32 也只需 4.88 GiB，远低于 baseline 的 9.21 GiB 上限。无内存压力，kvcached 无用武之地。

---

## 实验 2: v0-sharegpt — Qwen3.6-35B-A3B + ShareGPT

max_model_len=8192, ShareGPT dataset (variable length), C=32.

| Metric                 | kvcached    | baseline  |
| ---------------------- | ----------- | --------- |
| workflow TTFT mean     | **3,282ms** | 6,796ms   |
| workflow TTFT p99      | **5,832ms** | 12,496ms  |
| E2E mean               | 191,690ms   | 199,584ms |
| KV cache usage %       | 97-100%     | 24-25%    |
| Preemption (Waiting>0) | 0           | 0         |

| Component         | kvcached    | baseline |
| ----------------- | ----------- | -------- |
| input_guard mean  | **1,551ms** | 2,303ms  |
| main_ttft mean    | **1,731ms** | 4,493ms  |
| output_guard mean | 965ms       | 1,170ms  |

**结论**：kvcached TTFT 快 2 倍。但存在混淆变量：

- kvcached guard util=0.25 vs baseline 0.15（guard 有更多 KV cache → 处理更快）
- kvcached 的 97% usage 是 `cudaMemGetInfo` 导致的虚高（见下面「内存机制分析」）
- 优势可能来自 kvcached 的 scheduler patches 而非内存共享本身

---

## 实验 3: v2-dense — Qwen3-32B (Dense) + Random

max_model_len=16384, input ~3000 tokens, output 10, 64 prompts.

### TTFT

| C    | kvcached     | baseline     | 比率      | 赢家         |
| ---- | ------------ | ------------ | --------- | ------------ |
| 1    | 3,591ms      | 3,596ms      | 1.00x     | 平           |
| 2    | 5,808ms      | 5,853ms      | 1.01x     | 平           |
| 4    | 9,783ms      | 9,848ms      | 1.01x     | 平           |
| 8    | **19,491ms** | **11,838ms** | **0.61x** | **baseline** |
| 16   | **50,136ms** | **39,928ms** | **0.80x** | **baseline** |

### Component breakdown (mean ms)

| C    | kv guard | bl guard | kv main_ttft | bl main_ttft |
| ---- | -------- | -------- | ------------ | ------------ |
| 1    | 863      | 868      | 2,728        | 2,728        |
| 4    | 2,738    | 2,839    | 7,045        | 7,009        |
| 8    | 2,528    | 2,587    | **16,963**   | **9,251**    |
| 16   | 4,624    | 4,760    | **45,512**   | **35,168**   |

### Preemption (per-concurrency delta)

| C    | kvcached | baseline |
| ---- | -------- | -------- |
| 4    | 5        | 5        |
| 8    | **26**   | **5**    |
| 16   | **26**   | **25**   |

**结论**：kvcached **反而更差**。C=8 时 kvcached 有 5.2 倍的 preemption（26 vs 5），导致 1.65x 更高的 TTFT。根因是 `cudaMemGetInfo` 在统一内存上的问题（见下面分析）。

---

## 实验 4: v2-moe — Qwen3-30B-A3B + Random (3000 tok)

max_model_len=16384, input ~3000 tokens, output 10, 64 prompts.

| C    | kvcached TTFT | baseline TTFT | 比率  |
| ---- | ------------- | ------------- | ----- |
| 1    | 1,546ms       | 1,585ms       | 1.03x |
| 2    | 2,530ms       | 2,571ms       | 1.02x |
| 4    | 4,469ms       | 4,656ms       | 1.04x |
| 8    | 7,013ms       | 7,298ms       | 1.04x |
| 16   | 12,200ms      | 12,758ms      | 1.05x |

Preemption：kvcached 1 次，baseline 2 次（均在 C=16）。

**结论**：持平。C=16 × 3000 tok × 96 KB = 4.4 GiB，远低于 baseline 上限。

---

## 实验 5: v2-moe-v1 — Qwen3-30B-A3B + Random (6000 tok)

max_model_len=16384, input ~6000 tokens, output 10, 64 prompts.

| C    | kvcached TTFT | baseline TTFT | 比率  |
| ---- | ------------- | ------------- | ----- |
| 1    | 2,776ms       | 2,840ms       | 1.02x |
| 2    | 4,576ms       | 4,706ms       | 1.03x |
| 4    | 8,946ms       | 8,773ms       | 0.98x |
| 8    | 13,677ms      | 14,582ms      | 1.07x |
| 16   | 24,750ms      | 25,653ms      | 1.04x |

Preemption：**双方全程零**。

**结论**：持平。C=16 × 6000 tok × 96 KB = 8.8 GiB，接近但未超过 baseline ~10 GiB 上限。

---

## 实验 6: v2-moe-v2 — Qwen3-30B-A3B + Random (12000 tok)

max_model_len=16384, input ~12000 tokens, output 10, 64 prompts, C=8,16 only.

| C    | kvcached TTFT | baseline TTFT | 比率  |
| ---- | ------------- | ------------- | ----- |
| 8    | 29,495ms      | 29,951ms      | 0.98x |
| 16   | 53,429ms      | 54,495ms      | 0.98x |

### Component breakdown (mean ms)

| C    | kv guard | bl guard | kv main_ttft | bl main_ttft |
| ---- | -------- | -------- | ------------ | ------------ |
| 8    | 23,509   | 24,048   | 5,986        | 5,903        |
| 16   | 47,486   | 48,731   | 5,943        | 5,764        |

Preemption：**双方全程零**。

**结论**：完全持平。尽管理论 KV 需求（C=16 × 12k × 96 KB = 17.6 GiB）远超 baseline 上限（~10 GiB），但 guard 的串行处理（每个请求 ~6s）严重限制了请求到达 main 的速率，main 实际并发远低于 16。Guard 成为了真正的瓶颈，而非 KV cache。

---

## 综合分析

### 为什么 kvcached 在 Spark 上没有优势？

#### 1. 统一内存 + cudaMemGetInfo = 误判

这是核心问题。kvcached 的 `available_size()`（`kv_cache_manager.py:392`）通过 `cudaMemGetInfo` 实时检查物理可用内存：

```python
physical_free_pages = page_allocator.get_avail_physical_pages()  # calls cudaMemGetInfo
free_pages = min(virtual_free_pages, physical_free_pages)
```

在离散 GPU 上，`cudaMemGetInfo` 返回的是 GPU 独立显存的空闲量，不受 CPU 端影响。但在 Spark 的统一内存上：

- CPU 和 GPU 共享同一个物理内存池
- Guard 模型处理请求时会映射物理页（model weights + KV cache）
- `cudaMemGetInfo` 看到的 free 减少了
- kvcached 误以为 KV cache 满了 → 触发 preemption → TTFT 变高

Baseline 在启动时一次性预分配好 KV cache tensor，之后不再查 `cudaMemGetInfo`，所以不受 guard 活动影响。

#### 2. MoE 模型 KV cache 太小

| Model           | KV/token | C=16 × 12k tok | baseline KV上限 | 压力？   |
| --------------- | -------- | -------------- | --------------- | -------- |
| Qwen3.6-35B-A3B | 20 KB    | 3.7 GiB        | ~9 GiB          | 无       |
| Qwen3-30B-A3B   | 96 KB    | 17.6 GiB       | ~10 GiB         | 理论上有 |
| Qwen3-32B       | 256 KB   | 47 GiB         | ~10 GiB         | 严重     |

对 Qwen3-30B-A3B（96 KB），即使 12k input × C=16 = 17.6 GiB 理论需求超过 baseline 上限，但因为请求是流水线式到达（不是瞬间 16 个同时 prefill），所以实际峰值远低于理论值，双方都零 preemption。

对 Qwen3-32B（256 KB），3k input × C=8 = 5.9 GiB 就已经触发了 kvcached 的 `cudaMemGetInfo` 问题。

#### 3. Guard 成为瓶颈

在 12k token input 实验中，guard 处理一个请求就要 ~6s（非流式），C=16 时累积到 47s。Guard 的串行处理天然限制了请求到达 main 的速率，使得 main 的并发度远低于标称并发度，进一步减少了内存压力。

### KV cache 用量指标差异

kvcached 报告的 KV cache usage% 远高于 baseline（97% vs 24%），但这不代表实际 KV 数据更多：

- **baseline**: `used_blocks / total_blocks`（pre-allocated tensor 中实际使用的比例）
- **kvcached**: `1 - available_size() / num_gpu_blocks`，其中 `available_size()` 受 `cudaMemGetInfo` 影响

启动后零请求时 kvcached 就报告 15.8% usage（"phantom usage"），这是 guard 的物理内存占用被 kvcached 误算为不可用。

### 启动顺序影响

| 启动顺序 | Main KV cache | Guard KV cache | 总 KV cache   |
| -------- | ------------- | -------------- | ------------- |
| Guard 先 | 18.99 GiB     | 14.08 GiB      | 33.07 GiB     |
| Main 先  | **22.18 GiB** | 12.40 GiB      | **34.58 GiB** |

先启动的模型在 profiling 时看到更多空闲内存 → 分配更多虚拟块。相差 17%。

### Page 碎片（32MB pages）

kvcached 使用 32MB page size，导致部分填充的 page 浪费空间：

```
Per layer: 12,085 tokens × 4,096 bytes = 49.5 MB → 2 pages (64 MB) → 14.5 MB 浪费
64 layers × 14.5 MB = ~0.93 GiB 碎片开销
实测：3.57 GiB (kvcached 32MB) vs 2.96 GiB (baseline) = 0.61 GiB 开销
```

---

## 结论与建议

### 对于 demo

1. **如果必须在 Spark 上 demo**：用 Qwen3.6-35B-A3B + ShareGPT 数据集（实验 2），kvcached 快 2x。但要注意这个优势可能来自 kvcached 的 scheduler patches 而非内存共享。
2. **更好的 demo 环境**：在有独立 GPU 显存的机器上（A100/H100），kvcached 的 `cudaMemGetInfo` 应该能正确反映 GPU 内存，此时 dense 模型 + 高并发应该能展现真正的内存共享优势。

### 对于产品

统一内存架构（如 Spark、Apple M 系列）上，`cudaMemGetInfo` 返回的是整个系统的物理可用内存，不区分 GPU 和 CPU 用途。kvcached 需要一个**统一内存感知**的 `available_size()` 实现：

- 方案 A：在 alloc 时扣除已知的 co-located model 内存，而不是依赖 `cudaMemGetInfo`
- 方案 B：使用 virtual_free_pages 而非 physical_free_pages 作为上限（但可能导致 OOM）
- 方案 C：在统一内存平台上回退到 baseline 的静态分配策略

### 数据归档

所有原始 JSON 结果在 `results_*/` 目录中，每个文件包含 per-request 详细 breakdown（`--save-detailed`）。Serve 日志在 `/tmp/serve_main.log` 和 `/tmp/serve_guard.log`（每次实验会覆盖）。
