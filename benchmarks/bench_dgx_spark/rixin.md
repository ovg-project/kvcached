# kvcached Benchmark on DGX Spark — Experiment Report

## TL;DR

在 DGX Spark（128 GB 统一内存）上，双模型（Main LLM + Guardrail）共享单 GPU。
kvcached 在 **baseline KV cache 被打满**的场景下展现显著优势：

- **C=8: TTFT 2.05x 加速**（31.5s vs 64.7s）
- **C=16: TTFT 3.52x 加速**（48.3s vs 170.0s）
- **Throughput 1.30x 提升**（0.086 vs 0.066 req/s）

关键条件：baseline 紧配置（main=0.49, guard=0.31）→ baseline KV cache 只有 4.85 GiB，C≥8 时打满。

---

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

### 模型

| Role      | Model                       | Type                      | Params                | Weights (BF16) | KV/token |
| --------- | --------------------------- | ------------------------- | --------------------- | -------------- | -------- |
| Main LLM  | Qwen/Qwen3-30B-A3B          | MoE (48 full attn layers) | 30B total / 3B active | ~57 GB         | 96 KB    |
| Guardrail | meta-llama/Llama-Guard-3-8B | Dense                     | 8B                    | ~15 GB         | 128 KB   |

---

## 实验 1: Qwen3-30B-A3B, 12k input + 500 output (C=1-16)

复现Xingqi Experiment 5 (output=500) 并扩展到完整并发范围。
结果目录：`results_rixin_exp5_full/`

### 配置

| 参数                         | kvcached    | baseline    |
| ---------------------------- | ----------- | ----------- |
| main gpu-memory-utilization  | 0.75 *| 0.49        |
| guard gpu-memory-utilization | 0.30*      | 0.31        |
| 启动顺序                     | guard-first | guard-first |
| max-model-len                | 16384       | 16384       |
| enforce-eager                | yes         | yes         |
| prefix-caching               | off         | off         |

\* kvcached 在运行时覆盖 util 值，动态分配所有可用物理内存。

| 参数        | 值                      |
| ----------- | ----------------------- |
| Dataset     | Random                  |
| Input       | 8192 words ≈ 12k tokens |
| Output      | 500 tokens              |
| Concurrency | 1, 2, 4, 8, 16          |
| Prompts     | 64 per level            |

### 结果

#### Mean TTFT

| C    | kvcached (ms) | baseline (ms) | speedup   |
| ---- | ------------- | ------------- | --------- |
| 1    | 5,639         | 5,505         | 0.98x     |
| 2    | 10,222        | 9,937         | 0.97x     |
| 4    | 17,978        | 11,836        | 0.66x     |
| 8    | **31,535**    | **64,691**    | **2.05x** |
| 16   | **48,311**    | **169,988**   | **3.52x** |

#### P99 TTFT

| C    | kvcached (ms) | baseline (ms) | speedup   |
| ---- | ------------- | ------------- | --------- |
| 1    | 5,648         | 6,233         | 1.10x     |
| 2    | 11,528        | 11,905        | 1.03x     |
| 4    | 23,271        | 21,465        | 0.92x     |
| 8    | **48,352**    | **91,277**    | **1.89x** |
| 16   | **111,840**   | **247,171**   | **2.21x** |

#### Mean E2E

| C    | kvcached (ms) | baseline (ms) | speedup   |
| ---- | ------------- | ------------- | --------- |
| 1    | 25,067        | 23,134        | 0.92x     |
| 2    | 34,983        | 36,390        | 1.04x     |
| 4    | 56,045        | 60,816        | 1.09x     |
| 8    | 94,815        | 117,403       | **1.24x** |
| 16   | 184,768       | 223,004       | **1.21x** |

#### P99 E2E

| C    | kvcached (ms) | baseline (ms) | speedup   |
| ---- | ------------- | ------------- | --------- |
| 1    | 25,109        | 24,909        | 0.99x     |
| 2    | 35,506        | 45,059        | 1.27x     |
| 4    | 56,963        | 73,653        | **1.29x** |
| 8    | 98,843        | 145,303       | **1.47x** |
| 16   | 276,213       | 293,652       | 1.06x     |

#### Request Throughput

| C    | kvcached (req/s) | baseline (req/s) | speedup   |
| ---- | ---------------- | ---------------- | --------- |
| 1    | 0.040            | 0.043            | 0.93x     |
| 2    | 0.057            | 0.055            | 1.04x     |
| 4    | 0.071            | 0.065            | 1.09x     |
| 8    | **0.084**        | **0.066**        | **1.27x** |
| 16   | **0.086**        | **0.066**        | **1.30x** |

Baseline 在 C≥8 后 throughput 打平（0.066 req/s），被 KV cache 瓶颈卡死。
kvcached 继续提升到 0.086 req/s。

### KV cache 容量分析

#### Baseline main KV cache

vLLM serve log 报告：

```
Model loading took 56.88 GiB memory
Available KV cache memory: 4.85 GiB
GPU KV cache size: 52,960 tokens
Maximum concurrency for 16,384 tokens per request: 3.23x
```

vLLM 的 KV cache 计算逻辑（`vllm/v1/worker/gpu_worker.py` + `utils.py`）：

```python
requested_memory = total_memory × gpu_memory_utilization
                 = 121.7 GiB × 0.49 = 59.6 GiB

available_kv_cache = requested_memory - non_kv_cache_memory
                   = 59.6 - 54.78 = 4.85 GiB
```

其中 `non_kv_cache_memory = weights + activation_peak + non_torch_overhead`。
注意：54.78 GiB 是反推值（59.6 - 4.85），小于 "Model loading took 56.88 GiB"。
差异可能因为 56.88 GiB 包含了加载过程中的临时 buffer（之后被释放），而
`weights_memory` 取的是模型最终驻留大小。

#### 每请求 KV cache 需求

```
Qwen3-30B-A3B: 48 layers × 4 KV heads × 128 head_dim × 2(K+V) × 2 bytes = 96 KB/token
每请求峰值: (12,000 + 500) × 96 KB = 1.17 GiB
```

#### 并发容量验证

| C    | 理论峰值 KV | 占 4.85 GiB | 实测 KV usage  | Waiting | 能装下？ |
| ---- | ----------- | ----------- | -------------- | ------- | -------- |
| 1    | 1.17 GiB    | 24%         | ~22%           | 0       | ✅        |
| 2    | 2.34 GiB    | 48%         | ~45%           | 0       | ✅        |
| 4    | 4.69 GiB    | **97%**     | **~90%**       | 0       | ✅ 刚好   |
| 8    | 9.38 GiB    | **193%**    | 90%+Waiting 4  | 3-4     | ❌        |
| 16   | 18.75 GiB   | **387%**    | 90%+Waiting 12 | 8-12    | ❌        |

计算与实测吻合。**C=4 是 baseline 容量极限（97%），C=8 需要 193% 直接爆。**

#### kvcached main KV cache

kvcached 动态分配所有可用物理内存。从实测反推：

- C=8 时 KV usage ~38%，8 × 1.17 GiB = 9.38 GiB → 总 KV cache ≈ **24.7 GiB**
- 能装 24.7 / 1.17 ≈ **21 个并发请求**
- C=16 还有富余（实测偶尔 Waiting 1-3，可能是 `cudaMemGetInfo` 瞬时波动）

### 关键观察

1. **C=1,2**：持平，无内存压力
2. **C=4**：baseline 97% 在边界但不 Waiting，kvcached guard 处理略慢导致 TTFT
   反而更高（0.66x）
3. **C=8**：**转折点**！baseline 需要 193% 容量 → Waiting 3-4，kvcached 仅用
   38% → 零 Waiting → **2.05x TTFT 加速**
4. **C=16**：baseline 需要 387% → Waiting 8-12，kvcached 用 73% → 偶尔
   Waiting → **3.52x TTFT 加速**

---

## 实验 2: Qwen3.6-35B-A3B + ShareGPT (C=32)

早期实验，用不同模型和数据集。结果目录：`/tmp/c32_fair_*`

### 配置

| 参数                         | kvcached                                  | baseline |
| ---------------------------- | ----------------------------------------- | -------- |
| Main model                   | Qwen3.6-35B-A3B (MoE hybrid, 20 KB/token) | same     |
| main gpu-memory-utilization  | 0.65                                      | 0.65     |
| guard gpu-memory-utilization | 0.25                                      | 0.15     |
| max-model-len                | 8192                                      | 8192     |
| Dataset                      | ShareGPT (variable length)                | same     |
| Concurrency                  | 32                                        | 32       |

### 结果

| Metric             | kvcached    | baseline  | speedup   |
| ------------------ | ----------- | --------- | --------- |
| workflow TTFT mean | **3,282ms** | 6,796ms   | **2.07x** |
| workflow TTFT p99  | **5,832ms** | 12,496ms  | **2.14x** |
| E2E mean           | 191,690ms   | 199,584ms | 1.04x     |

| Component         | kvcached    | baseline |
| ----------------- | ----------- | -------- |
| input_guard mean  | **1,551ms** | 2,303ms  |
| main_ttft mean    | **1,731ms** | 4,493ms  |
| output_guard mean | 965ms       | 1,170ms  |

### 注意事项

这组实验存在混淆变量：

- **guard util 不同**（kvcached 0.25 vs baseline 0.15）→ guard 处理速度不同
- 双方都**零 preemption**，KV cache 未打满（Qwen3.6-35B-A3B 只有 20 KB/token）
- TTFT 优势可能来自 kvcached 的 scheduler patches 或 guard util 差异，而非内存共享

---

## 综合分析

### kvcached 优势的必要条件

kvcached 动态内存共享的优势需要同时满足：

1. **Baseline KV cache 被打满**（出现 Waiting/preemption）
2. **kvcached 的动态内存足够覆盖峰值需求**

在 DGX Spark 上制造这个条件需要：

- **长 decode (output≥500)**：请求持续占用 KV cache ~20s，增加同时在线 KV cache 量
- **Baseline 紧配置**：guard-first + 高 guard util → 挤压 main KV cache
- **高并发 (C≥8)**：超过 baseline 容量极限

### 为什么 output 长度是关键？

| output | 每请求 decode 时间 | C=8 同时在线 KV         | baseline 够用？ |
| ------ | ------------------ | ----------------------- | --------------- |
| 10     | ~0.5s              | 很低（请求快速释放）    | ✅ 够用          |
| 500    | ~20s               | 8 × 1.17 GiB = 9.38 GiB | ❌ 超 4.85 GiB   |

短 output 时，即使标称 C=8，main 的实际并发很低（请求来了就走），KV cache
来不及积累。长 output 时，请求在 main 停留 ~20s，KV cache 持续积累直到打满。

### 局限性

1. **绝对延迟偏高**：C=8 时 kvcached TTFT 也有 31.5s，因为 12k input prefill
   本身就慢。这是 Spark 单 GPU + 统一内存的硬件限制。
2. **Prefix caching 未开启**：实验中所有请求用重复 filler 文本，开启 prefix caching
   后 TTFT 会大幅下降，但 baseline 也可能不再 memory-bound。
3. **Baseline 配置的公平性**：baseline 用了较紧的 main=0.49（留更多给 guard），
   这是为了制造 memory-bound 场景。更宽松的 baseline 配置（如 main=0.62, guard=0.19）
   不会 memory-bound，但也无法同时给 guard 足够资源。

---

## 结论

### Demo 推荐配置

实验 1 的配置：

- Model: Qwen3-30B-A3B + Llama-Guard-3-8B
- Input: 8192 words (~12k tokens), Output: 500 tokens
- kvcached: main=0.75, guard=0.30
- baseline: main=0.49, guard=0.31, guard-first
- **C=8: TTFT 2.05x, Throughput 1.27x**
- **C=16: TTFT 3.52x, Throughput 1.30x**

### 关键发现

| 条件                                 | kvcached 表现        | 原因                   |
| ------------------------------------ | -------------------- | ---------------------- |
| 低并发 (C≤4)                         | 持平或略差           | 无内存压力             |
| 高并发 + 短 output                   | 持平                 | KV cache 占用时间太短  |
| 高并发 + 长 output + baseline 紧配置 | **2-3.5x TTFT 加速** | baseline KV cache 打满 |

### 数据归档

| 实验                 | 结果目录                   |
| -------------------- | -------------------------- |
| 实验 1 (exp5-full)   | `results_rixin_exp5_full/` |
| 实验 2 (v0-sharegpt) | `/tmp/c32_fair_*`          |
