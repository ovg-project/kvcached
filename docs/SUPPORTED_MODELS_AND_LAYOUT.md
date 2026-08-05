# kvcached 支持的模型架构与 KV-Cache 布局

本文档说明 kvcached 当前支持哪些模型架构,以及 **contiguous / non-contiguous** 两种 KV-cache 布局的含义、默认值和适用范围。所有结论均基于源码核实(文末附关键文件),并在 NVIDIA DGX Spark(GB10)上有运行实证(见 [§5](#5-实测验证))。

> 适用范围:本文对应 `main` / `integration-with-new-vllm` 分支。行号为参考,可能随版本变动。

---

## 1. kvcached 如何判断一个模型是否支持

kvcached **不按模型名识别**,而是检查推理引擎给出的 **KV-cache spec**,据此推断内部 `attention_type`。能识别其 spec 的模型即透明支持,识别不了的在 setup 阶段直接报错。

- **vLLM**:`_infer_attention_type`(`kvcached/integration/vllm/patches.py`)把整个 KV-cache 配置归为一种内部类型,优先级:`MLA` > `HYBRID_LINEAR` > `MHA`(默认)。
- **SGLang**:不做推断,而是用**专用弹性池(Elastic*Pool)子类替换** SGLang 原生池(`kvcached/integration/sglang/patches.py`)。

---

## 2. 支持的模型架构

### 2.1 按注意力类型

| 注意力类型 | 模型例子 | vLLM 触发的 spec | vLLM 内部类型 | SGLang 弹性池 |
|---|---|---|---|---|
| **MHA / GQA**(稠密) | Llama 3.1/3.3、Qwen 2.5/3 | `FullAttentionSpec` | `MHA`(GQA 共用,KV 头更少) | `ElasticMHATokenToKVPool` |
| **MLA**(潜在注意力) | DeepSeek-V2/V3 | `MLAAttentionSpec`(≥0.11)/ 旧版 `use_mla=True` | `MLA`,单融合 buffer,`num_kv_buffers=1` | `ElasticMLATokenToKVPool`(含 NSA / fp8 KV) |
| **滑窗 / 注意力混合** | **openai/gpt-oss-20b** | `SlidingWindowSpec`(+`FullAttentionSpec`) | 归 `MHA` 路径,多 group | `ElasticMHATokenToKVPool`(SWA 池独立 VM 预留) |
| **Hybrid-linear / Mamba** | Jamba、Bamba、NemotronH、Zamba2、Plamo2、Qwen3-Next/GDN | `FullAttentionSpec` + `MambaSpec` 同时存在 | `HYBRID_LINEAR` | `ElasticHybridLinearKVPool`(全注意力池 + `ElasticMambaPool`) |

vLLM 端**只接受 4 种 spec**:`FullAttentionSpec`、`SlidingWindowSpec`、`MLAAttentionSpec`、`MambaSpec`;其它类型报错
`kvcached only supports FullAttentionSpec, SlidingWindowSpec, MLAAttentionSpec, and MambaSpec, got {type} in group {idx}`。
`MambaSpec` 被接受但**不作为注意力管理**——其状态以 int8 原始 buffer 分配,再用 `torch.as_strided` 重塑。

### 2.2 关键约束:多注意力 group 必须共享「物理 block 几何」

`_validate_kv_cache_groups`(`patches.py` ~78–116)要求**所有注意力 group 的 `block_mem_size`(= `block_size × cell_size`,每块每 K-或-V 的字节数)和 `num_kv_buffers` 相同**,否则报错。**注意:不再要求 `block_size`/`cell_size` 各自相等** —— 只要物理块大小一致即可(见下「异构几何」支持)。

```
kvcached requires all attention KV cache groups to share one physical block
geometry (block_mem_size and num_kv_buffers). ...
```

- 根因:kvcached 的弹性池用**一套统一的物理 page/block 布局**服务所有 group(单池假设),只要 `block_mem_size` 一致,一个物理块就能同时被不同 group 以不同的 `as_strided` 视图解读。
- Mamba / 非注意力 group **豁免**此检查。
- **会被拒绝的情况**:
  1. 出现不支持的 spec 类型;
  2. **物理块几何真正冲突**:`block_mem_size` 不同,或 `num_kv_buffers` 不同(例如把 MLA 的 `num_kv_buffers=1` 与 MHA 的 `=2` 混在一起)。
  3. **纯 Mamba**(完全没有注意力 group):校验提前返回,但后续报 `kvcached is enabled but the KV cache config contains no attention groups`。

#### 异构几何混合模型(Gemma 3/4)—— 已支持

滑窗层与全注意力层 **`block_size`/`cell_size` 不同但 `block_mem_size` 相同** 的模型(如 **Gemma 3/4**:滑窗 `block_size=16`/`cell=4096`(8 KV 头×256)、全注意力 `block_size=64`/`cell=1024`(1 共享 KV 头×512),两者 `block_mem_size=65536` 相同)**现已支持**。

- 机制:一份统一物理池,`_reshape_kv_cache_tensors_from_kvcached` 按**每个 group 自己的** `(block_size, num_kv_heads, head_size)` 用 `interfaces.build_kv_views()` 建 `as_strided` 视图(`patches.py` 的异构分支 + `interfaces.py:build_kv_views`)。
- **要求**:vLLM ≥0.9(`KVCacheCoordinatorPatch` 多-group 路径)、且 **`KVCACHED_CONTIGUOUS_LAYOUT=false`**(异构目前仅支持非连续布局,连续布局会报错提示)。**不要**加 `--disable-hybrid-kv-cache-manager`(那会把各层并成 `UniformTypeKVCacheSpecs` 且 per-layer page 大小不一,反而无法承载)。
- 已 byte-exact 验证:见 [§5](#5-实测验证)。

> 对照:`gpt-oss` 也是滑窗混合,但其滑窗层与全注意力层 block 几何本就**一致**(单 group / 统一),走同构快路径。滑窗/多-group 基础支持由上游 PR #259 引入(放开到 `SlidingWindowSpec` + 多 group,前提是 `block_mem_size` 一致);**异构几何**的按-group 视图支持是在 `integration-with-new-vllm` 分支追加的。

### 2.3 MoE vs 稠密

**MoE 透明支持**——kvcached 只管 KV cache,与 FFN/专家路由正交,代码中无任何 MoE 逻辑。只要其注意力 spec 被支持即可(如 gpt-oss-20b)。注:README/examples 未对 MoE 做明文承诺,属「按架构支持」。

### 2.4 暂不支持 / 未声明

- **异构且 `block_mem_size` 也不同**的注意力 group(真正需要多物理池):当前单池承载不了,fallback 是「每 group 独立池」(未实现)。Gemma 3/4 这类 `block_mem_size` 相同的异构模型**已支持**(见 §2.2)。
- **异构几何 + 连续布局**:目前异构只支持 `KVCACHED_CONTIGUOUS_LAYOUT=false`;连续布局下的按-group 重排未实现。
- **纯 Mamba**(无任何全注意力层)。
- **多模态 LLM 的 KV-cache**、**MTP**:README/examples 未声明支持(视觉/扩散模型仅作为**同卡共置**负载出现,不由 kvcached 管理 KV)。

---

## 3. KV-Cache 布局:Contiguous vs Non-Contiguous

### 3.1 控制变量与默认值

环境变量 **`KVCACHED_CONTIGUOUS_LAYOUT`**,在 import 时读取一次并冻结为 `kvcached/utils.py` 的全局 `CONTIGUOUS_LAYOUT`(`_default_contiguous_layout`,~150–171):

```python
explicit = os.getenv("KVCACHED_CONTIGUOUS_LAYOUT")
if explicit is not None:                 # 显式设置永远优先
    return explicit.lower() == "true"
if getattr(torch.version, "hip", None):  # ROCm/HIP
    return False                         # → 非连续
return True                              # CUDA → 连续
```

| 平台 | 默认布局 |
|---|---|
| **CUDA** | `True`(连续) |
| **HIP / ROCm** | `False`(非连续) |

显式设置 `KVCACHED_CONTIGUOUS_LAYOUT` 在两个平台上都会覆盖自动默认值。

### 3.2 两种布局的定义

- **连续(contiguous)**:所有层 + K/V 打包进**一个**大 tensor,每个 block 由单个「复合页」backing——**一次 `map` 调用即覆盖某 block 跨所有层和 K/V**(复合页大小 = `kPageSize * num_layers * num_kv_buffers`)。每层 KV 是对这个大 tensor 的**跨步视图**(block 在最外维,block 间 stride 比紧凑布局大 `num_layers * num_kv_buffers` 倍),层在物理上交错。
- **非连续(non-contiguous)**:**每层各自**一个 tensor / 各自 VM 预留,每层 KV 是标准紧凑 contiguous。物理映射对每层分别 map K 和 V → 每 block `2 * num_layers` 次 VMM 操作(`unified_pool` 时每层 1 次)。

### 3.3 哪种架构/后端需要哪种布局

| 场景 | 要求 | 原因 |
|---|---|---|
| **vLLM hybrid-linear / Mamba** | 两者皆可¹(连续支持本分支新加入) | 连续下 attention 视图按 block 交织、mamba 视图按 `num_layers×page` 步幅重塑;`contiguous + ratio>1`(`kernel_block_size≠block_size`)亦已支持:attention 属主的块在块内按 kernel 粒度跨槽交错(全局线性),mamba 属主的块保持槽位顺序。详见 [HYBRID_LINEAR_CONTIGUOUS_LAYOUT_PLAN](./HYBRID_LINEAR_CONTIGUOUS_LAYOUT_PLAN.md) |
| **AMD ROCm / HIP**(任何模型) | **必须非连续**(自动默认) | 连续的交错布局喂给 ROCm 的 `PagedAttention.split_kv_cache` + paged kernel 会读错;CUDA 的 FlashAttention/FlashInfer 能容忍跨步视图 |
| **vLLM NIXL PD 分离** | **必须非连续** | `NixlConnector` 按 block-contiguous 注册每层 K/V,连续布局的跨层交错与之冲突 |
| **SGLang hybrid-linear / Mamba 投机解码** | **必须连续** | `fused_mamba_state_scatter` / MTP-verify 需要单个 `(num_layers, slots, *)` tensor |

> ¹ vLLM hybrid-linear 的**连续布局支持是本分支新加入的**(代码 + CPU 单测 `tests/test_hybrid_contiguous_layout.py`)。**GPU token 对齐验证已在 Mamba2 hybrid 上通过**:Zamba2-1.2B / vLLM 0.13.0 / A100,连续 vs 非连续 vs 原生 vLLM **逐 token 一致**(`contiguous_layout=1, num_kv_buffers=1`,无 `FTensor::map` 崩溃)。**GDN 也已通过**:Qwen3-Next(tiny)/ vLLM 0.13 / A100,连续 vs 非连续 vs 原生 **逐 token 一致** —— Mamba2 与 GDN 两种内核家族均验证过。仅本机 27B `Qwen3.6-27B-AWQ` 的 `Qwen3_5` 架构此 vLLM 未注册、未在该具体 checkpoint 上验(内核同 GDN,已覆盖)。验证细节/recipe 见 [方案文档](./HYBRID_LINEAR_CONTIGUOUS_LAYOUT_PLAN.md)。
| **NVIDIA 稠密 MHA/GQA/MLA** | **两者皆可** | FlashAttention/FlashInfer 读取路径对布局不敏感;CUDA 默认连续 |

### 3.4 正确性 / 性能权衡

| | 连续(CUDA 默认) | 非连续(ROCm 默认) |
|---|---|---|
| 物理映射 | 每 block 一次复合 map 覆盖所有层 → **VMM 操作更少**、map 并行更好、单一 tensor | 每 block `2*num_layers` 次 map(unified_pool 时每层 1 次) |
| 每层 KV tensor | 跨步视图(大 stride) | 紧凑标准 contiguous |
| ROCm paged kernel 正确 | ❌ | ✅ |
| NIXL PD | ❌ | ✅ |
| vLLM hybrid-linear/Mamba | ✅ 本分支新加入,已 GPU token-parity 验证(vLLM 0.22.1,ratio=5,L=1/2;含 `ratio>1`) | ✅(当前安全默认) |
| SGLang hybrid-linear 投机解码 | ✅(必需) | ❌ |

**净取舍**:连续 = 用「跨步的每层视图」换「更少的 VMM 操作」,CUDA 稠密/MLA 首选;非连续 = 紧凑每层 tensor,ROCm / NIXL 必需(vLLM hybrid-linear 连续支持已加入但待 GPU 验证,验证前仍以非连续为安全默认),代价是更多 map 调用。

> 另:每个 block(或 Mamba 每槽 super-cell)必须放进一个 `PAGE_SIZE`(默认 2MB),否则 `kv_cache_manager.py` 抛 `KVCachedConfigError`(对 GDN/Mamba 这类 recurrent state 较大的模型,需调大 `KVCACHED_PAGE_SIZE_MB`)。

---

## 4. 引擎与版本支持

| 引擎 | 版本范围(代码) | 已测 | 注意力类型 | 备注 |
|---|---|---|---|---|
| **vLLM** | `>=0.8.4`(`ALL`);多 group/混合需 `>=0.9.0` | README 测到 0.19;本仓库已验证 0.20–0.24 | MHA/GQA、MLA、滑窗、hybrid-linear | 0.8.x 路径仅单 group 且禁用 prefix cache |
| **SGLang** | `>=0.4.9`(`SGLANG_ALL_RANGE`) | README 测到 0.5.10,近期支持 0.5.11+ | MHA/GQA、MLA、Mamba、hybrid-linear | 仅 GPU |

其它:仅 **GPU**(CUDA/HIP);仅 **NHD** KV layout(非 NHD 的 FlashInfer 后端会报错);prefix caching **支持**(受 `KVCACHED_MAX_CACHED_TOKENS` 约束,默认 16000;vLLM 0.8.x 除外)。

---

## 5. 实测验证

在 NVIDIA DGX Spark(GB10,aarch64,CUDA 13)上,kvcached + vLLM(pip 安装,torch 2.11.0+cu130)实测:

| 模型 | 架构 | vLLM | 结果 |
|---|---|---|---|
| Qwen2.5-0.5B / Qwen3-8B | 稠密 MHA/GQA | 0.20.2 / 0.21.0 / 0.22.1 | ✅ patch + 输出与 baseline 字节一致 + on-demand map/unmap |
| **openai/gpt-oss-20b** | **滑窗混合(几何统一)** | 0.24.0 | ✅ patch 全过、`contiguous_layout=1`、MAP +1.0GB / UNMAP −0.8GB、答案正确、**0 次 block-geometry 拒绝** |
| google/gemma-4-12B-it | 滑窗混合(几何异构) | 0.24.0(原生加载) | ✅ **(方案 D 实现后)** `KVCACHED_CONTIGUOUS_LAYOUT=false`、hybrid manager 产生 16/64 异构组、越过 validate、kvcached engage;与 baseline **逐 token 一致**(裸长 prompt 4/4 + chat 长上下文连贯输出 2/2) |

验证脚本:`tests/test_kvcached_map_unmap.py`(map/unmap + 正确性,pytest 与 CLI 双模式)。

---

## 附录:关键文件与环境变量

**关键文件**
- `kvcached/integration/vllm/patches.py` — vLLM patch、`_infer_attention_type`、`_validate_kv_cache_groups`
- `kvcached/integration/vllm/interfaces.py` — vLLM 分配/reshape、布局分支、hybrid-linear 守卫
- `kvcached/integration/sglang/patches.py` / `interfaces.py` — SGLang 弹性池
- `kvcached/utils.py` — `CONTIGUOUS_LAYOUT` / `_default_contiguous_layout`、`MAX_CACHED_TOKENS`
- `kvcached/kv_cache_manager.py` — block 须放进单页的校验(`KVCachedConfigError`)
- `csrc/allocator.cpp` / `csrc/page_allocator.cpp` — 连续/非连续的 C++ 实现
- `kvcached/integration/vllm/nixl_compat.py` — NIXL PD 与布局约束

**相关环境变量**
- `ENABLE_KVCACHED` / `KVCACHED_AUTOPATCH` — 启用 + import 时自动 patch
- `KVCACHED_CONTIGUOUS_LAYOUT` — 布局(见 §3.1)
- `KVCACHED_GPU_UTILIZATION` — 物理池上限
- `KVCACHED_PAGE_SIZE_MB` — page 大小(默认 2)
- `KVCACHED_MAX_CACHED_TOKENS` — prefix cache 上限(默认 16000)
