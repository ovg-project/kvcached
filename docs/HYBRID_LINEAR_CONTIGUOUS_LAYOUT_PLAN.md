# kvcached 注意力类型 / KV-Cache 布局支持，与 Qwen3-Next(GDN)Contiguous 布局落地方案

> 分支:`integration-with-new-vllm`。所有结论均基于源码核实,关键处标注 `文件:行号`(行号可能随版本漂移)。
> 姊妹文档:[`docs/SUPPORTED_MODELS_AND_LAYOUT.md`](./SUPPORTED_MODELS_AND_LAYOUT.md)(注意力类型/布局的完整背景)。

---

## 0. 术语勘误(重要,请先读)

任务里写的是「为什么 **Qwen3.6 目前只能使用 contiguous layout**」以及「实现 Qwen3.6 的 **contiguous** layout support」。这两句放在一起是自相矛盾的(既然只能用 contiguous,又何谈"实现 contiguous 支持")。核对源码后,真实的极性是**反的**:

- 在 **vLLM** 里,hybrid-linear(即 Qwen3-Next / 代码里写的 `Qwen3.5/3.6 GDN`)**当前被强制走 non-contiguous**,contiguous 会在启动时直接报错:

  ```python
  # kvcached/integration/vllm/interfaces.py:249
  if is_hybrid_linear and _contiguous_layout:
      raise ValueError("... which requires the non-contiguous KV layout.
                        Re-launch with KVCACHED_CONTIGUOUS_LAYOUT=false. ...")
  ```

所以本文按**唯一自洽的读法**来写:**Qwen3-Next(GDN)目前在 vLLM 只能用 non-contiguous;要做的是给它加 contiguous 布局支持。** 若你本意另有所指,请指正。

---

## 总览:一站式梳理(2026-07-02)

> 这一节把整件事按「做了什么 + 难度 → 对性能的影响 → vLLM 的默认与 offloading」三段讲清楚,细节见下面的 Part 1/2/3 与附录。

### A. 给 Qwen3.6(GDN hybrid)加 contiguous:做了什么 + 难在哪

**一句话**:这**不是「从零写一套 contiguous」**,而是**把一条本已快通的路上的三个障碍清掉**——真正花功夫的是诊断一个只在 contiguous 下才发作的隐藏 crash,不是写代码。5 个步骤的具体改动见 [Part 3](#part-3--给-qwen3-nextgdn加-contiguous-布局支持的落地方案)。

**「支持 contiguous」要解决什么**:hybrid 模型有两类层——全注意力层(存 K/V)和 GDN/mamba 层(存循环状态)。要让它们都能在 block-major 的连续布局下正确工作(即「一个 block 跨所有层的数据」是一整块连续内存)。

**为什么说「路本已快通」**:连续布局下,**全注意力那一侧的视图 kvcached 本来就构造对了**。真正缺的只有三样:

1. **拆掉人为拦路**:之前有一道 guard,只要「hybrid + 连续布局」就直接报错,把这条路堵死。
2. **补回 mamba 侧视图**:全注意力侧对了,但 GDN/mamba 状态那侧的视图要按连续布局重算(相邻 block 在内存里隔 `num_layers` 个、而不是 1 个)。这段代码其实以前写过、后来当死代码删了,这次复活即可。
3. **修一个隐藏的配置 bug(真正的难点)**:有个 `num_kv_buffers` 参数在两条独立代码路径里一个填 1、一个填 2。**非连续布局下这个不一致无害、从没暴露过**;可一旦切到连续布局,C++ 那边算出的物理页大了一倍,底层 map 的地址对齐断言**必崩**。诊断这个「只在连续下发作、又横跨 Python/C++」的问题是最费劲的地方——而改动本身只有一行。

**顺带澄清上一版含糊的一句**(「CUDA 默认 contiguous 所以直接报错」):kvcached 有个环境变量 `KVCACHED_CONTIGUOUS_LAYOUT` 决定布局。**用户不设时,是 kvcached 自己的默认策略在 CUDA 上选连续、在 ROCm 上选非连续**(这是 kvcached 的默认值,不是「硬件的默认」)。所以 CUDA 上跑 hybrid、又没手动设成非连续的人,才会一定命中「hybrid+连续」这道 guard 报错。**报错的只是「hybrid + 连续」这一个组合——非连续一直能正常跑,不存在「所有非连续都报错」。**

改动是**纯 Python、零 C++**。

**验证**:CPU 单测 38/38 过;GPU token 对齐在**两种 linear-attn 内核家族**上都**逐 token 一致**(原生==非连续==连续,`contiguous_layout=1, num_kv_buffers=1`,无崩溃):**Mamba2**(Zamba2-1.2B)和 **GDN**(Qwen3-Next tiny,10 层/2 全注意力层,与 Qwen3.6 同一 GDN 内核家族),均 vLLM 0.13 / A100。注:本机 27B `Qwen3.6-27B-AWQ` 的 `Qwen3_5` 架构此 vLLM 未注册、加载不了,故用 Qwen3-Next 验其 GDN 内核(内核相同)。

### B. contiguous 对 end-to-end 性能的影响(两股相反的力)

- **正向(map/分配)**:contiguous 每块 **1 次** `cuMemMap`(覆盖所有层),non-contiguous **`num_layers` 次**。冷路径微基准(16 层)contiguous **~4.2× 快**(256/1024/4096 块:3.5/14.4/57.4 ms vs 15.1/59.5/242.3 ms),**随层数放大**。受益:init、涨池尾延迟/TTFT、TP 多卡广播。
- **反向(注意力读)**:contiguous 是 block-major → 单层内相邻 block 步幅 ≈ 一个 2MB 页 → 每块各占一页 → paged-attention gather 时 TLB/L2 崩塌,FlashAttention KV-读 kernel **+56%**(bench_layout dense 实测),写不受影响。
- **净结果**:**dense 净亏 30~50%**(每层都吃读惩罚);**hybrid 净赢**(Zamba2 实测 +4.3%,5129 vs 4916 tok/s,init 持平)——因为 hybrid 只有少数层是全注意力,读惩罚被稀释,而 map 收益对所有层生效。**Qwen3.6 这类 GDN hybrid 大概率持平到小赢**,最实在的好处在 init / 分配尾延迟 / TP。

### C. vLLM 默认是 non-contiguous;只有 offloading 才切 contiguous(为了传输快)

**默认(per-layer / layer-major)**:vLLM 给**每层单独**分配一块 KV buffer,没有跨层复合体,每层内部 block 紧挨着 block。在「跨 block / 跨 layer 怎么摆」这个维度上,这等价于 kvcached 的 **non-contiguous**。vLLM 保持这个默认,正是为了保住 B 里说的注意力读局部性(层内 block 连续 → 一页装很多块)。差别只在「同一层里相邻两个 block 在内存里隔多远」(把「一块一层的 KV」当一个格子 `C`,num_layers=2/num_blocks=3):

```
per-layer(默认/非连续):layer0 的 buffer   [ b0 | b1 | b2 ]          相邻 block 隔 1 格,~32 块共享一页
block-major(offloading/连续):共用 buffer  [ b0L0 b0L1 | b1L0 b1L1 | b2L0 b2L1 ]
                                             layer0 的块在 0,2,4 → 相邻 block 隔 num_layers 格 ≈ 一页
```

**只有 offloading 才用 block-major**:`prefer_cross_layer_blocks` 是 connector 类上的 `ClassVar`,基类默认 `False`,**只有 `OffloadingConnector`(CPU KV 卸载)设 `True`**(Nixl/Mooncake/LMCache 等都 False)。它为 True 才走 `allocate_uniform_kv_caches` 构造 block-major 复合体 `(num_blocks, num_layers, 2, block, H, D)`。**为什么**:把 `num_layers` 紧跟 `num_blocks`,某块的所有层 KV 就是一整块连续 slab,卸载/搬运能把「第 N 块跨所有层」当**一个连续块**一次搬走。

**点题**:kvcached 的 contiguous == vLLM 的 offloading(跨层)布局,出于**同一个理由**(让一个块的跨层数据连续 → 传输/映射一次搞定);区别只是 **vLLM 把它留给 offloading 按需用,kvcached 把它提拔成 CUDA 默认并用 VMM 分页撑**——代价就是 B 里说的注意力读惩罚。

---

## 实现状态(本分支已落地代码 + CPU 单测,GPU 验证已通过 Mamba2)

**已完成(纯 Python,无 C++ 改动):**

| 步骤 | 文件 | 状态 |
|---|---|:--:|
| Step 1 · `num_kv_buffers` 对齐(`compound_num_kv_buffers = 1 if unified_pool else num_k_or_v`) | `kvcached/integration/vllm/interfaces.py` | ✅ |
| Step 2 · 删除 `is_hybrid_linear and _contiguous_layout` 的 `ValueError` guard | `interfaces.py` | ✅ |
| Step 3 · 恢复 contiguous 的 mamba `raw_info` 分支(单 base buffer + `num_layers×page` 步幅 + `is_contiguous`) | `interfaces.py` | ✅ |
| Step 4 · 恢复 `_reshape_mamba_contiguous` + reshape 分派 | `kvcached/integration/vllm/patches.py` | ✅ |
| Step 5 · `contiguous + ratio>1` 精准 `NotImplementedError`(仅 hybrid) | `interfaces.py` | ✅ |
| Step 6 · 文档更新(本文 + `SUPPORTED_MODELS_AND_LAYOUT.md` + `examples/08` + 修 `bench_layout` 极性 bug) | docs | ✅ |
| Step 7a · CPU 单测(`num_kv_buffers` 不变量 + 字节级 aliasing + ratio>1 guard) | `tests/test_hybrid_contiguous_layout.py` | ✅ 8 用例全过 |

CPU 验证结果:`tests/test_hybrid_contiguous_layout.py` + 既有 `tests/test_alloc_kv_cache_alignment.py` 共 **38 用例全过**(含 MHA/MLA 无回归)。验证了:hybrid 在两种布局下都传 `num_kv_buffers=1`(MHA=2/MLA=1 不变);contiguous 下 mamba 视图与 attention 视图逐字节 alias 于 `(N*num_pools+L)*page_size_bytes`;pool 间按 `page_size` 交织、block 间按 `num_layers*page_size` 步进;`contiguous+ratio>1` fail-loud、non-contiguous 不受影响。

**GPU token 对齐验证:已通过(Mamba2 hybrid)。** 2026-07-02 在 A100 80GB 上跑通了三方对齐:

| run | `contiguous_layout` | `num_kv_buffers` | 输出 |
|---|:--:|:--:|---|
| 原生 vLLM(无 kvcached) | — | — | baseline |
| kvcached 非连续 | 0 | 1 | 与 baseline 一致 |
| kvcached **连续**(本次新路径) | **1** | **1** | **与前两者逐 token 一致** |

- 模型:`Zyphra/Zamba2-1.2B`(Mamba2 + 全注意力 hybrid,`Zamba2ForCausalLM`),vLLM 0.13.0,greedy,4 个 prompt(含跨多 block 的长 prompt)× 48 token,**三者 token id 完全相同**,输出连贯(" Paris. Paris is the capital of France…")。
- 关键证据:连续 run 的 C++ 日志 `Init C++ PageAllocator: ... contiguous_layout=1, ... num_kv_buffers=1, page_size=16MB` —— 正是 Step 1 修复后的配置。**没有触发 `FTensor::map` 的奇数-pid 断言崩溃**(修复前用 `num_kv_buffers=2` 必崩),证明 Step 1 生效;且 Mamba2 的 selective-state / causal-conv1d kernel **确实容忍** `num_layers×page_size` 的大 state 步幅 —— 这正是 §3.4 预警的承重风险,现已被实测排除。
- 环境注意:该 env(`triattn_sg`)默认会启 TriAttention 压缩插件,验证时用 `VLLM_PLUGINS=""` 关掉以免混淆;GDN/Mamba 的 per-block state 大于默认 2MB,用 `KVCACHED_PAGE_SIZE_MB=16`(两种 layout 一致)。

**残留风险:已基本关闭(2026-07-02 补测 GDN)。** 之前担心 Mamba2 通过不代表 GDN 通过(GDN 用的是不同的 linear-attn kernel:`chunk_gated_delta_rule` 等)。现已在 **Qwen3-Next(GDN,tiny,10 层/2 全注意力层)** 上补跑同样的三方对齐:原生==非连续==连续 **逐 token 一致**,连续 run `contiguous_layout=1, num_kv_buffers=1`、无崩溃 —— **GDN 内核确实容忍 `num_layers×page` 的大 state 步幅**。仅剩两点非承重项:①本机 27B `Qwen3.6-27B-AWQ`(`Qwen3_5` 架构)此 vLLM 0.13 未注册,未在**该具体 checkpoint** 上验(内核同 GDN,已覆盖);②超深 GDN(数十层)下 `num_layers×page` 步幅更大,`as_strided` 视图用 int64 步幅、理论不溢出,真上大模型时顺带看一眼即可。

### GPU 验证 recipe(用本机 `Qwen3.6-27B-AWQ` = `Qwen3_5ForConditionalGeneration` GDN 混合,`full_attention_interval=4`)

> ⚠️ 环境提醒:本机 env(如 `triattn_sg`)的 kvcached editable 指向的是**另一个 checkout**(`kvcached-pr356-v13`,C++ 与本分支不同),且本分支尚未 build `vmm_ops.so`。要验证**本分支**的改动,必须让所有 vLLM 进程都用本分支的源码 + **本分支自己 build 的** `vmm_ops`,否则结果被污染。

```bash
# 0) 在干净 env 里从本分支源码构建 kvcached(含本分支 csrc 的 vmm_ops)
cd /mnt/permanent/src/kvcached          # 本分支 integration-with-new-vllm
pip install -e . --no-build-isolation   # 构建 vmm_ops.cpython-*.so 到本 checkout

# 1) 三方对齐:同一 greedy prompt,比较 token id 是否逐 token 一致
#    (a) 原生 vLLM(不启 kvcached)
#    (b) kvcached 非连续  KVCACHED_CONTIGUOUS_LAYOUT=false
#    (c) kvcached 连续    KVCACHED_CONTIGUOUS_LAYOUT=true   ← 本次新路径
MODEL=/mnt/permanent/data/Qwen3.6-27B-AWQ
COMMON="--model $MODEL --enforce-eager --max-model-len 4096 --gpu-memory-utilization 0.85"
# GDN state 较大时若报 KVCachedConfigError(block>page),按提示加 KVCACHED_PAGE_SIZE_MB=<N>

# (a) baseline
python -c "from vllm import LLM,SamplingParams; m=LLM('$MODEL',enforce_eager=True,max_model_len=4096); \
print([o.outputs[0].token_ids for o in m.generate(['讲讲注意力机制的原理。'],SamplingParams(temperature=0,max_tokens=128))])"
# (b) ENABLE_KVCACHED=1 KVCACHED_CONTIGUOUS_LAYOUT=false  python ...(同上)
# (c) ENABLE_KVCACHED=1 KVCACHED_CONTIGUOUS_LAYOUT=true   python ...(同上)
```

**判定**:(c) 的 token id 序列必须与 (a) 和 (b) **逐 token 一致**(裸长 prompt + 多轮 chat 各一);同时观察 kvcached 是否 engage(日志 `Init C++ PageAllocator ... contiguous_layout=1`),page 增长/收缩无 `FTensor::map` assert。全过 → gate 通过,可把文档从「待验证」改为「已支持」。若 (c) 与 (a)/(b) 不一致而 (a)==(b) 一致 → 正是 §3.4 预警的 GDN kernel 大步幅问题,需回到 non-contiguous 或在 kernel 侧处理。

另:`Qwen3.6` ≙ **Qwen3-Next 系列的 GDN(Gated DeltaNet)线性注意力混合架构**——这个名字直接出现在 `kvcached/kv_cache_manager.py:101,113` 的报错文案里("hybrid linear-attention models (e.g. Qwen3.5/3.6 GDN, Mamba)")。

---

## Part 1 · kvcached 支持哪些注意力类型,以及哪些必须用 non-contiguous 布局

### 1.1 kvcached 如何归类一个模型(vLLM)

kvcached **不看模型名**,只看推理引擎给出的 KV-cache spec,推断出一个内部 `attention_type`。vLLM 端逻辑在 `_infer_attention_type`(`patches.py:137-167`),优先级 **MLA > HYBRID_LINEAR > MHA**:

- `_is_mla_kv_cache_spec`(`patches.py:245-257`):任一 group 带 `use_mla=True`(≤0.10)或是 `MLAAttentionSpec`(≥0.11)→ **MLA**;
- 同时存在 `FullAttentionSpec` group **和** `MambaSpec` group → **HYBRID_LINEAR**;
- 其余(含纯滑窗、滑窗+全注意力多 group,如 gpt-oss)→ **MHA**(GQA 走同一路径,KV 头更少)。

只接受 4 种 spec:`FullAttentionSpec`、`SlidingWindowSpec`、`MLAAttentionSpec`、`MambaSpec`;其它类型在 `_validate_kv_cache_groups` 报错(`patches.py:91-94`)。`MambaSpec` 被接受但**不作为注意力管理**(其 recurrent/conv state 以 int8 raw buffer 分配后 `as_strided` 重塑)。

### 1.2 注意力类型 × 布局需求总表

| 注意力类型 | 模型例子 | 内部类型 | `num_kv_buffers` | vLLM 布局需求 | SGLang 布局需求 |
|---|---|---|:--:|---|---|
| MHA / GQA(稠密) | Llama 3.x、Qwen2.5/3 稠密 | `MHA` | 2 | **两者皆可**(CUDA 默认 contiguous) | 两者皆可 |
| MLA(潜在注意力) | DeepSeek-V2/V3 | `MLA` | 1 | **两者皆可** | 两者皆可 |
| 滑窗 / 注意力混合 | gpt-oss-20b | `MHA`(多 group) | 2 | **两者皆可**(几何统一走同构快路径) | 两者皆可 |
| **异构几何**注意力混合 | **Gemma 3/4** | `MHA`(多 group,块几何不同但 `block_mem_size` 相同) | 2 | **仅 non-contiguous**(`patches.py:1290-1296`) | — |
| **Hybrid-linear / Mamba** | **Qwen3-Next/GDN**、Jamba、Bamba、NemotronH、Zamba2、Plamo2 | `HYBRID_LINEAR` | 1 | **仅 non-contiguous**(`interfaces.py:249`)← 本文要改的对象 | 两者皆可,**投机解码必须 contiguous** |

**跨切面的布局强制**(与注意力类型正交):

| 场景 | 布局要求 | 出处 |
|---|---|---|
| AMD ROCm / HIP(任何模型) | 自动默认 **non-contiguous** | `utils.py:_default_contiguous_layout()`(HIP → False) |
| vLLM NIXL PD 分离 | **必须 non-contiguous** | `nixl_compat.py:143-149` |
| CUDA 稠密 MHA/GQA/MLA | 默认 **contiguous** | `utils.py`(非 HIP → True) |

### 1.3 一句话记忆:哪些**必须** non-contiguous

> **vLLM 的 hybrid-linear/Mamba**、**vLLM 的异构几何混合(Gemma 3/4)**、**ROCm/HIP**、**NIXL PD** —— 这四类必须 non-contiguous。其余(稠密 MHA/GQA、MLA、几何统一的滑窗)两种布局都行,CUDA 上默认 contiguous。

> ⚠️ **文档 bug 提醒**:`benchmarks/bench_layout/README.md`(在 `main` 上,#323 引入)第 3 节写反了极性——它声称 hybrid-linear "refuses non-contiguous / 需要 contiguous"。这与实际代码(`interfaces.py:249` 拒绝 contiguous)**相反**,是过时/笔误。本分支的 `docs/SUPPORTED_MODELS_AND_LAYOUT.md` 是对的。建议顺手修掉 bench 那段。

---

## Part 2 · 为什么 Qwen3-Next(GDN)在 vLLM 目前只能用 non-contiguous

先厘清两种布局(细节见 `SUPPORTED_MODELS_AND_LAYOUT.md §3`):

- **contiguous**:所有层 + K/V 打包进**一个**大 tensor,每 block 由一个「复合页」backing,一次 `map` 覆盖该 block 跨所有层。block N、pool L 的物理落点是 `(N*num_layers + L) * page_size_bytes`。
- **non-contiguous**:**每层各自**一个 FTensor / 各自 VM 预留,block N 落在本层 buffer 的 `N * page_size_bytes`。

强制 non-contiguous 有 **1 个表层原因 + 1 个深层机械原因 + 2 个次要阻碍**:

### 2.1 表层(那道显式 guard)

`interfaces.py:249-257` 直接在 `is_hybrid_linear and _contiguous_layout` 时抛 `ValueError`。而 CUDA 默认恰恰是 contiguous(`utils.py`,非 HIP 返回 True),所以 GDN 用户不显式设 `KVCACHED_CONTIGUOUS_LAYOUT=false` 就会在启动时撞上这道墙。报错只说"requires the non-contiguous layout",没给机械原因——这道 guard 是"防止走进一条没修好的路",不是"物理上不可能"。

**这道 guard 从 hybrid 支持第一天(#310,`253bd36`)就存在**——vLLM hybrid 从来只支持过 non-contiguous。

### 2.2 深层(真正让 contiguous 崩掉的机械原因):`num_kv_buffers` 1-vs-2 不一致

这是本次调查最关键的发现——**即便把 guard 删掉,contiguous 也会立刻 assert 崩溃**,根因是一个只在 contiguous 下暴露的 `num_kv_buffers` 不一致:

- HYBRID_LINEAR 的 `_get_kv_cache_params` 返回 **`num_kv_buffers=1`**(K/V 交织进单 buffer,`patches.py:236`)。这个 1 一路流到 `KVCacheManager` → C++ `PageAllocator`,contiguous 分支按 `pid * page_size_ * num_layers_ * num_kv_buffers_` = **×1** 算偏移(`page_allocator.cpp:625,656`)。
- 但 `alloc_kv_cache` 调 `create_kv_tensors(..., num_kv_buffers=num_k_or_v=2, ...)`(`interfaces.py:333`,因为 hybrid 的 `is_mla=False` 所以 `num_k_or_v=2`)。C++ 按这个 2 把复合页尺寸算成 `kPageSize * num_layers * num_kv_buffers` = **×2**(`allocator.cpp:142`)。
- 结果:PageAllocator 交给 `FTensor::map` 的偏移是 `pid*kPageSize*num_layers*1`,而 FTensor 的复合页 `page_size_` 是 `kPageSize*num_layers*2`。于是 `assert offset % page_size_ == 0`(`ftensor.cpp:101`)**对每个奇数 pid 都失败**。

关键点:**non-contiguous 下这个 1-vs-2 是无害的**——`create_kv_tensors` 只在 `if(contiguous_layout_)` 分支里读 `num_kv_buffers`(`allocator.cpp:142`),non-contiguous 分支根本不看它。这正是 non-contiguous hybrid 能正常出货、而没人去修这个不一致的原因。

### 2.3 两个次要阻碍(#315 顺手删掉的东西)

`#310` 当年其实写了一条 contiguous 的 mamba reshape 路径,但被上面的 guard 挡住、从出生就不可达。`#315`(`c5ce8ec`)于是把它当 **dead code 删了**:

1. **`_reshape_mamba_contiguous` + contiguous 的 raw_info 分支** 被删(`interfaces.py` 的 `buffers=[base]`、`block_stride_bytes=num_layers*page_size_bytes`、`is_contiguous`;`patches.py` 的整个函数与 dispatch)。**经对抗验证:这段被删的代码是正确的,不是 buggy**——它算出的 block N、pool L 偏移 `(N*num_pools+L)*page_size_bytes` 与 aliasing 目标一致,`inner_offset` 递进方式与 non-contiguous 版逐字节相同。删它只因不可达,不因错。
2. **`ratio>1`(`kernel_block_size != block_size`)的 contiguous 注意力视图** 当时是 `NotImplementedError`,也一并删了。这个是**真的表达不了**(见 §3.3(b))。

> 小结:vLLM 选 non-contiguous 给 hybrid,一半是历史/工程取舍(non-contiguous 每 2MB 页装 ~32 个 block,attention 读路径 TLB/L2 命中更好),一半是 contiguous 路径当年没接完(`num_kv_buffers` 不一致 + dead code 被删)。**没有一条是"物理上不可能"**。

---

## Part 3 · 给 Qwen3-Next(GDN)加 contiguous 布局支持的落地方案

### 3.1 核心洞察

contiguous 对 hybrid **天然是对的**,不需要动 C++:

- vLLM 让 attention-layer-i 和 mamba-layer-i **共享同一个 pool i、但用不相交的 block id**;`_update_hybrid_attention_mamba_layout` 把 attention 重排成「每 block 内 K/V 交织」,使 block N 成为一个连续的 `page_size_bytes` cell,mamba 层可原样复用。
- kvcached **现有**的 contiguous 注意力视图 `contiguous_tensor[:,L].permute(...)`(`interfaces.py:405-407`)**已经**产出这种「跨层交织 + block 内 K/V 交织」的排布(block 步幅 `num_layers*2*hidden`、K/V 步幅 `hidden`)。所以**注意力侧零新增代码**。
- mamba 侧只需要那条被删的 `_reshape_mamba_contiguous`(把 block 步幅从 `page_size_bytes` 换成 `num_layers*page_size_bytes`)。

**经对抗验证的关键等式**:contiguous 下 `contiguous_tensor[:,L]` 的 block N/pool L 落点 `(N*num_layers+L)*page_size_bytes` == 复原后 mamba 视图的落点 `(N*num_pools+L)*page_size_bytes`,**逐字节 alias**。于是 **contiguous 的正确性 ≘ non-contiguous 的正确性**(后者已出货、已测),只差一个"更大 block 步幅"的经验风险(见 §3.4)。

### 3.2 分步方案(纯 Python,无 C++ 改动)

| # | 步骤 | 改动文件 | 关键内容 | 风险 |
|:--:|---|---|---|:--:|
| 1 | **对齐 `num_kv_buffers`**(核心机械修复) | `interfaces.py` | `create_kv_tensors(..., num_kv_buffers=(1 if unified_pool else num_k_or_v))`(≈:333)。让复合页因子(C++ `×num_kv_buffers`)与 manager/PageAllocator 的 `1` 一致。FTensor **尺寸**仍保留 `num_k_or_v=2`(`ftensor_bytes_per_layer` 不变),只改复合页因子。对 non-contiguous hybrid 是 no-op,对 MHA(2)/MLA(1) 不受影响(用 `unified_pool` 门控)。 | 中 |
| 2 | **删掉 hybrid+contiguous 的 guard** | `interfaces.py` | 删 `:249-257`。这是对 #310 那道 guard 的直接 revert。之后 hybrid 落入现有 contiguous 分支(注意力)+ 复原的 contiguous raw_info(mamba)。 | 低 |
| 3 | **复原 contiguous 的 mamba raw_info** | `interfaces.py` | 在 raw_info 构造处按 `_contiguous_layout` 分支:contiguous → `buffers=[raw_kv_tensors[0].view(int8)]`(单 base)、`block_stride_bytes=num_layers*page_size_bytes`、加回 `is_contiguous`。**#315 该 hunk 的逆操作**。 | 低 |
| 4 | **复原 `_reshape_mamba_contiguous` 及其 dispatch** | `patches.py` | 在 `_reshape_mamba_non_contiguous` 旁加回该函数;在 `_reshape_kv_cache_tensors_from_kvcached`(≈:1384)按 `mamba_info['is_contiguous']` 分派。**#315 该 hunk 的逆操作**;数学与 SGLang 已验证的 contiguous 视图(`[slot][layer][cell]` 步幅)一致。 | 低 |
| 5 | **针对 `ratio>1` 精准 fail-loud** | `interfaces.py` | 在 ratio 算出后(≈:271)加 `if is_hybrid_linear and _contiguous_layout and ratio>1: raise NotImplementedError(...)`。把 #315 删掉的"一刀切 ratio>1 报错"收窄到**仅此不支持的组合**。(验证者建议:`build_kv_views` 里的镜像 guard 是死代码,可省。) | 中 |
| 6 | **更新文档 / 示例** | `docs/SUPPORTED_MODELS_AND_LAYOUT.md`、`examples/08_.../README.md` | 把"vLLM hybrid-linear 必须 non-contiguous"改成"两种布局皆可,CUDA 默认 contiguous 可用",注明残留的 `ratio>1` 限制、以及 NIXL-PD/ROCm 仍强制 non-contiguous。 | 低 |
| 7 | **回归 + aliasing 测试** | `tests/test_alloc_kv_cache_alignment.py`、`tests/test_paged_allocator_aliasing.py` | CPU 单测:stub `torch.cuda`、拦截 `create_kv_tensors`,断言 hybrid+contiguous 下它收到 `num_kv_buffers=1`(锁死 Step 1 不变量,无需 GPU)。第二个 CPU 单测:构造假 contiguous base,跑复原的 mamba 视图 + 注意力视图,写 attention[:,L] 的 block N、从 mamba pool L 的 block N 读回,断言 `(N*num_pools+L)*page_size_bytes` 处字节相同。 | 低 |

工作量估计:**~90% 是对 #310/#315 的干净 revert**,真正的新代码只有 Step 1 的一行 `num_kv_buffers` 修复和 Step 5 的窄化 guard。

### 3.3 最难/最需盯的三个点

- **(a) attention/mamba aliasing —— 已解决(设计上)**:两种布局下 block N/pool L 都是一个连续 `page_size_bytes` cell,注意力交织视图与 page 步幅的 mamba 视图逐字节 alias;contiguous 注意力视图本就是 block 交织的。**残留经验风险见 §3.4。**
- **(b) `ratio>1` 真表达不了**:把 `block_size` 拆成 `(ratio, kernel_bs)` 后,虚拟 block 步幅(`num_layers*2*hidden`)与 ratio 步幅(`kernel_bs*H*D`)不构成单一等差序列,`as_strided` 拼不出 kernel 要的 `(2, num_kernel_blocks, kbs, H, D)`。→ 必须 fail-loud(Step 5)。是否可达取决于 vLLM 给 Qwen3-Next 全注意力层选的 backend 的 `kernel_block_size`。
- **(c) `num_kv_buffers` 是最易搞错的结构性点**:**manager 必须保持 `1`**(这样 `block_mem_size` = 整个 `page_size_bytes` = 一个 mamba cell = 一个 block id = 一个 slot,且 `kv_cache_manager.py:104` 的 block>page 守卫按真实 cell 尺寸触发);**FTensor 尺寸必须保持 `2`**(K/V 维)。方案的做法是**解耦二者**(尺寸 ×2、复合页因子 ×1)。诱人的"把 manager 也改成 2"会破坏 block-id↔slot 对应关系(block 变半 cell、mamba tensor 越界)。

### 3.4 对抗验证结论与**必须做的 GPU gate**

对抗验证结论:**sound-with-caveats**。逐条核实通过:Step 1 的 assert 根因、字节级 aliasing、被删代码"正确非 buggy"、contiguous 确实减少 VMM map 次数(复合单次 map vs `unified_pool` 每层一次,`allocator.cpp:174-177` vs `181-187`)、`ratio>1` 确实不可表达、ROCm/NIXL 仍正确走 non-contiguous。

**唯一承重的未知 —— 已于 2026-07-02 验证通过(Mamba2 + GDN 三方 token 对齐一致,见「实现状态」)。** 曾经的门槛:vLLM 的 **GDN/mamba2 CUDA kernel**(`causal_conv1d_update`、`selective_state_update`、`chunk_gated_delta_rule` 等)是否容忍 `num_layers*page_size_bytes` 这么大的 state block 步幅。这一点**无法从本仓库代码证明**——non-contiguous 从不 exercise 这个大步幅。缓解:①稠密 MHA/MLA 的 contiguous **已是 CUDA 默认且出货**,证明全注意力 backend 本就容忍大 contiguous 步幅,于是残留风险**收窄到 mamba/GDN kernel 这一处**;②SGLang 已出货的 contiguous hybrid 是先例。但仍须**在 GPU 上验**:

- **GPU token 对齐测(必测,对齐 Gemma "逐 token 一致" 的验收标准)**:小 GDN/Mamba2 混合(Qwen3-Next,或 Bamba/Zamba2/NemotronH),vLLM ≥0.9,greedy,`KVCACHED_CONTIGUOUS_LAYOUT=true`,断言与 (i) `=false` 和 (ii) 原生 vLLM **逐 token 一致**(裸长 prompt + 多轮 chat 各一)。
- **补充**:page 增长/收缩压测(命中 contiguous map/unmap 的 `num_kv_buffers=1` 偏移缩放,确认无 `FTensor::map` assert);spec-decode(MTP/Eagle)不回退;大 state 触发 `KVCACHED_PAGE_SIZE_MB` bump 下复合页仍能 map;顺带检查 GDN kernel 在大步幅下**无 int32 索引溢出**(40–80 层 × MB 级 state,步幅可达千万级元素)。

### 3.5 值不值得做

**值得,但定位是"易用性/一致性"收益,不是性能刚需——而且便宜。**

- **易用性**:contiguous 是 CUDA 默认,稠密 MHA/GQA/MLA 都能用,唯独 hybrid-linear 是拒绝默认的"异类",GDN 用户用 stock 配置就撞启动报错。GDN 混合(Qwen3-Next、Qwen3.5/3.6)是快速增长的一族,让默认"开箱即用"能去掉一个真实的坑。
- **更少 VMM map**:non-contiguous `unified_pool` 每次涨页发 `num_layers` 次 map,contiguous 只发 1 次复合 map;48+ 层模型约 **48× 更少的 map 系统调用**,降低分配尾延迟。
- **不是理由(诚实说)**:spec-decode。**与 SGLang 不同**(SGLang 的 `fused_mamba_state_scatter` 硬要 contiguous 的单 `(num_layers,slots,*)` tensor),vLLM 的 GDN spec-decode 已按 per-layer `spec_state_indices_tensor` 工作,在 non-contiguous 下就能跑,contiguous 在这点上不带来新能力——只需别回退。

**成本/风险:低。** 唯一新代码是 Step 1 一行 + Step 5 窄 guard;mamba reshape/raw_info 是 #315 的逐字节 revert,guard 删除是 #310 的 revert,**无 C++ 改动**。集中风险是 §3.4 那个经验问题,由 GPU token 对齐测直接了结。

---

## 附录 · 关键代码位置(便于 review)

| 主题 | 位置 |
|---|---|
| 注意力类型推断(MLA>HYBRID_LINEAR>MHA) | `kvcached/integration/vllm/patches.py:137-167` |
| group 几何校验 / 接受的 spec | `patches.py:78-129` |
| `num_kv_buffers`/`cell_size` 计算(hybrid→1) | `patches.py:219-242` |
| **hybrid+contiguous guard(要删)** | `kvcached/integration/vllm/interfaces.py:249-257` |
| contiguous 注意力视图(已产出 block 交织) | `interfaces.py:401-408` |
| **`create_kv_tensors` 传入 `num_kv_buffers`(Step 1 要改)** | `interfaces.py:331-335` |
| mamba raw_info 构造(Step 3 要加分支) | `interfaces.py:~418-434` |
| `_reshape_mamba_non_contiguous`(Step 4 参照) | `patches.py:178-208` |
| mamba reshape 分派点(Step 4 要加 dispatch) | `patches.py:1384-1388` |
| block > page 守卫(GDN 大 state 报错含 "Qwen3.5/3.6 GDN") | `kvcached/kv_cache_manager.py:96-116` |
| 复合页尺寸 `kPageSize*num_layers*num_kv_buffers` | `csrc/allocator.cpp:142` |
| contiguous 单次复合 map vs unified_pool 每层 map | `csrc/allocator.cpp:174-177` / `178-187` |
| contiguous 偏移 `pid*page_size*num_layers*num_kv_buffers` | `csrc/page_allocator.cpp:625,656` |
| `FTensor::map` 对齐 assert(奇数 pid 崩溃点) | `csrc/ftensor.cpp:101` |
| 默认布局(CUDA→contiguous,HIP→non) | `kvcached/utils.py:_default_contiguous_layout` |
| NIXL PD 强制 non-contiguous | `kvcached/integration/vllm/nixl_compat.py:143-149` |
| SGLang 参照(mamba pool `num_kv_buffers=1`、contiguous `[slot][layer][cell]`) | `kvcached/integration/sglang/interfaces.py`,`patches.py:1117+` |
| **被删的 contiguous 路径(可 revert 参照)** | `git show c5ce8ec`(#315) |
| hybrid guard 引入 | `git show 253bd36`(#310) |

**相关历史提交**:#310(`253bd36`,vLLM hybrid non-contiguous + guard)、#314(`0ed866e`,SGLang hybrid contiguous)、#315(`c5ce8ec`,删 dead contiguous 路径)、#318(`d747ab5`,SGLang hybrid non-contiguous)、#367(`fea40de`,block>page fail-loud)。
