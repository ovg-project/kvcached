# 快速开始

本指南将帮助你在 5 分钟内实现**两个 LLM 模型在单 GPU 上弹性共享内存**。

## 第 1 步：启用 kvcached

```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
```

## 第 2 步：启动两个模型服务

打开两个终端，分别启动一个模型：

**终端 1（vLLM）：**
```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
vllm serve meta-llama/Llama-3.2-1B-Instruct --no-enable-prefix-caching --port 12346
```

**终端 2（vLLM）：**
```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
vllm serve Qwen/Qwen3-0.6B --no-enable-prefix-caching --port 12347
```

## 第 3 步：发送请求验证

```bash
curl -s -X POST http://127.0.0.1:12346/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-1B-Instruct","prompt":"解释 LLM 如何工作","max_tokens":128}'
```

## 原理说明

没有 kvcached 时，每个模型会静态预留所有可用 GPU 内存，无法在同一 GPU 上运行两个模型。使用 kvcached 后：

1. 每个引擎预留一大段**虚拟**地址空间（不消耗物理内存）
2. 物理 GPU 内存在请求到达时**按需分配**
3. 模型空闲时，其物理内存被回收供其他模型使用
4. 总物理内存使用量始终在 GPU 容量内

!!! tip "无需设置 `--gpu-memory-utilization`"
    启用 kvcached 后，**不需要**设置内存利用率限制。kvcached 会根据实际需求自动管理内存分配。

## 下一步

- [多模型服务](../user-guide/multi-model.md) — 高级多模型配置
- [内存控制 CLI](../user-guide/memory-control.md) — 使用 `kvctl` 管理内存
- [系统架构](../core-concepts/architecture.md) — 理解虚拟内存系统
