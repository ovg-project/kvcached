# kvcached

**面向 LLM KV 缓存的 GPU 虚拟内存管理系统**

---

kvcached（KV cache daemon）是一个面向 LLM 推理引擎的 GPU 虚拟内存管理系统。它将操作系统级别的虚拟内存抽象引入 GPU 内存管理，实现**弹性、按需**的 KV 缓存分配，提升动态工作负载下的 GPU 利用率。

## 为什么选择 kvcached？

当前的 LLM 推理引擎（vLLM、SGLang）在启动时会静态预留大量 GPU 内存用于 KV 缓存。这种刚性分配方式导致 GPU 无法被高效共享，当工作负载动态变化或多个模型共存于同一 GPU 时会造成内存浪费。

kvcached 通过**将 GPU 虚拟地址空间与物理内存分配解耦**来解决这个问题。推理引擎可以预先保留虚拟内存，但只有当缓存被实际使用时才分配物理 GPU 内存。

## 核心特性

- **弹性 KV 缓存** — 根据实时负载动态分配和回收 KV 内存
- **GPU 虚拟内存** — 通过运行时映射将逻辑 KV 与物理 GPU 内存解耦
- **内存控制 CLI** — 使用 `kvctl` 管理内存限制，使用 `kvtop` 实时监控
- **前端路由与休眠模式** — 将请求路由到目标模型，空闲模型自动休眠释放内存
- **多引擎支持** — 透明集成 SGLang 和 vLLM
- **前缀缓存** — 支持自动前缀缓存（APC），可配置内存上限

## 应用场景

| 场景 | 说明 |
|------|------|
| **多 LLM 服务** | 多个 LLM 弹性共享 GPU 内存，无需刚性分区 |
| **无服务器 LLM** | 模型按需启停，仅在使用时分配 KV 缓存 |
| **复合 AI 系统** | 在流水线中跨专用模型弹性分配内存 |
| **GPU 工作负载混部** | LLM 推理与训练、微调或视觉模型共存 |

## 快速导航

- [安装指南](getting-started/installation.md) — 通过 PyPI、源码或 Docker 快速安装
- [快速开始](getting-started/quick-start.md) — 5 分钟在单 GPU 上运行两个 LLM
- [系统架构](core-concepts/architecture.md) — 理解系统设计和组件交互
- [用户指南](user-guide/multi-model.md) — 所有支持场景的使用指南

## 支持的引擎和模型

| 引擎 | 版本 | 注意力类型 | 示例模型 |
|------|------|-----------|---------|
| SGLang | >= v0.4.9（测试至 v0.5.10） | MHA / GQA / MLA | Llama 3.1/3.3, Qwen 2.5, DeepSeek-V3, GPT-OSS |
| vLLM | >= v0.8.4（测试至 v0.19.0） | MHA / GQA / MLA | Llama 3.1/3.3, Qwen 2.5, DeepSeek-V3, GPT-OSS |

## 企业用户

Google、LinkedIn、Intel、AMD、Red Hat、Adobe、Sony、字节跳动、阿里云、腾讯等。

## 引用

如果 kvcached 对您的工作有帮助，请引用我们的论文：

```bibtex
@article{yu2026prism,
  title={Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning},
  author={Yu, Shan and Qiao, Yifan and others},
  journal={OSDI},
  year={2026}
}
```
