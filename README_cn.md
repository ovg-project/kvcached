[English](README.md) | 中文版

<div align="center">
  <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/logo-v2.svg" alt="kvcached logo" height="96" />
</div>

<h2 align="center">让 GPU 共享变得灵活且简单</h2>

kvcached（KV cache daemon）是一个面向共享 GPU 上 LLM 推理/训练的 KV 缓存库。通过将操作系统级别的**虚拟内存**抽象引入 LLM 系统，它实现**弹性、按需**的 KV 缓存分配，提升动态工作负载下的 GPU 利用率。

kvcached 通过将 GPU 虚拟地址与物理内存分配解耦来实现这一目标。推理引擎在初始时仅保留虚拟内存，之后在缓存被实际使用时再用物理 GPU 内存进行支撑。这种解耦实现了按需分配和灵活共享。

### 核心特性

- **弹性 KV 缓存**：根据实时负载动态分配和回收 KV 内存
- **GPU 虚拟内存**：通过运行时映射将逻辑 KV 与物理 GPU 内存解耦
- **内存控制 CLI**：使用 kvcached CLI 管理内存限制
- **前端路由与休眠模式**：将请求路由到目标模型，空闲模型自动休眠
- **支持主流推理引擎**：透明集成 SGLang 和 vLLM
- **前缀缓存**：支持自动前缀缓存（APC），可配置内存上限

## 支持的引擎和模型

| 引擎 | 版本 | 注意力类型 | 示例模型 |
|------|------|-----------|---------|
| SGLang | >= v0.4.9（测试至 v0.5.10） | MHA / GQA / MLA | Llama 3.1/3.3, Qwen 2.5, DeepSeek-V3 等 |
| vLLM | >= v0.8.4（测试至 v0.19.0） | MHA / GQA / MLA | Llama 3.1/3.3, Qwen 2.5, DeepSeek-V3 等 |

## 应用场景

| 场景 | 说明 |
|------|------|
| **多 LLM 服务** | 多个 LLM 弹性共享 GPU 内存，无需刚性分区 |
| **无服务器 LLM** | 仅在需要时分配 KV 缓存，支持模型按需启停 |
| **复合 AI 系统** | 在有限硬件上跨专用模型弹性分配内存 |
| **GPU 工作负载混部** | LLM 推理与训练、微调或视觉模型共存 |

## 安装

### 通过 PyPI 安装

```bash
pip install kvcached --no-build-isolation
```

### 从源码安装

```bash
pip install -e . --no-build-isolation --no-cache-dir
python tools/dev_copy_pth.py
```

### 使用 Docker

```bash
docker pull ghcr.io/ovg-project/kvcached-sglang:latest
docker pull ghcr.io/ovg-project/kvcached-vllm:latest
docker pull ghcr.io/ovg-project/kvcached-dev:latest
```

## 文档

完整文档位于 [`docs/`](./docs) 目录下，可构建为本地文档网站：

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

文档涵盖：
- [系统架构](docs/zh/core-concepts/architecture.md)
- [快速开始](docs/zh/getting-started/quick-start.md)
- [用户指南](docs/zh/user-guide/multi-model.md)
- [配置参考](docs/zh/configuration/environment.md)
- [性能基准](docs/zh/performance/benchmarks.md)
- [常见问题](docs/zh/faq.md)

> kvcached 也在 [DeepWiki](https://deepwiki.com/ovg-project/kvcached) 上提供 AI 驱动的文档探索。

## 贡献

我们欢迎任何形式的贡献和合作。

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 联系方式

通过 [Slack 频道](https://join.slack.com/t/ovg-project/shared_invite/zt-3fr01t8s7-ZtDhHSJQ00hcLHgwKx3Dmw)或 [GitHub Issues](https://github.com/ovg-project/kvcached/issues) 与我们联系。

## 引用

```bibtex
@article{yu2026prism,
  title={Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning},
  author={Yu, Shan and Qiao, Yifan and Ma, Mingyuan and others},
  journal={OSDI},
  year={2026}
}
```

## 许可证

kvcached 使用 Apache License 2.0 许可证。详见 [LICENSE](LICENSE)。
[English](README.md) | 中文版

<div align="center">
  <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/logo-v2.svg" alt="kvcached logo" height="96" />

  <br>
  <br>
  <p>
    <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.9%E2%80%933.13-blue"></a>
    <img alt="Engines" src="https://img.shields.io/badge/Engines-SGLang%20%7C%20vLLM-blueviolet">
    <a href="https://arxiv.org/abs/2505.04021"><img alt="arXiv: Multi LLM Serving" src="https://img.shields.io/badge/arXiv-Multi%20LLM%20Serving-b31b1b?logo=arxiv&logoColor=white&labelColor=555555"></a>
    <a href="https://join.slack.com/t/ovg-project/shared_invite/zt-3fr01t8s7-ZtDhHSJQ00hcLHgwKx3Dmw"><img alt="Slack Join" src="https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&logoColor=white&labelColor=555555"></a>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  </p>
</div>

<h2 align="center">让 GPU 共享变得灵活且简单</h2>

kvcached（KV cache daemon）是一个面向共享 GPU 上 LLM 推理/训练的 KV 缓存库。通过将操作系统级别的**虚拟内存**抽象引入 LLM 系统，它实现**弹性、按需**的 KV 缓存分配，提升动态工作负载下的 GPU 利用率。

kvcached 通过将 GPU 虚拟地址与物理内存分配解耦来实现这一目标。它允许推理引擎在初始时仅保留虚拟内存，之后在缓存被实际使用时再用物理 GPU 内存进行支撑。这种解耦实现了按需分配和灵活共享，在动态和混合工作负载下带来更好的 GPU 内存利用率。

### 核心特性

- **弹性 KV 缓存**：根据实时负载动态分配和回收 KV 内存
- **GPU 虚拟内存**：通过运行时映射将逻辑 KV 与物理 GPU 内存解耦
- **内存控制 CLI**：使用 kvcached CLI 管理内存限制
- **前端路由与休眠模式**：将请求路由到目标模型，空闲模型自动休眠
- **支持主流推理引擎**：透明集成 SGLang 和 vLLM
- **前缀缓存**：支持自动前缀缓存（APC），可配置内存上限

## 支持的引擎和模型

| 引擎 | 版本 | 注意力类型 | 示例模型 |
|------|------|-----------|---------|
| SGLang | >= v0.4.9（测试至 v0.5.10） | MHA / GQA / MLA | Llama 3.1/3.3, Qwen 2.5, DeepSeek-V3 等 |
| vLLM | >= v0.8.4（测试至 v0.19.0） | MHA / GQA / MLA | Llama 3.1/3.3, Qwen 2.5, DeepSeek-V3 等 |

## 应用场景

| 场景 | 说明 |
|------|------|
| **多 LLM 服务** | 多个 LLM 弹性共享 GPU 内存，无需刚性分区，提高利用率 |
| **无服务器 LLM** | 仅在需要时分配 KV 缓存，支持模型按需启停 |
| **复合 AI 系统** | 在有限硬件上跨专用模型弹性分配内存 |
| **GPU 工作负载混部** | LLM 推理与训练、微调或视觉模型共存 |

## 安装

### 前置条件

- Python（测试 3.9 - 3.13）
- SGLang（测试 v0.5.10）或 vLLM（测试 v0.19.0）

### 通过 PyPI 安装

```bash
pip install kvcached --no-build-isolation
```

### 从源码安装

```bash
pip install -e . --no-build-isolation --no-cache-dir
python tools/dev_copy_pth.py
```

### 使用 Docker

```bash
docker pull ghcr.io/ovg-project/kvcached-sglang:latest   # kvcached-v0.1.5-sglang-v0.5.10
docker pull ghcr.io/ovg-project/kvcached-vllm:latest     # kvcached-v0.1.5-vllm-v0.19.0
docker pull ghcr.io/ovg-project/kvcached-dev:latest      # 开发版（包含两个引擎）
```

## 文档

完整文档位于 [`docs/`](./docs) 目录下，可构建为本地文档网站：

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

文档涵盖：
- [系统架构](docs/zh/core-concepts/architecture.md) — 系统设计和 GPU 虚拟内存模型
- [快速开始](docs/zh/getting-started/quick-start.md) — 安装、快速开始、Docker 部署
- [用户指南](docs/zh/user-guide/multi-model.md) — 多模型服务、内存控制、路由、休眠管理
- [配置参考](docs/zh/configuration/environment.md) — 环境变量和引擎选项
- [性能基准](docs/zh/performance/benchmarks.md) — TTFT 基准测试和调优指南
- [常见问题](docs/zh/faq.md) — 常见问题和故障排除

> kvcached 也在 [DeepWiki](https://deepwiki.com/ovg-project/kvcached) 上提供 AI 驱动的文档探索。

## 测试

启用 kvcached：

```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
```

运行引擎基准测试：

```bash
# vLLM
vllm serve meta-llama/Llama-3.2-1B-Instruct --port=12346
vllm bench serve --model meta-llama/Llama-3.2-1B-Instruct --request-rate 10 --num-prompts 1000 --port 12346
```

## 贡献

我们欢迎任何形式的贡献和合作。

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 联系方式

通过 [Slack 频道](https://join.slack.com/t/ovg-project/shared_invite/zt-3fr01t8s7-ZtDhHSJQ00hcLHgwKx3Dmw)或 [GitHub Issues](https://github.com/ovg-project/kvcached/issues) 与我们联系。

## 引用

```bibtex
@article{yu2026prism,
  title={Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning},
  author={Yu, Shan and Qiao, Yifan and Ma, Mingyuan and others},
  journal={OSDI},
  year={2026}
}

@article{xing2025towards,
  title={Towards Efficient and Practical GPU Multitasking in the Era of LLM},
  author={Xing, Jiarong and Qiao, Yifan and Mo, Simon and others},
  journal={arXiv preprint arXiv:2508.08448},
  year={2025}
}
```
