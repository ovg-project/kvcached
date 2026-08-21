# 安装指南

## 前置条件

- **Python**：3.9 – 3.13
- **PyTorch**：>= 2.6.0（支持 CUDA 或 ROCm）
- **推理引擎**：SGLang (>= v0.4.9) 或 vLLM (>= v0.8.4)
- **GPU**：支持 CUDA 的 NVIDIA GPU，或支持 ROCm/HIP 的 AMD GPU

kvcached 作为**插件**安装在现有的 SGLang 或 vLLM 环境中，不会替换或冲突现有的推理引擎。

## 通过 PyPI 安装

```bash
pip install kvcached --no-build-isolation
```

!!! note "`--no-build-isolation` 为什么是必须的？"
    kvcached 包含一个需要链接 PyTorch CUDA 库的 C++ 扩展。`--no-build-isolation` 确保构建过程能找到已安装的 PyTorch。

## 从源码安装

```bash
git clone https://github.com/ovg-project/kvcached.git
cd kvcached
pip install -e . --no-build-isolation --no-cache-dir
python tools/dev_copy_pth.py
```

## 使用 Docker

```bash
# vLLM 版本
docker pull ghcr.io/ovg-project/kvcached-vllm:latest

# SGLang 版本
docker pull ghcr.io/ovg-project/kvcached-sglang:latest

# 开发版本（包含 vLLM + SGLang）
docker pull ghcr.io/ovg-project/kvcached-dev:latest
```

## 验证安装

```bash
python -c "import kvcached; print(f'kvcached 版本: {kvcached.__version__}')"
```

## 下一步

- [快速开始](quick-start.md) — 运行你的第一个多模型部署
- [系统架构](../core-concepts/architecture.md) — 理解 kvcached 的工作原理
