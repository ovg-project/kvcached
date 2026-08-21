# 常见问题

## 基本问题

### kvcached 是什么？

kvcached 是一个面向 LLM 推理的 GPU 虚拟内存管理库，让多个 LLM 能够动态共享 GPU 内存。

### kvcached 会替换 vLLM 或 SGLang 吗？

不会。kvcached 是一个**插件**，与现有推理引擎协同工作。它只透明替换内存管理层，其他引擎功能完全不受影响。

## 安装问题

### 为什么需要 `--no-build-isolation`？

kvcached 的 C++ 扩展需要链接 PyTorch 的 CUDA 库。构建隔离会创建一个无法访问 PyTorch 的独立环境，导致构建失败。

## 使用问题

### 需要设置 `--gpu-memory-utilization` 吗？

不需要。启用 kvcached 后，它会根据实际需求自动管理 GPU 内存分配。

### 如何验证 kvcached 正在工作？

1. 检查启动日志中的补丁信息
2. 使用 `nvidia-smi` 观察内存使用量随负载变化（而非静态分配）
3. 使用 `kvtop` 查看 KV 缓存实时内存使用

## 故障排除

### 补丁未被应用

确保两个环境变量都已设置：
```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
```

### 多模型运行时内存不足

使用 `kvctl` 设置每个模型的内存上限：
```bash
kvctl
kvcached> limit-percent VLLM 50
kvcached> limit-percent SGLANG 50
```
