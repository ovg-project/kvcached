# Compatibility Guide

This document describes version compatibility, known issues, and troubleshooting for kvcached.

## Supported Versions

### vLLM

| vLLM Version | Status | Notes |
|--------------|--------|-------|
| 0.8.4 - 0.8.5 | Supported | Use `VLLM_USE_V1=1` |
| 0.9.0 - 0.9.2 | Supported | Recommended |
| 0.10.x | Supported | Requires latest kvcached |
| 0.11.x | Experimental | Requires CUDA 12.8+ |

### SGLang

| SGLang Version | Status | Notes |
|----------------|--------|-------|
| 0.4.x | Supported | Use `--disable-radix-cache` |
| 0.5.x | Experimental | May require updates |

### PyTorch

| PyTorch Version | Status | Notes |
|-----------------|--------|-------|
| 2.4.x | Supported | |
| 2.5.x | Supported | |
| 2.6.x | Supported | |
| 2.7.x | Supported | |
| 2.8.x | Known Issue | See below |

## Quantization Support

### FP16/BF16

Full support for standard floating-point formats:
- FP16 (float16): Fully supported
- BF16 (bfloat16): Fully supported

### FP8 Quantization

| Feature | Status | Notes |
|---------|--------|-------|
| FP8 E4M3 | Experimental | May have type mismatch issues |
| FP8 E5M2 | Experimental | Limited testing |
| Qwen-FP8 | Known Issue | See issue #203 |

**Known Issues with FP8:**

1. **Type Mismatch Error** (Issue #214)
   ```
   RuntimeError: Expected tensor of type Float8_e4m3fn but got Float16
   ```
   **Cause**: kvcached may not correctly detect FP8 dtype
   **Workaround**: Ensure model and KV cache dtypes match explicitly

2. **Out-of-Bounds Error** (Issue #203)
   ```
   IndexError: index out of bounds
   ```
   **Cause**: Size calculation differences with FP8 tensors
   **Status**: Under investigation

### FP4/NF4 Quantization

| Feature | Status | Notes |
|---------|--------|-------|
| NF4 (4-bit NormalFloat) | Not Supported | Planned |
| nvfp4 | Not Supported | Issue #214 |

**Note**: 4-bit quantization typically applies to model weights, not KV cache. However, some implementations use quantized KV caches which require special handling.

## Multi-GPU Configuration

### Tensor Parallelism

kvcached supports tensor parallelism with proper configuration:

```bash
# Required for multi-GPU
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1

# Each GPU needs same IPC configuration
export KVCACHED_IPC_NAME=shared_cache
```

### Multi-Instance on Multi-GPU (Issue #221)

Running multiple model instances across multiple GPUs requires careful configuration.

**Scenario**: 4 instances on 2 x A100 GPUs

**Configuration**:
```bash
# Instance 1 & 2 on GPU 0
CUDA_VISIBLE_DEVICES=0 KVCACHED_IPC_NAME=gpu0_cache vllm serve model-a --port 8001 &
CUDA_VISIBLE_DEVICES=0 KVCACHED_IPC_NAME=gpu0_cache vllm serve model-b --port 8002 &

# Instance 3 & 4 on GPU 1
CUDA_VISIBLE_DEVICES=1 KVCACHED_IPC_NAME=gpu1_cache vllm serve model-c --port 8003 &
CUDA_VISIBLE_DEVICES=1 KVCACHED_IPC_NAME=gpu1_cache vllm serve model-d --port 8004 &
```

**Common Issues**:

1. **IPC Name Collision**
   - Each GPU should have a unique IPC name
   - Use `KVCACHED_IPC_NAME` to differentiate

2. **Memory Contention**
   - Set appropriate `KVCACHED_GPU_UTILIZATION` per GPU
   - Monitor with `kvctl watch`

3. **Race Conditions**
   - Stagger instance startup by a few seconds
   - Use controller for coordinated launches

### Troubleshooting Multi-GPU

```bash
# Check all IPC segments
kvctl list

# Monitor specific GPU's segment
kvctl watch kvcached_gpu0_cache

# Clean up stale segments
kvctl delete kvcached_gpu0_cache
```

## Known Issues

### PyTorch 2.8.0 - Undefined Symbol Error

**Issue**: When using PyTorch 2.8.0, you may encounter:
```
undefined symbol: _ZNK2at10TensorBase4nameEv
```

**Cause**: ABI incompatibility between kvcached's C++ bindings and PyTorch 2.8.0.

**Solutions**:
1. **Recommended**: Use PyTorch 2.7.x or earlier
2. **Alternative**: Rebuild kvcached from source with PyTorch 2.8.0:
   ```bash
   pip install -e . --no-build-isolation
   ```

### Dynamic Memory Management Errors

**Issue**: `ValueError: Cannot get N free blocks from the pool`

**Cause**: Race condition when multiple engines compete for memory.

**Solutions**:
1. Reduce concurrent requests
2. Increase `KVCACHED_GPU_UTILIZATION`
3. Use controller for coordinated memory management

### Output Corruption After Long Sequences (Issue #201)

**Issue**: Output becomes corrupted after ~2000 tokens.

**Status**: Under investigation

**Workaround**:
- Limit max sequence length
- Enable sanity checks: `KVCACHED_SANITY_CHECK=true`

## CUDA Version Requirements

| kvcached Feature | Minimum CUDA |
|------------------|--------------|
| Basic functionality | CUDA 11.8 |
| Flash Attention support | CUDA 12.0 |
| vLLM 0.11.x support | CUDA 12.8 |

## GPU Support

### NVIDIA GPUs

| GPU Architecture | Status |
|------------------|--------|
| Ampere (A100, A10) | Fully Supported |
| Hopper (H100, H200) | Fully Supported |
| Ada Lovelace (L40S, RTX 4090) | Fully Supported |
| Blackwell (B100, B200) | Untested |

### AMD GPUs

AMD GPU support is planned but not yet available. See issue #94.

### ARM64 Support

ARM64 (aarch64) support is in progress. See issue #225.

## Container/Kubernetes

### Docker

kvcached works in Docker containers with:
- `/dev/shm` mounted with sufficient size
- NVIDIA runtime configured

```yaml
services:
  model:
    image: your-image
    shm_size: '16gb'  # Important!
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### Kubernetes

For Kubernetes deployments:

1. **Shared Memory**: Ensure adequate `/dev/shm`:
   ```yaml
   volumes:
     - name: shm
       emptyDir:
         medium: Memory
         sizeLimit: 16Gi
   ```

2. **GPU Resources**: Use NVIDIA device plugin:
   ```yaml
   resources:
     limits:
       nvidia.com/gpu: 1
   ```

3. **Dynamic Scaling**: kvcached automatically detects available GPU memory.

See issue #87 for more details on Kubernetes integration.

## Troubleshooting

### Check Versions

```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Check CUDA version
nvcc --version

# Check vLLM version
python -c "import vllm; print(vllm.__version__)"

# Check SGLang version
python -c "import sglang; print(sglang.__version__)"
```

### Verify kvcached Installation

```bash
# Check if C++ extensions are built correctly
python -c "from kvcached import _C; print('C++ extensions OK')"

# Check VMM operations
python -c "from kvcached.vmm_ops import is_available; print(f'VMM: {is_available()}')"
```

### Common Fixes

1. **Rebuild with matching PyTorch**:
   ```bash
   pip uninstall kvcached
   pip install kvcached --no-cache-dir
   ```

2. **Clear pip cache**:
   ```bash
   pip cache purge
   ```

3. **Check shared memory**:
   ```bash
   df -h /dev/shm
   ls -la /dev/shm/kvcached*
   ```

### Debug Logging

Enable detailed logging for troubleshooting:

```bash
export KVCACHED_LOG_LEVEL=DEBUG
```

### Memory Monitoring

```bash
# Real-time memory view
kvctl kvtop

# List all segments
kvctl list

# Watch specific segment
kvctl watch <ipc_name>
```
