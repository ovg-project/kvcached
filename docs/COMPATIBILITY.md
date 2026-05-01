# Compatibility Guide

This document describes version compatibility and known issues with kvcached.

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

### CUDA Version Requirements

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
| Ada Lovelace (L40S) | Fully Supported |
| Blackwell (B100, B200) | Untested |

### AMD GPUs

AMD GPU support is planned but not yet available. See issue #94.

### ARM64 Support

ARM64 (aarch64) support is in progress. See issue #225.

## Memory Requirements

### Minimum GPU Memory

kvcached requires sufficient GPU memory for:
1. Model weights
2. KV cache (managed by kvcached)
3. Activation memory during inference

**Recommended**: At least 2GB free after loading model weights.

### Multi-Model Recommendations

| Total GPU Memory | Recommended Models |
|------------------|-------------------|
| 24GB | 2-3 small models (7B) |
| 48GB | 2 medium models (13B) or 4 small |
| 80GB | 2 large models (70B) or 4+ medium |

## Container/Kubernetes Compatibility

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

3. **Dynamic Scaling**: kvcached automatically detects available GPU memory. When Kubernetes scales GPU resources, kvcached will utilize the new capacity on next initialization.

See issue #87 for more details on Kubernetes integration.

## Troubleshooting Compatibility Issues

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
