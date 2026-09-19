# Troubleshooting Guide

This guide helps diagnose and resolve common issues with kvcached.

## Quick Diagnostics

Run these commands to check your setup:

```bash
# 1. Verify installation
python -c "from kvcached import _C; print('OK')"

# 2. Check VMM support
python -c "from kvcached.vmm_ops import is_available; print(f'VMM: {is_available()}')"

# 3. Check GPU
nvidia-smi

# 4. Check shared memory
df -h /dev/shm
```

## Common Issues

### 1. ImportError: No module named 'kvcached._C'

**Symptoms**:
```
ImportError: No module named 'kvcached._C'
```

**Causes**:
- C++ extensions not built
- PyTorch version mismatch

**Solutions**:
```bash
# Rebuild from source
pip uninstall kvcached
pip install kvcached --no-build-isolation

# Or rebuild in development mode
pip install -e . --no-build-isolation
```

### 2. FileNotFoundError: /dev/shm/...

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: '/dev/shm/kvcached_...'
```

**Causes**:
- IPC segment was deleted while in use
- Race condition between multiple engines
- Improper shutdown

**Solutions**:
```bash
# Clean up stale segments
rm /dev/shm/kvcached_*

# Restart engines
# Use unique IPC names for each engine group
export KVCACHED_IPC_NAME=my_unique_name
```

### 3. ValueError: Cannot get N free blocks from the pool

**Symptoms**:
```
ValueError: Cannot get 31 free blocks from the pool
```

**Causes**:
- GPU memory exhausted
- Too many concurrent requests
- Memory fragmentation

**Solutions**:
```bash
# Increase GPU utilization limit
export KVCACHED_GPU_UTILIZATION=0.98

# Or reduce concurrent requests
# Or use smaller batch sizes
```

**Monitoring**:
```bash
# Check current memory usage
kvctl kvtop
```

### 4. RuntimeError: CUDA out of memory

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

**Causes**:
- Model weights + KV cache exceed GPU memory
- Other processes using GPU

**Solutions**:
```bash
# Check GPU memory usage
nvidia-smi

# Free other processes
# Or reduce kvcached allocation
export KVCACHED_GPU_UTILIZATION=0.80

# Or use smaller models
```

### 5. AssertionError in block allocation

**Symptoms**:
```
AssertionError: block_ids is not None and len(block_ids) == num_blocks
```

**Causes**:
- Race condition in memory allocation
- Concurrent engines competing for memory

**Solutions**:
```bash
# Use controller for coordinated allocation
python -m kvcached.controller --config config.yaml

# Or stagger engine startup
```

### 6. Type Mismatch with FP8 Models

**Symptoms**:
```
RuntimeError: Expected tensor of type Float8_e4m3fn but got Float16
```

**Causes**:
- FP8 dtype not correctly detected
- Mixed precision configuration

**Solutions**:
```bash
# Ensure consistent dtype
# Check model configuration

# Disable kvcached for FP8 temporarily
unset ENABLE_KVCACHED
```

### 7. Output Corruption with Long Sequences

**Symptoms**:
- Garbage output after ~2000 tokens
- Repetitive or nonsensical text

**Causes**:
- Memory corruption (under investigation)
- Possible overflow in block tracking

**Workaround**:
```bash
# Enable sanity checks
export KVCACHED_SANITY_CHECK=true

# Limit max sequence length
vllm serve model --max-model-len 2000
```

### 8. Slow Performance with kvcached

**Symptoms**:
- Higher latency than without kvcached
- Reduced throughput

**Causes**:
- Dynamic allocation overhead
- Page fault handling

**Solutions**:
```bash
# Pre-allocate pages
export KVCACHED_PAGE_PREALLOC_ENABLED=true
export KVCACHED_MIN_RESERVED_PAGES=10

# Use larger page sizes for stable workloads
export KVCACHED_PAGE_SIZE_MB=4
```

### 9. IPC Segment Access Denied

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied: '/dev/shm/kvcached_...'
```

**Causes**:
- Different user created the segment
- Docker permission issues

**Solutions**:
```bash
# Remove old segments (as correct user)
sudo rm /dev/shm/kvcached_*

# In Docker, ensure proper user mapping
docker run --user $(id -u):$(id -g) ...
```

### 10. kvctl Command Not Found

**Symptoms**:
```
kvctl: command not found
```

**Solutions**:
```bash
# Ensure kvcached is installed
pip show kvcached

# Add to PATH if installed with --user
export PATH="$HOME/.local/bin:$PATH"

# Or use module directly
python -m kvcached.cli.kvctl list
```

## Debugging

### Enable Debug Logging

```bash
export KVCACHED_LOG_LEVEL=DEBUG
```

### Check Memory State

```python
from kvcached.cli.utils import get_all_ipc_names, RwLockedShm

# List all segments
for name in get_all_ipc_names():
    print(name)

# Check specific segment
with RwLockedShm(f"/dev/shm/{name}", "rb") as f:
    data = f.read()
    print(len(data), "bytes")
```

### Trace Allocations

```python
import logging
logging.getLogger("kvcached").setLevel(logging.DEBUG)

# Your code here
```

## Performance Tuning

### For Low Latency

```bash
export KVCACHED_PAGE_PREALLOC_ENABLED=true
export KVCACHED_MIN_RESERVED_PAGES=20
export KVCACHED_CONTIGUOUS_LAYOUT=true
```

### For Maximum Memory Efficiency

```bash
export KVCACHED_GPU_UTILIZATION=0.98
export KVCACHED_PAGE_SIZE_MB=2
export KVCACHED_MIN_RESERVED_PAGES=2
```

### For Multi-Model

```bash
export KVCACHED_GPU_UTILIZATION=0.95
export KVCACHED_PAGE_PREALLOC_ENABLED=true
```

## Getting Help

If you can't resolve your issue:

1. **Search existing issues**: https://github.com/ovg-project/kvcached/issues

2. **Open a new issue** with:
   - kvcached version: `pip show kvcached`
   - PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - CUDA version: `nvcc --version`
   - GPU model: `nvidia-smi -L`
   - Full error traceback
   - Steps to reproduce

3. **Include debug output**:
   ```bash
   export KVCACHED_LOG_LEVEL=DEBUG
   # Run your failing command
   # Copy the full output
   ```
