# FAQ

## General

### What is kvcached?

kvcached (KV cache daemon) is a GPU virtual memory management library for LLM serving. It enables elastic KV cache allocation so multiple LLMs can share GPU memory dynamically.

### Does kvcached replace vLLM or SGLang?

No. kvcached is a **plugin** that works alongside existing serving engines. It transparently replaces the memory management layer while keeping all other engine functionality intact.

### What GPUs are supported?

- **NVIDIA**: Any GPU with CUDA support (requires CUDA VMM APIs, available since CUDA 10.2+)
- **AMD**: GPUs with ROCm/HIP support

## Installation

### Why do I need `--no-build-isolation`?

kvcached includes a C++ extension (`vmm_ops`) that links against PyTorch's CUDA libraries. Build isolation would create a separate environment without access to your PyTorch installation, causing the build to fail.

### Can I use kvcached with multiple Python versions?

Yes. kvcached supports Python 3.9 through 3.13.

## Usage

### Do I need to set `--gpu-memory-utilization`?

No. When kvcached is enabled, it automatically manages GPU memory allocation based on actual demand. Setting a memory utilization limit is unnecessary and may conflict with kvcached's elastic management.

### Why is prefix caching enabled by default?

kvcached supports prefix caching with a bounded memory budget (default: 16000 tokens). This provides prefix reuse benefits while maintaining memory elasticity. You can adjust or disable it via `KVCACHED_MAX_CACHED_TOKENS`.

### How do I verify kvcached is working?

1. Check startup logs for patching messages
2. Use `nvidia-smi` to observe that memory usage tracks actual load (not statically allocated)
3. Use `kvtop` for a real-time view of KV cache memory usage

## AMD / ROCm

### Why does kvcached default to non-contiguous layout on ROCm?

The contiguous KV cache layout produces strided per-layer tensors that vLLM's ROCm attention backend cannot read correctly. The non-contiguous layout matches what the backend expects. You can override with `KVCACHED_CONTIGUOUS_LAYOUT=true|false`, but contiguous is not recommended on ROCm.

## Troubleshooting

### kvcached patches are not being applied

Ensure both environment variables are set:

```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
```

Also verify that `kvcached_autopatch.pth` is in your `site-packages` directory. If installed from source, run `python tools/dev_copy_pth.py`.

### Out of memory errors with multiple models

Check that the combined memory demand of all models does not exceed physical GPU capacity. Use `kvctl` to set memory limits:

```bash
kvctl
kvcached> limit-percent VLLM 50
kvcached> limit-percent SGLANG 50
```

### Engine version not supported

kvcached maintains patches for specific engine version ranges. Check the [supported versions table](core-concepts/autopatch.md) and upgrade/downgrade your engine if needed. If you encounter issues with a specific version, [open an issue](https://github.com/ovg-project/kvcached/issues).
