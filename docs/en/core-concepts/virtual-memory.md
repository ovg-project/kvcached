# GPU Virtual Memory

## The Problem: Static Memory Allocation

Traditional LLM serving engines manage GPU memory at the application level. At startup, they allocate a large, fixed block of GPU physical memory as the KV cache pool. This creates several issues:

1. **Wasted memory when idle**: Even with zero requests, the allocated memory cannot be used by other processes
2. **No cross-model sharing**: Multiple models cannot dynamically share the same physical memory pool
3. **Over-provisioning required**: Each model must reserve for peak usage, leading to poor average utilization

## The Solution: CUDA Virtual Memory Management

kvcached leverages CUDA's Virtual Memory Management (VMM) APIs to implement OS-style virtual memory for GPU:

```
GPU Memory Layout (per model with kvcached)
┌─────────────────────────────────────────┐
│  Virtual Address Space    (reserved)     │ ← Large, contiguous, no physical cost
├─────────────────────────────────────────┤
│  Mapped Physical Pages    (on-demand)    │ ← Only what's actively used
├─────────────────────────────────────────┤
│  Unmapped Virtual Range   (free)         │ ← Available for future growth
└─────────────────────────────────────────┘
```

### Key CUDA VMM Operations

| Operation | CUDA API | Purpose |
|-----------|----------|---------|
| Reserve virtual space | `cuMemAddressReserve` | Reserve contiguous virtual address range (no physical memory) |
| Allocate physical page | `cuMemCreate` | Create a 2MB physical memory page |
| Map page | `cuMemMap` + `cuMemSetAccess` | Link a physical page to a virtual address |
| Unmap page | `cuMemUnmap` | Disconnect physical page from virtual address |
| Free physical page | `cuMemRelease` | Return physical memory to the system |

### Page Size: 2MB Granularity

kvcached operates at **2MB page granularity**, the minimum supported by CUDA VMM. This provides:

- Fine-grained memory control (allocate/free in 2MB increments)
- Low fragmentation (physical pages can be freely remapped)
- Fast redistribution (millisecond-level overhead for page operations)

## How It Works

### Virtual Address Reservation

At engine startup, kvcached reserves a large virtual address space for each model:

```python
# Conceptual flow (simplified)
virtual_space = cuMemAddressReserve(size=MAX_KV_CACHE_SIZE)
# No physical memory consumed yet!
```

The engine sees this as if it has dedicated GPU memory, but no physical resources are committed.

### On-Demand Physical Allocation

When the engine actually needs memory (new requests arrive, KV cache grows):

```python
# When new KV blocks are needed:
physical_page = cuMemCreate(size=2MB)
cuMemMap(virtual_address, physical_page)
cuMemSetAccess(virtual_address, access_flags)
```

### Elastic Reclamation

When memory is no longer needed (requests complete, model goes idle):

```python
# When KV blocks are freed:
cuMemUnmap(virtual_address)
cuMemRelease(physical_page)
# Physical memory now available for other models!
```

## Optimizations

### Pre-allocation Buffer

To avoid the latency of `cuMemCreate` on the critical path, kvcached maintains a **pre-allocation thread** that asynchronously prepares a small buffer of GPU pages:

- New page requests draw from this buffer (fast path)
- Released pages return to the buffer instead of being immediately freed
- Pages are only physically freed when the buffer exceeds its limit or memory must be reclaimed

### Contiguous Virtual Layout

Mainstream engines maintain separate K and V tensors per layer, requiring `2L` page allocations (where L = number of layers). kvcached reorganizes the memory layout so all layers' K and V vectors for a token are stored in contiguous virtual space, requiring only **one batch allocation** for all pages.

### Elastic Tensor (eTensor)

kvcached introduces the eTensor abstraction via PyTorch's extension interface:

- Behaves exactly like a regular PyTorch tensor
- Internally backed by virtual memory with on-demand physical mapping
- Compatible with CUDA graph optimizations
- Requires no modifications to attention kernels

## AMD ROCm Support

On AMD GPUs with ROCm/HIP builds, kvcached uses the equivalent HIP VMM APIs:

- `hipMemAddressReserve` / `hipMemMap` / `hipMemUnmap`
- Defaults to **non-contiguous** KV cache layout (matching ROCm attention backend expectations)
- Configurable via `KVCACHED_CONTIGUOUS_LAYOUT=true|false`

## Further Reading

- [Architecture](architecture.md) — How virtual memory fits into the overall system
- [Prefix Caching](prefix-caching.md) — How prefix caching works with elastic memory
- [Environment Variables](../configuration/environment.md) — Memory-related configuration options
