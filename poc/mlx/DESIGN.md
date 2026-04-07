# KVCached on Apple MLX — Design & PoC Results

## PoC Results (Validated)

All tests pass on Linux. The same mmap logic applies to macOS ARM64.

### What We Proved

1. **mmap(MAP_FIXED) atomically remaps pages in a reserved VA range** — data appears
   immediately at the new location, old mapping is replaced cleanly.

2. **Base pointer is stable across map/unmap cycles** — the VA range start address
   never changes. This means a Metal `newBufferWithBytesNoCopy` wrapping this
   pointer stays valid as we remap pages underneath.

3. **True zero-copy shared memory** (memfd/shm_open + MAP_SHARED) — writes through
   the "physical page" pointer are visible through the arena pointer and vice versa.
   No memcpy needed.

4. **~40K-90K map/unmap ops/sec** — each remap costs ~11-25μs. This is fast enough
   for KV cache block management (typical inference does O(10) page ops per request).

5. **Block-level elastic allocation works** — the KVCacheManager correctly maps pages
   on first block allocation, unmaps when all blocks in a page are freed, and reclaims
   memory between requests.

### What Needs macOS Validation (test_metal_vmm.m)

The Metal GPU test (`test_metal_vmm.m`) validates the critical assumption:
**After mmap(MAP_FIXED) remaps pages within a Metal buffer's VA range,
does the GPU see the remapped data?**

Expected: **Yes**, because Apple Silicon UMA shares page tables between CPU and GPU.
The test dispatches a Metal compute kernel that reads from the buffer before/after
page remapping and verifies the GPU sees the correct data.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  MLX Inference Engine (mlx-lm)                       │
│  KV cache uses mx.array backed by arena              │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────┴────────────────────────────────┐
│  Python Layer (mlx_vmm.py)                           │
│                                                      │
│  MLXKVCacheManager ─── block alloc/free              │
│  MLXArena ─────────── VA reservation + page mapping  │
│  PhysicalPage ─────── shm-backed physical pages      │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────┴────────────────────────────────┐
│  C/ObjC Layer (metal_vmm.c + metal_vmm_metal.m)     │
│                                                      │
│  mmap(MAP_ANONYMOUS) ─── reserve VA range            │
│  shm_open/memfd ──────── physical page backing       │
│  mmap(MAP_FIXED) ─────── page mapping/unmapping      │
│  MTLBuffer(bytesNoCopy) ─ Metal GPU access           │
└──────────────────────────────────────────────────────┘
                      │
┌─────────────────────┴────────────────────────────────┐
│  Apple Silicon Hardware                              │
│                                                      │
│  Unified Memory ── CPU & GPU share physical memory   │
│  Shared Page Tables ── mmap changes visible to GPU   │
│  HW Cache Coherence ── no manual flushes needed      │
└──────────────────────────────────────────────────────┘
```

## Key Mappings: CUDA VMM → macOS VMM

| CUDA VMM API | macOS Equivalent | Notes |
|---|---|---|
| `cuMemAddressReserve` | `mmap(MAP_ANONYMOUS)` | Reserve VA range |
| `cuMemCreate` | `shm_open` + `ftruncate` | Allocate physical page |
| `cuMemMap` | `mmap(MAP_FIXED \| MAP_SHARED, fd)` | Map page into VA |
| `cuMemUnmap` | `mmap(MAP_FIXED \| MAP_ANONYMOUS)` | Replace with zeros |
| `cuMemSetAccess` | N/A | UMA — always accessible |
| `cuMemRelease` | `close(fd)` | Release physical page |
| `torch::from_blob(vaddr)` | `MTLBuffer(bytesNoCopy)` → `mx.array` | Tensor creation |

## Page Size Considerations

| Platform | HW Page | KVCached Page | Rationale |
|---|---|---|---|
| CUDA (current) | 2 MB (granularity) | 2 MB | CUDA VMM minimum |
| macOS ARM64 | 16 KB | 64-256 KB recommended | Finer granularity OK; larger pages amortize mmap overhead |
| macOS ARM64 | 16 KB | 2 MB (compat mode) | Matches CUDA for feature parity |

## Synchronization Contract

**CRITICAL**: Page remapping MUST NOT occur while the GPU is reading the affected region.

```
Safe sequence:
  1. mx.eval(pending_computation)     # flush MLX graph
  2. mx.synchronize()                 # wait for GPU idle
  3. arena.map(page_idx, new_page)    # remap pages
  4. # GPU can now access new data
```

In KVCached's architecture, this naturally aligns with the scheduler:
- Pages are mapped during block allocation (before inference starts)
- Pages are unmapped during block deallocation (after inference completes)
- The scheduler already serializes allocation/computation

## Implementation Roadmap

### Phase 1: Core VMM (this PoC, done)
- [x] mmap-based VA reservation
- [x] memfd/shm page backing
- [x] mmap(MAP_FIXED) page mapping
- [x] Block-level KV cache manager
- [x] C implementation (metal_vmm.c)
- [x] Python implementation (mlx_vmm.py)
- [ ] Metal buffer wrapping (macOS test needed)
- [ ] MLX array creation from Metal buffer

### Phase 2: C++ Extension Module
- [ ] Build as Python extension (like current vmm_ops)
- [ ] Metal/ObjC++ for GPU buffer management
- [ ] Python bindings via pybind11 or nanobind
- [ ] Integrate with existing kvcached Python layer

### Phase 3: mlx-lm Integration
- [ ] Analyze mlx-lm's KVCache class
- [ ] Create ElasticKVCache wrapper
- [ ] Autopatch mechanism (like vLLM/SGLang integration)
- [ ] Prefix caching support

### Phase 4: Multi-model & Memory Pressure
- [ ] Shared memory coordinator between model instances
- [ ] macOS memory pressure notifications (dispatch_source)
- [ ] Cooperative eviction with OS VM system

## Files

```
poc/mlx/
├── DESIGN.md              # This document
├── Makefile               # Build system
├── metal_vmm.h            # C API header
├── metal_vmm.c            # Core VMM implementation (portable)
├── metal_vmm_metal.m      # Metal buffer integration (macOS)
├── test_vmm_remap.c       # C VMM validation test (Linux/macOS)
├── test_metal_vmm.m       # Metal GPU validation test (macOS)
├── mlx_vmm.py             # Python VMM + KV cache manager
└── test_mlx_vmm.py        # Python test suite
```
