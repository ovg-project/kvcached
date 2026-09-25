# TensorRT-LLM Integration Design Document

**Status**: Proposed
**Issue**: #199
**Author**: Community
**Last Updated**: December 2024

## Overview

This document explores the feasibility and design of integrating kvcached with NVIDIA's TensorRT-LLM inference engine.

## Background

### What is TensorRT-LLM?

TensorRT-LLM is NVIDIA's high-performance inference library for LLMs:

- **Optimized Kernels**: Custom CUDA kernels for transformers
- **Quantization**: INT8, FP8, INT4 support
- **Paged Attention**: Built-in paged KV cache
- **Inflight Batching**: Continuous batching support
- **Multi-GPU**: Tensor and pipeline parallelism

### Current Architecture

```
┌─────────────────────────────────────────┐
│          TensorRT-LLM Runtime           │
│  ┌─────────────────────────────────┐   │
│  │       Batch Manager             │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │    KV Cache Manager (Built-in)  │   │
│  │  ┌─────────────────────────┐    │   │
│  │  │   Block Manager         │    │   │
│  │  └─────────────────────────┘    │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │     TensorRT Engine             │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Motivation

### Benefits of kvcached Integration

1. **Multi-Model Serving**: Share GPU memory across TRT-LLM models
2. **Dynamic Memory**: Better memory utilization
3. **Unified Management**: Single memory manager for mixed deployments

### Challenges

1. **C++ Codebase**: TRT-LLM is primarily C++
2. **Tightly Coupled**: KV cache deeply integrated
3. **Proprietary Elements**: Some closed-source components
4. **Different Design**: Block-based vs. page-based

## Feasibility Analysis

### KV Cache in TensorRT-LLM

TRT-LLM's KV cache management:

```cpp
// tensorrt_llm/runtime/kvCacheManager.h
class KVCacheManager {
public:
    void allocate(SizeType32 numBlocks);
    void free(std::vector<SizeType32> const& blockIds);
    ITensor::SharedPtr getKCacheBlockPointers();
    ITensor::SharedPtr getVCacheBlockPointers();

private:
    std::vector<void*> mBlockPointers;
    SizeType32 mBlockSize;
    SizeType32 mNumLayers;
};
```

### Integration Points

1. **Block Allocator**: Replace memory allocation
2. **Pool Manager**: Override pool management
3. **IPC Layer**: Add shared memory support

## Proposed Approaches

### Approach 1: Memory Allocator Hook

Override the CUDA memory allocator used by TRT-LLM.

```cpp
// Custom allocator
class KVCachedAllocator : public nvinfer1::IGpuAllocator {
public:
    void* allocate(uint64_t size, uint64_t alignment, uint32_t flags) override {
        return kvcached_alloc(ctx_, size, alignment);
    }

    void free(void* memory) override {
        kvcached_free(ctx_, memory);
    }

private:
    kvcached_context_t ctx_;
};
```

**Pros**:
- Minimal TRT-LLM changes
- Works with existing code

**Cons**:
- Limited control
- May miss some allocations

### Approach 2: KV Cache Manager Replacement

Replace TRT-LLM's KVCacheManager entirely.

```cpp
class KVCachedCacheManager : public KVCacheManager {
public:
    void allocate(SizeType32 numBlocks) override {
        // Use kvcached for allocation
        for (int i = 0; i < numBlocks; ++i) {
            void* ptr = kvcached_alloc_block(ctx_);
            mBlockPointers.push_back(ptr);
        }
    }

    void free(std::vector<SizeType32> const& blockIds) override {
        for (auto id : blockIds) {
            kvcached_free_block(ctx_, mBlockPointers[id]);
        }
    }
};
```

**Pros**:
- Full control over KV cache
- Better memory management

**Cons**:
- Requires TRT-LLM modification
- Maintenance burden

### Approach 3: External Coordinator

Run kvcached as external service coordinating memory.

```
┌──────────────┐     ┌──────────────┐
│  TRT-LLM #1  │ ◄───│   kvcached   │
└──────────────┘     │   Service    │
                     │              │
┌──────────────┐     │              │
│  TRT-LLM #2  │ ◄───│              │
└──────────────┘     └──────────────┘
```

**Pros**:
- No TRT-LLM modifications
- Works with any version

**Cons**:
- Higher latency
- Complex coordination

## Technical Considerations

### Memory Layout Compatibility

TRT-LLM KV cache layout:
```
[B, num_heads, max_seq_len, head_dim]  // K
[B, num_heads, max_seq_len, head_dim]  // V
```

kvcached layout:
```
[2, num_blocks, block_size, num_heads, head_dim]  // KV interleaved
```

**Conversion needed** or layout adaptation.

### Block Size Alignment

TRT-LLM uses different block sizes than kvcached:
- TRT-LLM: Typically larger blocks (128-256 tokens)
- kvcached: Configurable (default 16 tokens)

Need to align or adapt block sizes.

### Quantization Support

TRT-LLM supports:
- FP16/BF16
- FP8
- INT8 (KV cache)
- INT4 (weights only)

kvcached must handle these formats correctly.

## C API Design

```c
// kvcached C API for TRT-LLM integration
#ifdef __cplusplus
extern "C" {
#endif

typedef struct kvcached_ctx* kvcached_ctx_t;

// Initialization
kvcached_ctx_t kvcached_init_trtllm(
    int device_id,
    size_t max_memory,
    int block_size
);

void kvcached_shutdown_trtllm(kvcached_ctx_t ctx);

// Allocation
void* kvcached_alloc_block_trtllm(kvcached_ctx_t ctx);
void kvcached_free_block_trtllm(kvcached_ctx_t ctx, void* ptr);

// Batch operations
int kvcached_alloc_blocks_trtllm(
    kvcached_ctx_t ctx,
    int num_blocks,
    void** out_ptrs
);

void kvcached_free_blocks_trtllm(
    kvcached_ctx_t ctx,
    void** ptrs,
    int num_blocks
);

// Stats
size_t kvcached_available_memory_trtllm(kvcached_ctx_t ctx);
size_t kvcached_used_memory_trtllm(kvcached_ctx_t ctx);

#ifdef __cplusplus
}
#endif
```

## Implementation Plan

### Phase 1: Research
- [ ] Deep dive into TRT-LLM KV cache code
- [ ] Identify all integration points
- [ ] Test memory allocator hooks

### Phase 2: Prototype
- [ ] Implement C API
- [ ] Create simple allocator hook
- [ ] Test with single model

### Phase 3: Integration
- [ ] Full KV cache manager replacement
- [ ] Multi-model support
- [ ] Performance optimization

### Phase 4: Production
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Upstream contribution attempt

## Performance Expectations

| Metric | Native TRT-LLM | With kvcached |
|--------|----------------|---------------|
| Single model throughput | Baseline | ~95-98% |
| Memory efficiency | Fixed | Dynamic |
| Multi-model capacity | 1x | 2-3x |
| Startup time | Baseline | +100-200ms |

## Risks and Mitigations

### Risk 1: TRT-LLM Version Coupling
- **Risk**: Integration breaks with TRT-LLM updates
- **Mitigation**: Version-specific patches, active monitoring

### Risk 2: Performance Regression
- **Risk**: Added overhead affects latency
- **Mitigation**: Careful benchmarking, optional bypass

### Risk 3: Compatibility Issues
- **Risk**: Memory layout differences cause errors
- **Mitigation**: Thorough testing, format conversion

## Alternatives

### 1. Triton Inference Server Integration
Use Triton's model management instead of direct TRT-LLM integration.

### 2. vLLM with TensorRT Backend
Use vLLM (which has kvcached support) with TensorRT as backend.

### 3. Custom Inference Engine
Build custom engine using TensorRT for computation and kvcached for memory.

## Open Questions

1. Is TRT-LLM's paged attention compatible with kvcached's VMM?
2. How to handle TRT-LLM's prefill/decode optimization with dynamic memory?
3. What's the minimum TRT-LLM version that's feasible to integrate?
4. Should we contribute upstream to TensorRT-LLM?

## References

- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
