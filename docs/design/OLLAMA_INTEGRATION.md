# Ollama Integration Design Document

**Status**: Proposed
**Issue**: #81
**Author**: Community
**Last Updated**: December 2024

## Overview

This document proposes adding Ollama integration to kvcached, enabling dynamic GPU memory management for Ollama's LLM serving capabilities.

## Background

### What is Ollama?

Ollama is a popular tool for running LLMs locally. Key characteristics:

- **Simple CLI**: `ollama run llama2`
- **Model Management**: Download, manage, and run models
- **REST API**: OpenAI-compatible endpoints
- **Cross-Platform**: macOS, Linux, Windows
- **llama.cpp Backend**: Uses llama.cpp for inference

### Current Architecture

```
┌─────────────────────────────────────┐
│           Ollama CLI                │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│         Ollama Server               │
│  ┌─────────────────────────────┐   │
│  │      Model Manager          │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │     llama.cpp Backend       │   │
│  │  ┌───────────────────────┐  │   │
│  │  │    KV Cache (static)  │  │   │
│  │  └───────────────────────┘  │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

## Motivation

### Problems with Current Ollama

1. **Static Memory Allocation**: Each model reserves fixed GPU memory
2. **Model Switching Overhead**: Loading/unloading models is slow
3. **Limited Concurrency**: Cannot efficiently run multiple models
4. **Memory Waste**: Idle models hold GPU memory

### Benefits of kvcached Integration

1. **Dynamic Memory**: Share GPU memory across models
2. **Fast Model Switching**: Keep models warm with minimal memory
3. **Multi-Model Support**: Run more models concurrently
4. **Memory Efficiency**: Release memory when not needed

## Integration Approach

### Option 1: llama.cpp Patch (Recommended)

Modify llama.cpp's KV cache allocator to use kvcached.

```cpp
// llama.cpp modification
struct llama_kv_cache {
    #ifdef USE_KVCACHED
    kvcached_handle_t kvcached;
    #else
    std::vector<ggml_tensor*> k;
    std::vector<ggml_tensor*> v;
    #endif
};
```

**Pros**:
- Works with any llama.cpp frontend
- Minimal Ollama changes
- Benefits other llama.cpp users

**Cons**:
- Requires llama.cpp fork or upstream changes
- C++ implementation complexity

### Option 2: Ollama Server Patch

Intercept memory allocation in Ollama's Go server.

```go
// Ollama modification
type KVCache struct {
    kvcached *kvcached.Manager
    // ... existing fields
}

func (c *KVCache) Allocate(size int) error {
    return c.kvcached.Alloc(size)
}
```

**Pros**:
- Go-level control
- Can add Ollama-specific optimizations

**Cons**:
- Tied to Ollama internals
- May break with Ollama updates

### Option 3: External Memory Manager

Run kvcached as a separate service that Ollama communicates with.

```
┌─────────────┐     ┌─────────────┐
│   Ollama    │ ◄───│  kvcached   │
│   Server    │     │   Service   │
└─────────────┘     └─────────────┘
       │                   │
       └───────┬───────────┘
               ▼
       ┌───────────────┐
       │  GPU Memory   │
       └───────────────┘
```

**Pros**:
- No Ollama modifications
- Can manage multiple Ollama instances

**Cons**:
- IPC overhead
- Complex coordination

## Technical Design

### Memory Mapping for llama.cpp

llama.cpp uses GGML tensors for KV cache. Integration requires:

1. **Tensor Interception**: Replace GGML allocator
2. **Memory Mapping**: Map kvcached pages to GGML tensors
3. **Synchronization**: Handle concurrent access

```cpp
class KVCachedAllocator : public ggml_backend_buffer_type {
public:
    void* alloc(size_t size) override {
        return kvcached_alloc(ctx_, size);
    }

    void free(void* ptr) override {
        kvcached_free(ctx_, ptr);
    }

private:
    kvcached_ctx_t ctx_;
};
```

### API Design

```c
// C API for llama.cpp integration
typedef struct kvcached_ctx* kvcached_ctx_t;

kvcached_ctx_t kvcached_init(int device_id);
void kvcached_shutdown(kvcached_ctx_t ctx);

void* kvcached_alloc(kvcached_ctx_t ctx, size_t size);
void kvcached_free(kvcached_ctx_t ctx, void* ptr);

size_t kvcached_available(kvcached_ctx_t ctx);
void kvcached_set_limit(kvcached_ctx_t ctx, size_t limit);
```

### IPC Protocol

For external service approach:

```protobuf
service KVCached {
    rpc Allocate(AllocRequest) returns (AllocResponse);
    rpc Free(FreeRequest) returns (FreeResponse);
    rpc GetStats(StatsRequest) returns (StatsResponse);
}

message AllocRequest {
    int64 size = 1;
    string model_id = 2;
}

message AllocResponse {
    int64 handle = 1;
    int64 gpu_ptr = 2;
}
```

## Configuration

### Environment Variables

```bash
# Enable kvcached for Ollama
export OLLAMA_KVCACHED_ENABLED=true

# GPU memory limit for all Ollama models
export OLLAMA_KVCACHED_MEMORY_GB=20

# Path to kvcached socket (for external service)
export OLLAMA_KVCACHED_SOCKET=/tmp/kvcached.sock
```

### Ollama Config

```yaml
# ~/.ollama/config.yaml
kvcached:
  enabled: true
  memory_limit: 20GB
  eviction_policy: lru
```

## Implementation Plan

### Phase 1: Research & Prototype
- [ ] Analyze llama.cpp memory management
- [ ] Prototype GGML allocator replacement
- [ ] Test with single model

### Phase 2: Core Integration
- [ ] Implement C API for llama.cpp
- [ ] Handle model loading/unloading
- [ ] Add memory limit enforcement

### Phase 3: Ollama Integration
- [ ] Fork/patch llama.cpp
- [ ] Update Ollama build
- [ ] Add configuration options

### Phase 4: Testing & Polish
- [ ] Multi-model testing
- [ ] Performance benchmarks
- [ ] Documentation

## Challenges

### 1. GGML Tensor Format

llama.cpp uses GGML's tensor format, which differs from PyTorch:
- Custom memory layout
- Different quantization types
- No CUDA managed memory

**Solution**: Create GGML-compatible allocator that wraps kvcached.

### 2. Quantization Support

llama.cpp supports many quantization formats (Q4_0, Q4_1, Q5_K, etc.):
- Variable bit widths
- Block-based quantization
- Different memory requirements

**Solution**: Calculate actual memory requirements per quantization type.

### 3. Context Window Management

llama.cpp manages context windows differently:
- Sliding window attention
- Ring buffer for long contexts
- Different from vLLM/SGLang

**Solution**: Adapt page allocation to match llama.cpp patterns.

## Performance Expectations

| Scenario | Without kvcached | With kvcached |
|----------|-----------------|---------------|
| Single model | Baseline | ~5% overhead |
| 2 models, alternating | 2-5s switch | <100ms switch |
| 3+ models | OOM | Works |
| Idle model memory | Full allocation | Minimal |

## Alternatives Considered

### 1. Ollama + vLLM Backend

Replace llama.cpp with vLLM as Ollama's backend.

- **Pros**: Immediate kvcached support
- **Cons**: Different model format, features

### 2. Memory-Mapped Files

Use mmap for KV cache instead of GPU memory.

- **Pros**: Automatic paging
- **Cons**: Slow, no GPU acceleration

## Open Questions

1. How to handle Ollama's model format vs. HuggingFace?
2. Should we support Apple Silicon (Metal)?
3. How to coordinate with Ollama's built-in model manager?
4. What's the right abstraction level for integration?

## References

- [Ollama GitHub](https://github.com/ollama/ollama)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [GGML Documentation](https://github.com/ggerganov/ggml)
