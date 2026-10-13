# CPU Memory Offloading Design Document

**Status**: Proposed
**Issue**: #93
**Author**: Community
**Last Updated**: December 2024

## Overview

This document proposes adding CPU memory offloading support to kvcached, enabling KV cache data to be moved between GPU and CPU memory based on access patterns and memory pressure.

## Motivation

### Current Limitations
- GPU memory is the primary bottleneck for serving large models
- When GPU memory is exhausted, requests fail rather than gracefully degrading
- Long sequences consume memory even when idle

### Benefits of CPU Offloading
- **Increased Capacity**: Serve more concurrent requests
- **Graceful Degradation**: Slow down rather than fail under memory pressure
- **Cost Efficiency**: Better utilize available CPU memory
- **Long Context Support**: Enable very long sequences without OOM

## Design Goals

1. **Minimal Latency Impact**: Offloading should not significantly affect active requests
2. **Transparent Operation**: No changes required to inference engines
3. **Configurable Policies**: Allow users to tune offloading behavior
4. **Memory Safety**: Prevent data corruption during transfers

## Architecture

### Memory Hierarchy

```
┌─────────────────────────────────────────┐
│           GPU Memory (Hot)              │
│  ┌─────────────────────────────────┐   │
│  │     Active KV Cache Pages       │   │
│  │   (currently in use by model)   │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
                    ↑↓ (page migration)
┌─────────────────────────────────────────┐
│          CPU Memory (Warm)              │
│  ┌─────────────────────────────────┐   │
│  │    Recently Evicted Pages       │   │
│  │   (quick restore if needed)     │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
                    ↑↓ (optional)
┌─────────────────────────────────────────┐
│         Disk/NVMe (Cold)                │
│  ┌─────────────────────────────────┐   │
│  │    Long-term Storage (future)   │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### Component Design

#### 1. OffloadManager

```python
class OffloadManager:
    """Manages CPU-GPU memory transfers."""

    def __init__(
        self,
        gpu_device: torch.device,
        cpu_memory_limit: int,
        transfer_stream: torch.cuda.Stream,
    ):
        self.gpu_device = gpu_device
        self.cpu_memory_limit = cpu_memory_limit
        self.transfer_stream = transfer_stream
        self.cpu_pages: Dict[int, torch.Tensor] = {}
        self.page_access_time: Dict[int, float] = {}

    def offload_page(self, page_id: int, gpu_tensor: torch.Tensor) -> None:
        """Move a page from GPU to CPU memory."""
        with torch.cuda.stream(self.transfer_stream):
            cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
            self.cpu_pages[page_id] = cpu_tensor

    def restore_page(self, page_id: int, gpu_tensor: torch.Tensor) -> None:
        """Move a page from CPU back to GPU memory."""
        if page_id not in self.cpu_pages:
            raise ValueError(f"Page {page_id} not in CPU memory")

        with torch.cuda.stream(self.transfer_stream):
            gpu_tensor.copy_(self.cpu_pages[page_id], non_blocking=True)

        del self.cpu_pages[page_id]
```

#### 2. EvictionPolicy

```python
class EvictionPolicy(ABC):
    """Abstract base class for eviction policies."""

    @abstractmethod
    def select_pages_to_evict(
        self,
        num_pages: int,
        page_info: Dict[int, PageInfo],
    ) -> List[int]:
        """Select pages to evict from GPU memory."""
        pass

class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""

    def select_pages_to_evict(
        self,
        num_pages: int,
        page_info: Dict[int, PageInfo],
    ) -> List[int]:
        # Sort by last access time
        sorted_pages = sorted(
            page_info.items(),
            key=lambda x: x[1].last_access_time
        )
        return [p[0] for p in sorted_pages[:num_pages]]

class PriorityEvictionPolicy(EvictionPolicy):
    """Priority-based eviction considering request importance."""

    def select_pages_to_evict(
        self,
        num_pages: int,
        page_info: Dict[int, PageInfo],
    ) -> List[int]:
        # Consider request priority and sequence position
        def score(page: PageInfo) -> float:
            recency = time.time() - page.last_access_time
            priority = page.request_priority
            position = page.sequence_position / page.total_length
            return recency * (1 / priority) * (1 - position)

        sorted_pages = sorted(
            page_info.items(),
            key=lambda x: score(x[1]),
            reverse=True
        )
        return [p[0] for p in sorted_pages[:num_pages]]
```

#### 3. TransferScheduler

```python
class TransferScheduler:
    """Schedules CPU-GPU transfers to minimize latency impact."""

    def __init__(self, offload_manager: OffloadManager):
        self.offload_manager = offload_manager
        self.pending_offloads: Queue[int] = Queue()
        self.pending_restores: Queue[int] = Queue()
        self.transfer_thread = Thread(target=self._transfer_loop)

    def _transfer_loop(self):
        """Background thread for handling transfers."""
        while True:
            # Prioritize restores over offloads
            if not self.pending_restores.empty():
                page_id = self.pending_restores.get()
                self._handle_restore(page_id)
            elif not self.pending_offloads.empty():
                page_id = self.pending_offloads.get()
                self._handle_offload(page_id)
            else:
                time.sleep(0.001)  # Avoid busy waiting
```

## Configuration

### Environment Variables

```bash
# Enable CPU offloading
export KVCACHED_OFFLOAD_ENABLED=true

# CPU memory limit for offloaded pages (in GB)
export KVCACHED_OFFLOAD_CPU_MEMORY_GB=32

# Eviction policy (lru, priority, hybrid)
export KVCACHED_OFFLOAD_POLICY=lru

# GPU memory threshold to trigger offloading (0.0-1.0)
export KVCACHED_OFFLOAD_THRESHOLD=0.9

# Prefetch pages when GPU memory available
export KVCACHED_OFFLOAD_PREFETCH=true
```

### Programmatic Configuration

```python
from kvcached.offload import OffloadConfig, configure_offload

config = OffloadConfig(
    enabled=True,
    cpu_memory_limit=32 * 1024**3,  # 32 GB
    eviction_policy="lru",
    gpu_threshold=0.9,
    prefetch_enabled=True,
)

configure_offload(config)
```

## Performance Considerations

### Transfer Bandwidth

| Transfer Type | Bandwidth | Latency |
|--------------|-----------|---------|
| GPU → CPU (PCIe 4.0 x16) | ~25 GB/s | ~10μs |
| CPU → GPU (PCIe 4.0 x16) | ~25 GB/s | ~10μs |
| GPU → CPU (PCIe 5.0 x16) | ~50 GB/s | ~8μs |
| NVLink (GPU-GPU) | ~600 GB/s | ~1μs |

### Page Size Tradeoffs

| Page Size | Transfer Time | Fragmentation | Overhead |
|-----------|---------------|---------------|----------|
| 2 MB | 0.08 ms | Low | High |
| 16 MB | 0.64 ms | Medium | Medium |
| 64 MB | 2.56 ms | High | Low |

### Latency Impact

Expected latency increase for offloaded pages:
- **Best case**: Page prefetched, 0 additional latency
- **Typical case**: 1-5ms for page restore
- **Worst case**: 10-20ms for cold page restore

## Implementation Plan

### Phase 1: Basic Offloading
- [ ] Implement OffloadManager with sync transfers
- [ ] Add LRU eviction policy
- [ ] Integrate with PageAllocator
- [ ] Add configuration options

### Phase 2: Async Transfers
- [ ] Implement async transfer using CUDA streams
- [ ] Add TransferScheduler
- [ ] Implement prefetching

### Phase 3: Advanced Policies
- [ ] Add priority-based eviction
- [ ] Implement hybrid policies
- [ ] Add per-request offload hints

### Phase 4: Optimization
- [ ] Profile and optimize transfer paths
- [ ] Add compression for CPU pages
- [ ] Implement NVMe tier (optional)

## Testing Strategy

### Unit Tests
- Page offload/restore correctness
- Eviction policy behavior
- Memory limit enforcement

### Integration Tests
- Multi-model offloading
- Concurrent transfers
- Error recovery

### Performance Tests
- Latency impact benchmarks
- Throughput under memory pressure
- Memory efficiency metrics

## Alternatives Considered

### 1. Unified Memory (CUDA UVM)
- **Pros**: Automatic migration, simple API
- **Cons**: Less control, potential thrashing
- **Decision**: May use as fallback for simpler cases

### 2. RDMA for Remote Memory
- **Pros**: Access remote GPU/CPU memory
- **Cons**: Requires special hardware, complex setup
- **Decision**: Future consideration for multi-node

### 3. KV Cache Compression
- **Pros**: Reduce memory without offloading
- **Cons**: Quality tradeoffs, compute overhead
- **Decision**: Complementary feature, not replacement

## Open Questions

1. How to handle in-flight requests when offloading their pages?
2. Should we support partial page offloading?
3. How to coordinate offloading across tensor parallel ranks?
4. What metrics should guide automatic policy tuning?

## References

- [vLLM CPU Offloading Discussion](https://github.com/vllm-project/vllm/discussions/...)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [FlexGen: High-Throughput Generative Inference](https://arxiv.org/abs/2303.06865)
