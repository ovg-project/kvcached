# kvcached Architecture

This document explains the architecture of kvcached and how engines, models, and memory management interact.

## System Overview

```
+-------------------+     +-------------------+     +-------------------+
|    vLLM/SGLang    |     |    vLLM/SGLang    |     |    vLLM/SGLang    |
|    Engine #1      |     |    Engine #2      |     |    Engine #3      |
+--------+----------+     +--------+----------+     +--------+----------+
         |                         |                         |
         |     KV Cache Requests   |                         |
         v                         v                         v
+------------------------------------------------------------------------+
|                        kvcached Memory Manager                          |
|                                                                         |
|  +------------------+  +------------------+  +------------------+       |
|  | Page Allocator   |  | KV Cache Manager |  | IPC Coordinator  |       |
|  +------------------+  +------------------+  +------------------+       |
|                                                                         |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
|                          GPU Physical Memory                            |
|  [Page 0] [Page 1] [Page 2] [Page 3] [Page 4] [Page 5] [Page 6] ...   |
+------------------------------------------------------------------------+
```

## Component Decoupling

### Engine Independence

Each inference engine (vLLM or SGLang instance) operates independently:

1. **Separate Process Space**: Each engine runs in its own process
2. **Shared Memory Communication**: Engines communicate via `/dev/shm`
3. **Virtual Memory Abstraction**: Each engine sees a virtual address space

### How Decoupling Works

```
Engine A (vLLM)                    Engine B (SGLang)
     |                                  |
     v                                  v
Virtual KV Cache                  Virtual KV Cache
[Block 0-99]                      [Block 0-99]
     |                                  |
     |   (kvcached mapping layer)       |
     v                                  v
Physical Pages (shared pool)
[Page 0] [Page 1] [Page 2] ...
```

**Key Insight**: Virtual block IDs are local to each engine, but physical pages are shared across all engines.

## Connecting to Pre-Started Engines

kvcached supports attaching to engines that are already running.

### Method 1: Environment Variable Naming

Start engines with the same IPC name:

```bash
# Terminal 1 - Engine A
export KVCACHED_IPC_NAME=my_shared_cache
export ENABLE_KVCACHED=true
vllm serve model-a --port 8001

# Terminal 2 - Engine B (connects to same IPC segment)
export KVCACHED_IPC_NAME=my_shared_cache
export ENABLE_KVCACHED=true
vllm serve model-b --port 8002
```

### Method 2: Controller Orchestration

Use the controller to manage multiple engines:

```yaml
# config.yaml
kvcached:
  kvcached_gpu_utilization: 0.95

instances:
  - name: model-a
    model: meta-llama/Llama-3.2-1B
    engine: vllm
    engine_args:
      - "--port=8001"
  - name: model-b
    model: Qwen/Qwen3-0.6B
    engine: sglang
    engine_args:
      - "--port=8002"
```

### Method 3: Manual Integration

For custom setups, initialize kvcached programmatically:

```python
from kvcached.integration.vllm import interfaces as kvi

# Connect to existing IPC segment
import os
os.environ["KVCACHED_IPC_NAME"] = "existing_segment"

# Initialize with existing configuration
kvi.init_kvcached(
    tp_rank=0,
    tp_size=1,
    device="cuda:0",
)

# Now engine will use the shared memory pool
```

## IPC Mechanism

### Shared Memory Segments

kvcached uses POSIX shared memory for coordination:

```
/dev/shm/
├── kvcached_vLLM_12345           # Memory info for engine group
├── kvcached_vLLM_12345_page_0    # Page allocation bitmap
└── kvcached_vLLM_12345_lock      # Coordination lock
```

### Memory Info Structure

Each IPC segment tracks:
- `total_size`: Maximum allocatable memory
- `used_size`: Currently allocated memory
- `prealloc_size`: Pre-allocated but unused pages

### Tensor Parallel Coordination

For multi-GPU tensor parallelism:

```
Rank 0 (Scheduler)          Rank 1 (Worker)          Rank 2 (Worker)
     |                           |                        |
     v                           v                        v
+-----------+              +-----------+            +-----------+
| Local VMM |              | Local VMM |            | Local VMM |
+-----------+              +-----------+            +-----------+
     |                           |                        |
     +------ Unix Socket IPC ----+------------------------+
     |
     v
+------------------+
| Broadcast alloc/ |
| free operations  |
+------------------+
```

## Memory Lifecycle

### Allocation Flow

1. **Request arrives**: Engine needs N blocks for new sequence
2. **Virtual allocation**: KVCacheManager assigns virtual block IDs
3. **Page mapping**: PageAllocator maps virtual blocks to physical pages
4. **Memory mapping**: VMM maps physical pages to GPU virtual addresses
5. **Ready for use**: KV cache tensors are ready for attention computation

### Deallocation Flow

1. **Sequence completes**: Engine frees block IDs
2. **Page unmapping**: VMM unmaps physical pages from virtual addresses
3. **Page return**: Physical pages return to free pool
4. **Available for reuse**: Memory available for other engines/sequences

## Configuration for Decoupled Operation

### Shared Configuration

All engines sharing memory should use consistent settings:

```bash
# Must match across all engines
export KVCACHED_IPC_NAME=shared_cache
export KVCACHED_GPU_UTILIZATION=0.95
export KVCACHED_PAGE_SIZE_MB=2
```

### Independent Configuration

These can vary per engine:

```bash
# Can differ per engine
export KVCACHED_LOG_LEVEL=INFO
export KVCACHED_PAGE_PREALLOC_ENABLED=true
```

## Monitoring Decoupled Engines

### View All Segments

```bash
kvctl list
```

Output:
```
IPC                      Limit        Used       %
kvcached_vLLM_12345      8.00 GB      2.50 GB    31.3%
kvcached_SGLang_67890    8.00 GB      1.20 GB    15.0%
```

### Monitor Specific Segment

```bash
kvctl watch kvcached_vLLM_12345
```

### Real-Time Visualization

```bash
kvctl kvtop
```
