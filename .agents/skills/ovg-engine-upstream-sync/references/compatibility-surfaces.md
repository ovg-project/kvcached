# kvcached engine compatibility surfaces

Read this file before repairing an upstream engine sync.

## vLLM

Primary patch modules:

- `vllm.v1.core.block_pool`: `KVCacheBlock`
- `vllm.v1.engine.core`: `EngineCore`
- `vllm.v1.core.kv_cache_coordinator`: `KVCacheCoordinator`
- `vllm.v1.core.kv_cache_manager`: `KVCacheManager`
- `vllm.v1.worker.gpu_model_runner`: `GPUModelRunner`
- `vllm.v1.worker.gpu_worker`: `Worker`
- `vllm.v1.kv_cache_interface`: attention specifications and `KVCacheTensor`

High-risk behavior:

- block ownership, reference counts, block hashes, and prefix-cache eviction;
- KV cache group geometry and hybrid-attention grouping;
- tensor allocation, reshape, layout, and block count;
- worker memory profiling and GPU utilization;
- tensor-parallel and pipeline-parallel initialization;
- NIXL registration, K/V splitting, strides, and physical block count.

When these areas change, run both GPU profiles and an output-correctness probe.

## SGLang

Primary patch modules:

- `sglang.srt.mem_cache.allocator`:
  `BaseTokenToKVPoolAllocator`, `alloc_extend_kernel`,
  `alloc_decode_kernel`
- `sglang.srt.mem_cache.memory_pool`:
  `KVCache`, `MHATokenToKVPool`, `MLATokenToKVPool`
- optional memory pools: `MambaPool`, `HybridLinearKVPool`
- `sglang.srt.managers.scheduler`: `Scheduler`
- `sglang.srt.mem_cache.radix_cache`: `RadixCache`

High-risk behavior:

- token/page allocation and free-group semantics;
- MHA, MLA, Mamba, and hybrid pool constructors;
- page size, null slot, data pointers, and layer ranges;
- overlap scheduling and memory-leak checks;
- RadixCache insertion, eviction parameters, and evictable size.

When allocator or memory-pool constructors change, test both page size 1 and a
paged configuration.

## Repair rules

- A renamed or moved symbol requires a compatibility adapter and a regression
  test.
- A changed constructor requires mapping every old argument to its new
  semantic equivalent; forwarding `*args/**kwargs` alone is not evidence.
- A changed tensor layout requires checking shape, stride, block count, and
  generated output.
- A removed upstream invariant must not be reintroduced only to satisfy an old
  patch.
- Update declared supported versions only after the matching runtime gate
  passes.
