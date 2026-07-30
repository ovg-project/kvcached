# CPU offloading milestone

`kvcached/cpu_offload.py` implements the control-plane foundation for
CPU-backed KV cache pages. It is intentionally not enabled in vLLM or SGLang
yet.

## Page model

One kvcached page id identifies the same byte range in every layer and KV
buffer. Offloading page `p` therefore means preserving a bundle of:

```text
num_layers * num_kv_buffers * page_size
```

bytes. Treating only one tensor slice as a complete page would silently lose
most of the model's KV state.

## Implemented now

- bounded CPU capacity with least-recently-used eviction;
- strict page geometry validation;
- GPU-to-CPU transaction ordering;
- restore rollback when a copy fails;
- explicit reporting of CPU pages evicted by the capacity policy;
- structured failure reporting, including page ids evicted before a later GPU
  release failure, so engine metadata can still be invalidated;
- a page-level planner that rejects active pages and respects CPU capacity;
- a transfer-versus-recompute break-even estimator;
- a transfer-backend protocol that keeps CUDA details outside policy code;
- a pinned-memory CUDA transfer microbenchmark;
- CPU-only tests for all of the above.

The safety rule is: save a complete CPU copy before releasing GPU memory. On
restore, keep the CPU copy until GPU allocation and copy-back both succeed.
Callers must invalidate prefix-cache metadata for every `evicted_page_ids`
entry returned by `OffloadResult` or `OffloadError`.

## GPU implementation milestone

The first GPU backend should:

1. allocate pinned host buffers for every layer/KV-buffer slice;
2. expose page-copy methods from `FTensorAllocator`;
3. copy on a dedicated CUDA stream and record a completion event;
4. unmap the GPU page only after the device-to-host event completes;
5. map a GPU page before host-to-device restore;
6. connect restored pages to vLLM's prefix-cache metadata;
7. invalidate prefix metadata when the CPU LRU evicts its only copy.

The initial benchmark should compare:

- CPU offload and restore latency;
- recomputing the same prefix;
- time to first token after a CPU-cache hit;
- GPU memory reclaimed;
- CPU cache hit rate and LRU eviction count.

Only pages whose blocks are inactive and reusable should enter this path.
Active request pages must never be offloaded.

Run the data-plane microbenchmark on a CUDA machine:

```bash
python tools/benchmark_cpu_offload.py \
  --page-size-mb 2 \
  --layers 32 \
  --kv-buffers 2 \
  --iterations 50 \
  --report cpu-offload-transfer.json
```

The measured H2D latency is the lower bound for a CPU-cache restore. Compare it
with prefix recomputation latency before deciding which pages are worth
keeping on CPU.
