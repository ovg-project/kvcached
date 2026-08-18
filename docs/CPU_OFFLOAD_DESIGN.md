# CPU offloading milestone

`kvcached/cpu_offload.py` implements the control plane and pinned-memory CUDA
data path for CPU-backed KV cache pages. The PageAllocator now preserves a
stable virtual page id while replacing its private GPU mapping with the shared
zero page, then maps the same id again before restore.

## Page model

One kvcached page id identifies the same byte range in every layer and KV
buffer. Offloading page `p` therefore means preserving a bundle of:

```text
num_layers * num_kv_buffers * page_size
```

bytes. Treating only one tensor slice as a complete page would silently lose
most of the model's KV state.

## Implemented now

- bounded CPU capacity with atomic rejection when the store is full;
- strict page geometry validation;
- GPU-to-CPU transaction ordering;
- restore rollback when a copy fails;
- recency tracking for an engine-level policy to choose explicit victims;
- structured failure reporting for GPU release and restore failures;
- a page-level planner that rejects active pages and respects CPU capacity;
- a transfer-versus-recompute break-even estimator;
- a pinned-memory CUDA transfer backend with a dedicated stream;
- non-contiguous and compound-page tensor span calculation;
- transactional `map -> H2D copy -> commit` restore visibility;
- PageAllocator offload/restore state, transition guards, memory accounting,
  and Python bindings;
- KVCacheManager tracking that removes offloaded pages from block allocation;
- a pinned-memory CUDA transfer microbenchmark;
- a real VMM round-trip correctness and memory-reclamation experiment;
- CPU-only tests for policy, tensor geometry, and C++ binding contracts.

The safety rule is: save a complete CPU copy before releasing GPU memory. On
restore, keep the CPU copy until GPU allocation and copy-back both succeed.
The store never drops the only CPU copy to make room implicitly. To reuse CPU
capacity in this milestone, the caller restores a chosen page, invalidates and
frees it through the engine's normal cache lifecycle, then retries the new
offload. A direct CPU-resident-page retirement API belongs with the later
engine integration because it must update prefix metadata and page ownership
atomically.

## Transaction

Offload executes:

1. copy every layer/KV-buffer slice to pinned host memory;
2. synchronize the transfer stream;
3. commit the complete page to the bounded CPU store;
4. replace the page's private GPU mapping with the shared zero page;
5. keep its virtual page id and block metadata unavailable to new allocation.

Restore executes:

1. map fresh GPU physical memory at the same virtual page id;
2. copy every pinned payload back on the transfer stream;
3. synchronize the transfer stream;
4. expose the page to block allocation only after the copy succeeds;
5. retain the CPU copy and undo the GPU mapping after any failure.

Only pages whose blocks are inactive and reusable should enter this path.
Active request pages must never be offloaded.

Run the complete provider-neutral VMM validation on a CUDA machine:

```bash
git clone --branch zixuan/cpu-offload-control-plane \
  https://github.com/Lanoxia/kvcached.git
cd kvcached
bash tools/run_cpu_offload_h20_validation.sh
```

The command builds the current checkout, runs repeated real-page round trips,
checks byte-level correctness and VMM state, runs the transfer benchmark, and
creates a checksummed artifact archive.

It also handles the two requirements found during H20 validation: when the
host compiler is older than GCC 9 it provisions an isolated Conda GCC 11.4
toolchain, and the extension build automatically links through the CUDA
toolkit stub when a container exposes only the versioned driver library.

The same validation is available through
`.github/workflows/cpu-offload-gpu.yml`. It supports manual page/cycle counts,
runs weekly by default, serializes access to the persistent GPU runner, and
uploads the checksummed bundle on success or failure.
