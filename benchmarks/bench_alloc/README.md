# `KVCacheManager` alloc/free microbenchmark

Times the `alloc(k) + free(handles)` hot path under three allocator implementations:

- **Python allocator** — baseline before PR #319.
- **C++ allocator** — PR #319 (`lianghao_c++`): allocator migrated to C++, `cudaMemGetInfo` dropped from `available_size()`, page-grouping moved into a single C++ call, `KVCacheBlock` object pool added.
- **C++ + restored resize** — PR #319 + `fix/pr319-restore-resize`: re-adds the elastic-resize poll and shm-name pin that PR #319 dropped.

NVIDIA GB10 (aarch64). All latency numbers are per call in μs (lower is better) unless noted.

For e2e vLLM serving numbers, see [`../bench_layout/README.md`](../bench_layout/README.md).

## Run

```bash
python bench_alloc.py
```

## Results

### 1. `available_size()` — the most frequently called allocator function

The Python path called `cudaMemGetInfo` on every invocation (~6 μs each). The C++ path skips it.

| | μs/call |
|--:|--:|
| Python | 6.52 |
| C++ | 0.52 |
| C++ + resize | 0.52 |

**12.5×.** Called once per scheduler step.

### 2. `group_indices_by_page` — called inside `free()`

Maps a list of N block indices to their owning pages. Python used a per-element loop + `defaultdict`; C++ replaces it with one call.

| N | Python | C++ | speedup |
|--:|--:|--:|--:|
| 64 | 3.4 | 1.3 | 2.6× |
| 1024 | 52.6 | 16.8 | 3.1× |
| 16384 | 834 | 292 | 2.9× |

**~3× across the range.** Restored-resize matches C++ within noise.

### 3. Slow-path alloc — `cuMemMap` per call

`KVCACHED_MIN/MAX_RESERVED_PAGES=0` forces every alloc to map a fresh 2 MB VMM page. `k` is blocks per alloc; k=128 ≈ one page.

| k | Python | C++ | C++ + resize |
|--:|--:|--:|--:|
| 128 | 4196 | 4023 | 4354 |
| 1024 | 33028 | 32488 | 34662 |
| 4096 | 134479 | 134430 | 137295 |

**All within 5%.** The CUDA driver syscall dominates; switching the surrounding code to C++ doesn't help.

### 4. Multi-thread throughput — Python contention dissolves

N Python threads, each in a tight `alloc(k=16) + free(h)` loop, `async_sched=True`. Aggregate ops/s (**higher is better**).

| threads | Python Kops/s | C++ Kops/s | C++ + resize Kops/s |
|--:|--:|--:|--:|
| 1 | 15.1 | 41.2 | 32.5 |
| 4 | 12.0 | 48.6 | 31.6 |
| 8 | 9.1 | 51.5 | 29.1 |

Python **degrades** under thread count (Python-level contention dominated the old hot path). C++ holds or improves. Restored-resize is flat ~30K — each alloc polls a resize shm descriptor that bare C++ skips.

(GIL is still held during C++ work, so gains come from shorter critical sections, not real parallelism. Real vLLM uses `async_sched=False` and doesn't exercise this path.)

### 5. `KVCacheBlock` object pool — C++ only

Pre-allocated pool of `KVCacheBlock` objects vs `new` per call. The Python baseline has no equivalent pool.

| N | no-pool | pool | speedup |
|--:|--:|--:|--:|
| 8 | 1.06 | 0.19 | 5.6× |
| 1024 | 147 | 17.4 | 8.5× |
| 4096 | 651 | 67.7 | 9.6× |

**5-10×**, speedup grows with N.

## Summary — what the C++ allocator delivers

- **12.5× on `available_size()`** — eliminates the per-scheduler-step `cudaMemGetInfo` cost.
- **~3× on `group_indices_by_page`** — flat across N from 64 to 16,384.
- **Multi-thread throughput scales** instead of degrading: 8 threads go from 9 Kops/s (Python) to 51 Kops/s (C++).
- **5-10× on `KVCacheBlock` allocation** via the new object pool (no Python equivalent).
- Slow-path `cuMemMap` is driver-bound and unaffected by the migration.

The restored-resize variant retains every gain except multi-thread (~70% of bare C++) because each alloc polls the resize shm descriptor.

These wins amortise to ~5% on e2e vLLM serving (per-token model forward dominates). The much larger e2e lever is unrelated — see [`../bench_layout/README.md`](../bench_layout/README.md).
