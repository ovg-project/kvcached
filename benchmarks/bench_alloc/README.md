# `KVCacheManager` alloc/free microbenchmark

Times the `alloc(k) + free(handles)` hot path under three allocator implementations:

- **Python allocator** — baseline before PR #319.
- **C++ allocator** — PR #319 (`lianghao_c++`): allocator migrated to C++, `cudaMemGetInfo` dropped from `available_size()`, page-grouping moved into a single C++ call, `KVCacheBlock` object pool added.
- **C++ allocator + restored resize** — PR #319 + `fix/pr319-restore-resize`: re-adds the elastic-resize poll and shm-name pin that PR #319 dropped.

For e2e vLLM numbers and the layout overhead, see [`../bench_layout/README.md`](../bench_layout/README.md).

## Run

```bash
python bench_alloc.py
```

NVIDIA GB10 (aarch64). 100 warmup iterations, then `alloc(k) + free(handles)` at varying `k`.

## Headline numbers

| operation | Python | C++ | C++ + resize |
|---|--:|--:|--:|
| `available_size()` μs | 6.52 | 0.52 | 0.52 |
| `group_indices_by_page` μs @ N=1024 | 52.6 | 16.8 | 16.9 |
| Slow-path alloc μs @ k=128 | 4196 | 4023 | 4354 |
| Multi-thread alloc+free @ 4 threads, k=16 (Kops/s) | 12.0 | 48.6 | 31.6 |
| `KVCacheBlock` pool μs @ N=1024 | — | 17.4 (vs 147 no-pool) | — |

C++ allocator gains **3-12×** on Python-heavy paths. Slow-path alloc is CUDA-driver-bound (`cuMemMap`) and unaffected. The restored-resize variant retains all gains except multi-thread (~70% of bare C++ — each alloc now polls a resize shm descriptor).

## Per-N detail

`group_indices_by_page` holds ~3× across the range:

| N | Python | C++ | speedup |
|--:|--:|--:|--:|
| 64 | 3.4 | 1.3 | 2.6× |
| 1024 | 52.6 | 16.8 | 3.1× |
| 16384 | 834 | 292 | 2.9× |

Multi-thread (Kops/s, `async_sched=True`, k=16) — Python degrades, C++ holds:

| threads | Python | C++ | C++ + resize |
|--:|--:|--:|--:|
| 1 | 15.1 | 41.2 | 32.5 |
| 4 | 12.0 | 48.6 | 31.6 |
| 8 | 9.1 | 51.5 | 29.1 |

`KVCacheBlock` object pool — speedup grows with N:

| N | no-pool | pool | speedup |
|--:|--:|--:|--:|
| 8 | 1.06 | 0.19 | 5.6× |
| 1024 | 147 | 17.4 | 8.5× |
| 4096 | 651 | 67.7 | 9.6× |

Note: GIL is still held during C++ work, so multi-thread gains come from shorter critical sections, not real parallelism. Real vLLM uses `async_sched=False` (NoOpLock) and doesn't hit this path.

## Bottom line

3-12× microbench wins amortise to **~5% on e2e vLLM serving** because per-token model forward dominates. The much larger lever is the KV layout default — see [`../bench_layout/README.md`](../bench_layout/README.md).
