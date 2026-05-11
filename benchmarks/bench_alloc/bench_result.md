# `bench_alloc` — KVCacheManager alloc/free microbench

NVIDIA GB10 (aarch64), clean rebuild between runs. `main` = `ovg-project/kvcached` main, `PR` = `lianghao_c++` + must-fix `98d9bb3`, `fix` = `fix/pr319-restore-resize`.

Driver: `bench_alloc.py` — times `KVCacheManager.alloc(k) + free(handles)` cycles after 100 warmup iterations.

For e2e vLLM serving numbers and the layout-overhead investigation, see [`../bench_layout/bench_result.md`](../bench_layout/bench_result.md).

## A. `available_size()`

PR drops `cudaMemGetInfo` from the hot path.

| | μs/call |
|--:|--:|
| main | 6.52 |
| PR | 0.52 |
| fix | 0.52 |

12.5×. Side effect: `available_size()` no longer caps by physical memory (later restored on PR-319 head — see commit `7e3fdb7`).

## B. `group_indices_by_page`

Called by `KVCacheManager.free`. Python loop + `defaultdict` → single C++ call.

| N | main | PR | fix | speedup |
|--:|--:|--:|--:|--:|
| 64 | 3.36 | 1.28 | 1.21 | 2.6× |
| 256 | 12.88 | 3.34 | 3.24 | 3.9× |
| 1024 | 52.57 | 16.84 | 16.86 | 3.1× |
| 4096 | 209.12 | 71.06 | 72.50 | 2.9× |
| 16384 | 834.13 | 291.87 | 296.72 | 2.9× |

(μs)

## C. Slow-path alloc

`KVCACHED_MIN/MAX_RESERVED_PAGES=0` forces a fresh `cuMemMap` per alloc. `k=128` ≈ one 2 MB page.

| k | main | PR | fix |
|--:|--:|--:|--:|
| 128 | 4196 | 4023 | 4354 |
| 256 | 8384 | 8113 | 8657 |
| 1024 | 33028 | 32488 | 34662 |
| 4096 | 134479 | 134430 | 137295 |

(μs) All within 5%. The syscall dominates; C++ doesn't help.

## D. Multi-thread throughput

N Python threads, each in a tight `alloc(k) + free(h)` loop, `async_sched=True`. Aggregate ops/s.

main:
| threads | k=1 | k=16 |
|--:|--:|--:|
| 1 | 15.7K | 15.1K |
| 2 | 15.3K | 14.4K |
| 4 | 13.3K | 12.0K |
| 8 | 10.4K | 9.1K |

PR:
| threads | k=1 | k=16 |
|--:|--:|--:|
| 1 | 45.9K | 41.2K |
| 2 | 41.1K | 37.8K |
| 4 | 38.5K | 48.6K |
| 8 | 43.6K | 51.5K |

fix:
| threads | k=1 | k=16 |
|--:|--:|--:|
| 1 | 31.5K | 32.5K |
| 2 | 30.6K | 30.7K |
| 4 | 29.5K | 31.6K |
| 8 | 27.8K | 29.1K |

main degrades with threads. PR holds. fix sits ~30% behind PR (per-alloc shm poll for kvctl resize).

GIL is still held during C++ work (no `gil_scoped_release`); gains come from shorter critical sections, not real parallelism. Real vLLM uses `async_sched=False` (NoOpLock) anyway.

## F. `KVCacheBlock` object pool

PR-only — main has no pool. Pre-allocated pool vs `new()` per call.

| N | no-pool | pool | speedup |
|--:|--:|--:|--:|
| 8 | 1.06 | 0.19 | 5.6× |
| 64 | 8.07 | 1.13 | 7.1× |
| 256 | 31.28 | 4.38 | 7.1× |
| 1024 | 147.4 | 17.44 | 8.5× |
| 4096 | 650.8 | 67.65 | 9.6× |

(μs)

## Summary (microbench only)

| | speedup vs main | fix retains |
|---|---|---|
| A. `available_size()` | 12.5× | full |
| B. `group_indices_by_page` | ~3× | full |
| C. slow-path alloc | 1× (driver-bound) | full |
| D. multi-thread | 2-5× | ~70% of PR |
| F. block-object pool | ~7× (PR-only) | full |

These microbench wins are real but amortise to ~5% on e2e vLLM serving because per-token model forward dominates the hot path — see [`../bench_layout/bench_result.md`](../bench_layout/bench_result.md) for the e2e picture and the much larger layout-related lever.
