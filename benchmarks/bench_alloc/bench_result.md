All measurements done on the same machine (NVIDIA GB10, aarch64) with clean rebuild + reinstall before each run. `main` = `ovg-project/kvcached` main, `PR HEAD` = `lianghao_c++` (incl. must-fix `98d9bb3`), `fix branch` = `fix/pr319-restore-resize`.

---

## A. `available_size()` overhead

main calls `cudaMemGetInfo` via `get_avail_physical_pages()`; PR removed that.

| | μs/call |
|--:|--:|
| main | 6.52 |
| PR HEAD | 0.52 |
| fix branch | 0.52 |

**12.5× speedup**, no fix-branch cost. Trade-off: PR's `available_size()` no longer caps by physical memory.

---

## B. `group_indices_by_page` (called by `KVCacheManager.free`)

main: Python `for + defaultdict` + Python `get_page_id` per iter. PR: single C++ call.

| N indices | main μs | PR HEAD μs | fix branch μs | speedup |
|--:|--:|--:|--:|--:|
| 64 | 3.36 | 1.28 | 1.21 | 2.6× |
| 256 | 12.88 | 3.34 | 3.24 | 3.9× |
| 1024 | 52.57 | 16.84 | 16.86 | 3.1× |
| 4096 | 209.12 | 71.06 | 72.50 | 2.9× |
| 16384 | 834.13 | 291.87 | 296.72 | 2.9× |

~3× across the board. Matters for large batch frees (e.g. burst sequence completion).

---

## C. Slow-path alloc (`cuMemMap` cost)

`KVCACHED_MIN_RESERVED_PAGES=0`, `KVCACHED_MAX_RESERVED_PAGES=0` → every alloc maps a fresh page, every free unmaps. `k=128` = 1 full 2 MB page.

| k | main μs | PR HEAD μs | fix branch μs |
|--:|--:|--:|--:|
| 128 | 4196 | 4023 | 4354 |
| 256 | 8384 | 8113 | 8657 |
| 1024 | 33028 | 32488 | 34662 |
| 4096 | 134479 | 134430 | 137295 |

All within ~5% of each other. **Slow path is dominated by `cuMemMap` syscalls**; the C++ migration doesn't help here. PR's gains are concentrated in the fast path (reserved-pool hit).

---

## D. Multi-threaded throughput (`async_sched=True`)

N Python threads each running `alloc(k)+free(h)` in a tight loop, aggregate ops/s.

**main (Python alloc + RLock + GIL):**
| threads | k=1 ops/s | k=16 ops/s |
|--:|--:|--:|
| 1 | 15.7K | 15.1K |
| 2 | 15.3K | 14.4K |
| 4 | 13.3K | 12.0K |
| 8 | 10.4K | 9.1K |

→ More threads = LESS throughput.

**PR HEAD (C++ alloc):**
| threads | k=1 ops/s | k=16 ops/s |
|--:|--:|--:|
| 1 | 45.9K | 41.2K |
| 2 | 41.1K | 37.8K |
| 4 | 38.5K | 48.6K |
| 8 | 43.6K | 51.5K |

→ ~3× single-thread, holds up to 8 threads.

**fix branch:**
| threads | k=1 ops/s | k=16 ops/s |
|--:|--:|--:|
| 1 | 31.5K | 32.5K |
| 2 | 30.6K | 30.7K |
| 4 | 29.5K | 31.6K |
| 8 | 27.8K | 29.1K |

→ ~2× single-thread, holds steady. ~30% slower than PR HEAD due to per-alloc shm poll.

**Caveat:** `page_allocator_alloc_page` etc. don't `py::gil_scoped_release`, so GIL is still held during C++ work. Multi-thread benefit comes from shorter critical sections, not actual parallelism. Real vLLM uses `async_sched=False` (NoOpLock), so the per-thread RLock is gone but contention via GIL + manager state remains.

---

## F. `KVCacheBlock` object pool (`patches.py`)

Synthetic test: object creation per-call vs object reuse from a pre-allocated pool. Independent of branch (PR introduces the pool; main has no pool).

| N | no-pool μs | pool μs | speedup |
|--:|--:|--:|--:|
| 8 | 1.06 | 0.19 | 5.6× |
| 64 | 8.07 | 1.13 | 7.1× |
| 256 | 31.28 | 4.38 | 7.1× |
| 1024 | 147.4 | 17.44 | 8.5× |
| 4096 | 650.8 | 67.65 | 9.6× |

~6-10×. Saves 7-27 μs per `get_new_blocks` call at typical batch sizes.

---

## E. End-to-end vLLM serving

`vllm bench serve --backend vllm --dataset-name random --random-input-len 512 --random-output-len 128 --num-prompts 500 --request-rate inf` against a locally running vLLM server with `ENABLE_KVCACHED=true`. Model: `Qwen/Qwen3-0.6B`, `gpu-memory-utilization=0.5`, `max-model-len=2048`. 100-prompt warmup discarded; 3 seeds (42, 99, 7) per branch, median reported.

| | Throughput (req/s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) |
|---|--:|--:|--:|--:|--:|
| **vanilla vLLM** (no kvcached) | **14.21** | **11575** | **26377** | **119.3** | **143.2** |
| main (kvcached on) | 9.42 (-34%) | 17284 (+49%) | 40464 (+53%) | 185.1 (+55%) | 239.6 (+67%) |
| PR HEAD | 9.86 (-31%) | 16472 (+42%) | 38756 (+47%) | 177.5 (+49%) | 230.2 (+61%) |
| fix branch | 9.83 (-31%) | 16602 (+43%) | 38746 (+47%) | 178.2 (+49%) | 231.7 (+62%) |

(Δ shown vs. vanilla vLLM.)

**Two distinct gaps:**

1. **kvcached vs. vanilla vLLM: ~30-50% overhead everywhere**, regardless of branch. This is the pre-existing kvcached cost (CUDA VMM map/unmap, page-level indirection, alloc bookkeeping) and is what issue #299 reported (their 81 ms vLLM vs 268 ms kvcached TTFT, 3.3× — same direction, larger magnitude).
2. **PR's C++ migration vs. main: ~5% closer to vanilla.** Real but modest at e2e on this setup; far smaller than #299's reported gap (would need their H20 + Llama2-7B + sharegpt setup to reproduce that).

**fix branch retains all of PR's 5% gain** (kvctl resize poll cost is invisible at e2e, <1%, within run-to-run noise).

The microbench wins (A: 12.5×, B: 3×, D: 2-5× under contention, F: 7×) wash down to ~5% e2e because per-token model forward (hundreds of μs of attention + FFN) dominates the request budget; allocator overhead is a few μs at fast path, ~5% of the total.

---

## Summary

| benchmark | what changed in PR | speedup vs main | retained on fix branch |
|---|---|---|---|
| A. `available_size()` | drop `cudaMemGetInfo` from hot path | **12.5×** | yes (full) |
| B. `group_indices_by_page` | Python loop → C++ unordered_map | **~3×** | yes (full) |
| C. slow-path alloc (`cuMemMap`) | unchanged (driver-bound) | **~1.0×** | yes (full) |
| D. multi-thread alloc/free | C++ allocator + shorter critical sections | **2-5×** under contention | yes, ~70% of PR's gain |
| E. e2e vLLM serving | sum of A–D | **~5%** | yes (full) |
| F. `KVCacheBlock` object pool | reuse instead of new() | **~7×** (independent) | yes (full) |

Microbench wins are real and broad. At e2e level they amortise to ~5% on this setup because model forward dominates the per-request time budget. On bigger models or more allocator-heavy workloads (PR #299's setup), the gap is reportedly larger; we did not reproduce that.

A separate, larger overhead remains: **kvcached as a whole adds ~30-50% e2e overhead vs vanilla vLLM** (pre-existing, on all kvcached branches), which the PR partially closes (~5%) but does not eliminate.

The kvctl resize poll on the fix branch costs ~7 μs/alloc in microbench (~30% in multi-thread alloc), but is invisible at e2e (<1%).
