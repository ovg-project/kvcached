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

---

## G. Investigating the 30–50% e2e gap — the layout flag is the entire story

Three hypotheses for the pre-existing kvcached overhead (Section E), tested on the fix branch with the same setup as Section E (Qwen3-0.6B, gpu-mem-util 0.5, max-model-len 2048, random 512in/128out, 500 prompts, 3 seeds, 100-prompt warmup discarded; all medians).

### G.1 Burst alloc is **not** the cause

`KVCACHED_MIN_RESERVED_PAGES=50` + `KVCACHED_MAX_RESERVED_PAGES=200` (vs default 5/10) — supposed to absorb burst alloc demand at request-rate=inf:

| | tput | mean TTFT | mean TPOT |
|---|--:|--:|--:|
| kvcached default (5/10) | 9.87 | 16555 ms | 177.51 ms |
| kvcached RESERVED=50/200 | 9.77 | 16548 ms | 179.38 ms |

Statistically identical. The reserved-pool size has **no e2e effect** at this workload — the slow-path `cuMemMap` cost (4 ms/page from Section C) amortises out across the run.

### G.2 KV layout is the **entire** cause

`KVCACHED_CONTIGUOUS_LAYOUT=false` switches from one big shared buffer with cross-layer interleaving to **per-layer separate VMM pools**, matching vanilla vLLM's allocation pattern. Per-layer block stride drops from `num_layers × per_block_bytes` (28 × 64 KB = 1.79 MB on Qwen3-0.6B) back to `per_block_bytes` (64 KB).

| rate=inf | tput (req/s) | mean TTFT | mean TPOT | P99 TPOT |
|---|--:|--:|--:|--:|
| **vanilla vLLM** | 14.25 | 11565 ms | 118.42 ms | 142.33 ms |
| kvcached default (LAYOUT=true) | 9.87 (-31%) | 16555 (+43%) | 177.51 (+50%) | 230.84 (+62%) |
| **kvcached LAYOUT=false** | **14.17 (-1%)** | **11642 (+1%)** | **119.03 (+0.5%)** | **142.72 (+0.3%)** |

| rate=16 | tput (req/s) | mean TTFT | mean TPOT | P99 TPOT |
|---|--:|--:|--:|--:|
| **vanilla vLLM** | 13.20 | 240 ms | 73.62 ms | 117.27 ms |
| kvcached default (LAYOUT=true) | 9.88 (-25%) | 1574 (6.6×) | 141.93 (+93%) | 226.60 (+93%) |
| **kvcached LAYOUT=false + RESERVED=200** | **13.17 (-0.2%)** | **246 (+2%)** | **73.67 (+0%)** | **117.01 (-0.2%)** |

`KVCACHED_CONTIGUOUS_LAYOUT=false` **eliminates the e2e gap entirely** — kvcached is statistically indistinguishable from vanilla vLLM on every metric, at both burst (rate=inf) and sustained (rate=16) load. **There is no residual "VMM mapping cost"; the entire 30-50% e2e overhead was the layer-interleaved layout.**

### G.3 Mechanism

`CONTIGUOUS_LAYOUT=true` (default) packs all layers' KV into one shared VM range with shape `[num_blocks, num_layers, 2, block_size, head_num, head_dim]`, and exposes per-layer tensors as strided views. Within one layer, block n+1 is `num_layers × per_block_bytes` bytes after block n — for Qwen3-0.6B that's 1.79 MB.

During decode, FlashAttention reads **all blocks of one layer** for each request before moving to the next layer. With layer-interleaved layout, each block read crosses a different 2 MB VMM page → TLB pressure goes up roughly `num_layers`× vs vanilla. With per-layer pools (`LAYOUT=false`), consecutive blocks of one layer are tightly packed in one buffer, just like vanilla.

This is consistent with the gap scaling with `num_layers` — small for shallow models (which is why Section E only saw +30-50% on a 28-layer model), and presumably much worse on the 32-layer Llama2-7B in PR #299's reported setup.

### G.4 Required fix to make `LAYOUT=false` actually work

`kvcached/kv_cache_manager.py:107` hardcoded `contiguous_layout=True` when constructing the C++ `PageAllocator`, regardless of the env var. With `KVCACHED_CONTIGUOUS_LAYOUT=false`, the FTensor side switched to per-layer buffers but the PageAllocator continued computing offsets as if layers were striped, leading to `cuMemUnmap (...) failed in CUDA driver (1): invalid argument` on the first eviction. Patched to read `CONTIGUOUS_LAYOUT` from `kvcached.utils`. Diff:

```python
-from kvcached.utils import DEFAULT_IPC_NAME, PAGE_SIZE, SANITY_CHECK, get_kvcached_logger
+from kvcached.utils import (CONTIGUOUS_LAYOUT, DEFAULT_IPC_NAME, PAGE_SIZE,
+                             SANITY_CHECK, get_kvcached_logger)
@@
-            contiguous_layout=True,
+            contiguous_layout=CONTIGUOUS_LAYOUT,
```

### G.5 Implication

The kvcached "pre-existing overhead" framed in Section E is not pre-existing at all — it's a default setting. **Changing one env var (`KVCACHED_CONTIGUOUS_LAYOUT=false`) closes 100% of the gap**, and the previously-reported "~5% PR speedup" becomes the marginal piece of a feature whose primary cost was elsewhere.

Open question: is there any reason `CONTIGUOUS_LAYOUT=true` is the default? The only place it's required is for hybrid-linear (mamba) models, which already check and bail out (`interfaces.py:138`). For all standard MHA/GQA/MLA workloads, `LAYOUT=false` should be the default.
