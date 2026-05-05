NVIDIA GB10 (aarch64), clean rebuild between runs. `main` = `ovg-project/kvcached` main, `PR` = `lianghao_c++` + must-fix `98d9bb3`, `fix` = `fix/pr319-restore-resize`.

## A. `available_size()`

PR drops `cudaMemGetInfo` from the hot path.

| | μs/call |
|--:|--:|
| main | 6.52 |
| PR | 0.52 |
| fix | 0.52 |

12.5×. Side effect: `available_size()` no longer caps by physical memory.

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

## E. End-to-end vLLM serving

`vllm bench serve` against a local server, `ENABLE_KVCACHED=true`. Qwen3-0.6B, gpu-mem-util 0.5, max-model-len 2048, random 512in/128out, 500 prompts, `request-rate=inf`. 100-prompt warmup discarded; 3 seeds (42, 99, 7), median.

| | tput | TTFT mean | TTFT P99 | TPOT mean | TPOT P99 |
|---|--:|--:|--:|--:|--:|
| vanilla | 14.21 | 11575 | 26377 | 119.3 | 143.2 |
| main | 9.42 (-34%) | 17284 (+49%) | 40464 | 185.1 | 239.6 |
| PR | 9.86 (-31%) | 16472 | 38756 | 177.5 | 230.2 |
| fix | 9.83 (-31%) | 16602 | 38746 | 178.2 | 231.7 |

(req/s, ms.) PR closes ~5% of the kvcached overhead; fix retains it. The remaining ~30% gap to vanilla is the layout default — see G.

## G. Where the 30-50% e2e gap actually comes from

Three hypotheses, tested on fix with the Section E setup.

### G.1 Reserved-pool size — no e2e effect

| | tput | TTFT mean | TPOT mean |
|---|--:|--:|--:|
| RESERVED=5/10 (default) | 9.87 | 16555 | 177.51 |
| RESERVED=50/200 | 9.77 | 16548 | 179.38 |

`cuMemMap` cost (4 ms/page from C) amortises across the run.

### G.2 KV layout — the entire cause

`KVCACHED_CONTIGUOUS_LAYOUT=false` switches from one shared buffer with cross-layer interleaving to per-layer VMM pools, matching vanilla's allocation pattern.

rate=inf:
| | tput | TTFT mean | TPOT mean | TPOT P99 |
|---|--:|--:|--:|--:|
| vanilla | 14.25 | 11565 | 118.42 | 142.33 |
| LAYOUT=true (default) | 9.87 (-31%) | 16555 | 177.51 | 230.84 |
| LAYOUT=false | 14.17 (-1%) | 11642 | 119.03 | 142.72 |

rate=16:
| | tput | TTFT mean | TPOT mean | TPOT P99 |
|---|--:|--:|--:|--:|
| vanilla | 13.20 | 240 | 73.62 | 117.27 |
| LAYOUT=true | 9.88 (-25%) | 1574 | 141.93 | 226.60 |
| LAYOUT=false + RESERVED=200 | 13.17 (-0.2%) | 246 | 73.67 | 117.01 |

`LAYOUT=false` is indistinguishable from vanilla on every metric, at burst (inf) and sustained (16) load. The whole 30-50% overhead is the layout default — there's no residual VMM mapping cost.

### G.3 Mechanism

`CONTIGUOUS_LAYOUT=true` packs all layers into one VM range with shape `[num_blocks, num_layers, 2, block_size, head_num, head_dim]`. Within a layer, block n+1 sits `num_layers × per_block_bytes` after block n — 1.79 MB on Qwen3-0.6B (28 layers × 64 KB).

FlashAttention reads all blocks of one layer before moving to the next. With this layout, each block read crosses a fresh 2 MB VMM page → ~`num_layers`× TLB pressure. Per-layer pools pack same-layer blocks contiguously.

Effect scales with `num_layers`: 28 layers gave +30-50%; deeper models (32-layer Llama2-7B in #299) should be worse.

### G.4 Patch

`kvcached/kv_cache_manager.py:107` hardcoded `contiguous_layout=True`. With `LAYOUT=false`, FTensor used per-layer buffers but `PageAllocator` still computed striped offsets → `cuMemUnmap (1): invalid argument` on first eviction.

```python
-from kvcached.utils import DEFAULT_IPC_NAME, PAGE_SIZE, SANITY_CHECK, get_kvcached_logger
+from kvcached.utils import (CONTIGUOUS_LAYOUT, DEFAULT_IPC_NAME, PAGE_SIZE,
+                             SANITY_CHECK, get_kvcached_logger)
@@
-            contiguous_layout=True,
+            contiguous_layout=CONTIGUOUS_LAYOUT,
```

## Summary

| | speedup vs main | fix retains |
|---|---|---|
| A. `available_size()` | 12.5× | full |
| B. `group_indices_by_page` | ~3× | full |
| C. slow-path alloc | 1× (driver-bound) | full |
| D. multi-thread | 2-5× | ~70% of PR |
| F. block-object pool | ~7× (PR-only) | full |
| E. e2e (LAYOUT=true) | ~5% | full |

Microbench wins amortise to 5% e2e because per-token model forward dominates.

The bigger lever is unrelated to the C++ migration: `KVCACHED_CONTIGUOUS_LAYOUT=false` closes the entire kvcached-vs-vanilla gap on standard MHA/GQA/MLA. Open question — why is `true` the default? Hybrid-linear/mamba models need it, but they already check and bail at `interfaces.py:138`.
