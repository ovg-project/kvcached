# H100 Verification: `KVCACHED_CONTIGUOUS_LAYOUT` Overhead

**Platform:** NVIDIA H100 80GB HBM3, x86_64  
**Model:** `Qwen/Qwen3-0.6B` (28 layers, 8 KV heads, head_dim 128, bf16)  
**vLLM:** 0.19.0  
**Setup:** `vllm serve --gpu-memory-utilization 0.5 --max-model-len 2048`  
**Bench:** `vllm bench serve` random 512in/128out, 500 prompts, 3 seeds (seeds 42/99/7), means reported

---

## 1. E2E Sweep Results

### rate=inf (throughput-bound)

| Config | Env | Throughput (req/s) | TTFT (ms) | TPOT (ms) |
|---|---|--:|--:|--:|
| A — vanilla | — | 94.91 | 2905 | 5.45 |
| B — kvcached default | `LAYOUT=true` | 95.24 (+0.3%) | 2829 | 6.43 |
| C — layout=false | `LAYOUT=false` | 75.96 **(-20%)** | 2846 | 17.09 |
| D — reserved200 | `LAYOUT=true` + reserved | 92.75 (-2%) | 2938 | 5.88 |
| E — both knobs | `LAYOUT=false` + reserved | 95.01 (+0.1%) | 2868 | 6.16 |

### rate=16 (latency-bound)

| Config | Env | Throughput (req/s) | TTFT (ms) | TPOT (ms) |
|---|---|--:|--:|--:|
| F — vanilla | — | 15.87 | 38.9 | 2.39 |
| G — kvcached default | `LAYOUT=true` | 15.87 (0%) | 40.5 | 2.45 |
| H — kvcached best | `LAYOUT=false` + reserved | 15.87 (0%) | 39.6 | 2.42 |

**Key observations:**
- `LAYOUT=true` (kvcached default) matches vanilla on H100 with no tuning needed.
- `LAYOUT=false` *without* reserved pages causes a **~20% throughput drop and 3× TPOT regression**.
- `LAYOUT=false` *with* reserved pages (`MIN=50, MAX=200`) fully recovers performance.
- At rate=16, all configs are functionally identical — the server is not the bottleneck.

---

## 2. Comparison with README (GB10/aarch64)

The README was benchmarked on a GB10 Grace Hopper (aarch64, unified CPU-GPU memory). Results are **inverted** on H100:

| Layout | GB10 throughput | H100 throughput |
|---|---|---|
| `LAYOUT=true` (default) | 9.87 req/s **(-31%)** vs vanilla | 95.24 req/s **(+0.3%)** vs vanilla |
| `LAYOUT=false` | 14.17 req/s **(−1%)** vs vanilla | 75.96 req/s **(-20%)** vs vanilla |

The recommendation in the README to flip the default to `LAYOUT=false` is **GB10-specific and does not apply to H100**.

---

## 3. Root Cause: CPU-side VMM Driver Overhead

### Hypothesis

`LAYOUT=false` requires one `cuMemMap`/`cuMemSetAccess`/`cuMemUnmap` call **per layer per K/V buffer** per page:

- `LAYOUT=true`: **1** `cuMemMap` call per page (compound page covers all layers)
- `LAYOUT=false`: **num_layers × 2 = 28 × 2 = 56** calls per page (per-layer K+V mapping)

Without reserved pages, these 56 calls happen **synchronously on the decode hot path**, stalling the scheduler between every decode step that needs a new KV block.

With reserved pages, a background thread pre-maps pages during GPU idle time. The decode hot path only pops from a pre-filled pool — **zero driver calls in the critical path**.

### nsys Verification

`nsys profile` run: 100 prompts at rate=inf, 30-prompt warmup, `--capture-range=cudaProfilerApi`.

#### GPU kernel time (total across capture window)

| | layout_false | layout_true | Δ |
|---|--:|--:|--:|
| `nvjet_tst_*` (attention) | ~80 ms | ~80 ms | ≈ 0 |
| Total kernel time | 258 ms | 284 ms | +10% |

`flash_fwd_splitkv_kernel` (the bottleneck on GB10) does not appear — this vLLM version uses FlashInfer's JIT-compiled `nvjet_tst_*` kernels on H100. Attention kernel time is **essentially identical** between layouts.

#### CPU-side CUDA driver calls (CUPTI RUNTIME)

| API | layout_false | layout_true | Ratio |
|---|--:|--:|--:|
| `cuMemSetAccess` | **359 ms**, 5992 calls | 8 ms, 100 calls | 60× |
| `cuMemUnmap` | **249 ms**, 5992 calls | 6 ms, 100 calls | 60× |
| `cuMemCreate` | **94 ms**, 3304 calls | 3 ms, 55 calls | 60× |
| `cuMemRelease` | **63 ms**, 2688 calls | 3 ms, 45 calls | 60× |
| `cuMemMap` | 20 ms, 5992 calls | 0.5 ms, 100 calls | 60× |
| **VMM total** | **~785 ms** | **~20 ms** | **39×** |

Call count ratio (~60×) matches the theoretical prediction of `num_layers × 2 = 56×` from `allocator.cpp:182-198`.

`cuMemSetAccess` is the dominant cost (359 ms), more expensive than `cuMemMap` itself (20 ms).

### Why the TLB issue (GB10) does not dominate on H100

On GB10, `LAYOUT=true` causes FlashAttention to read KV blocks with a stride of `num_layers × block_size = 1.75 MB`, which matches the 2 MB VMM page size. Every block read is a TLB miss. On GB10's unified memory architecture, TLB misses require coordinating CPU+GPU page tables, making them very costly.

On H100 (discrete HBM3, 3.35 TB/s), the same stride pattern exists but:
- The GPU's 50 MB L2 cache absorbs many misses
- TLB miss penalty is lower with discrete GPU memory
- The `nvjet_tst_*` kernels (FlashInfer) may handle paged access differently than `flash_fwd_splitkv_kernel`

As a result, the attention kernel times are nearly identical across layouts on H100. The bottleneck shifts entirely to the CPU-side VMM calls.

---

## 4. Summary

| Finding | GB10 (README) | H100 (this run) |
|---|---|---|
| Bottleneck | GPU: FlashAttention TLB miss (`flash_fwd_splitkv_kernel` +56%) | CPU: `cuMemSetAccess` / `cuMemUnmap` (60× more calls with `LAYOUT=false`) |
| Bad layout | `LAYOUT=true` (default) | `LAYOUT=false` without reserved pages |
| Fix | `LAYOUT=false` alone | `LAYOUT=true` (default) OR `LAYOUT=false` + reserved pages |
| Attention kernel | `flash_fwd_splitkv_kernel` | FlashInfer `nvjet_tst_*` |

**Recommendation for H100:** Keep the kvcached default (`LAYOUT=true`). It matches vanilla vLLM with no extra configuration. If `LAYOUT=false` is needed (e.g., debugging, non-hybrid models with very deep layer counts), pair it with `KVCACHED_MIN_RESERVED_PAGES=50 KVCACHED_MAX_RESERVED_PAGES=200` to eliminate the allocation hot-path stall.
