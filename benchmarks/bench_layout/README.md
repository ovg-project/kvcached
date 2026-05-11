# vLLM e2e + `KVCACHED_CONTIGUOUS_LAYOUT` overhead

Why kvcached is 30-50% slower than vanilla vLLM by default, and why `KVCACHED_CONTIGUOUS_LAYOUT=false` fixes it.

For the alloc/free microbench, see [`../bench_alloc/README.md`](../bench_alloc/README.md).

## Setup

GB10 (aarch64). `Qwen/Qwen3-0.6B` (28 layers, 8 KV heads, head_dim 128, bf16). `vllm serve --gpu-memory-utilization 0.5 --max-model-len 2048`. Bench: `vllm bench serve` random 512in/128out, 500 prompts, 3 seeds.

## Run

```bash
# E2E sweep (vanilla vs kvcached × LAYOUT × reserved pool)
bash run_sweep.sh && python parse_results.py sweep_results/

# Kernel-level profile under both layouts
bash run_nsys_layout.sh
python diff_nsys_kernels.py nsys_runs/layout_false.nsys-rep nsys_runs/layout_true.nsys-rep
```

Intermediate outputs aren't tracked in git — reproducible from the scripts.

## 1. The gap

500 prompts at `rate=inf`:

| | tput (req/s) | TTFT mean (ms) | TPOT mean (ms) |
|---|--:|--:|--:|
| vanilla | 14.21 | 11575 | 119.3 |
| kvcached (`LAYOUT=true`, default) | 9.87 (-31%) | 16555 | 177.5 |
| kvcached + `LAYOUT=false` | 14.17 (-1%) | 11642 | 119.0 |

`LAYOUT=false` matches vanilla on every metric, also at `rate=16` (sustained load). The C++ allocator from PR #319 only buys back ~5%; reserve-pool size doesn't help either. **It's all the layout.**

## 2. Where the gap actually is

### Stride math

`CONTIGUOUS_LAYOUT=true` lays out KV as `[num_blocks, num_layers, k/v, token, head, dim]` (`interfaces.py:282-289`). When you slice down to one layer, block n→n+1 stride is `num_layers × per_block_bytes`. For Qwen3-0.6B:

- per-block K+V, one layer = 16·8·128·2 = **64 KB**
- stride under `LAYOUT=true` = 28 × 64 KB = **1.75 MB** (≈ VMM page = 2 MB)
- stride under `LAYOUT=false` = **64 KB** (~32 blocks share a page)

So under contiguous, every FlashAttention block read lands on its own fresh 2 MB page. Non-contiguous packs 32 blocks per page. The attention kernel can't hide that.

### nsys per-kernel breakdown

Same workload as Section 1. Going from `LAYOUT=false → true` adds **+8,043 ms (+34.8%)** total GPU kernel time, all in one kernel:

| kernel | calls | LAYOUT=false ms | LAYOUT=true ms | Δ |
|---|--:|--:|--:|--:|
| `flash::flash_fwd_splitkv_kernel` (KV-read) | 3948 | 14,666 | 22,879 | **+8,213 (+56%)** |
| `vllm::reshape_and_cache_flash_kernel` (KV-write) | 3948 | 302 | 271 | -32 (-11%) |
| everything else | — | ~8,000 | ~8,000 | ~0 |

That one kernel exceeds the entire gap. Worth noting: the KV-*write* kernel isn't affected — only the multi-block read path is. Writes are sequential per-position so they never hit the cross-page stride.

### Scales with working set

Per-call attention time:

- 100 prompts: 1163 vs 851 μs (**+37%**)
- 500 prompts: 5795 vs 3715 μs (**+56%**)

More concurrent requests → larger working set → more distinct 2 MB pages touched → worse TLB/L2 hit rate. `LAYOUT=false` stays flat because 32 blocks share one page. Deeper models (Llama2-7B at 32 layers, Llama3-70B at 80) cross the page boundary even harder.

## 3. Where `LAYOUT=true` still wins

Three things to put on the other side of the scale.

**Hybrid linear / mamba: required.** Mamba state shares the KV buffer and indexes by virtual block across layers. `interfaces.py:138` outright refuses non-contiguous for hybrid-linear configs.

**Init time: ~1.4 s faster at server boot.** Contiguous reserves one big VM range; non-contiguous reserves `num_layers` separate ones. Measured `alloc_kv_cache` (16 layers, 1 GB/layer): 635 ms vs 2055 ms. ~99% of that 1.4 s is `FTensor::init_with_zero_()` mapping the zero-page over the entire VM range — contiguous uses a 64 MB compound page so it makes 1947 `cuMemMap` calls (~325 μs each); non-contiguous uses 2 MB pages and makes 62,304 calls (~33 μs each). CUDA driver per-call overhead is the dominant cost, and bigger pages amortise it better.

The gap stays roughly flat across `num_layers ∈ {8..80}` (1.3–1.5 s), one-shot at startup.

**Alloc/free hot path: ~2× faster.** Each page mapping under contiguous = 1 `cuMemMap`; under non-contiguous = `num_layers × (K+V)` FTensor `map()` calls. Cold path (`RESERVED=0`) shows a consistent 2.1× ratio; steady-state at small `k` is similar, collapsing to ~1× at `k=256`.

### When does the trade-off flip?

Attention overhead hits every decode step. Startup hits once. For the Section 1 workload:

- `LAYOUT=true` startup advantage: ~1.4 s
- `LAYOUT=false` throughput advantage: 14.17 vs 9.83 req/s, ≈ 31 ms/req

Break-even at **~45 requests**. Above that, non-contiguous wins on total wall-clock; below, contiguous's faster boot wins. Deeper models shift the break-even down further.

So contiguous still wins for: smoke tests, single-shot inference, request-level autoscaling, boot-SLA workloads, hybrid linear/mamba (forced). Everything else: non-contiguous.

## Summary

The kvcached default `CONTIGUOUS_LAYOUT=true` costs ~30% e2e throughput on standard MHA/GQA/MLA because every FlashAttention block read crosses a fresh 2 MB VMM page. Flipping to `LAYOUT=false` closes the gap entirely, at the price of ~1.4 s extra startup that's paid off in tens of requests.

The default should flip to `false` for non-hybrid models; `interfaces.py:138` already handles the hybrid-linear case where contiguous is mandatory.
