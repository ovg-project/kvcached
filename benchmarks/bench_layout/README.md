# vLLM e2e + `KVCACHED_CONTIGUOUS_LAYOUT` overhead

Where the 30-50% kvcached-vs-vanilla e2e overhead comes from, and why `KVCACHED_CONTIGUOUS_LAYOUT=false` closes it.

For the alloc/free microbench, see [`../bench_alloc/README.md`](../bench_alloc/README.md).

## Setup

NVIDIA GB10 (aarch64). `Qwen/Qwen3-0.6B` (28 layers, 8 KV heads, head_dim 128, bf16). `vllm serve --gpu-memory-utilization 0.5 --max-model-len 2048`. Bench: `vllm bench serve` random 512in/128out, 500 prompts, 3 seeds, median.

Variants compared: **vanilla** vs **kvcached** (Python allocator / C++ allocator / C++ + restored resize — all three behave identically at e2e).

## Run

```bash
# E2E sweep (vanilla vs kvcached × LAYOUT × reserved pool)
bash run_sweep.sh
python parse_results.py sweep_results/

# Kernel-level profile under both layouts
bash run_nsys_layout.sh
python diff_nsys_kernels.py nsys_runs/layout_false.nsys-rep nsys_runs/layout_true.nsys-rep
```

Intermediate outputs (`sweep_results/`, `nsys_runs/`, ...) are reproducible from the scripts and not tracked in git.

## 1. The gap

`rate=inf`, 500 prompts:

| | tput (req/s) | TTFT mean (ms) | TPOT mean (ms) |
|---|--:|--:|--:|
| vanilla | 14.21 | 11575 | 119.3 |
| kvcached default (`LAYOUT=true`) | 9.87 (-31%) | 16555 | 177.5 |
| kvcached + `LAYOUT=false` | 14.17 (-1%) | 11642 | 119.0 |

`LAYOUT=false` matches vanilla on every metric, also at `rate=16` (sustained load). The C++ allocator from PR #319 only closes ~5% of the kvcached overhead; reserved-pool size has no measurable effect. **The entire 30% gap is the layout default.**

## 2. Mechanism

### Stride math (from `interfaces.py:282-289`)

Under `CONTIGUOUS_LAYOUT=true`, the raw KV tensor has shape `[num_blocks, num_layers, k/v, token, head, dim]`. When sliced to layer `i`, block n→n+1 stride is `num_layers × per_block_bytes`. For Qwen3-0.6B:

- per-block K+V, one layer = 16 × 8 × 128 × 2 = **64 KB**
- stride **`LAYOUT=true`** = 28 × 64 KB = **1.75 MB** (≈ VMM page size = 2 MB)
- stride **`LAYOUT=false`** = **64 KB** (~32 blocks share one 2 MB page)

So every FlashAttention block read crosses a fresh 2 MB VMM page under `LAYOUT=true`; non-contiguous packs them densely.

### nsys per-kernel attribution

Same workload as section 1. Total GPU kernel time grows by **+8,043 ms (+34.8%)** going from `LAYOUT=false → true`:

| kernel | calls | `LAYOUT=false` ms | `LAYOUT=true` ms | Δms | Δ% |
|---|--:|--:|--:|--:|--:|
| `flash::flash_fwd_splitkv_kernel` (KV-read) | 3948 | 14,666 | **22,879** | **+8,213** | **+56.0%** |
| `cutlass_80_..._gemm_relu` | 7756 | 2,277 | 2,214 | -63 | -2.8% |
| `vllm::reshape_and_cache_flash_kernel` (KV-write) | 3948 | 302 | 271 | -32 | -10.5% |
| (everything else) | — | ~6,000 | ~6,000 | ~0 | flat |

**One kernel — FlashAttention's split-KV read — accounts for more than the entire e2e gap.** Every other kernel is unchanged or noise. The KV-*write* kernel is even slightly faster under contiguous (writes one position per request, no stride pattern).

### Scaling

Per-call attention slowdown grows with concurrent working-set size — consistent with TLB/L2 pressure:

| workload | `LAYOUT=false` μs/call | `LAYOUT=true` μs/call | Δ |
|---|--:|--:|--:|
| 100 prompts | 851 | 1163 | +37% |
| 500 prompts | 3,715 | 5,795 | **+56%** |

`LAYOUT=false` is flat with workload because ~32 blocks share each page. Deeper models (32-layer Llama2-7B, 48+ layer Llama3-70B) cross the 2 MB page boundary even more sharply.

## 3. Patch needed for `LAYOUT=false`

`kvcached/kv_cache_manager.py:107` previously hardcoded `contiguous_layout=True`. With `LAYOUT=false`, FTensor used per-layer buffers but `PageAllocator` still computed striped offsets → `cuMemUnmap (1): invalid argument` on first eviction. Fix: wire `CONTIGUOUS_LAYOUT` env var into the `KVCacheManager` constructor.

## Recommendation

`CONTIGUOUS_LAYOUT=true` is the kvcached default but is only needed for hybrid-linear/mamba models (`interfaces.py:138` already detects those and bails out of the non-contiguous path). On standard MHA/GQA/MLA, flipping the default to `false` eliminates the entire kvcached-vs-vanilla e2e overhead measured here.
