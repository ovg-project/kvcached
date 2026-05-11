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

## 3. When does `LAYOUT=true` win?

The previous sections show non-contiguous wins big on attention. The contiguous layout still pays off on three things:

### 3.1 Hybrid linear / mamba models — required

Mamba layers share the same flat KV buffer as attention layers and index into it per virtual block across layers. `interfaces.py:138` detects hybrid-linear configs and refuses to run with `CONTIGUOUS_LAYOUT=false`. So for `HYBRID_LINEAR` model types there is no choice — contiguous is mandatory.

### 3.2 Init time — `LAYOUT=true` is ~3× faster

Contiguous reserves one big VM range; non-contiguous reserves `num_layers` separate ranges + a shared zero-page handle. Measured `alloc_kv_cache` time (ms) at varying `num_layers`, 1 GB per layer:

| `num_layers` | `LAYOUT=true` | `LAYOUT=false` | Δ |
|--:|--:|--:|--:|
| 8 | 658 | 2,058 | +1,400 |
| 16 | 640 | 1,899 | +1,259 |
| 28 | 568 | 2,058 | +1,490 |
| 32 | 632 | 1,936 | +1,304 |
| 80 | 613 | 1,890 | +1,277 |

The extra cost is a roughly flat **~1.4 s**, not linear in `num_layers` — looks like fixed-cost driver / metadata overhead per VM range plus a small per-layer term, with the per-layer term cheap enough that it stays in the noise up to 80 layers. One-shot at server startup; doesn't recur per request.

### 3.3 Alloc/free hot path — `LAYOUT=true` is ~2× faster

Each PageAllocator "page" maps to 1 cuMemMap call under contiguous, vs `num_layers × (K+V)` FTensor `map()` calls under non-contiguous. Measured with the same `bench_alloc.py` harness (16 layers):

Cold path — `KVCACHED_MIN/MAX_RESERVED_PAGES=0` forces a fresh mapping every alloc:

| k | `LAYOUT=true` μs | `LAYOUT=false` μs | ratio |
|--:|--:|--:|--:|
| 1 | 4527 | 9512 | 2.1× |
| 16 | 4554 | 9567 | 2.1× |
| 64 | 4526 | 9638 | 2.1× |
| 256 | 9240 | 19261 | 2.1× |

Steady state — default `RESERVED=5/10`, mappings cached:

| k | `LAYOUT=true` μs | `LAYOUT=false` μs | ratio |
|--:|--:|--:|--:|
| 1 | 43 | 87 | 2.0× |
| 16 | 37 | 94 | 2.5× |
| 64 | 38 | 64 | 1.7× |
| 256 | 52 | 59 | 1.1× |

Holds ~2× at small k; collapses to ~1× at larger k as bookkeeping amortises.

### 3.4 Net effect: startup vs steady-state

Per-request attention overhead hits every decode step. Startup overhead hits once. So which layout wins on **total wall-clock including server boot** depends on how many requests you serve before tearing down.

For the Section 1 workload:
- `LAYOUT=true` startup advantage: ~1.4 s (Section 3.2)
- `LAYOUT=false` throughput advantage: 14.17 vs 9.83 req/s, i.e. 31 ms saved per request on average

Break-even:

```
t_true(N)  =       N / 9.83
t_false(N) = 1.4 + N / 14.17
N* ≈ 45 requests
```

For **>45 requests** total, `LAYOUT=false` is faster wall-clock even counting startup. For **<45**, `LAYOUT=true`'s faster boot wins on total time. The 45 figure depends on the model — deeper layers shift the attention gap upward (Section 2 "Scaling") and would lower the break-even further.

So the trade-off is genuine but heavily favours `LAYOUT=false` for any real serving workload. Where `LAYOUT=true` actually pays off:

- **Very short-lived runs** — smoke tests, debugging, single-shot inference with <50 requests
- **Frequent server restarts** — request-level autoscaling, model reloading
- **Boot-time SLA** — when every second to first-ready matters
- **Hybrid linear / mamba** — forced regardless

For all other cases (long-running serving, throughput-bound deployments, latency-bound deployments under sustained load), the 1.4 s of extra startup is paid off within tens of requests and `LAYOUT=false` is strictly better thereafter.

## Recommendation

For standard MHA/GQA/MLA, `CONTIGUOUS_LAYOUT=false` is the right default — it eliminates the entire kvcached-vs-vanilla e2e overhead at the cost of a few seconds of extra startup and microseconds of extra alloc-path latency, neither of which shows up in steady-state throughput.

`CONTIGUOUS_LAYOUT=true` should remain the default only for hybrid-linear/mamba, which `interfaces.py:138` already detects.
