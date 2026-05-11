# `bench_layout` — vLLM e2e + `KVCACHED_CONTIGUOUS_LAYOUT` overhead

NVIDIA GB10 (aarch64). `main` = `ovg-project/kvcached` main, `PR` = `lianghao_c++` + must-fix `98d9bb3`, `fix` = `fix/pr319-restore-resize` (PR #319 head + restore-resize + bench scripts).

For the alloc/free microbench numbers (Sections A-D, F), see [`../bench_alloc/bench_result.md`](../bench_alloc/bench_result.md).

Driver scripts:
- `run_sweep.sh` / `run_kvcached_configs.sh` / `run_layout_retest.sh` — `vllm bench serve` sweeps over branch × layout × reserved-pool size
- `run_nsys_layout.sh` — same workload under `nsys profile`, brackets capture window via `/start_profile` + `/stop_profile`
- `diff_nsys_kernels.py` — per-kernel time diff between two nsys traces
- `run_ncu_attn.sh` — ncu memory-workload analysis on `flash_fwd_splitkv_kernel` (needs `NVreg_RestrictProfilingToAdminUsers=0`)

## E. End-to-end vLLM serving

`vllm bench serve` against a local server, `ENABLE_KVCACHED=true`. Qwen3-0.6B, gpu-mem-util 0.5, max-model-len 2048, random 512in/128out, 500 prompts, `request-rate=inf`. 100-prompt warmup discarded; 3 seeds (42, 99, 7), median.

| | tput | TTFT mean | TTFT P99 | TPOT mean | TPOT P99 |
|---|--:|--:|--:|--:|--:|
| vanilla | 14.21 | 11575 | 26377 | 119.3 | 143.2 |
| main | 9.42 (-34%) | 17284 (+49%) | 40464 | 185.1 | 239.6 |
| PR | 9.86 (-31%) | 16472 | 38756 | 177.5 | 230.2 |
| fix | 9.83 (-31%) | 16602 | 38746 | 178.2 | 231.7 |

(req/s, ms.) PR closes ~5% of the kvcached overhead vs main; fix retains it. The remaining ~30% gap to vanilla is the layout default — see G.

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

#### G.3.1 The stride (from code)

`CONTIGUOUS_LAYOUT=true` packs all layers into one VM range. At `kvcached/integration/vllm/interfaces.py:282-289`:

```python
contiguous_shape = [num_blocks_per_layer, num_layers] + layer_elem_shape
contiguous_tensor = raw_kv_tensors[0].view(dtype=dtype)[:num_eles].view(contiguous_shape)
kv_tensors = [contiguous_tensor[:, i].permute(*permute_order) for i in range(num_layers)]
```

So the raw memory order is `(block, layer, k/v, token, head, dim)`. When the per-layer view `[:, i]` is sliced out for layer `i`, the stride between block `n` and block `n+1` in that view is `num_layers × per_block_bytes`. For Qwen3-0.6B:

| field | value |
|---|---|
| `num_layers` | 28 |
| `num_kv_heads` | 8 |
| `head_dim` | 128 |
| `block_size` | 16 tokens |
| dtype | bf16 (2 B) |
| per-block K (or V), one layer | 16·8·128·2 = **32 KB** |
| per-block K+V, one layer | **64 KB** |
| **block stride, contiguous=true** | 28 × 64 KB = **1.75 MB** |
| **block stride, contiguous=false** | **64 KB** (from `interfaces.py:274`: `strides[1] = hidden_size_eles`) |
| VMM page size | **2 MB** |

So contiguous=true makes every block read of FlashAttention land within ~1 byte of a new 2 MB VMM page; non-contiguous packs ~32 blocks per page.

#### G.3.2 nsys per-kernel diff

Same workload as Section E (Qwen3-0.6B, gpu-mem 0.5, rand 512/128, rate=inf, 500 prompts, 100 warmup discarded, seed=42). vLLM started under `nsys profile --capture-range=cudaProfilerApi`; profiler window bracketed by `/start_profile`+`/stop_profile`.

Total GPU-kernel time inside the profiler window:

| | total kernel time | delta |
|---|--:|--:|
| LAYOUT=false | 23,121 ms | — |
| LAYOUT=true | 31,164 ms | **+8,043 ms (+34.8%)** |

Top kernels by absolute change (file: `nsys_runs/kernel_diff_500p.txt`):

| kernel | calls | LAYOUT=false ms | LAYOUT=true ms | delta ms | delta % |
|---|--:|--:|--:|--:|--:|
| `flash::flash_fwd_splitkv_kernel` *(KV-read)* | 3948 | 14,666 | **22,879** | **+8,213** | **+56.0%** |
| `cutlass_80_tensorop_bf16_s16816gemm_relu` | 7756 | 2,277 | 2,214 | -63 | -2.8% |
| `vllm::reshape_and_cache_flash_kernel` *(KV-write)* | 3948 | 302 | 271 | -32 | -10.5% |
| `nvjet_sm121_tst_mma_128x176x64` | 3752 | 580 | 552 | -28 | -4.8% |
| `triton_red_fused_3` | 3780 | 118 | 95 | -23 | -19.8% |

The +8,213 ms FlashAttention regression alone exceeds the +8,043 ms total — every other kernel is unchanged or slightly faster (likely L2 noise from cross-run variation). One kernel explains the entire e2e gap.

Per-call attention cost:

| | LAYOUT=false μs/call | LAYOUT=true μs/call | delta μs/call | delta % |
|---|--:|--:|--:|--:|
| 100 prompts (smaller WS) | 851 | 1163 | +313 | +37% |
| 500 prompts (Section E WS) | 3,715 | 5,795 | **+2,080** | **+56%** |

The per-call gap *and* the per-call slowdown both grow with concurrent working-set size — consistent with TLB/cache misses: more in-flight requests → more distinct 2 MB pages touched per attention call → cache pressure → linear-ish blow-up in kernel time. Non-contiguous is flat because ~32 blocks share each page.

Note the asymmetry: only the **read** kernel regresses. `reshape_and_cache_flash_kernel` (which writes one new token per request per step) is unaffected — writes are sequential per-position so they don't trigger the cross-page stride pattern.

#### G.3.3 Implication

Effect scales with `num_layers`: 28 layers gave +56% per attention call; deeper models (32-layer Llama2-7B in #299) should be slightly worse, and a model with ≥48 layers crosses the 2 MB page boundary even more sharply.

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

The bigger lever is unrelated to the C++ migration: `KVCACHED_CONTIGUOUS_LAYOUT=false` closes the entire kvcached-vs-vanilla gap on standard MHA/GQA/MLA.

Open question — why is `true` the default? Hybrid-linear/mamba models need it, but they already check and bail at `interfaces.py:138`. On standard attention layouts, flipping the default would eliminate the kvcached e2e overhead measured here.
