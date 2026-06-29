# gpt-oss-120b on DGX Spark: same performance, smaller KV footprint

Two experiments on one **NVIDIA GB10 (128 GB unified memory)**:

- **Phase 1 — footprint A/B (run this first).** One model (`openai/gpt-oss-120b`,
  MXFP4 weights, **fp8 KV**), kvcached vs vanilla vLLM at the **same**
  `gpu-memory-utilization` cap, driven by ShareGPT over a concurrency sweep.
  Result: **identical** throughput/TTFT/TPOT, but kvcached's **physical** KV
  footprint is far below vanilla's static pool. → motivates Phase 2.
- **Phase 2 — co-location.** Add a guardrail model (`Qwen/Qwen2-VL-7B-Instruct`)
  on the same GPU. Baseline must statically wall off memory; kvcached lets
  gpt-oss grow into the headroom *and* fit the guard → converts the freed
  memory into throughput/latency.

## Why this works (the mechanism)

Vanilla vLLM V1 **physically allocates the entire KV pool at startup**
(`gpu_mem_util·total − weights − activation`) and never shrinks it — footprint
is constant no matter how full it is. kvcached reserves the same *virtual* pool
but maps physical pages **on demand** (CUDA VMM) and unmaps on free. Under
ShareGPT (short requests), live KV is a few GiB while the static pool is tens of
GiB → that gap is the result. gpt-oss KV is intrinsically small too (GQA, and
18 of 36 layers are sliding-window capped at 128 tokens), so the gap is large.

## ⚠️ Three things that invalidate the result if wrong

1. **Use the `kvcached` conda env (vLLM 0.19.2 + patches), not the base
   `vllm` (0.23.0).** The base build is unpatched; kvcached would silently back
   the whole pool (zero elasticity). `config.sh` activates the right env.
   After launch, confirm kvcached is actually elastic: `kvtop` (or
   `c*_mem.json`) must show `used` *breathing* with load, not pinned at the cap.
2. **MXFP4 correctness on GB10 (sm_121).** The default Marlin MXFP4 kernel is
   buggy on sm_121 (corrupt first token → `null` content). We set
   `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1` (FlashInfer FP8×MXFP4 path) and
   `launch_main.sh` runs a coherence gate ("capital of France") that aborts on
   garbage. If it fails, FlashInfer/CUTLASS FP4 isn't available in this build —
   rebuild or fall back to gpt-oss-20b to prototype.
3. **Don't saturate the pool.** If load fills the pool, kvcached maps it all
   (footprint == vanilla) and latency blows up (no longer iso-perf). Keep
   `num_requests_waiting` and `num_preemptions_total` at ~0 (the monitor records
   them; `plot_results.py` warns if not). The footprint gap shrinking toward 0
   as concurrency rises *is* the curve to show.

## Fairness: held identical across both modes
same vLLM build · same model/dtype · TP=1 · **same `--gpu-memory-utilization`** ·
same `--max-model-len`/`--max-num-seqs`/`--block-size` · `--enforce-eager` ·
`--no-enable-prefix-caching` · `--disable-hybrid-kv-cache-manager` (kvcached's
documented gpt-oss route; both sides use it so the SWA saving is given up
equally and we isolate elasticity) · `--kv-cache-dtype fp8` · same ShareGPT
file + seed + concurrency points. The **only** difference is the kvcached env.

## Run

```bash
cd benchmarks/bench_dgx_spark_gptoss

# Phase 1 (footprint): baseline + kvcached + plots
./run_phase1.sh                       # -> results/{baseline,kvcached}/, results/*.png

# Phase 2 (co-location): baseline + kvcached
./run_phase2.sh                       # -> results_phase2/{baseline,kvcached}/
```

Key knobs (env-overridable, see `config.sh`): `GPU_UTIL` (0.80, both modes),
`MAX_MODEL_LEN` (16384), `CONCURRENCIES` ("1 4 8 16 32 64"), `KV_CACHE_DTYPE`
(fp8), `SHAREGPT_OUTPUT_LEN` (empty = dataset-natural).

## What to measure

- **Same performance**: `results/<mode>/cN.json` (from `vllm bench serve`) —
  `request_throughput`, `mean/p99_ttft_ms`, `mean_tpot_ms`, `mean_itl_ms`,
  `*_e2el_ms`. Baseline and kvcached should coincide within noise.
- **Smaller usage**: `results/<mode>/cN_mem.json` — kvcached
  `kvcached_peak_physical_gib` (= `used_size + prealloc_size` high-water) vs the
  vanilla static pool (`logs/main_baseline_pool.txt`, the
  "reserved for KV Cache" / "GPU KV cache size" line). Saving = pool − peak.
- Live view during a run: `kvtop` (shows used/prealloc/total + GPU memory).

## Files
`config.sh` shared config · `launch_main.sh` gpt-oss server (baseline|kvcached) ·
`launch_guard.sh` Qwen2-VL guard · `kv_monitor.py` peak physical-KV sampler ·
`bench_single.sh`/`run_phase1.sh` Phase 1 · `workflow_benchmark.py` +
`bench_workflow.sh`/`run_phase2.sh` Phase 2 · `plot_results.py` · `stop.sh`.
