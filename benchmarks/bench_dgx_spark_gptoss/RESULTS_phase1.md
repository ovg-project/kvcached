# Phase 1 results — gpt-oss-120b, same performance, smaller KV footprint

**Hardware/stack:** NVIDIA GB10 (DGX Spark, 121.7 GiB unified), vLLM 0.19.2.dev0
(kvcached env) + kvcached 0.1.5. Model `openai/gpt-oss-120b` (MXFP4 weights via
**Marlin**, **fp8 KV**), `--max-model-len 16384`, `--max-num-seqs 128`,
`--enforce-eager`, `--no-enable-prefix-caching`, `--disable-hybrid-kv-cache-manager`.
**Identical `--gpu-memory-utilization 0.70` on both modes** → ~16.6 GiB virtual
KV pool either way. Driven by **ShareGPT** over `/v1/completions`, 256-token
outputs, concurrency sweep, run 2026-06-28.

## Same performance (within run-to-run noise; 0 failed, 0 preemptions)

| C | throughput (req/s) base→kvc | mean TTFT (ms) base→kvc | mean TPOT (ms) base→kvc |
|--:|:--:|:--:|:--:|
| 1  | 0.123 → 0.123 | 225 → 184 | 31.0 → 31.1 |
| 8  | 0.554 → 0.560 | 882 → 856 | 53.1 → 52.7 |
| 32 | 1.362 → 1.367 | 1300 → 1362 | 86.9 → 86.4 |

## Smaller KV footprint (the win)

Vanilla statically reserves the whole KV pool at startup; kvcached maps physical
pages on demand (peak of `used+prealloc` from the shm `MemInfoStruct`).

| C | baseline reserved | kvcached peak physical | **saved** | KV pool logical fill |
|--:|:--:|:--:|:--:|:--:|
| idle | 16.58 GiB | 0.84 GiB | 15.7 GiB | 0% |
| 1  | 16.58 GiB | 0.84 GiB | **15.7 GiB (95%)** | 0.22% |
| 8  | 16.58 GiB | 0.98 GiB | **15.6 GiB (94%)** | 0.93% |
| 32 | 16.58 GiB | 1.41 GiB | **15.2 GiB (92%)** | 3.40% |

**Takeaway:** at identical throughput/latency, kvcached's physical KV footprint
is **0.8–1.4 GiB vs the 16.6 GiB vanilla nails up — ~92–95% less.** The vanilla
pool sits 96–99% idle under this load; kvcached returns that memory to the device.
Artifacts: `results/phase1_summary.csv`, `results/perf_vs_concurrency.png`,
`results/kv_footprint_vs_concurrency.png`.

## Notes / gotchas hit and resolved (GB10-specific)
- **MXFP4 backend:** FlashInfer TRTLLM/CUTLASS kernels don't support sm_121 (and
  flashinfer isn't importable); the OAI triton_kernels path needs capability
  <11.0. Forced **Marlin** (`VLLM_MXFP4_USE_MARLIN=1`, and *unset* the FlashInfer
  envs). Marlin dequants FP4→bf16: correct, portable, somewhat slower — fine for
  an A/B where both sides pay it.
- **Harmony vocab:** gpt-oss needs the o200k_base tiktoken vocab; auto-download is
  flaky here → pre-fetched to `~/.cache/harmony_encodings` and pointed
  `TIKTOKEN_ENCODINGS_BASE` at it.
- **Chat vs completions:** on this build the gpt-oss **Harmony chat** path returns
  null content, but **raw `/v1/completions` is coherent** ("…the capital of France
  is Paris."). Compute is correct; we benchmark via completions.
- **Unified-memory caveat:** `nvidia-smi`/`torch.cuda.mem_get_info` report the whole
  shared device (incl. desktop ~24 GiB), so they are not a clean per-engine signal.
  The real signals: vanilla pool from the serve log ("Available KV cache memory"),
  kvcached physical from the shm (`used+prealloc`).
- **Regime:** preemptions=0 at every C (KV never the bottleneck), so this is
  genuinely iso-performance. The footprint gap narrows as C rises — extrapolated,
  it closes only near pool saturation.
