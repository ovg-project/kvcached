# gpt-oss-120b on DGX Spark — same performance, less committed GPU memory

kvcached maps KV-cache pages on demand instead of statically reserving the whole
pool, so at identical serving performance it commits far less GPU memory than
vanilla vLLM.

## Hardware / software

| Component | Detail |
|---|---|
| GPU | NVIDIA GB10 (DGX Spark), 121.7 GiB unified LPDDR5X |
| Engine | vLLM 0.19.2.dev0 (kvcached-patched) |
| kvcached | 0.1.5 |
| Model | `openai/gpt-oss-120b` (MXFP4 weights, fp8 KV) |
| Python / CUDA | 3.12 / 13.0 |

## Configuration

Both servers use **identical** engine flags and the **same
`--gpu-memory-utilization 0.70`**. The only difference is the kvcached env block
— so the comparison isolates *static* vs *on-demand* KV backing.

### Shared environment (both modes)

```bash
conda activate kvcached                         # vLLM 0.19.2 (NOT the base 0.23.0 on PATH)

# MXFP4 on GB10 (sm_121): Marlin is the only working MoE backend
# (FlashInfer TRTLLM/CUTLASS unsupported on sm_121; OAI triton_kernels need cap <11.0).
export VLLM_MXFP4_USE_MARLIN=1
unset  VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8 VLLM_USE_FLASHINFER_MXFP4_MOE

# gpt-oss Harmony vocab (auto-download is flaky here; pre-fetch o200k_base)
export TIKTOKEN_ENCODINGS_BASE=~/.cache/harmony_encodings   # sha256 446a9538...

export VLLM_ATTENTION_BACKEND=FLASH_ATTN        # auto-falls to TRITON_ATTN under fp8 KV
export VLLM_PLUGINS=""                           # disable the env's TriAttention plugin
                                                 # (crashes on long seqs; unrelated to kvcached)
```

### Common serve flags (identical for both modes)

```bash
COMMON="--port 12346 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 16384 --max-num-seqs 128 --block-size 16 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --disable-hybrid-kv-cache-manager"
```

### Launch — vanilla vLLM (baseline)

```bash
vllm serve openai/gpt-oss-120b $COMMON
```

### Launch — kvcached (Fluxion)

```bash
ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 \
KVCACHED_IPC_NAME=gptoss KVCACHED_CONTIGUOUS_LAYOUT=true VLLM_USE_V1=1 \
vllm serve openai/gpt-oss-120b $COMMON
```

### Workload (driven against each server)

```bash
vllm bench serve \
  --backend openai --base-url http://localhost:12346 --endpoint /v1/completions \
  --model openai/gpt-oss-120b \
  --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --sharegpt-output-len 256 \
  --num-prompts 128 --max-concurrency 8 --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90 --ignore-eos
```

Notes: gpt-oss is driven via `/v1/completions` (its Harmony *chat* path returns
null content on this build; raw completions are coherent). KV-cache memory is
sampled at 5 Hz from the kvcached POSIX-shm `MemInfoStruct`
(`used_size + prealloc_size`); the vanilla pool is read from the serve-log line
`Available KV cache memory: …`. `nvidia-smi` is not used (on GB10 unified memory
it reports the whole shared device, not the per-engine footprint).

## Results — single model: vanilla vLLM vs kvcached

ShareGPT, 256-token outputs, max-concurrency 8, 128 prompts, seed 0.

| | vanilla vLLM | kvcached (Fluxion) |
|---|---|---|
| TTFT p50 / p90 | 0.72 / 0.92 s | 0.57 / 0.80 s |
| TPOT p50 | 0.054 s | 0.054 s |
| output throughput | 142.5 tok/s | 144.0 tok/s |
| **avg committed GPU memory** | **86.6 GiB (flat)** | **66.9 GiB** |
| — of which KV cache | 20.67 GiB (flat) | **0.96 GiB** |

**Takeaway:** serving performance is identical (TTFT/TPOT/throughput within
run-to-run variance). But vanilla vLLM **statically commits its entire KV pool
(20.67 GiB) at startup and holds it flat regardless of load**, while kvcached
maps KV pages on demand and commits only **~0.96 GiB** for the same workload —
**~95% less KV memory** (~20 GiB / ~23% off total committed GPU memory). The
freed memory is returned to the device.

## Results — co-locating a guardrail model (same config)

Same engine config as above (**main `--gpu-memory-utilization 0.70`**), now adding
`meta-llama/Llama-Guard-3-8B` to screen every request on the same single GB10
(`User → guardrail → gpt-oss-120b → response`).

| | vanilla vLLM | kvcached (Fluxion) |
|---|---|---|
| main gpt-oss-120b alone | ✅ runs (commits ~86 GiB, flat) | ✅ runs (commits ~67 GiB) |
| free memory left for a 2nd model | ~8 GiB | ~25 GiB |
| **+ guardrail (Llama-Guard-8B, needs ~18 GiB)** | **❌ CUDA out of memory** | **✅ both models serve** |
| Chat (main) TTFT mean / p99 | — (cannot launch) | 0.65 / 2.80 s |
| workflow TTFT (guard → chat) mean | — | 1.59 s |
| guardrail check latency (median) | — | 0.30 s |
| workflow throughput | — | 0.50 req/s |

**Takeaway:** at the *identical* memory config, vanilla vLLM's static KV
reservation leaves only ~8 GiB free — not enough for the guardrail, which **OOMs
on startup**. kvcached commits KV on demand, leaving ~25 GiB, so **both models
run on the one GPU**. The ~18–20 GiB kvcached frees in the single-model table is
exactly what makes this multi-model agentic workflow (chat + safety model, or
voice → chat, etc.) possible on a memory-constrained box like the DGX Spark.

(Measured 2026-06-29: vanilla `main 0.70` → add guard → `torch.AcceleratorError:
CUDA error: out of memory`; kvcached `main 0.70` → add guard → both healthy,
128/128 requests completed. ShareGPT, 256-token chat outputs, max-concurrency 8.)

## Caveat (why total saving is "only" ~23%)

gpt-oss-120b weights (66 GiB) dominate total committed memory and are identical
in both modes — kvcached only reclaims the **KV** portion, hence the headline is
the KV row (95% less). On a model with smaller weights or a larger KV
reservation, the total-memory gap is proportionally larger.
