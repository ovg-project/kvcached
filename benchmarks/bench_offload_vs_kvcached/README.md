# kvcached vs. vLLM CPU offloading (multi-LLM serving)

This benchmark compares three KV-memory strategies for serving **two LLM
instances on one GPU** under staggered bursty load:

| mode | GPU memory | when KV memory runs out |
|---|---|---|
| `static` | fixed 0.45 / 0.45 split | evict prefix cache, queue/preempt requests |
| `offload` | fixed 0.45 / 0.45 split | same, but KV blocks are offloaded to CPU RAM and restored over PCIe (vLLM native `OffloadingConnector`) |
| `kvcached` | elastic shared pool | grow into the idle instance's free GPU memory |

The offloading baseline is deliberately given its best case: every burst
re-sends conversations whose prefix KV was previously computed (multi-turn
pattern), so evicted-but-offloaded KV can be restored from CPU instead of
recomputed. Prefix caching (APC) is enabled in all three modes.

## Setup

- 1x NVIDIA A100-40G, 2 instances of `Qwen/Qwen3-4B` (vLLM 0.19.0, the newest
  version current kvcached supports; vLLM >= 0.21 defaults to the V2 model
  runner whose KV allocation path kvcached does not yet patch)
- Workload per instance: 44 conversations, turn-1 prompt 2048 tokens,
  turn-2 adds 192 tokens (2240 total), 128 output tokens, temperature 0
- Timeline: A seeds (2 req/s) -> A bursts all turn-2 (8 req/s) while B idles
  (trickle probes) -> B seeds -> B bursts while A idles
- TTFT measured client-side via streaming completions

## Run

```bash
./run_one.sh static   --num-convs 44 --gap 30
./run_one.sh offload  --num-convs 44 --gap 30
./run_one.sh kvcached --num-convs 44 --gap 30
python analyze.py static=logs/static/results.jsonl \
                  offload=logs/offload/results.jsonl \
                  kvcached=logs/kvcached/results.jsonl
```

## Results

Burst-phase TTFT across both instances (n=80), A100-40G, vLLM 0.19.0:

| mode | mean | p50 | p99 |
|---|---|---|---|
| `static` | 2177 ms | 164 ms | 8573 ms |
| `offload` | 1447 ms | 102 ms | 5838 ms |
| `kvcached` | **186 ms** | 166 ms | **460 ms** |

- **kvcached vs offload: 7.8x mean, 12.7x p99 TTFT reduction.**
- Offloading does help the baseline (1.5x mean over static; its p50 is even the
  best of the three because evicted prefixes restore from CPU at ~17 GB/s
  instead of being recomputed — the offload run moved 19.1 GiB GPU->CPU and
  13.1 GiB CPU->GPU per instance). But offloading cannot raise the *GPU KV
  capacity* available to running requests, so burst-induced queueing keeps the
  tail at seconds. kvcached grows the bursting instance's KV pool into the
  idle instance's free memory (observed per-process peak ~24-25 GiB vs the
  ~9.4 GiB static partition), eliminating the queue: zero preemptions, p99
  under half a second.
- Off-peak traffic is unaffected: seed phases are equal (offload pays a small
  eager-store overhead: 140 ms vs 122 ms mean) and trickle probes on the idle
  instance stay ~30 ms in all modes.

Notes for reproduction:
- The kvcached mode sets `KVCACHED_MAX_CACHED_TOKENS=0` (evict-on-free).
  With the default retention (16000 tokens), freed-but-cached prefix blocks
  fragment across 2 MiB pages and pin ~9 GiB extra on the idle instance,
  which starves the second burst at the 0.95 shared-pool cap (observed: 6
  preemptions, near-static p99). Evict-on-free is also conservative for
  kvcached: the baselines keep full prefix-cache retention inside their
  partitions, kvcached gives it up.
- kvcached currently requires vLLM <= 0.19.x: vLLM >= 0.21 ships a V2 model
  runner (default since 0.22) whose KV allocation path kvcached does not
  patch — the engine starts but physically backs the entire virtual KV cache
  (silently no elasticity), and the 0.22 V1-runner path crashes in
  `_reshape_kv_cache_tensors` (signature change + new
  `profile_cudagraph_memory` minimal-KV profiling path).
