# kvcached vs. vLLM native CPU offloading for multi-LLM serving on a shared GPU

Experiment report, 2026-06-10.

## 1. Question

kvcached's published benchmarks compare elastic KV sharing against *static GPU
memory partitioning*. A reasonable objection is that static partitioning is a
weak baseline: vanilla vLLM can already mitigate KV-memory shortage by
**offloading KV blocks to CPU RAM** and restoring them over PCIe (native
`OffloadingConnector`, no extra dependencies). This experiment answers:

> When two LLM instances share one GPU under bursty, staggered load, how much
> does kvcached's elastic GPU sharing still win over a static split *plus*
> CPU offloading?

## 2. Setup

| | |
|---|---|
| GPU | 1x NVIDIA A100-SXM4-40GB |
| Host | 115 GiB RAM, 32 cores, PCIe ~17 GB/s effective (measured via offload transfer counters) |
| Models | 2 instances of `Qwen/Qwen3-4B` (bf16, 7.56 GiB weights each) |
| Engine | vLLM 0.19.0, torch 2.10.0+cu128 |
| kvcached | HEAD (`3cc92be`, v0.1.5), installed editable |
| Client | streaming `/v1/completions`, token-id prompts, TTFT = time to first SSE chunk |

**Why vLLM 0.19.0 and not the latest (0.22.1):** current kvcached does not
support vLLM >= 0.21. The V2 model runner (default since 0.22) bypasses
kvcached's allocation patches — the engine starts normally but physically
backs the *entire* virtual KV cache, silently disabling elasticity (we
observed a single instance holding 37 GiB and the second instance failing to
load weights). Forcing the V1 runner (`VLLM_USE_V2_MODEL_RUNNER=0`) instead
crashes in `_patched_reshape_kv` because 0.22 changed
`_reshape_kv_cache_tensors` signatures and added a
`profile_cudagraph_memory()` minimal-KV profiling path. 0.19.0 is the newest
version kvcached works on (it is also the version listed in the README), and
it ships the same native `OffloadingConnector`, so the baseline is not
handicapped. This deserves its own issue.

## 3. Compared systems

All modes: `--max-model-len 8192 --max-num-seqs 64 --block-size 16`, prefix
caching (APC) enabled (vLLM default), identical workload and client.

| mode | GPU memory | KV capacity per instance | when KV runs out |
|---|---|---|---|
| `static` | fixed `--gpu-memory-utilization 0.45` each | 68,000 / 72,096 tokens (~9.4 GiB) | evict prefix cache; queue/preempt |
| `offload` | same static split | same | same, but evicted/preempted KV is offloaded to a 16 GiB pinned CPU pool per instance and restored over PCIe (`--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":17179869184}}'`) |
| `kvcached` | elastic shared pool (`KVCACHED_GPU_UTILIZATION=0.95`, default util 0.9 per engine, `KVCACHED_MAX_CACHED_TOKENS=0` — see section 6) | virtual 200,480 / 156,960 tokens; physical on demand | grow into the idle peer's free GPU memory |

## 4. Workload design (and why it is fair to the offloading baseline)

Per instance: 44 -> final 40 multi-turn conversations.

- **Seed phase** (2 req/s): each conversation sends turn-1 — a unique
  2048-token prompt, 512 output tokens.
- **Burst phase** (8 req/s): all 40 conversations send turn-2 — the same
  2048-token prefix + 192 new tokens (2240 total), 512 output tokens.
- Bursts are **staggered**: while A bursts, B receives only trickle probes
  (128-token prompt every 5 s), and vice versa. Timeline:
  `A_seed@0s, A_burst@50s, B_seed@85s, B_burst@135s`.

Design rationale:

1. **Give offloading its best case.** Burst requests *reuse* a long prefix
   whose KV was computed during seeding. The seed working set
   (40 x 2176 ≈ 87K tokens ≈ 12.5 GiB) exceeds the static partition
   (~9.4 GiB / ~70K tokens), so part of the prefix cache is evicted before
   the burst. Without offloading those prefixes are recomputed; with
   offloading they restore from CPU at PCIe speed. A burst of *fresh* random
   prompts would have made the offloading baseline a strawman (nothing to
   restore).
2. **Make the burst exceed the static partition but fit the elastic pool.**
   Peak burst KV demand = 40 x (2240 prompt + 512 output) ≈ 110K tokens
   ≈ 15.5 GiB: ~1.6x the static partition's capacity (queueing pressure for
   the baselines), but within what one instance can borrow when its peer is
   idle (observed elastic peak: ~25 GiB per process).
3. **Long outputs (512 tokens)** make requests hold KV for ~10 s, so
   capacity-induced queueing — not prefill compute — dominates the baseline's
   tail. With 128-token outputs the same burst showed only ~1.4 s p99 on
   static; the pressure was too mild to differentiate systems.
4. **Trickle probes** verify the idle instance is not collateral damage.

Token-id prompts (`prompt: [ids...]`) guarantee exact prefix matches for APC
and the offload connector's hash lookup, avoiding tokenizer round-trip drift.

## 5. Results

Burst-phase TTFT, both instances pooled (n=80), zero client errors in all runs:

| mode | mean | p50 | p99 | preemptions (A+B) |
|---|---|---|---|---|
| `static` | 2177 ms | 164 ms | 8573 ms | 10 |
| `offload` | 1447 ms | **102 ms** | 5838 ms | 12 |
| `kvcached` | **186 ms** | 166 ms | **460 ms** | **0** |

**kvcached vs. offload: 7.8x mean, 12.7x p99.** Per-instance numbers are
symmetric (A/B burst mean 186/186 ms, p99 448/423 ms).

Off-peak behavior:

| phase | static | offload | kvcached |
|---|---|---|---|
| seed mean | 122 ms | 140 ms | 123 ms |
| trickle mean (idle instance during peer burst) | ~30 ms | ~33 ms | ~29 ms |

Offloading is *not* a strawman here — it genuinely worked:

- Transfer counters per instance: **19.1 GiB GPU->CPU stored, 13.1 GiB
  CPU->GPU restored** (`vllm:kv_offload_total_bytes_total`), ~17 GB/s.
- Its p50 is the best of all three modes: requests whose prefix restores
  from CPU skip recomputation entirely, beating even kvcached's p50 (which,
  with evict-on-free retention, re-prefills every burst request).
- Seed-phase overhead of eager CPU stores is modest (140 vs 122 ms mean).

But offloading cannot raise the *GPU KV capacity available to running
requests*: the burst needs ~15.5 GiB resident while the partition holds
~9.4 GiB, so ~half the burst queues behind ~10-second decodes regardless of
how fast prefixes restore. That queueing is the entire tail. kvcached
removes it by letting the bursting instance grow into the idle peer's
memory: per-process traces show the burster peaking at ~24-25 GiB and
returning to its ~10.5 GiB baseline within ~8 s of the burst draining.

## 6. Pitfalls we hit (read before reproducing)

1. **CPU pool sizing vs. system RAM.** `cpu_bytes_to_use` is allocated as
   pinned shared memory at startup. 40 GiB per instance x2 on a 115 GiB host
   got instance A's EngineCore killed by the kernel OOM killer while
   instance B initialized its pool. 16 GiB per instance holds the full
   benchmark working set (~12.5 GiB) safely.
2. **`KVCACHED_MAX_CACHED_TOKENS` default starves the second burster.**
   With the default (16000 tokens ≈ 2.2 GiB), the idle instance retained
   **18.6 GiB** (~8.9 GiB above its baseline): freed-but-cached prefix
   blocks are kept *in place*, scattered across 2 MiB pages (64 blocks per
   page for this model), and a page with even one cached block cannot be
   unmapped. The pinned set is bounded by pages touched, not tokens cached.
   Consequence: the *first* instance to burst kept its fragmented pages, the
   *second* burst hit the 0.95 shared-pool cap (sum 38.3 GiB), took 6
   preemptions, and degraded to near-static tail latency (B: mean 1710 ms,
   p99 9045 ms, while A enjoyed mean 154 ms / p99 309 ms). Setting
   `KVCACHED_MAX_CACHED_TOKENS=0` (evict-on-free) restored symmetry — and
   note this is *conservative* for kvcached, which then gives up the
   cross-request prefix reuse that both baselines keep inside their
   partitions. This fragmentation behavior deserves its own issue/fix
   (page-aware eviction would likely let kvcached keep its cache *and*
   return memory).
3. **kvcached + vLLM >= 0.21 silently loses elasticity** (see section 2).
   Always verify elasticity by checking idle per-process GPU memory after
   startup; "server healthy" is not evidence.

## 7. Reproduction

Scripts in this directory:

```bash
./run_one.sh static   --num-convs 40 --gap 30
./run_one.sh offload  --num-convs 40 --gap 30
./run_one.sh kvcached --num-convs 40 --gap 30
python analyze.py static=logs/static/results.jsonl \
                  offload=logs/offload/results.jsonl \
                  kvcached=logs/kvcached/results.jsonl
```

`run_one.sh` snapshots each instance's `/metrics` (offload transfer and
preemption counters) before shutdown. For memory traces, sample
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits`
every 2 s during a run.

## 8. Limitations

- Single GPU type (A100-40G), single model (Qwen3-4B x2), one run per config
  (the effect sizes — 7.8x/12.7x — dwarf run-to-run noise, but multi-seed
  repetition would tighten the numbers).
- Synthetic random-token conversations; real multi-turn traffic has shared
  system prompts that would raise APC hit rates for all modes.
- TTFT-focused; decode throughput (TPOT) was not analyzed (raw per-request
  latencies are in `results.jsonl` if needed).
- vLLM 0.19.0, not latest — forced by kvcached compatibility (section 2).
  The OffloadingConnector interface is unchanged through 0.22, and we
  verified identical behavior of the offload smoke test on 0.22.1, so the
  baseline conclusion should transfer.
