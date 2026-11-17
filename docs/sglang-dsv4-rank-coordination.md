# Rank-local DeepSeek-V4 backing

The SGLang 0.5.20 DSV4 adapters can coordinate rank-local backing with
tensor and pipeline parallelism. Enable the existing KVCached integration and set
`KVCACHED_RANK_COORDINATION_ENABLE=true`.

Each TP/PP worker continues to map its own storage. The coordinator uses the
instance's TP CPU groups and PP CPU columns to agree on native token budgets,
C128 request-row reservations, admission decisions, preparation outcomes, and
the actual
batch before forward. Native request priority and native page/row IDs stay
with SGLang. There is no collective between different serving instances.

A capacity miss during prefill returns only that attempt's newly allocated
pages and rows on every rank. Existing prefixes and request state remain
owned. Unused rows reserved by the attempt are released before another pass.
The scheduler can continue decoding and retry prefill after capacity
returns. Decode checks reserve backing for the native allocator's next pages;
a capacity miss on any rank releases every rank's unused reservations before
retraction or retry. A backend error, mismatched phase, or different actual
batch is fatal; it must not be retried as an ordinary capacity miss.

The coordinated path supports ordinary text generation with
attention-DP disabled and fixed, separate prefill/decode batches. Speculative
generation, beam or multimodal requests, preemption, LoRA, HiSparse,
disaggregation, dynamic chunking, prefill/slot delayers and storage-tier cache
are outside this contract. Unsupported launch settings reject before managed
backing is constructed; unsupported requests reject before preparation. Unset
the flag to retain the existing native-storage fallback for multi-rank DSV4.

C128 request-state storage stays native with prefill CUDA graph capture.
Managed request-state backing requires `--cuda-graph-backend-prefill=disabled`;
an unknown capture configuration also keeps native state storage.

CPU regressions cover protocol ordering, asymmetric allocation failures,
rollback ownership, native admission source and four-process Gloo transport.
They do not establish full-model CUDA serving correctness or performance.

## Pipeline parallelism

PP uses the native SGLang pipeline and commits the logical batch across all
stages before receiving activations. The forward guard stays within each TP
stage. Physical page IDs and per-stage layer counts can differ, including
stages with no C128 layers. CUDA, `--pp-async-batch-depth=0` and disabled
hierarchical cache are required. Other restrictions above still apply.

Before admission, the adapter relays the previous output and retains its
asynchronous send handles until the receivers have traversed the ring. The
queue is bounded by the number of stages. Retired KV pages wait for the model
and scheduling streams before unmapping; they do not wait for outstanding
output-ring communication through a device-wide barrier.

CPU tests exercise the SGLang 0.5.20 loop with simulated transport at PP2/3/4,
including idle rounds, skipped chunk outputs and a delayed stage. Real Gloo
tests cover TP1/PP4 and TP2/PP2 preparation failures, retries, heterogeneous
request-row backing and stage-local forward guards. These are protocol and
ownership tests, not CUDA/NCCL or full-model validation. GPU validation remains
pending; keep this path experimental until the following checks pass.

## Bounded GPU validation

Prepare the pinned engine, built extension, weights and request corpus before
starting a paid session. Record both repository SHAs, weight revision, CUDA,
PyTorch, driver and topology. Use the same model, quantization, attention
backend, graph settings and token limits for each native/managed pair.

1. Start with TP8/PP1, then TP4/PP2 and TP2/PP4 where model memory permits.
   Test one topology at a time; a native failure stops that pair. Keep the
   native baseline with `ENABLE_KVCACHED=false` and the managed launch with
   `ENABLE_KVCACHED=true KVCACHED_RANK_COORDINATION_ENABLE=true`. Use
   `--cuda-graph-backend-prefill=disabled` to exercise managed C128 rows.
2. Bound startup with an external timeout chosen for the prepared weights.
   After readiness, allow at most two minutes per smoke phase. Stop on timeout,
   worker failure, consensus error or output mismatch; preserve every rank's
   log instead of repeatedly restarting the same failing configuration.
3. Run the same short deterministic corpus sequentially and concurrently;
   compare output token IDs and completion counts with that topology's native
   baseline. Include prompts crossing the chunk boundary and idle-to-busy
   transitions. Exercise multiple microbatch slots and mixed request lengths.
4. Lower one managed rank's capacity using the existing control interface.
   Verify prefill/decode pressure, retries, unchanged live prefixes, completion
   after capacity restoration, and an idle drain without leaked backing.
   Check both a C128-owning stage and a stage without C128 when present.
5. Only after correctness and pressure recovery pass, compare steady-state
   throughput and latency against the same-topology native run. Record CPU
   consensus time, mapped bytes and per-rank GPU memory alongside throughput.
   Repeat with decode graphs enabled and disabled before claiming support.

Retain requests, token outputs, launch arguments, timings, memory snapshots
and all rank logs. CPU CI passing is not a substitute for these GPU checks.
