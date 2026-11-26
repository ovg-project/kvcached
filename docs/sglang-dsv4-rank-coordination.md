# Rank-local DeepSeek-V4 backing

The SGLang 0.5.20 DSV4 adapters can coordinate rank-local backing with
tensor parallelism. Enable the existing KVCached integration and set
`KVCACHED_RANK_COORDINATION_ENABLE=true`.

Each TP worker continues to map its own storage. The coordinator uses the
instance's TP CPU group to agree on native token budgets, C128 request-row
reservations, admission decisions, preparation outcomes, and the actual
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

The coordinated path supports ordinary text generation with PP=1,
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
