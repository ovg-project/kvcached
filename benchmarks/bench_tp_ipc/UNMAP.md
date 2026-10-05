# Transactional unmap latency

From the repository root, with the native extension built:

```bash
ENABLE_KVCACHED=false KVCACHED_AUTOPATCH=0 PYTHONPATH=. \
  python benchmarks/bench_tp_ipc/benchmark_unmap_transaction.py --devices 0

# Two workers on different GPUs:
ENABLE_KVCACHED=false KVCACHED_AUTOPATCH=0 PYTHONPATH=. \
  python benchmarks/bench_tp_ipc/benchmark_unmap_transaction.py --devices 0,1
```

The benchmark compares three paths with the same native extension and real Unix
sockets: a one-phase unmap control, explicit prepare/commit, and the production
automatic path. The control intentionally has no cross-worker prepare barrier;
it is a successful-operation timing reference, not a safe failure-handling option.
It does not recreate an older commit's native allocator implementation.

Each worker owns real CUDA VMM tensors. Mapping precedes every measured unmap;
data write/readback checks run outside the timing window. The default is three
rounds with rotated protocol order, 20 warmups and 200 measured iterations per
protocol, two layers and one 2 MiB offset. JSON output includes per-round map and
unmap mean, P50 and P95 latency and worker exit codes. `--pages` changes batch size;
`--layers` changes the physical mapping fan-out.

`--devices 0,0` can exercise two worker processes on one GPU, but is not a two-GPU
TP result. This isolates protocol/VMM latency without serving traffic: it cannot
establish end-to-end throughput or latency under GPU inference contention.
