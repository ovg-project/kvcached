# Coordinator

Several kvcached instances can share one GPU, each sized as if it had the GPU
to itself. kvcached accounts the physical KV memory they use and, as the
pieces below land, lets a coordinator decide which instance gets memory under
pressure: kvcached's own local coordinator, or an external one such as AIBrix.

## 1. Model

- **GPU**: one physical GPU. Every kvcached process on it shares one ledger.
- **Instance**: one engine server, named by `KVCACHED_IPC_NAME`; its TP
  workers share the name.
- **Pool**: one KV cache of an instance; an instance can have several.

| Role | Owns |
| --- | --- |
| Engine | scheduling, which blocks are prefix cache, preempting requests |
| kvcached | the mechanisms: accounting, per-instance controls, reclaim, offload |
| Coordinator | the policy: the value of each instance's controls |

```
engine ──alloc/free──▶ kvcached ◀──controls / state── coordinator
                          │                        (local, or AIBrix)
                    per-GPU ledger
```

## 2. Mechanisms

| Mechanism | Status |
| --- | --- |
| 2.1 Ledger | implemented (#487) |
| 2.2 Per-instance controls | `max` exists as `kvctl limit`; the rest planned |
| 2.3 Watermarks | planned |
| 2.4 Reclaim | planned, on #552's shrink path |
| 2.5 CPU tier | planned |

### 2.1 Ledger

One file per GPU in `/dev/shm`, named by its PCI bus ID. It records only KV:
each pool's charge, summed under a short lock.

- **Grant**: before creating physical pages, a process charges the batch. It
  gets the grant only if the GPU's total stays within the capacity and the
  instance's total within its limit; otherwise the allocation misses without
  any CUDA call. Pages reused from the local reserve are not charged again.
- **Refund**: after the pages are physically released.
- **Crash**: a dead process's lease closes, and the next registration drops
  its record.

Everything else on the GPU (weights, activations, other processes) comes from
the driver. About once a second, while the others keep allocating,

```
capacity = charged + free − margin
```

with `free` from `cudaMemGetInfo` and the margin 5% of the GPU
(`KVCACHED_GPU_UTILIZATION`). On a unified-memory GPU such as GB10, `free`
is `MemAvailable` from `/proc/meminfo` instead, because the driver's count
leaves out page cache that the kernel reclaims for device allocations.
A sample overstates the capacity, until the
next one, when it counts a batch both in `charged` and in `free`:

- a batch that is charged but whose pages are not created yet;
- pages that are released but not yet refunded.

The margin absorbs this. A real allocation failure still rolls back.

All kvcached processes on a GPU must see the same `/dev/shm`.

### 2.2 Per-instance controls

| Control | Meaning | Default |
| --- | --- | --- |
| `max` | upper limit | startup size |
| `min` | not reclaimed below this | 0 |
| priority | order of watermarks and reclaim | equal |
| `headroom` | reserve kept from lower priorities | 0 |
| preemption | whether reclaim may preempt running requests | off |

With the defaults, behavior is today's: first come, first served.

### 2.3 Watermarks

A grant also fails if it would cut into what higher-priority instances need:
their `headroom` plus their unmet demand (`shortfall`). The requests then wait
in the engine.

### 2.4 Reclaim

An instance short of memory posts its `shortfall`. Lower-priority instances
above their `min` shrink: free pre-mapped pages, then evict prefix cache,
then, if allowed, preempt running requests through the engine's own
preemption.

### 2.5 CPU tier

Evicted prefixes and the KV of preempted requests go to host memory through
the engines' own offloading (vLLM KV connectors, SGLang HiCache) instead of
being dropped.

## 3. Coordination

### 3.1 API for an external coordinator

Planned. A coordinator such as AIBrix sets each instance's controls, reads its
usage, limits and pending reclaim, and asks it to shrink or grow, with
separate acknowledgements for accepted and completed. Reads take no lock and
promise no consistency across fields. Today it has `kvctl limit`.

### 3.2 Local coordinator

Planned. kvcached's own coordinator applies a default policy to the controls
above. It can be turned off so that an external coordinator owns the
controls; only one of them writes them at a time.
