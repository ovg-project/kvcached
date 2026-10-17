# Physical KV growth budget

The native allocator charges a shared per-GPU budget before creating a batch of
physical KV pages. CUDA allocation, mapping, IPC and rank waits do not hold the
accounting lock. Taking a block or page from the existing local reserve does not
acquire an additional budget lock or charge the same page again.

Batch grants serialize the GPU balance, owner charge and instance limit under
one short accounting lock. Existing local allocator synchronization remains.
This PR provides native growth accounting and resource lifecycle handling.
A narrow SGLang prefill guard reserves existing manager blocks before request
publication. Distributed admission and transaction integration remain separate work.
No minimum reservation, priority, reclaim policy or CPU offload is introduced.

## Accounting and identity

All participating workers on one GPU must share the same underlying file, not
merely matching pathnames in separate container mounts. In particular, private
container `/dev/shm` mounts do not coordinate; configure a shared mount when needed.
The v2 ledger protocol requires all participating workers to restart together;
mixing v1 and v2 processes would create independent budgets. The file uses the PCI bus
ID and `KVCACHED_PHYSICAL_GROWTH_LOCK_DIR` (default `/dev/shm`). GPU partitions with
the same PCI identity are not separate accounting domains in this version.
Use a consistent `KVCACHED_IPC_NAME` across an instance's workers. Its fallback is
the existing process-group naming convention. Pool IDs remain separate records;
the instance limit applies to their sum on that GPU. Clean shutdown reclaims an
instance slot only after no owned or charged pool references it. A surviving pool
therefore retains its limit and revision across another participant's shutdown.

The ledger grants a whole native batch if both the GPU and instance budgets
permit it. Each allocation gets a receipt. Unused grant bytes are returned;
issued receipts are returned only after successful physical release and removal
of their mappings. Local reserve reuse and map commit do not charge again.
Startup zero-page backing and non-KV allocations are reflected in device samples
rather than tracked as dynamic KV growth receipts.

Partial map failure rolls back this transaction's newly attached pages and
preserves pre-existing adopted pages. Failed cleanup retains its owner for abort
retry. An uncertain VMM mapping remains charged and quarantined; worker restart
is required when its state cannot be recovered. Finalized map transaction IDs do
not provide a durable replay protocol; that remains part of distributed
integration. Duplicate operations on live transactions do not charge twice.

## Device-memory calibration

Initialization samples device free memory and subtracts the safety margin from
`KVCACHED_GPU_UTILIZATION`. This is a fixed safety margin for device usage, not
a per-priority reservation.

Physical growth batches attempt another sample at most once per second, when no
create/release operation is in flight. `recalibrate_physical_budget()` also permits
an explicit sample. A kernel-managed lease excludes new grants while sampling;
the budget data lock is released before the CUDA call. Concurrent releases change
a version counter, causing that sample to be discarded. Active grants therefore
cannot be mistaken for already resident memory and added back as extra credit.

A stable sample sets `budget = max(0, charged_bytes + sampled_free_after_safety_margin)`.
The available credit is `budget - charged_bytes`: recovering external free memory
therefore restores exactly the sampled capacity. Pool registration can retire an idle dead owner after its kernel lease closes.
Retirement subtracts the charge from both the budget (saturating at zero) and
accounting, so it never increases available GPU credit. A later coherent sample
recovers only capacity actually observed free. Live owner leases retain their
charges and instance limits. Interrupted mutations still prevent calibration. Negative headroom is
preserved until after addition so releasing pages cannot regrant credit needed
to restore the safety margin.

Single-process admission queries use both the GPU and instance budget and refresh
at the same one-second cadence. Worker-IPC estimates remain driver-based because
the ledger belongs to the workers. Estimates do not reserve capacity; grants are
still authoritative. Grants racing calibration wait on its kernel lease outside
the budget data mutex. A sampler cannot recursively wait on its own lease.

A separate initialization lease serializes joining connections through the
initial sample. Failed initialization cannot publish the previous epoch's header.
A crashed in-flight operation retains its uncertain credit and may prevent
calibration until all participants exit. A crashed sampler releases its kernel
lease automatically.

Unrelated CUDA work can consume memory after a sample. Real allocation failures
must still roll back and return a miss where the engine supports it. This protocol
does not guarantee that every external allocation or engine request avoids OOM.

## Native interfaces and validation

- `physical_budget_state(group_id=0)` returns GPU and instance charges, budgets,
  and the count of in-flight physical mutations.
- `set_instance_budget_limit(bytes, revision, group_id=0)` updates an instance's
  GPU limit and returns `applied`, `stale` or `conflict`. Shrinking below current
  usage blocks new growth without taking away existing pages.
- `recalibrate_physical_budget(group_id=0)` returns whether a sample was applied.

`tests/cpp/run_tests.sh` covers accounting, multi-process contention, calibration
races, process death, old epochs and file replacement without requiring a GPU.
`tests/test_vmm_transaction.py`, `tests/test_vmm_failure_policy.py` and
`tests/test_native_unmap_retry.py` exercise real VMM allocation/mapping/release,
injected failures, multi-pool limits and external CUDA allocation calibration.
These tests do not constitute a serving throughput benchmark or multi-GPU proof.

## Synchronization measurement

The `sync_*` fields in `physical_budget_state()` are process-local counters for
this budget connection. Acquisition time includes the synchronization primitive's
uncontended cost. Wait time starts after its first contended attempt; it excludes
the allocator's local mutex. Snapshot queries also acquire the accounting lock.
`sync_cas_retries` is zero for the flock backend.
`sync_dead_owner_recoveries` counts retired pool records, not confirmed CUDA frees.
These counters describe accounting synchronization, not end-to-end token latency.

Native pool registration uses the resolved Python IPC instance name, including
engine tags and collision suffixes. Direct dispatcher callers can pass an explicit
instance name; their fallback is process-local unless `KVCACHED_IPC_NAME` is set.
If ledger access fails during teardown, native unmap/release is still attempted,
but unconfirmed refunds remain charged to the old epoch.

## SGLang prefill failure boundary

For SGLang 0.5.20 single-rank page-size-one prefill, the adapter reserves physical
KV and Mamba blocks before `prepare_for_extend` allocates request rows. A failed
reservation releases unused reservations and returns unpublished fresh requests
to the normal queue. An existing chunk retains its row, Mamba state and previous
inflight count. No allocation retry runs inside the failed call.

This guard excludes sessions, multi-rank/pipeline execution, speculative/dLLM,
hierarchical caches and other page sizes. It does not catch errors from the
original preparation function after publication begins. Native unsafe-state
errors remain fatal. It is not a general transactional scheduler or a guarantee
against unrelated CUDA OOM.
