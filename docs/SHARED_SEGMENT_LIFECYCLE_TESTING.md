# Shared-segment shutdown validation

This plan covers removal of the small shared-memory capacity/control segment.
It does not change or validate the ordering of CUDA KV-page unmapping. A missing
control segment is a correctness failure even if HTTP requests still succeed.

## Contract

A pathname is a locator, not ownership. An engine process handle identifies a
process, not the generation currently stored at that pathname. A cleanup owner
must retain the identity of the segment belonging to its own engine generation
while that ownership can still be established. First shutdown is not necessarily
early enough: the engine may already have exited and a replacement may be ready.

The following invariants are the test oracle, independent of implementation:

1. A non-owning API client never removes an engine's segment.
2. An owner never removes a replacement generation, including on its first call.
3. Cleanup waits for every relevant owned process and local initialization or
   preallocation task to stop. One exited child is insufficient.
4. The original owned segment is eventually removed when shutdown and cleanup
   succeed. Keeping every segment is not an acceptable safety fix.
5. Unknown identity or liveness preserves the file and produces a diagnostic.
   A retry must not silently adopt whichever file now occupies the old path.
6. Retry retains the original identity and does not repeat successful stop steps.
   Concurrent cleanup and native destruction must obey the same ownership rule.
7. Cleanup preserves upstream return values and exceptions. Disabled integration
   must not add filesystem work. Descriptors must be bounded and close-on-exec.

No finite test suite proves absence of all races. These invariants define the
supported lifecycle; untested topology and namespace behavior must remain named
limitations rather than being inferred from green CI.

## Ownership boundaries

- Native pools retain the opened inode at creation. Explicit release and the
  destructor use that same identity; neither adopts a shutdown-time pathname.
- Pools in one process sharing an explicit name share cleanup ownership. Release
  waits for the last stopped pool. This does not redesign the existing metadata
  format or provide cross-process reference counting for unrelated engines.
- The supervisor constructor returns before READY. For normal launches the
  parent retains identity after `wait_for_engine_startup` succeeds and while its children are live.
  A modern MPClient reuses that record; a legacy client captures after its own
  constructor completes the startup handshake.
- Headless launches have no parent READY handshake. A startup-only observer
  pins a newly created segment while all owned children are live, and then exits.
  The name must be absent before spawning the children; a pre-existing name is
  never adopted. Monitor/shutdown exit cancels the observer. Failure to start or
  stop observation must not skip the upstream monitor/shutdown operation.
- If the applicable startup identity capture never succeeds, parent cleanup is unavailable.
  A killed startup may leave a segment for manual cleanup. This is preferable to
  deleting an unverified replacement and must not be described as leak-free.
- Identity comparison is not an atomic unlink-by-inode operation. Independent
  live engines must not concurrently replace the same explicit name between
  identity verification and unlink. Distinct instances need distinct IPC names.
- The lifecycle locks run at segment creation/release and Python cleanup only;
  they do not wrap page allocation, mapping, or memory-usage updates.

## State and transition coverage

Track these independently, rather than collapsing them into a `closed` boolean:

| Dimension | States |
| --- | --- |
| Caller | EngineCore pool, owning client, legacy owning client, non-owner frontend, supervisor, native destructor |
| Initialization | no segment yet, segment created, waiting for tensors, reserving null block, starting preallocation, ready |
| Segment | absent, original generation, replacement generation, identity unreadable |
| Children | all live, some exited, all exited, unknown/closed handle |
| Cleanup | never called, stopping, unlink failed, completed, concurrent retry |

Enumerate every meaningful transition in each path. Do not claim an exhaustive
Cartesian product: unsupported combinations need an explanation, and additional
independent risk axes need targeted combinations. Use Events/barriers and fake
clocks for ordering. Timeout values are deadlock bounds, not correctness oracles.
Use an independent open descriptor in replacement tests to prevent incidental
OS inode reuse from invalidating the test setup. Never prepopulate private
ownership fields merely to make production cleanup pass.

## CPU regression matrix

Tests live in `tests/test_vllm_engine_core_shutdown.py`, already in the CPU CI
manifest. Existing tests and the new lifecycle matrix are both required.

| Requirement | Coverage |
| --- | --- |
| Original removed, including child exiting before first shutdown | `test_lifecycle_owned_original_is_eventually_removed` |
| Replacement before first call, during teardown, after success or failed unlink | `test_lifecycle_replacement_is_never_claimed` |
| Capture fails, old child exits, replacement exists before retry | `test_lifecycle_capture_failure_cannot_claim_replacement_on_retry` |
| Stat/fstat/unlink fails once; original survives then retry succeeds | `test_lifecycle_identity_or_unlink_failure_preserves_retry` |
| Cleanup failure preserves upstream return/exception identity | `test_lifecycle_cleanup_failure_preserves_upstream_contract` |
| Each of four child positions is last to exit | `test_lifecycle_waits_for_every_owned_process` |
| Identity-query errors are visible | `test_identity_error_warns_and_does_not_unlink` |
| Missing capture cannot claim a later file | `test_absent_identity_never_claims_later_file` |
| Descriptor lifetime and exec inheritance | `test_cleanup_releases_pinned_descriptor_without_inheriting_it` |
| Overlapping calls on the same cleanup identity | `test_concurrent_cleanup_of_one_identity_is_idempotent` |
| Non-owner/unknown/disabled/legacy behavior | Existing client and process-state tests |
| Cancellation while tensors/null block/thread start are pending | Existing post-init barrier tests |
| Registry retained after incomplete pool shutdown | Existing interface retry tests |
| Real Python pool construction followed by late first cleanup | `test_ready_pool_first_shutdown_preserves_replacement` |
| Shared segment survives while one registered pool cannot stop | `test_registry_keeps_shared_segment_until_every_pool_stops` |
| Native destructor and explicit/group names | `tests/test_native_shm_cleanup.py` on the native tier |
| Old/current READY signatures and no-handshake startup failure | `test_supervisor_captures_only_after_ready_handshake`, `test_supervisor_without_ready_handshake_does_not_claim_later_file` |
| Client reuses the supervisor record | `test_client_reuses_supervisor_identity_without_recapturing` |
| Headless creation during/after constructor, stale names, early exit, no segment, monitor/thread errors | `test_headless_supervisor_without_ready_handshake` |
| Observer stop failure preserves upstream shutdown exception | `test_headless_watch_stop_failure_keeps_upstream_shutdown` |

The new owner matrix runs through client, legacy client and supervisor boundaries.
Process-count tests exercise liveness aggregation only; they are not GPU TP/PP
tests. Mocked upstream constructors are also not proof that real engine readiness
occurs at the same point: production readiness binding needs integration tests.

Run the red stage on the unmodified implementation:

```sh
python -m pytest tests/test_vllm_engine_core_shutdown.py -v
```

Record the exact base SHA, diff, collected cases, failures and diagnostics.
Keep failure logs; do not mark a missing safety contract `xfail` to report green.

## Native and single-GPU tier

After the CPU contracts are green, use the rebuilt extension from the final
candidate. Verify any compatibility overlay leaves the cleanup source unchanged.

- Real processes create/remove original and replacement native segments. Delay
  the old supervisor's first cleanup until the replacement is ready.
- Exercise native descriptor capture/query failures and delayed destruction.
- Exercise real engine readiness, failed initialization and local thread-stop
  barriers; do not replace the actual readiness handshake with a fake one.
- Serve a small model before termination. Test engine-only and process-group
  SIGTERM with shutdown timeouts 0 and nonzero, plus controlled abnormal exit.
- Run one frontend and multiple frontends. A non-owning frontend's exit must
  preserve segment identity/content and the surviving serving path.
- Run a headless supervisor without the parent READY helper. Verify capture,
  killed-child cleanup and preservation of pre-existing/unknown segments. A
  multiple-frontend run is not a substitute for this distinct startup path.
- Restart using the same explicit name. Check the new segment and full generated
  output after old-owner cleanup. HTTP 200 alone is not semantic success.
- Check segment/descriptor/process leftovers and recovery, including after
  injected failures. Remove only test-owned resources.

## TP/PP tier

Before changing cleanup for distributed runs, inspect and record the actual
mapping: EngineCore PID, owning supervisor/client, TP rank, PP rank, pool group,
IPC name, file identity, and which processes read or write that segment. TP
workers are not automatically independent segment owners; four simulated process
handles do not establish four GPU ranks. A nonzero pool group is not a PP rank.

The native naming implementation currently appends a group suffix only when its
explicit `ipc_name` argument is empty. Python pool construction passes a name.
Consequently, do not assume separate group IDs imply separate segments. Verify
the supported shared-versus-independent naming contract before asserting that
each pool may independently unlink; no broad naming redesign is implied here.

| Hardware | Topology | Required checks |
| --- | --- | --- |
| T4 | TP=1, PP=1 | Real-process/native/model lifecycle above |
| Two GPUs | TP=2, PP=1 | Delayed last worker, rank failure during startup/serving/shutdown, final cleanup and same-name restart |
| Two GPUs | TP=1, PP=2 | Stage failure and partial teardown, segment-owner mapping, final cleanup and same-name restart |
| Four GPUs, if combined topology is claimed | TP=2, PP=2 | Cross-stage/rank partial failure and cleanup ordering |

For TP/PP, also vary one versus multiple API frontends and exercise supported
multiple KV groups. A failure in one rank/stage must not make a non-owner delete
another still-live owner's segment. Conversely, after full owned teardown there
must be no unexplained stale segment. Record actual injection hits, all rank
exit states, complete token counts, restarts, CUDA/NCCL errors and recovery.
If segment ownership cannot be distinguished for a supported topology, that is
a design blocker, not a reason to weaken the assertion.

Do not infer MPS isolation or multi-GPU readiness from single-GPU success. Add
those environment-specific tests only when the PR claims or changes that scope.

## TDD and release gates

1. **Red:** tests first, run against the unmodified implementation. Classify each
   failure as product behavior or a harness/setup failure; retain both records.
2. **Green:** make the smallest coherent ownership/lifecycle change that passes
   both safety and eventual-cleanup assertions. Do not relax tests to match the
   implementation or introduce a blanket skip/delete-never workaround.
3. **Refactor:** remove duplicated cleanup decisions only after contracts pass.
   Re-run the full CPU suite with a different test order to catch module-stub
   pollution. Re-run focused native and model tests on the final source.
4. **Challenge:** temporarily break identity comparison, ownership/liveness checks
   and cancellation in disposable sources. The corresponding tests must fail.
   These are test-sensitivity checks, not proof of exhaustive race coverage.
5. **Qualify topology:** complete each applicable GPU row, leaving unsupported
   or untested rows explicit. Benchmark steady-state inference and shutdown
   latency against the same baseline/configuration if new locks or background
   work are introduced. A lifecycle lock must not enter the allocation hot path.
6. **Publish once:** full applicable CI on the exact candidate, real-device
   evidence, reviewer inspection, then one verified branch update. Tests-only
   red-stage work is kept local; do not use upstream CI to discover these reds.
   Approval and merge require separate authorization.
