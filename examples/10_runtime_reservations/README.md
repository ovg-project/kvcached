# Runtime-owned memory reports

Serving runtimes own memory outside kvcached's elastic KV pools, including
workspace, graph pools and model-specific state. The shared reporting API records
these bytes by integration, device, pool category and live owner. SGLang and vLLM
expose the same interface.

Registration only records accounting facts. It does not allocate physical memory,
change a KV limit or subtract from an engine's profiled budget. A consuming adapter
must decide where a reservation belongs in its budget and avoid counting memory
that profiling already included. Registering after KV backing creation does not
retroactively resize that backing.

## Report from the owning process

Choose the shim used by the running engine:

```python
from kvcached.integration.vllm import interfaces as runtime
# For SGLang: from kvcached.integration.sglang import interfaces as runtime
```

After successful construction of a runtime-owned workspace, report its bytes:

```python
runtime.register_runtime_owned_reservation(
    "cuda:0", "workspace", workspace.numel() * workspace.element_size(),
    owner=workspace,
)

total = runtime.get_runtime_owned_reservation_bytes("cuda:0")
breakdown = runtime.get_runtime_owned_reservation_breakdown("cuda:0")
snapshots = runtime.runtime_reservation_snapshot_dicts()
```

The owner must support weak references. It remains owned by the runtime; neither
the registry nor returned snapshots keep it alive. Registration replaces the
reported byte count for that owner, device and category. Different owners of the
same category are summed, so a draft pool cannot overwrite its target pool.

Use a fully qualified device such as `cuda:0`. `hip:0` normalizes to `cuda:0`.
Bare `cuda` is rejected because its meaning depends on the calling thread's
current device. The reporting API never queries CUDA to resolve it.

Pool names should be stable categories, not request IDs. Only report bytes still
owned by the runtime, exclude kvcached-managed backing, and deduplicate aliased
buffers before summing their sizes.

## Updates and removal

Report a replacement size using the same owner, device and category. Report zero
to remove that entry while leaving other owners intact:

```python
runtime.register_runtime_owned_reservation(
    "cuda:0", "workspace", 0, owner=workspace,
)
```

Entries also disappear when their owner is collected. A report is tied to the
owner's lifetime, not to the lifetime of its individual buffers; if the owner
remains alive after releasing or transferring a buffer, update its report.

Successful integration shutdown clears that integration's reports. Failed or
incomplete shutdown keeps them for retry. The other engine's reports remain
untouched.

## Read-only snapshots and capability discovery

Each immutable snapshot aggregates one integration/device/category combination:

```json
{
  "schema_version": "kvcached.observability.v1",
  "integration": "vllm",
  "device": "cuda:0",
  "pool_name": "workspace",
  "num_bytes": 1048576,
  "owner_count": 2
}
```

`num_bytes` is the sum of the callers' live reports, not a measurement of all
device memory. Previously returned snapshots remain unchanged when reports are
updated or owners disappear. Empty categories are omitted.

Consumers can check
`get_capabilities()["features"].get("runtime_reservation_reporting", False)` from
`kvcached.observability`. The `runtime_reservation_snapshot_fields` entry lists the
available fields. Missing capabilities or optional fields remain unsupported,
following the existing additive observability contract.

The engine-neutral functions live in `kvcached.runtime_reservations`; registration
takes an explicit `integration` and snapshot reads can filter by integration and
device. Integration shims bind their own integration name automatically.

All records are process-local. A controller or an API-server process cannot read
another process's registry by importing this module. The consuming integration
owns snapshot transport and exporter registration. This API adds neither a
metrics server nor a second IPC protocol.
