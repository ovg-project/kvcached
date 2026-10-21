# Memory monitoring and control via kvcached CLI

kvcached includes a built-in CLI tool that allows you to monitor GPU memory usage and manage memory limits across different applications. A command `kvctl` is installed along with kvcached package:

```bash
kvctl

# if kvcached is installed in source
cd <kvcached-source-dir>/kvcached/cli
python kvctl.py
```

Once inside the CLI, type `help` to view all supported commands:

```
kvcached> help
Available commands:
  list [ipc ...]               List IPC segments and usage
  limit <ipc> <size>           Set absolute limit (e.g. 512M, 2G)
  limit-percent <ipc> <pct>    Set limit as percentage of total GPU RAM
  watch [-n sec] [ipc ...]     Continuously display usage table
  kvtop [ipc ...] [--refresh r]  Launch curses kvtop UI (q to quit)
  !<shell cmd>                 Run command in system shell
  help                         Show this help message
  delete <ipc>                 Delete IPC segment and its limit entry
  exit | quit                  Exit the shell

kvcached>
```

Lowering a limit evicts cached prefixes (vLLM prefix cache, SGLang radix cache)
until the instance fits, including while it is idle. Pages that running
requests use are never revoked; they are released as those requests finish.
SGLang with tensor or pipeline parallelism is not covered: its ranks would
evict at different steps, so cached prefixes still hold a lower limit back.

## Embedding revisioned limits in a production controller

`kvctl limit` is useful for manual operation. A production control layer can
instead apply a revisioned instance budget through the Python API:

```python
from kvcached.control import get_instance_memory_limit, set_instance_memory_limit

instance = "my-model"  # The engine's KVCACHED_IPC_NAME.
snapshot = get_instance_memory_limit(instance)
if not snapshot["complete_membership"]:
    raise RuntimeError("instance participants are unavailable")

generations = {gpu["gpu"]: gpu["generation"] for gpu in snapshot["gpus"]}
revisions = {gpu["gpu"]: gpu["revision"] + 1 for gpu in snapshot["gpus"]}
result = set_instance_memory_limit(
    4 * 1024**3, instance_name=instance,
    revision=revisions, generations=generations,
)
for gpu in result["gpus"]:
    print(gpu["gpu"], gpu["status"])  # Per-GPU publication result.
if result["status"] != "accepted":
    # partial: some GPUs accepted; rejected: none accepted.
    # Reconcile the per-GPU results before retrying selected GPUs.
    raise RuntimeError(f"publication incomplete: {result['status']}")

# Query again on later controller ticks; acceptance is not convergence.
progress = get_instance_memory_limit(instance)
if not progress["complete_membership"] or any(
    gpu["generation"] != generations.get(gpu["gpu"])
    or gpu["revision"] != revisions.get(gpu["gpu"])
    for gpu in progress["gpus"]
):
    raise RuntimeError("membership or assignment changed; reconcile")
if progress["status"] == "failed":
    for gpu in progress["gpus"]:
        if gpu["failed"]:
            print(gpu["gpu"], gpu["revision"], gpu["failure_reason"])
elif progress["converged"]:
    print("assignment converged")
else:
    print([(gpu["gpu"], gpu["convergence"]) for gpu in progress["gpus"]])
```

The API uses the shared physical-GPU ledger. A separate controller process can
query and publish limits when it can access the same ledger directory
(`KVCACHED_PHYSICAL_GROWTH_LOCK_DIR`, default `/dev/shm`) and instance name.
Initialize the engine and wait for all runtime-control participants first.
The engine thread performs the resize; publication does not resize synchronously.

Explicit revisions require the per-GPU generations from a query. Use increasing
revisions for new assignments and preserve the original generations and revisions
when retrying an assignment. Omitting `revision` allocates a new revision on each
selected GPU, even when the limit is unchanged. GPUs publish independently;
`partial` does not roll back the GPUs that accepted. Inspect each result and use
`gpu_ids` to retry selected GPUs with their original tuple where appropriate.

| Publication status | Meaning | Controller action |
| --- | --- | --- |
| `accepted` | All selected GPUs accepted or returned `idempotent`. | Query progress for the accepted generations and revisions. |
| `partial` | Some selected GPUs accepted. | Reconcile each GPU; do not assume an atomic instance update. |
| `rejected` | No selected GPU accepted. | Inspect per-GPU `stale`, `conflict`, `instance_changed`, or error details. |
| `unavailable` | Complete instance membership was not discovered. | Wait for readiness or inspect participant availability. |

| Query convergence | Meaning | Controller action |
| --- | --- | --- |
| `pending` | Execution or release of active KV pages is still outstanding. | Query again, checking generation and revision. |
| `converged` | Every participant acknowledged and charges fit its target. | Record completion. |
| `failed` | A resize encountered quarantined pages. | Inspect `failed` participant count and `failure_reason` (`quarantined_pages_prevent_resize`); stop waiting for this revision. |
| `unsupported` | A participant cannot safely perform the requested shrink. | Inspect per-GPU convergence and choose a supported assignment. |
| `unavailable` | A participant disappeared or membership is incomplete. | Reconcile membership before publishing again. |

Both immediate and deferred quarantine failures are bound to the executing
generation, revision, and participant. The accepted cap remains in force and
continues to restrict growth; a failure does not roll it back. Retrying the same
tuple does not rerun the failed resize or clear the failure. A new revision clears
the previous revision's progress, but does not repair quarantine.

Limits are bytes per instance per GPU. Each GPU divides its budget in proportion
to its pools' original capacities, rounding each pool's target down to physical
page bundles. Query `gpus` for the requested `limit_bytes`, aligned
`effective_limit_bytes`, `charged_bytes`, and convergence. A higher limit can
regrow a pool within its original reservation.

kvcached applies the assigned budget but does not choose it. Quota policy,
fairness, admission, and cross-instance arbitration remain the responsibility
of the external control layer.

Use the `kvtop` command for real-time visualization of memory usage:

<!-- KVCache memory monitor (muted colours) -->
<pre>
<span style="color:#009ACD; font-weight:bold;">KVCache Memory Usage</span>

<span style="color:#009ACD;">IPC: SGLANG</span>
<span style="color:#009ACD;">[</span><span style="color:#B7A800;">==</span><span style="color:#009E8F;">##################</span><span style="color:#666666;">----------------------------------------</span><span style="color:#009ACD;">]</span>
Prealloc: 792.0&nbsp;MB | Used: 11.2&nbsp;GB / 39.9&nbsp;GB (30.1%) | Free: 27.9&nbsp;GB

<span style="color:#009ACD;">IPC: VLLM</span>
<span style="color:#009ACD;">[</span><span style="color:#B7A800;">==</span><span style="color:#009E8F;">#######</span><span style="color:#666666;">--------------------------------------------------- </span><span style="color:#009ACD;">]</span>
Prealloc: 768.0&nbsp;MB | Used: 3.6&nbsp;GB / 37.4&nbsp;GB (11.7%) | Free: 33.0&nbsp;GB

<span style="color:#009ACD;">GPU Memory Usage</span>
<span style="color:#009ACD;">[</span><span style="color:#B7A800;">########################################</span><span style="color:#666666;">--------------------</span><span style="color:#009ACD;">]</span>
Used: 52.9&nbsp;GB / 79.2&nbsp;GB (66.8%) | Free: 26.3&nbsp;GB

<span style="color:#555555;">Press 'q' to quit</span>
</pre>
