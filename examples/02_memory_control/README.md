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

## Embedding revisioned limits in a production controller

`kvctl limit` is useful for manual operation. A production control layer can
instead apply a revisioned instance budget through the Python API:

```python
from typing import Any

from kvcached.control import set_instance_memory_limit


def apply_memory_assignment(limit_bytes: int, revision: int) -> dict[str, Any]:
    result = set_instance_memory_limit(limit_bytes, revision=revision)

    if result["status"] not in {"applied", "deferred"}:
        raise RuntimeError(
            "memory assignment was not accepted: "
            f"{result['reason'] or result['status']}"
        )
    return result
```

This API is process-local. Call it from a control handler running in the same
process as the vLLM EngineCore or SGLang scheduler after kvcached has registered
its live KV pools. Calling it from a separate sidecar or controller process will
return `unavailable` because that process has a different pool registry. The
transport that delivers assignments to the engine process (for example, RPC or
a Unix socket) is intentionally left to the production integration.

Use a monotonically increasing, non-negative revision for each new assignment:

| Status | Meaning | Controller action |
| --- | --- | --- |
| `applied` | The aligned limit is active. | Record the acknowledgement. |
| `deferred` | Active mappings exceed the new limit; no active mapping was revoked. | Retry the same limit and revision to poll until it becomes `applied`. |
| `stale` | A newer revision is already active. | Discard this response and reconcile with the newer assignment. |
| `conflict` | The same revision was reused with a different limit. | Allocate a new revision; do not retry the conflicting tuple. |
| `unavailable` | No live KV pool is registered in this process. | Wait for engine initialization or fix the control-handler placement. |

Retries of the same `(limit_bytes, revision)` tuple are idempotent. A lower
limit converges through the existing `resize()` / `in_shrink` path as requests
release pages; a later higher revision can regrow the pool within its original
reservation. The returned `pools` list reports each pool's aligned share and
current mapped bytes. When an engine owns multiple KV pools, kvcached splits the
instance budget deterministically in proportion to their original capacities.

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
