# Reproduce the PageAllocator GPU capacity-query mismatch

This uses normal SGLang startup and public HTTP requests. An independent
Torch process holds memory on GPU 0; the model runs on GPU 1. Only the
upstream kvcached plugin patches are enabled.

## Setup

Tested with two RTX 4090s, driver 580.142, Python 3.12.13, SGLang 0.5.20,
Torch 2.13.0+cu130 and CUDA 13.0. `uv pip check` reported compatible packages.
Build both sources with the same environment and CUDA toolkit. Full toolkit
include paths must be available for ordinary SGLang JIT compilation.

From a checkout containing this PR, use fresh directories for both sources:

```bash
export REPRO_HELPER="$PWD/tools/repro_page_allocator_device.py"
export REPRO_ORIGINAL="$(realpath ../pr589-original)"
export REPRO_FIXED="$(realpath ../pr589-fixed)"
git worktree add --detach "$REPRO_ORIGINAL" 04ccd1da0653d251622122d0695d72bb288fb8ec
git worktree add --detach "$REPRO_FIXED" 1a09801843f1d170d96d749a8e494fa47b26ff0e
(cd "$REPRO_ORIGINAL" && python setup.py build_ext --inplace)
(cd "$REPRO_FIXED" && python setup.py build_ext --inplace)
python -m pip install -e "$REPRO_ORIGINAL" --no-build-isolation --no-deps
```

The normal editable install provides `kvcached_autopatch.pth`; `PYTHONPATH`
below selects which compiled source to exercise. Install the normal kvcached
runtime dependencies, including `wrapt` and `posix_ipc`, in this environment.

Use the same local Qwen/Qwen2.5-0.5B-Instruct snapshot for every case. Set
`REPRO_MODEL` to its absolute directory in each terminal. The tested weight
file, `model.safetensors`, has SHA-256:

```text
fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe
```

## Normal startup with unrelated GPU pressure

Use the same Python environment in both terminals and preserve the source
and model variables above. GPU 0 must be idle enough to leave half its memory
free after the helper's allocation.

Terminal A:

```bash
CUDA_VISIBLE_DEVICES=0 ENABLE_KVCACHED=false KVCACHED_AUTOPATCH=0 \
  python "$REPRO_HELPER"
```

Wait for its `"state": "ready"` JSON. The helper holds a real Torch allocation
of 35% of CUDA's reported total memory. Ctrl-C releases that process's memory.

Terminal B, with the pressure still present:

```bash
CUDA_VISIBLE_DEVICES=0,1 SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=false \
ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_GPU_UTILIZATION=0.25 \
KVCACHED_IPC_NAME=pr589_original_pressure PYTHONPATH="$REPRO_ORIGINAL" \
HF_HUB_OFFLINE=1 \
python -m sglang.launch_server --model-path "$REPRO_MODEL" \
  --host 127.0.0.1 --port 18879 --base-gpu-id 1 --tp-size 1 \
  --random-seed 42 --enable-cache-report --dtype bfloat16 \
  --context-length 16384 --max-total-tokens 32768 --max-running-requests 16 \
  --mem-fraction-static 0.8 --attention-backend triton \
  --sampling-backend pytorch --disable-cuda-graph --chunked-prefill-size 4096
```

Keep other kvcached settings at their defaults, including page preallocation.
Use a fresh IPC name for each server. The 25% utilization setting deliberately
makes GPU 0's capacity budget zero under the bounded 35% pressure allocation;
it leaves substantial physical memory free and does not require an OOM.

From another terminal, inspect readiness and device memory:

```bash
curl -i http://127.0.0.1:18879/health
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv
```

On original main, observe the null-block capacity wait for at least 20 seconds.
Then stop only Terminal A's helper. Keep the same server running: it becomes
ready once the unrelated GPU 0 pressure disappears.

This is a bounded observation of a startup/readiness stall. A generic timeout,
JIT error or unrelated exception is inconclusive. No allocator method or
scheduler state is changed by the reproduction.

## Controls and complete outputs

Stop each server before the next case, and use a fresh IPC name. Keep all
other settings and inputs identical:

| Case | Changes to the command above |
| --- | --- |
| Original, no pressure | Leave Terminal A stopped |
| Original, pressure | Original command above |
| Fixed, pressure | Set `PYTHONPATH="$REPRO_FIXED"` |
| Original, one visible GPU | Set `CUDA_VISIBLE_DEVICES=1`, `--base-gpu-id 0` |
| Native, pressure | Set `ENABLE_KVCACHED=false`, `KVCACHED_AUTOPATCH=0` |

For the one-visible-device control, keep Terminal A on physical GPU 0. The
server sees only physical GPU 1, remapped to logical GPU 0. These are TP=1
tests; two visible devices do not imply two model-parallel ranks.

After `/health` returns 200, set `REPRO_CASE` to a fresh name such as
`original-pressure` or `native-pressure`. Record three requests, including a
repeated prefix, and compare every complete output token array with native:

```bash
python - "$REPRO_CASE.json" <<'PY'
import json
import sys
import urllib.request

outputs = []
for ids in ([1000, 1001, 1002, 1003] * 32,
            [1000, 1001, 1002, 1003] * 32,
            [1100, 1101, 1102, 1103] * 96):
    payload = {"input_ids": ids, "sampling_params": {
        "temperature": 0, "max_new_tokens": 32, "ignore_eos": True}}
    request = urllib.request.Request("http://127.0.0.1:18879/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    assert len(result["output_ids"]) == 32
    outputs.append(result["output_ids"])
with open(sys.argv[1], "x") as stream:
    json.dump(outputs, stream)
PY
```

For original under pressure, send these requests after
releasing the helper and observing the same server recover. Compare files:

```bash
cmp original-pressure.json native-pressure.json
cmp original-no-pressure.json native-pressure.json
cmp fixed-pressure.json native-pressure.json
cmp original-single-visible.json native-pressure.json
```

After stopping the server and helper, use `kvctl delete <its-ipc-name>` to
remove that exact idle control file under the project's OFD lock protocol.
Check that the owned GPU processes have exited; retain failed logs.
The tested standalone Python lacked OFD constants for Python CLI cleanup;
cleanup used the unchanged control-file protocol with host Python instead.
Use a Python build with Linux OFD support when running `kvctl delete`.

## Recorded result

Source commits: original `04ccd1d`, fixed `1a09801`. The model and parameters
above are from the executed normal-entry run. GPU 1 had 22471 MiB free
according to NVML, while the original pool reported zero available physical
pages. Its `/health` responses remained 503 during the observed capacity wait.

Original server log excerpts, in order, line-wrapped for readability:

```text
[kvcached][WARNING][2026-10-18 07:09:42][kv_cache_manager.py:626] Waiting for
  physical KV capacity to reserve null block: reason=no_effective_capacity,
  elapsed=0.2s, loops=1, alloc_attempts=0, group_id=0, pp_rank=0,
  available_before=0, available_now=0, num_avail_blocks=0, reserved_blocks=0,
  avail_pages=0, virtual_free_pages=4, physical_free_pages=0,
  reserved_physical_pages=0
[kvcached][WARNING][2026-10-18 07:09:52][kv_cache_manager.py:626] Waiting for
  physical KV capacity to reserve null block: reason=no_effective_capacity,
  elapsed=10.2s, loops=966, alloc_attempts=0, group_id=0, pp_rank=0,
  available_before=0, available_now=0, num_avail_blocks=0, reserved_blocks=0,
  avail_pages=0, virtual_free_pages=4, physical_free_pages=0,
  reserved_physical_pages=0
[kvcached][WARNING][2026-10-18 07:10:02][kv_cache_manager.py:626] Waiting for
  physical KV capacity to reserve null block: reason=no_effective_capacity,
  elapsed=20.2s, loops=1958, alloc_attempts=0, group_id=0, pp_rank=0,
  available_before=0, available_now=0, num_avail_blocks=0, reserved_blocks=0,
  avail_pages=0, virtual_free_pages=4, physical_free_pages=0,
  reserved_physical_pages=0
[kvcached][INFO][2026-10-18 07:10:03][kv_cache_manager.py:660] Reserved null
  block after waiting for physical KV capacity: elapsed=21.7s, loops=2108,
  alloc_attempts=1
```

The pressure process was stopped after 20.63 seconds of observed
waiting. The same server became ready 3.26 seconds later.
The timestamps in the log above use the host's UTC+8 clock.

| Case | Initial readiness | Requests | Full output IDs |
| --- | --- | ---: | --- |
| Original / no pressure | Ready | 3 | Match native |
| Original / pressure | Waits; recovers on release | 3 | Match native |
| Fixed / pressure | Ready | 3 | Match native |
| Original / one visible GPU | Ready | 3 | Match native |
| Native / pressure | Ready | 3 | Match native |

All 15 explicit requests completed with 32 output tokens each, matching the
native reference exactly. The three original-pressure requests completed
after pressure release. This validates the tested TP=1 configuration, not
multi-rank, HIP, CUDA graphs or performance. Startup times include different
JIT cache states and are not a speed comparison.
