# P2P NCCL vLLM Baseline Validation

## PR Description Draft

This PR adds debug-only instrumentation and a reproducible harness for investigating
`P2pNcclConnector` hangs in PD disaggregation.

No behavioral fix is included. The goal is to determine whether the observed hang is
caused by kvcached integration or by upstream vLLM P2P NCCL request/key behavior.

## What This Adds

- Diagnostic logging around `P2pNcclConnector` and `P2pNcclEngine`.
- A runnable 2-GPU P2P NCCL harness:
  - kvcached-enabled mode for instrumented debugging.
  - kvcached-disabled mode for plain vLLM baseline validation.
- Logs that show:
  - producer and consumer request IDs
  - generated `tensor_id = request_id#layer_name`
  - ZMQ registration and `PUT`/`GET` flow
  - NCCL send/recv begin/end
  - consumer wait heartbeat when a tensor key is missing

## Current Finding

The observed hang is not at NCCL send, NCCL recv, or ZMQ ack. In the instrumented
run, producer sends all layer tensors successfully and decode receives entries in
`recv_store`.

Decode then waits forever for a different `tensor_id`.

Example producer key:

```text
cmpl-___prefill_addr_172.23.0.2:21001___decode_addr_172.23.0.2:22001_<uuid>-0-b5ce4b84#model.layers.0.self_attn.attn
```

Example consumer key:

```text
cmpl-___prefill_addr_172.23.0.2:21001___decode_addr_172.23.0.2:22001_<uuid>-0-a08a4814#model.layers.0.self_attn.attn
```

The embedded external request address prefix matches, but the vLLM internal
randomized request suffix differs between prefill and decode.

## Validation Needed

Before proposing a fix, run pure upstream vLLM P2P NCCL without kvcached/autopatch.

Decision tree:

- If plain vLLM hangs the same way, this is likely an upstream `P2pNcclConnector`
  keying issue.
- If plain vLLM succeeds, kvcached integration is changing request identity,
  scheduling, or transfer behavior.
- If `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` makes the run pass, that strongly
  confirms randomized internal request ID mismatch as the root cause.

## Scope

This PR is intentionally observational only.

No layout changes, tensor reshaping, `.contiguous()`, timeout, exception, or
production fix is included.

## Run 1: Same Harness, kvcached Disabled

This keeps our exact proxy/topology/request shape while disabling kvcached. It is the
fastest controlled baseline.

```bash
cd /root/kvcached
git pull

RUN_WITH_KVCACHED=0 \
TIMEOUT_REQUEST=300 \
./experiments/10_p2p_nccl_debug.sh
```

Expected notes:

- No `kvcached p2p debug` lines should appear.
- The script should skip kvcached install/autopatch checks.
- If this hangs, the issue is likely reproducible in plain vLLM P2P NCCL.
- If this succeeds, compare against the kvcached-enabled run.

## Run 2: Same Harness, Disable vLLM Request ID Randomization

This tests the current leading hypothesis directly.

```bash
cd /root/kvcached

RUN_WITH_KVCACHED=0 \
VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 \
TIMEOUT_REQUEST=300 \
./experiments/10_p2p_nccl_debug.sh
```

Expected interpretation:

- If Run 1 hangs and Run 2 passes, the internal randomized request suffix is almost
  certainly the problem.
- If both hang, the problem is deeper than request ID randomization.
- If both pass, the kvcached-enabled debug run is changing behavior and needs a
  narrower comparison.

## Run 3: Official vLLM P2P NCCL Example

Use this as a sanity check that we are not accidentally testing only our harness.
The official example is documented here:

<https://docs.vllm.ai/en/latest/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/>

The example source lives here:

<https://github.com/vllm-project/vllm/tree/main/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd>

Automated local wrapper:

```bash
cd /root/kvcached
git pull

TIMEOUT_REQUEST=300 \
./experiments/11_vllm_p2p_nccl_direct.sh
```

Repeat with request ID randomization disabled:

```bash
cd /root/kvcached

DISABLE_REQUEST_ID_RANDOMIZATION=1 \
TIMEOUT_REQUEST=300 \
./experiments/11_vllm_p2p_nccl_direct.sh
```

That wrapper uses the upstream vLLM proxy from the cloned vLLM repo and direct
`vllm serve` producer/decode processes. It does not use the kvcached debug proxy.

Manual equivalent:

Setup:

```bash
cd /root
git clone https://github.com/vllm-project/vllm.git vllm-upstream
cd /root/vllm-upstream

pip install -U "vllm==0.19.0" quart pyzmq msgpack aiohttp pandas datasets
```

The upstream script currently checks `HF_TOKEN`, even for models that may not need
auth. If it refuses to start, export a token:

```bash
export HF_TOKEN=hf_...
```

Run a 1P1D A100 baseline:

```bash
cd /root/vllm-upstream/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd

MODEL=Qwen/Qwen2.5-1.5B-Instruct \
PREFILL_GPUS=0 \
DECODE_GPUS=1 \
PREFILL_PORTS=20003 \
DECODE_PORTS=20005 \
PROXY_PORT=30001 \
TIMEOUT_SECONDS=1200 \
bash disagg_example_p2p_nccl_xpyd.sh
```

In another shell, send a request through the proxy:

```bash
curl --max-time 300 -sS http://localhost:10001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "The capital of France is",
    "max_tokens": 16,
    "temperature": 0
  }'
```

Then repeat with request ID randomization disabled:

```bash
cd /root/vllm-upstream/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd

VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 \
MODEL=Qwen/Qwen2.5-1.5B-Instruct \
PREFILL_GPUS=0 \
DECODE_GPUS=1 \
PREFILL_PORTS=20003 \
DECODE_PORTS=20005 \
PROXY_PORT=30001 \
TIMEOUT_SECONDS=1200 \
bash disagg_example_p2p_nccl_xpyd.sh
```

## What To Capture

For the kvcached harness runs, capture the final summary and the log directory:

```bash
ls -ltr /root/kvcached/experiments/logs_p2p_debug
```

For the upstream vLLM example, capture:

```bash
tail -n 200 proxy.log
tail -n 200 prefill1.log
tail -n 200 decode1.log
```

Also capture request ID randomization state:

```bash
python3 - <<'PY'
import os
print("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=", os.getenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"))
PY
```

## Expected Root-Cause Signal

The strongest confirmation is:

1. Plain vLLM P2P NCCL hangs.
2. The same run passes with `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`.
3. The kvcached debug logs show producer and consumer using the same embedded
   external address prefix but different internal randomized suffixes.

If that happens, the next discussion should be about how to key P2P transfer tensors
by a stable transfer request ID rather than the per-engine randomized internal vLLM
request ID.
