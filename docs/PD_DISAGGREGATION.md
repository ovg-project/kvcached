# Prefill/Decode Disaggregation Support

This document tracks kvcached support for vLLM prefill/decode (P/D)
disaggregation connectors. P/D disaggregation is connector-specific: each
connector makes different assumptions about KV-cache layout, block counts, and
transport registration.

Related issues: [#302](https://github.com/ovg-project/kvcached/issues/302),
[#311](https://github.com/ovg-project/kvcached/issues/311).

## Support Matrix

| Connector | Transport / storage path | kvcached status | Validation entrypoint | Notes |
|---|---|---|---|---|
| `NixlConnector` | NIXL / UCX | Smoke harness available | `tools/run_vllm_nixl_pd_smoke.sh` | Requires `KVCACHED_CONTIGUOUS_LAYOUT=false` because NIXL registers each layer's K/V blocks as independently block-contiguous regions. |
| `P2pNcclConnector` | NCCL peer-to-peer | Untested | TBD | Needs a minimal baseline run, then a kvcached run that checks NCCL with VMM-backed KV tensors. |
| `LMCacheConnectorV1` | LMCache | Untested | TBD | Needs a minimal multi-component configuration and a kvcached correctness run. LMCache may have different tensor ownership/layout assumptions from NIXL. |
| `MooncakeConnector` | Mooncake | Not in current scope | TBD | Revisit after the three connectors above are understood. |
| `SharedStorageConnector` | Shared storage | Not in current scope | TBD | Revisit after the three connectors above are understood. |
| `MultiConnector` | Composite connector | Not in current scope | TBD | Should be tested only after the individual component connectors have known behavior. |

## NIXL Smoke Test

Run from the repository root on a CUDA machine:

```bash
bash tools/run_vllm_nixl_pd_smoke.sh
```

Useful overrides:

```bash
INSTALL_VLLM=0 \
MODEL=Qwen/Qwen2.5-1.5B-Instruct \
PREFILL_GPU=0 \
DECODE_GPU=1 \
GPU_MEMORY_UTILIZATION=0.35 \
BLOCK_SIZE=128 \
NUM_REQUESTS=3 \
EXPECTED_SUBSTRING=Paris \
MIN_REMOTE_BLOCKS=2 \
bash tools/run_vllm_nixl_pd_smoke.sh
```

The script runs a baseline vLLM+NIXL pass first unless `RUN_BASELINE=0`.
It then runs the kvcached pass with `KVCACHED_CONTIGUOUS_LAYOUT=false`.

## Pass Criteria

A successful NIXL run should provide all of the following:

- the prefill and decode servers both become ready;
- each prefill response returns `kv_transfer_params`;
- the remote block count is at least `MIN_REMOTE_BLOCKS`;
- the decode response contains `EXPECTED_SUBSTRING` when strict checking is
  enabled;
- the kvcached logs show the NIXL compatibility patch path;
- the logs do not contain known failure signatures such as `set_stride`,
  inconsistent KV block counts, or NIXL transfer failures.

The smoke script writes one JSON line per completed case to:

```text
${LOG_DIR}/summary.jsonl
```

Those lines are intended to be pasted into GitHub issues or PR comments after a
GPU run.

## Known Failure Signatures

| Signature | Likely cause | Next check |
|---|---|---|
| `set_stride is not allowed` | NIXL forced an HND layout over kvcached VMM tensors. | Confirm the NIXL compatibility patch loaded and layout was overridden to NHD. |
| `All kv cache tensors must have the same number of blocks` | NIXL registered tensors using vLLM's profiled block count instead of kvcached's virtual block count. | Confirm the `NixlConnector num_blocks` compatibility log appears. |
| `KVCACHED_CONTIGUOUS_LAYOUT=false` error | NIXL was started with an incompatible contiguous layout. | Set `KVCACHED_NIXL_CONTIGUOUS_LAYOUT=false` or `KVCACHED_CONTIGUOUS_LAYOUT=false`. |
| `NIXL connector module was not installed/importable` | vLLM/NIXL dependencies are incomplete or mismatched. | Re-run with `INSTALL_VLLM=1` or check the active environment. |
| `NIXL transfer failure` | Transport or UCX registration issue. | Check UCX/NIXL versions, GPU topology, and transport variables such as `UCX_TLS` and `UCX_NET_DEVICES`. |

## Next Validation Targets

1. Run the NIXL smoke test on a known two-GPU CUDA host and post the generated
   `summary.jsonl` plus log paths.
2. Add the smallest P2P NCCL baseline/kvcached smoke test that can distinguish
   NCCL transport failures from kvcached VMM issues.
3. Add the smallest LMCache baseline/kvcached smoke test and document any
   tensor layout or ownership assumptions.
4. Promote each connector from `Untested` to `Smoke-tested` only after a
   reproducible command and pass/fail criteria exist in the repository.
