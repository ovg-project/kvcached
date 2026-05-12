# LMCacheConnectorV1 PD Debug Validation

This branch validates whether plain vLLM `LMCacheConnectorV1` can run a minimal
1-prefill / 1-decode disaggregated-prefill setup and produce real LMCache KV
hits. The first goal is simply "does LMCache PD work at all?"

The harness launches one prefiller, one decoder, and a local proxy. The proxy
sends the same client request to prefiller first with `max_tokens=1`, then to
decoder for the full completion.
For LMCache PD, the proxy also attaches prefiller-side
`kv_transfer_params.disagg_spec` with the decoder receiver host/init/alloc
ports; without that per-request metadata, the prefiller reaches
`lmcache_engine.store(...)` with a null transfer spec.

The default `PROMPT_MODE=long` uses deterministic synthetic context long enough
to cross the LMCache chunk boundary. Use `PROMPT_MODE=short` only for fast
startup checks; short prompts can complete successfully while still reporting
`LMCache hit tokens: 0`.

## Files

- `experiments/12_lmcache_connector_v1_debug.sh`
- `experiments/collect_lmcache_connector_v1_evidence.sh`
- `experiments/lmcache_connector_v1_validation.md`

## Bring-Up

Run on a 2-GPU node:

```bash
cd /root/kvcached
git checkout pd-disagg-LMCacheConnectorV1-1
git pull
```

Plain vLLM LMCacheConnectorV1. Leave vLLM request-id settings alone:

```bash
RUN_WITH_KVCACHED=0 \
GPU_MEM_UTIL=0.45 \
PROMPT_MODE=long \
TIMEOUT_REQUEST=300 \
./experiments/12_lmcache_connector_v1_debug.sh
```

This run is good only if the summary includes:

```text
Classifier: decoder reported non-zero LMCache hit tokens.
```

After plain LMCache shows non-zero hits, test kvcached:

```bash
RUN_WITH_KVCACHED=1 \
GPU_MEM_UTIL=0.45 \
PROMPT_MODE=long \
TIMEOUT_REQUEST=300 \
./experiments/12_lmcache_connector_v1_debug.sh
```

## Interpretation

If the plain run reports zero LMCache hits, debug LMCache PD/cache lookup before
testing kvcached. If the plain run reports non-zero hits and the kvcached run
does not, the regression is in the kvcached path.

## Observed Results So Far

Test environment:

```text
vLLM: 0.19.0
LMCache: 0.4.4
NIXL: 1.1.0
GPU: 2x NVIDIA H100 80GB HBM3
Model: Qwen/Qwen2.5-1.5B-Instruct
```

Plain vLLM `LMCacheConnectorV1` baseline passes with real LMCache hits:

```text
run_id: plain_lmcache_hits_1
prefill request 1: Total tokens 546, LMCache hit tokens: 0
prefill request 1: Stored 546 out of total 546 tokens
prefill request 2: Total tokens 545, LMCache hit tokens: 512
prefill request 2: Retrieved 512 out of 512 required tokens
decode request 1: LMCache hit tokens: 512
decode request 1: Retrieved 512 out of 512 required tokens
decode request 2: LMCache hit tokens: 512
decode request 2: Retrieved 512 out of 512 required tokens
```

The same harness with kvcached enabled fails on the first prefiller save:

```text
run_id: kvcached_lmcache_hits_1
prefill request 1: Total tokens 546, LMCache hit tokens: 0
HTTP status: 500
```

The failure is not caused by missing PD request metadata. The failing scheduler
output includes:

```text
extra_args={'kv_transfer_params': {'ret_first_tok': True, 'disagg_spec': {
  'req_id': 'lmcache-disagg-...',
  'receiver_host': 'localhost',
  'receiver_init_port': [7300],
  'receiver_alloc_port': [7400]}}}
disagg_spec=DisaggSpec(... receiver_init_port=[7300], receiver_alloc_port=[7400])
```

The root stack is in LMCache's GPU store path:

```text
lmcache_engine.store
  -> gpu_connector.batched_from_gpu
  -> initialize_kvcaches_ptr
  -> permute_kv_caches_to_contiguous
ValueError: tensor is non-contiguous for reasons other than permutation
```

The relevant contrast is the KV cache shape/layout seen by LMCache:

```text
plain:    KV layer groups shape=torch.Size([2, 75835, 16, 2, 128])
kvcached: KV layer groups shape=torch.Size([2, 185088, 16, 2, 128])
```

In the kvcached run, kvcached reports `contiguous_layout=True`, but its vLLM
allocation path returns layer views built from a layer-interleaved backing
tensor:

```python
contiguous_tensor = raw_kv_tensors[0].view(dtype=dtype)[:num_eles].view(contiguous_shape)
kv_tensors = [
    contiguous_tensor[:, i].permute(*permute_order) for i in range(num_layers)
]
```

That view is valid for vLLM's attention path, but LMCache assumes any
non-contiguous KV tensor can be recovered by a metadata-only permutation. The
kvcached view still has hidden layer spacing in its block stride, so LMCache
rejects it as slicing/as_strided-style non-contiguity.

Evidence bundles from these runs were collected with:

```bash
./experiments/collect_lmcache_connector_v1_evidence.sh \
  experiments/logs_lmcache_v1_debug/plain_lmcache_hits_1

./experiments/collect_lmcache_connector_v1_evidence.sh \
  experiments/logs_lmcache_v1_debug/kvcached_lmcache_hits_1
```

The resulting tarballs contain `summary.md`, package versions, system info,
configs, request/response JSON, and prefill/decode/proxy logs.

## Notes

- The script defaults to the newer LMCache disaggregated-prefill config style:
  `enable_pd: True`, `transfer_channel: "nixl"`, and `pd_role`.
- If the installed LMCache expects the older vLLM example config, rerun with:

```bash
LMCACHE_CONFIG_STYLE=nixl_legacy \
./experiments/12_lmcache_connector_v1_debug.sh
```

- Generated logs are under:

```text
experiments/logs_lmcache_v1_debug/latest/
```

- Important files:

```text
proxy.log
prefill.log
decode.log
request_1_request.json
request_1_response.json
request_1_curl_status.txt
```

## External References

- vLLM LMCache disaggregated example:
  https://docs.vllm.ai/en/latest/examples/disaggregated/lmcache/
- LMCache 1P1D NIXL docs:
  https://docs.lmcache.ai/disaggregated_prefill/nixl/1p1d.html
