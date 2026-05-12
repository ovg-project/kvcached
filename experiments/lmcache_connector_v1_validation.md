# LMCacheConnectorV1 PD Debug Validation

This branch repeats the P2P NCCL request-ID experiment with vLLM
`LMCacheConnectorV1`.

The harness launches one prefiller, one decoder, and a local proxy. The proxy
sends the same client request to prefiller first with `max_tokens=1`, then to
decoder for the full completion. It forwards a shared `X-Request-Id` to both
servers by default so the experiment is comparable to the P2P NCCL run.
For LMCache PD, the proxy also attaches prefiller-side
`kv_transfer_params.disagg_spec` with the decoder receiver host/init/alloc
ports; without that per-request metadata, the prefiller reaches
`lmcache_engine.store(...)` with a null transfer spec.

## Files

- `experiments/12_lmcache_connector_v1_debug.sh`
- `experiments/lmcache_connector_v1_validation.md`

## Baseline Matrix

Run on a 2-GPU node:

```bash
cd /root/kvcached
git checkout pd-disagg-LMCacheConnectorV1-1
git pull
```

Plain vLLM LMCacheConnectorV1, request ID randomization enabled:

```bash
RUN_WITH_KVCACHED=0 \
GPU_MEM_UTIL=0.45 \
TIMEOUT_REQUEST=300 \
./experiments/12_lmcache_connector_v1_debug.sh
```

Plain vLLM LMCacheConnectorV1, request ID randomization disabled:

```bash
RUN_WITH_KVCACHED=0 \
DISABLE_REQUEST_ID_RANDOMIZATION=1 \
GPU_MEM_UTIL=0.45 \
TIMEOUT_REQUEST=300 \
./experiments/12_lmcache_connector_v1_debug.sh
```

kvcached-enabled LMCacheConnectorV1, request ID randomization enabled:

```bash
RUN_WITH_KVCACHED=1 \
GPU_MEM_UTIL=0.45 \
TIMEOUT_REQUEST=300 \
./experiments/12_lmcache_connector_v1_debug.sh
```

kvcached-enabled LMCacheConnectorV1, request ID randomization disabled:

```bash
RUN_WITH_KVCACHED=1 \
DISABLE_REQUEST_ID_RANDOMIZATION=1 \
GPU_MEM_UTIL=0.45 \
TIMEOUT_REQUEST=300 \
./experiments/12_lmcache_connector_v1_debug.sh
```

## Interpretation

For `P2pNcclConnector`, randomization broke tensor matching because the tensor key
was `request_id#layer_name`, and prefill/decode got different internal vLLM
suffixes.

`LMCacheConnectorV1` should be tested separately because it transfers via
LMCache/NIXL and keying is not the same as P2P NCCL. If the randomization-on and
randomization-off LMCache runs behave the same, request ID randomization is likely
not the relevant failure mode for this connector.

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
