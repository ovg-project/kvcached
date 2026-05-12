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
