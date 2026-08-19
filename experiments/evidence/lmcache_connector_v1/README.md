# LMCacheConnectorV1 evidence bundles

These tarballs capture the evidence used by the LMCacheConnectorV1
diagnostic PR. The filenames and top-level archive directories are
timestamp-free on purpose so they remain stable in the repository and in PR
discussion.

## Files

- `lmcache_connector_v1_plain_hits.tar.gz`
  - Plain vLLM + LMCacheConnectorV1 baseline.
  - Shows successful prefiller/decoder requests and non-zero LMCache hit
    tokens.

- `lmcache_connector_v1_kvcached_default_failure.tar.gz`
  - Initial kvcached run with the default compound layout.
  - Shows the first prefiller request failing in LMCache's GPU KV store path.

- `lmcache_connector_v1_plain_layout_diag.tar.gz`
  - Plain vLLM layout diagnostic control.
  - Shows LMCache receiving contiguous per-layer KV tensors.

- `lmcache_connector_v1_kvcached_compound_layout_failure_diag.tar.gz`
  - kvcached default compound-layout diagnostic.
  - Shows layer-interleaved non-contiguous KV views and LMCache raising:
    `ValueError: tensor is non-contiguous for reasons other than permutation`.

- `lmcache_connector_v1_kvcached_noncompound_layout_fix_pass.tar.gz`
  - kvcached run with `KVCACHED_CONTIGUOUS_LAYOUT=false`.
  - Shows LMCache receiving contiguous per-layer KV tensors, successful
    prefiller/decoder requests, and non-zero LMCache hit tokens.

## Validated compatibility mode

For vLLM `LMCacheConnectorV1`, kvcached should use the non-compound layout:

```bash
KVCACHED_CONTIGUOUS_LAYOUT=false
```
