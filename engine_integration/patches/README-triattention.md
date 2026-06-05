# TriAttention external integration

kvcached does not vendor TriAttention source.  To enable TriAttention, install
the upstream repository separately and apply the kvcached compatibility patch:

```bash
git clone https://github.com/WeianMao/triattention.git
cd triattention
git apply /path/to/kvcached/engine_integration/patches/kvcached-triattention-main.patch
pip install -e .
pip install flash-attn --no-build-isolation
```

The patch keeps only the changes needed by kvcached's vLLM integration:

- reject vLLM automatic prefix caching when TriAttention reclaim is active;
- keep effective KV length anchored after compression;
- handle vLLM V1 tensor/numpy input-buffer variants;
- make KV group/tensor resolution more tolerant across vLLM layouts.
- emit optional JSONL compression events when
  `TRIATTN_RUNTIME_EVENT_SINK_PATH` is set, so benchmark scripts can count
  applied/skipped compression actions without depending on log formatting.

SGLang uses the upstream TriAttention integration directly.
