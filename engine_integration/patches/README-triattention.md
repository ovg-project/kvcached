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

## Benchmark scripts

Run the comparison scripts from the kvcached repository root after installing
kvcached, the target serving engine, and the external TriAttention package in
the same Python environment. The scripts start and stop the serving process by
themselves, so make sure the selected port is free before running them.

### vLLM

Use `tools/compare_vllm_triattention.py` to compare kvcached-only serving with
kvcached plus TriAttention. The script disables vLLM automatic prefix caching,
samples GPU memory with `nvidia-smi`, sends concurrent long-context OpenAI
chat requests, and writes one CSV row per run.

```bash
python tools/compare_vllm_triattention.py \
  --concurrencies 4 \
  --budgets 1024 \
  --triattention-root /root/triattention-main \
  --stats-path /root/triattention-main/triattention/calibration/for_aime25_experiment/qwen3_8b.pt \
  --model /root/data/models/Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --max-model-len 32768 \
  --max-tokens 2048 \
  --trust-remote-code
```

Use `--skip-baseline` to run only the TriAttention row, or
`--skip-triattention` to run only the kvcached baseline. Results are written to
`results/results_vllm_tri_compare.csv` by default, and server logs are written
under `/tmp/triattn-compare`.

### SGLang

Use `tools/compare_sglang_triattention.py` to run the same style of comparison
on SGLang. The script starts SGLang with `--disable-radix-cache` and
`--disable-overlap-schedule`, which are required for TriAttention's SGLang
compression path.

SGLang TriAttention compression is decode-time driven, so the recommended
benchmark uses a short prompt and a long forced decode. This lets compression
and reclaim happen while decode memory is still growing, making the effect more
visible in GPU peak memory.

```bash
python tools/compare_sglang_triattention.py \
  --workload-profile decode-stress \
  --concurrencies 4 \
  --budgets 1024 \
  --triattention-root /root/triattention-main \
  --stats-path /root/triattention-main/triattention/calibration/for_aime25_experiment/qwen3_8b.pt \
  --model /root/data/models/Qwen/Qwen3-8B \
  --max-model-len 32768 \
  --prompt-repeat 64 \
  --max-tokens 8192 \
  --min-tokens 8192 \
  --ignore-eos \
  --trust-remote-code
```

Results are written to `results/results_sglang_tri_compare.csv` by default, and
server logs are written under `/tmp/triattn-sglang-compare`. The SGLang parser
counts log lines such as `TriAttention compression complete ... (freed N slots)`
as compression and reclaim events.
