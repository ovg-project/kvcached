# bench_gemma_hetero — kvcached correctness on heterogeneous-KV-group models (Gemma 3/4)

Byte-exact correctness of kvcached on a **heterogeneous-KV-group** hybrid model,
where sliding-window and full-attention layers have different block geometry
(`block_size`/`cell_size`) but the same `block_mem_size`. `google/gemma-4-12B-it`
is the reference: sliding-window layers `block_size=16/cell=4096`, full-attention
layers `block_size=64/cell=1024` (full layers use 1 shared KV head x head_dim 512). Support for this landed on branch
`integration-with-new-vllm` (per-group `as_strided` KV views over one shared pool).

Verified on **NVIDIA GB10 (DGX Spark)**, vLLM **0.24.0**, torch 2.11.0+cu130.

## Requirements
- A CUDA vLLM **>= 0.24** (Gemma 4 native support) + kvcached in the active env.
- Gemma is **gated** — `huggingface-cli login` first.
- Heterogeneous Gemma requires kvcached's **non-contiguous** layout
  (`KVCACHED_CONTIGUOUS_LAYOUT=false`); do **not** pass
  `--disable-hybrid-kv-cache-manager` (it unifies the groups and defeats the test).
- Gemma is multimodal; the check runs it text-only (`limit_mm_per_prompt` all 0).

## How to run

```bash
# baseline (no kvcached)
python correctness_check.py --tag baseline
# kvcached (heterogeneous -> non-contiguous, hybrid manager on)
ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_CONTIGUOUS_LAYOUT=false \
  python correctness_check.py --tag kvcached
# compare token ids
python correctness_check.py --compare baseline kvcached
```

The prompts are long (> the 1024 sliding window) so the sliding-window layers are
exercised, not just the prefix.

## Result — PASS (byte-exact)
kvcached (non-contiguous, hybrid manager on → real 16/64 heterogeneous groups)
produced token-for-token **identical** output to the no-kvcached baseline:
- 4/4 long (>1024-token, sliding-window-spanning) raw prompts, and
- 2/2 chat-template coherent long-context generations (128 tokens each).

This confirms the per-group KV views read correctly, including on the
sliding-window layers at long context — the one residual risk (that the sliding
group's `as_strided` stride matches what the attention kernel expects) that only a
live forward pass can close.
