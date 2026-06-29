# Performance Tuning

## Memory Configuration

### Prefix Cache Budget

Adjust `KVCACHED_MAX_CACHED_TOKENS` based on your workload:

- **High prefix reuse** (system prompts, RAG): Increase to 32000+
- **Diverse requests** (no common prefixes): Set to 0 or low value
- **Maximum elasticity**: Set to 0 (disable prefix caching)

### Memory Limits

Use `kvctl` to set per-model memory limits:

```bash
kvctl
kvcached> limit-percent VLLM 60
kvcached> limit-percent SGLANG 40
```

## Engine Tuning

### vLLM

- Use `VLLM_USE_V1=1` for best performance with kvcached
- Set `VLLM_ATTENTION_BACKEND=FLASH_ATTN` for optimal attention

### SGLang

- Default settings work well with kvcached
- Consider `--page-size 1` for fine-grained memory control

## Monitoring

Use `kvtop` to identify memory bottlenecks in real-time and adjust configurations accordingly.
