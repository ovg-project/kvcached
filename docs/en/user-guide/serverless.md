# Serverless LLM

kvcached enables serverless LLM deployments where models spin up on demand and scale to zero when idle.

## How kvcached Helps

- **Elastic KV cache**: Allocates and reclaims KV memory on demand, so idle models consume near-zero GPU memory
- **GPU virtual memory abstraction**: Decouples logical KV from physical GPU memory for dynamic remapping
- **Lower TTFT and cost**: Reduces time-to-first-token and saves money compared to static allocation

## Prism Integration

[Prism](https://arxiv.org/abs/2505.04021) is a multi-LLM serverless serving system built on kvcached that achieves 2x cost savings and 3.3x more SLO attainment through dynamic GPU sharing.

See `examples/06_serverless_serving/` for details.
