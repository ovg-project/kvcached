# Enhance serverless LLM serving with kvcached

Serverless LLM spins up models on demand and scales to zero when idle. Serverless LLM platforms manage provisioning and scaling, so customers only pay for actual usage. This is ideal for bursty or sporadic traffic, but it introduces challenges like cold starts, memory pressure, and tail-latency under dynamic workloads.

## How kvcached helps
- **Elastic KV cache**: Allocates and reclaims KV memory on demand, so idle models consume near‑zero GPU memory and can scale to zero cleanly.
- **GPU virtual memory abstraction**: Decouples logical KV from physical GPU memory, enabling dynamic remapping and higher GPU utilization across mixed, bursty workloads.
- **Lower TTFT and cost**: Thanks to the above features, kvcached reduces time‑to‑first‑token (TTFT) and saves money compared to conventional systems that can't adapt to dynamic traffic.

## Example: Prism integration

Prism is a multi-LLM serving (serverless) system that achieves more than 2× cost savings and 3.3× more SLO attainment through kvcached-enabled dynamic GPU sharing.

<p align="center">
  <img src="https://raw.githubusercontent.com/Multi-LLM/prism-research/main/pic/prism_overview.png" alt="Prism" width="600" />
</p>

The above figure shows the system architecture of Prism. It has two key designs:
1. kvcached-enabled elastic KV cache
2. A two-level scheduling algorithm

For more details, please refer to the Prism paper and its code repository.

- Prism paper: [Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving](https://arxiv.org/abs/2505.04021)
- Code repository: https://github.com/Multi-LLM/prism-research
