# kvcached

**GPU Virtual Memory Management for Elastic LLM KV Cache**

---

kvcached (KV cache daemon) is a GPU virtual memory management system for LLM serving engines. It brings OS-style virtual memory abstraction to GPU memory, enabling **elastic and demand-driven** KV cache allocation that improves GPU utilization under dynamic workloads.

## Why kvcached?

Today's LLM serving engines (vLLM, SGLang) statically reserve a large portion of GPU memory at startup for KV cache. This rigid allocation prevents efficient GPU sharing and wastes memory when workloads are dynamic or when multiple models coexist on the same GPU.

kvcached solves this by **decoupling GPU virtual addressing from physical memory allocation**. Serving engines reserve virtual memory upfront but only back it with physical GPU memory when the cache is actively used.

## Key Features

- **Elastic KV Cache** — Allocate and reclaim KV memory dynamically to match live load
- **GPU Virtual Memory** — Decouple logical KV from physical GPU memory via runtime mapping
- **Memory Control CLI** — Enforce memory limits with `kvctl` and monitor with `kvtop`
- **Frontend Router & Sleep Mode** — Route requests to target models and put models to sleep when idle
- **Multi-Engine Support** — Integrate transparently with SGLang and vLLM
- **Prefix Caching** — Support automatic prefix caching (APC) with configurable memory bounds

## Use Cases

| Scenario | Description |
|----------|-------------|
| **Multi-LLM Serving** | Multiple LLMs share a GPU's memory elastically without rigid partitioning |
| **Serverless LLM** | Models spin up and down on demand, allocating KV cache only when needed |
| **Compound AI Systems** | Elastically allocate memory across specialized models in a pipeline |
| **GPU Workload Colocation** | LLM inference coexists with training, fine-tuning, or vision models |

## Quick Navigation

<div class="grid cards" markdown>

- :material-download: **[Installation](getting-started/installation.md)**

    Get kvcached running in minutes via PyPI, source, or Docker.

- :material-rocket-launch: **[Quick Start](getting-started/quick-start.md)**

    Run two LLMs on a single GPU in 5 minutes.

- :material-cube-outline: **[Architecture](core-concepts/architecture.md)**

    Understand the system design and how components interact.

- :material-book-open-variant: **[User Guide](user-guide/multi-model.md)**

    Step-by-step guides for all supported use cases.

</div>

## Supported Engines and Models

| Engine | Versions | Attention Types | Example Models |
|--------|----------|-----------------|----------------|
| SGLang | >= v0.4.9 (tested up to v0.5.10) | MHA / GQA / MLA | Llama 3.1/3.3, Qwen 2.5, DeepSeek-V3, GPT-OSS |
| vLLM | >= v0.8.4 (tested up to v0.19.0) | MHA / GQA / MLA | Llama 3.1/3.3, Qwen 2.5, DeepSeek-V3, GPT-OSS |

## Trusted and Deployed By

Google, LinkedIn, Intel, AMD, Red Hat, Adobe, Sony, ByteDance, Alibaba Cloud, Tencent, and more.

## Citation

If you find kvcached useful, please cite our papers:

```bibtex
@article{yu2026prism,
  title={Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning},
  author={Yu, Shan and Qiao, Yifan and others},
  journal={OSDI},
  year={2026}
}

@article{xing2025towards,
  title={Towards Efficient and Practical GPU Multitasking in the Era of LLM},
  author={Xing, Jiarong and Qiao, Yifan and others},
  journal={arXiv preprint arXiv:2508.08448},
  year={2025}
}
```
