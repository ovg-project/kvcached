<div align="center">
  <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/logo-v2.svg" alt="kvcached logo" height="96" />

  <br>
  <br>
  <p>
    <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.9%E2%80%933.13-blue"></a>
    <img alt="Engines" src="https://img.shields.io/badge/Engines-SGLang%20%7C%20vLLM-blueviolet">
    <a href="https://yifanqiao.notion.site/Solve-the-GPU-Cost-Crisis-with-kvcached-289da9d1f4d68034b17bf2774201b141"><img alt="Blog" src="https://img.shields.io/badge/Blog-Read-FF5722?logo=rss&logoColor=white&labelColor=555555"></a>
    <a href="https://arxiv.org/abs/2508.08448"><img alt="arXiv: GPU OS vision" src="https://img.shields.io/badge/arXiv-GPU%20OS%20vision-b31b1b?logo=arxiv&logoColor=white&labelColor=555555"></a>
    <br>
    <a href="https://arxiv.org/abs/2505.04021"><img alt="arXiv: Multi LLM Serving" src="https://img.shields.io/badge/arXiv-Multi%20LLM%20Serving-b31b1b?logo=arxiv&logoColor=white&labelColor=555555"></a>
    <a href="https://join.slack.com/t/ovg-project/shared_invite/zt-3fr01t8s7-ZtDhHSJQ00hcLHgwKx3Dmw"><img alt="Slack Join" src="https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&logoColor=white&labelColor=555555"></a>
    <a href="https://deepwiki.com/ovg-project/kvcached"><img alt="DeepWiki" src="https://img.shields.io/badge/DeepWiki-Docs-6B46C1?logo=book&logoColor=white&labelColor=555555"></a>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  </p>

</div>

<h2 align="center">Make GPU Sharing Flexible and Easy </h2>

<p align="center">
  <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/ads.jpg" alt="Make GPU Sharing Flexible and Easy" width="500" />
</p>

kvcached (KV cache daemon) is a KV cache library for LLM serving/training on **shared GPUs**.  By bringing OS-style **virtual memory** abstraction to LLM systems, it enables **elastic and demand-driven** KV cache allocation, improving GPU utilization under dynamic workloads.

kvcached achieves this by decoupling GPU virtual addressing from physical memory allocation for KV caches. It allows serving engines to initially reserve virtual memory only and later back it with physical GPU memory when the cache is actively used. This decoupling enables on-demand allocation and flexible sharing, bringing better GPU memory utilization under dynamic and mixed workloads. Check out more details in the [blog](https://yifanqiao.notion.site/Solve-the-GPU-Cost-Crisis-with-kvcached-289da9d1f4d68034b17bf2774201b141).

<!-- <p align="center">
  <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/vmm_v2.svg" alt="kvcached virtual memory model" width="600" />
</p> -->

<h3 align="left">Key Features</h3>

- **Elastic KV cache**: allocate and reclaim KV memory dynamically to match live load.
- **GPU virtual memory**: decouple logical KV from physical GPU memory via runtime mapping.
- **Memory control CLI**: enforce memory limits with kvcached CLI.
- **Frontend router and sleep mode**: route requests to the target models and put models to sleep when idle.
- **Prefix caching**: support automatic prefix caching (APC) for vLLM (including hybrid attention models) and RadixCache for SGLang, with configurable memory bounds.
- **Support mainstream serving engines**: integrate with SGLang and vLLM.

## 📢 Updates

- **[2026-04]** <img src="https://img.shields.io/badge/Featured%20by-Red%20Hat-EE0000?logo=redhat&logoColor=white" alt="Featured by Red Hat" /> kvcached is **featured by Red Hat** for running LLMs dynamically in production under limited resources! Red Hat's [Sardeenz](https://github.com/rh-aiservices-bu/sardeenz) builds on kvcached to provide dynamic multi-model serving with Kubernetes and OpenShift support. See the [blog post](https://www.redhat.com/en/blog/running-llms-dynamically-production-limited-resources-hard-we-think-theres-room-another-approach) for more details.
  [[▶ View Demo]](https://app.arcade.software/share/xZoDfo1vyDbZrbZTK2gv?ref=share-link)

- **[2026-04]** Added **prefix caching** support. kvcached now supports **automatic prefix caching (APC)** for vLLM and **RadixCache** for SGLang, enabling cross-request prefix reuse while maintaining elastic memory management. The cached token budget can be controlled via `KVCACHED_MAX_CACHED_TOKENS` (default: `16000`).
  - **vLLM**: Cached blocks are retained in an evictable pool and freed on demand when memory pressure occurs (lazy eviction). The token limit is converted to blocks internally (`KVCACHED_MAX_CACHED_TOKENS // block_size`).
  - **SGLang**: After each request finishes, RadixCache proactively evicts entries that exceed the token budget. Works with both `page_size=1` and `page_size>1`.

- **[2026-03]** Added **pipeline parallelism** support.
MLA models (DeepSeek-V3, DeepSeek-V2 etc.) and GPT-OSS hybrid attention models (`openai/gpt-oss-20b`) are now also supported in **vLLM**.
GPT-OSS support in SGLang updated to **v0.5.9**.

- **[2026-02]** kvcached now supports **vLLM v0.16.0** and **SGLang v0.5.9**.
MLA models (DeepSeek-V3, DeepSeek-V2 etc.) are supported in SGLang with both `page_size=1` and `page_size>1`.
GPT-OSS hybrid attention models (`openai/gpt-oss-20b`) are supported in SGLang.

### Supported engines and models

| Engine | Versions | Attention types | Example models |
|--------|----------|-----------------|----------------|
| SGLang | ≥ v0.4.9 (tested up to v0.5.9) | MHA / GQA / MLA | Llama 3.1/3.3, Qwen 2.5, DeepSeek-V3, openai/gpt-oss-20b, etc. |
| vLLM | ≥ v0.8.4 (tested up to v0.16.0) | MHA / GQA / MLA | Llama 3.1/3.3, Qwen 2.5, DeepSeek-V3, openai/gpt-oss-20b |

## Example use cases

<div align="center">
  <table border="0" cellspacing="0" cellpadding="0" style="border: none; border-collapse: collapse; width: auto;">
    <tr>
      <td align="left" style="border: none; vertical-align: middle; width: 196px;">
        <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/uc-multillm.svg" alt="Multi‑LLM serving" width="196" />
      </td>
      <td align="left" style="border: none; vertical-align: middle; padding-left: 8px;">
        <b>Multi‑LLM serving</b><br>kvcached allows multiple LLMs to share a GPU's memory elastically, enabling concurrent deployment without the rigid memory partitioning used today. This improves GPU utilization and saves serving costs.
      </td>
    </tr>
    <tr>
      <td align="left" style="border: none; vertical-align: middle; width: 196px;">
        <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/uc-serverless.svg" alt="Serverless LLM" width="196" />
      </td>
      <td align="left" style="border: none; vertical-align: middle; padding-left: 8px;">
        <b>Serverless LLM</b><br>By allocating KV cache only when needed, kvcached supports serverless deployments where models can spin up and down on demand.
      </td>
    </tr>
    <tr>
      <td align="left" style="border: none; vertical-align: middle; width: 196px;">
        <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/uc-compound.svg" alt="Compound AI systems" width="196" />
      </td>
      <td align="left" style="border: none; vertical-align: middle; padding-left: 8px;">
        <b>Compound AI systems</b><br>kvcached makes compound AI systems practical on limited hardware by elastically allocating memory across specialized models in a pipeline (e.g., retrieval, reasoning, and summarization).
      </td>
    </tr>
    <tr>
      <td align="left" style="border: none; vertical-align: middle; width: 196px;">
        <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/uc-colocate.svg" alt="GPU workload colocation" width="196" />
      </td>
      <td align="left" style="border: none; vertical-align: middle; padding-left: 8px;">
        <b>GPU workload colocation</b><br>kvcached allows LLM inference to coexist with other GPU workloads such as training jobs, fine-tuning, or vision models.
      </td>
    </tr>
  </table>

</div>

See concrete examples here: [kvcached/examples](https://github.com/ovg-project/kvcached/tree/main/examples).

## kvcached in action

The following simple example shows how kvcached enables an unmodified vLLM engine run with dynamically allocated memory.

<p align="center">
  <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/kvcached-example.gif" alt="kvcached in action" width="90%">
</p>

## Performance: Multi-LLM serving

kvcached enables dynamic memory sharing between LLMs, allowing them to share the same GPU memory elastically. As a comparison, the current serving engines need to statically reserve GPU memory at startup.

This benchmark shows the performance benefits of kvcached when serving three `Llama-3.1-8B` models on an A100-80G GPU under workloads with intermittent peaks. kvcached can achieve **2-28x TTFT reduction** compared to the current serving engines. This performance gain can be converted to **significant cost savings** for LLM serving. Without kvcached, the systems have to provision more GPUs to achieve the same performance.
Details can be found in [benchmarks/bench_latency_benefit](https://github.com/ovg-project/kvcached/tree/main/benchmarks/bench_latency_benefit).

<p align="center">
  <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/ttft_results/ttft_mean.svg" alt="TTFT mean" width="49%" />
  <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/ttft_results/ttft_p99.svg" alt="TTFT p99" width="49%" />
</p>

## Installation

### Prerequisites

- Python (tested with 3.9 - 3.13)
- SGLang (tested with v0.5.9) or vLLM (tested with v0.16.0)

kvcached can be installed as a plugin with existing SGLang or vLLM environment.

### Install from PyPI

```bash
pip install kvcached --no-build-isolation
```

### Install from source

```bash
# under the project root folder

pip install -e . --no-build-isolation --no-cache-dir
python tools/dev_copy_pth.py
```

### Using Docker

kvcached installed with original engine dockers.

```bash
docker pull ghcr.io/ovg-project/kvcached-sglang:latest   # kvcached-v0.1.4-sglang-v0.5.9
docker pull ghcr.io/ovg-project/kvcached-vllm:latest     # kvcached-v0.1.4-vllm-v0.16.0
```

We prepare an all-in-one docker for developers:

```bash
docker pull ghcr.io/ovg-project/kvcached-dev:latest
```

More instructions can be found [here](https://github.com/ovg-project/kvcached/blob/main/docker/README.md). GB200 dockers are on the way.

## Documentation

kvcached is indexed on [DeepWiki](https://deepwiki.com/ovg-project/kvcached) for LLM-powered documentation.

The documentation covers:
- Core architecture and memory management system
- Integration with vLLM and SGLang
- Multi-model serving and controller system
- Deployment guides and configuration reference
- Performance benchmarking and analysis
- Development tools and testing

## Testing

kvcached can be enabled by setting the following environmental variables:

```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
```

If you are using the engine-specific dockers, you can test kvcached by running the original engines' benchmark scripts. For example:

```bash
# for sglang
python -m sglang.launch_server --model meta-llama/Llama-3.2-1B-Instruct --disable-radix-cache --port 30000
python -m sglang.bench_serving --backend sglang-oai --model meta-llama/Llama-3.2-1B-Instruct --dataset-name sharegpt --request-rate 10 --num-prompts 1000 --port 30000

# for vllm
vllm serve meta-llama/Llama-3.2-1B-Instruct --no-enable-prefix-caching --port=12346
vllm bench serve --model meta-llama/Llama-3.2-1B-Instruct --request-rate 10 --num-prompts 1000 --port 12346
```

> [!NOTE]
> kvcached now supports **prefix caching** for both vLLM (APC) and SGLang (RadixCache). You can enable prefix caching as usual (the engines' defaults apply). Cached blocks are retained for cross-request prefix reuse and evicted on demand when memory is needed. Set `KVCACHED_MAX_CACHED_TOKENS` to control the cached token budget for both engines (default: `16000`; `0` means unlimited). If you prefer to disable prefix caching, use `--no-enable-prefix-caching` for vLLM and `--disable-radix-cache` for SGLang.
>
> When kvcached is enabled, there is NO need to set memory utilization limit (e.g., using `--gpu-memory-utilization`) as kvcached will automatically manage the memory.

If you installed kvcached using its source code, you can also do the following:

```bash
cd benchmarks/simple_bench
./start_server.sh [sglang|vllm] --venv-path $VENV_PATH --model meta-llama/Llama-3.2-1B-Instruct
# Wait until LLM server is ready
./start_client.sh [sglang|vllm] --venv-path $VENV_PATH --model meta-llama/Llama-3.2-1B-Instruct
```

The benchmark scripts automatically set `ENABLE_KVCACHED=true`. Please refer to each script for instructions on how to run inference with kvcached.

> [!TIP]
> Starting from transformers >= 4.44, there is no fallback “default” chat template. If the tokenizer does not define a chat_template, `apply_chat_template` cannot be used without explicitly providing one. If you encounter chat template errors during its chat warmup at startup, use an Instruct model (e.g., `meta-llama/Llama-3.2-1B-Instruct`) instead of the base model.

> [!NOTE]
> We haven’t fully tested kvcached with every version of SGLang and vLLM (there are too many!). If you run into issues with a specific version, please open an issue---we'll look into it and fix it within a few hours.

## Roadmap

The latest roadmap is also tracked in [issue #125](https://github.com/ovg-project/kvcached/issues/125).

- **Engine integration**
  - [x] SGLang and vLLM
  - [ ] Ollama (in progress)
  - [ ] llama.cpp and LMStudio
- **Features**
  - [x] Tensor parallelism
  - [x] Prefix caching
  - [ ] KV cache offloading to host memory
  - [ ] More attention types (sliding window attention, linear attention, vision encoder, etc.)
- **Performance optimizations**
  - [x] Contiguous KV tensor layout
  - [x] Physical memory management
- **Hardware**
  - [x] NVIDIA GPUs
  - [ ] AMD GPUs

## Contributing

We are grateful for and open to contributions and collaborations of any kind.

We use pre-commit to ensure a consistent coding style. You can set it up by

```
pip install pre-commit
pre-commit install
```

Before pushing your code, please run the following check and make sure your code passes all checks.

```
pre-commit run --all-files
```

## Contacts

kvcached is developed by many contributors from the community. The best way to contact us for questions, issues, and contributions, is through our [Slack channel](https://join.slack.com/t/ovg-project/shared_invite/zt-3fr01t8s7-ZtDhHSJQ00hcLHgwKx3Dmw) or [GitHub Issues](https://github.com/ovg-project/kvcached/issues).

## Citation

If you find kvcached useful, please cite our paper:

```bibtex
@article{xing2025towards,
  title={Towards Efficient and Practical GPU Multitasking in the Era of LLM},
  author={Xing, Jiarong and Qiao, Yifan and Mo, Simon and Cui, Xingqi and Sela, Gur-Eyal and Zhou, Yang and Gonzalez, Joseph and Stoica, Ion},
  journal={arXiv preprint arXiv:2508.08448},
  year={2025}
}

@article{yu2025prism,
  title={Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving},
  author={Yu, Shan and Xing, Jiarong and Qiao, Yifan and Ma, Mingyuan and Li, Yangmin and Wang, Yang and Yang, Shuo and Xie, Zhiqiang and Cao, Shiyi and Bao, Ke and others},
  journal={OSDI},
  year={2026}
}
```
