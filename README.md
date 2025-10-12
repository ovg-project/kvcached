<div align="center">
  <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/logo-v2.svg" alt="kvcached logo" height="96" />

  <br>
  <br>
  <p>
    <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.9%E2%80%933.12-blue"></a>
    <img alt="Engines" src="https://img.shields.io/badge/engines-SGLang%20%7C%20vLLM-blueviolet">
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  </p>

</div>

<h2 align="center">Virtualized Elastic KV Cache for Dynamic GPU Sharing and Beyond </h2>

kvcached is a KV cache library for LLM serving/training on **shared GPUs**.  By bringing OS-style **virtual memory** abstractions to LLM systems, it supports **elastic and demand-driven** KV cache allocation and reclamation, improving utilization under dynamic workloads.

As shown in the figure below, kvcached decouples GPU virtual addressing from physical memory allocation for KV caches. Serving engines initially reserve virtual address space only and later back it with physical GPU memory when the cache is actively used. This decoupling allows on-demand allocation and release of KV cache, leading to better GPU memory utilization under dynamic and mixed workloads.

<p align="center">
  <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/vmm_v2.svg" alt="kvcached virtual memory model" width="600" />
</p>

<h3 align="left">Key Features</h3>

- **Elastic KV cache**: allocate and reclaim KV memory dynamically to match live load.
- **GPU virtual memory**: decouple logical KV from physical GPU memory via runtime mapping.
- **Memory control CLI**: enforce memory limits with kvcached CLI.
- **Frontend router and sleep manager**: route requests to the corresponding backend and put models to sleep when idle.
- **Support SGLang and vLLM**: integrate with SGLang and vLLM.

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

See concrete examples here: [kvcached/examples](./examples).

## Performance: Multi-LLM serving

kvcached enables dynamic memory sharing between LLMs, allowing them to share the same GPU memory elastically. As a comparison, the current serving engines need to statically reserve GPU memory at startup.

This benchmark shows the performance benefits of kvcached when serving three `Llama-3.1-8B` models on an A100-80G GPU under workloads with intermittent peaks. kvcached can achieve **2-28x TTFT reduction** compared to the current serving engines. This performance gain can be converted to **significant cost savings** for LLM serving. Without kvcached, the systems have to provision more GPUs to achieve the same performance.
Details can be found in [benchmarks/bench_latency_benefit](./benchmarks/bench_latency_benefit).

<p align="center">
  <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/ttft_results/ttft_mean.svg" alt="TTFT mean" width="49%" />
  <img src="https://raw.githubusercontent.com/ovg-project/kvcached/refs/heads/main/assets/ttft_results/ttft_p99.svg" alt="TTFT p99" width="49%" />
</p>

## Installation

### Prerequisites

- Python (tested with 3.9 - 3.12)
- PyTorch (compatible with SGLang and vLLM)

kvcached can be installed as a plugin with SGLang and vLLM.

### Quick Install

To install kvcached into an existing SGLang or vLLM environment:

```bash
uv pip install kvcached --no-build-isolation
```

### All-in-One Setup Script

For a complete setup with kvcached and a specific inference engine version, use our automated setup script. This script creates a separate virtual environment (using `uv`) and installs kvcached with your chosen engine:

```bash
cd engine_integration/scripts
# check installation instructions
./setup.sh --help

# install kvcached with SGLang v0.4.9
./setup.sh --engine sglang --engine-version 0.4.9
# install kvcached with vLLM v0.10.1
./setup.sh --engine vllm --engine-version 0.10.1
```

This script will install the specified versions of engines, create separate venv environments (using `uv`), and install kvcached. Should you require more versions, please let us know by opening an issue.

## Run kvcached with Docker

You can test or develop kvcached with Docker.

To test kvcached with SGLang or vLLM.

```bash
docker pull ghcr.io/ovg-project/[kvcached-sglang|kvcached-vllm]:latest
```

For developmenet:

```bash
docker pull ghcr.io/ovg-project/kvcached-dev:latest
```

More instructions can be found [here](./docker/README.md).

## Testing

kvcached can be enabled or disabled by `export ENABLE_KVCACHED=true` or `false`. To verify the successful installation and benchmark the performance of SGLang/vLLM with kvcached, run:

```bash
cd benchmarks/simple_bench
export VENV_PATH=../../engine_integration/[sglang|vllm]-kvcached-venv
./start_server.sh [sglang|vllm] --venv-path $VENV_PATH --model meta-llama/Llama-3.2-1B
# Wait until LLM server is ready
./start_client.sh [sglang|vllm] --venv-path $VENV_PATH --model meta-llama/Llama-3.2-1B
```

The benchmark scripts automatically set `ENABLE_KVCACHED=true`. Please refer to each script for instructions on how to run inference with kvcached.

## Roadmap

The latest roadmap is also tracked in [issue #125](https://github.com/ovg-project/kvcached/issues/125).

- **Engine integration**
  - [x] SGLang and vLLM
  - [ ] Ollama (in progress)
  - [ ] llama.cpp and LMStudio
- **Features**
  - [x] Tensor parallelism
  - [ ] Prefix caching
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

kvcached is developed by many contributors from the community. Feel free to contact us for contributions and collaborations.

```
Jiarong Xing (jxing@rice.edu)
Yifan Qiao (yifanqiao@berkeley.edu)
Shan Yu (shanyu1@g.ucla.edu)
```

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
  journal={arXiv preprint arXiv:2505.04021},
  year={2025}
}
```
