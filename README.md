# kvcached

kvcached is a KV cache management system for serving multiple Large Language Models (LLMs) on a single GPU. It manages and coordinates the use of KV cache resources among co-located LLM instances, enabling them to share GPU memory. kvcached is compatible with popular LLM serving engines, including vLLM and SGLang.

## Prerequisites

* Python (tested with 3.11)
* PyTorch (tested with 2.6.0 and 2.7.0)
* Virtual environment tools (scripts are provided for uv==0.7.12)

**Important Note:** The scripts will create separate virtual environments using `uv` for vLLM and sglang.

## All-in-One Installation (vLLM+SGLang+kvcached, Recommended)

kvcached now supports both vLLM and SGLang. Currently, it requires modifications to the LLM engine's code. To facilitate this process, we provide patches for vLLM version 0.8.4 and SGLang version 0.4.6.post2, along with detailed setup instructions, in the `engine_integration/` directory.

You can install everything (vLLM+SGLang+kvcached) by running the following commands:

```bash
cd engine_integration/scripts
./setup.sh
```

This script will download the specified versions of vLLM and SGLang, create separate venv environments, compile the code, apply the necessary patches, and install kvcached.

## Installation from Source

If you want to install kvcached on your own, you can install it from source, run the following command:

```bash
pip install -r requirements.txt # install build dependencies
pip install -e . --no-build-isolation
```

This will compile and install kvcached. If you have the right versions of vLLM and SGLang, you can apply the patches in `engine_integration/scripts`, and it should work.

NOTE: `--no-build-isolation` is required for kvcached to find the right PyTorch in the current virtual environment.

### Manual Compilation

kvcached includes a CPP-based library called `vmm_ops` for managing low-level CUDA virtual memory operations. This library is typically built and installed automatically during the kvcached installation process. However, one can rebuild the `vmm_ops` library locally by running:

```python
python setup.py build_ext --inplace
```

## Testing

To verify the successful installation and benchmark the performance of vLLM/SGLang with kvcached, run:

```bash
cd engine_integration/benchmark
./start_server.sh [vllm|sgl]
# Wait until LLM server is ready
./start_client.sh [vllm|sgl]
```

Please refer to each script for instructions on how to run vLLM/SGLang with kvcached.

## Known issues

If you observe the SGLang server crashing after the benchmark client finishes, please refer to [this issue](https://github.com/ovg-project/kvcached/issues/5) for a detailed explanation of the root cause and solutions.

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

Feel free to contact us for contributions and collaborations.

```
Jiarong Xing (jxing@rice.edu)
Yifan Qiao (yifanqiao@berkeley.edu)
```
