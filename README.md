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
pip install -e . --no-build-isolation
```

This will compile and install kvcached. If you have the right versions of vLLM and SGLang, you can apply the patches in `engine_integration/scripts`, and it should work.

NOTE: `--no-build-isolation` is required for kvcached to find the right PyTorch in the current virtual environment.


## Testing

To verify the successful installation and benchmark the performance of vLLM/SGLang with kvcached, run:

```bash
cd engine_integration/benchmark
./start_server.sh [vllm|sgl]
# Wait until LLM server is ready
./start_client.sh [vllm|sgl]
```

Please refer to each script for instructions on how to run vLLM/SGLang with kvcached.