# Installation

## Prerequisites

- **Python**: 3.9 – 3.13
- **PyTorch**: >= 2.6.0 (with CUDA or ROCm support)
- **Serving Engine**: SGLang (>= v0.4.9) or vLLM (>= v0.8.4)
- **GPU**: NVIDIA GPU with CUDA support, or AMD GPU with ROCm/HIP support

kvcached is installed as a **plugin** alongside your existing SGLang or vLLM environment. It does not replace or conflict with your serving engine installation.

## Install from PyPI

The simplest way to install kvcached:

```bash
pip install kvcached --no-build-isolation
```

!!! note "Why `--no-build-isolation`?"
    kvcached includes a C++ extension that links against PyTorch's CUDA libraries. The `--no-build-isolation` flag ensures the build can find your existing PyTorch installation.

## Install from Source

For development or the latest features:

```bash
git clone https://github.com/ovg-project/kvcached.git
cd kvcached

pip install -e . --no-build-isolation --no-cache-dir
python tools/dev_copy_pth.py
```

The `dev_copy_pth.py` script installs the `.pth` file that enables the autopatch mechanism.

## Using Docker

Pre-built Docker images are available with kvcached installed alongside the serving engines:

=== "vLLM"

    ```bash
    docker pull ghcr.io/ovg-project/kvcached-vllm:latest
    # Based on: kvcached-v0.1.5 + vLLM v0.19.0
    ```

=== "SGLang"

    ```bash
    docker pull ghcr.io/ovg-project/kvcached-sglang:latest
    # Based on: kvcached-v0.1.5 + SGLang v0.5.10
    ```

=== "Development"

    ```bash
    docker pull ghcr.io/ovg-project/kvcached-dev:latest
    # Contains both vLLM and SGLang for development
    ```

### Running a Container

```bash
docker run -itd \
  --shm-size 32g \
  --gpus all \
  --env "HF_TOKEN=<your-token>" \
  -v /dev/shm:/shm \
  --ipc=host \
  --network=host \
  --privileged \
  --name kvcached-vllm \
  ghcr.io/ovg-project/kvcached-vllm:latest \
  bash
```

## Verify Installation

After installation, verify kvcached is working correctly:

```bash
python -c "import kvcached; print(f'kvcached version: {kvcached.__version__}')"
```

When you start a serving engine with kvcached enabled, you should see patching messages:

```
[kvcached][INFO] Applying 6 patches for vllm
[kvcached][INFO] Successfully patched vllm: elastic_block_pool, engine_core, ...
```

## Next Steps

- [Quick Start](quick-start.md) — Run your first multi-model deployment
- [Architecture](../core-concepts/architecture.md) — Understand how kvcached works
