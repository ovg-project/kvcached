# Try kvcached Using Docker Images

This directory contains Dockerfiles for running **kvcached** together with either [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/lm-sys/sglang). Both images are built upon existing SGLang or vLLM images, and installed with needed patched and kvcached.

We also provide a development image that has both vLLM and SGLang.

---

## 1. Image names & tags

| Engine | Public image | Default tag |
| ------ | ------------ | ----------- |
| vLLM   | `ghcr.io/ovg-project/vllm-v0.10.2-kvcached`     | `latest` |
| SGLang | `ghcr.io/ovg-project/sglang-v0.5.2-kvcached`   | `latest` |
| vLLM+SGLang | `ghcr.io/ovg-project/kvcached-dev`   | `latest` |

## 2. Pulling a pre-built image

```bash
# vLLM engine
docker pull ghcr.io/ovg-project/vllm-v0.10.2-kvcached:latest

# SGLang engine
docker pull ghcr.io/ovg-project/sglang-v0.5.2-kvcached:latest

# vLLM+SGLang kvcached development
docker pull ghcr.io/ovg-project/kvcached-dev:latest
```

## 3. Running the containers

We use vLLM as an example.

```bash
docker run -itd \
  --shm-size 32g \
  --gpus all \
  --env "HF_TOKEN=<secret>" \
  -v /dev/shm:/shm \
  --ipc=host \
  --network=host \
  --privileged \
  --name kvcached-vllm \
  ghcr.io/ovg-project/vllm-v0.10.2-kvcached \
  bash
```

## 4 Performing a benchmark

Attach to the container first.

```bash
docker exec -it kvcached-vllm bash
```

Then, you can use it as a normal vLLM container, e.g., running benchmarks.

For example, you can run the following command to start a vLLM server and run benchmarks.

```bash
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export KVCACHED_IPC_NAME=VLLM
vllm serve meta-llama/Llama-3.2-1B --disable-log-requests --no-enable-prefix-caching --port=12346 --tensor-parallel-size=1
vllm bench serve --model meta-llama/Llama-3.2-1B --request-rate 10 --num-prompts 1000 --port 12346
```

NOTE: If installed correctly, you should see that kvcached patches the vLLM:
```
[kvcached][INFO][2025-10-15 23:01:33][patch_base.py:98] Applying 6 patches for vllm
INFO 10-15 23:01:35 [__init__.py:216] Automatically detected platform cuda.
[kvcached][INFO][2025-10-15 23:01:37][version_utils.py:189] Detected vllm version: 0.10.2
[kvcached][INFO][2025-10-15 23:01:37][version_utils.py:189] Detected vllm version: 0.10.2
W1015 23:01:39.249000 598 torch/utils/cpp_extension.py:2425] TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.
W1015 23:01:39.249000 598 torch/utils/cpp_extension.py:2425] If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'] to specific architectures.
[kvcached][INFO][2025-10-15 23:01:39][version_utils.py:189] Detected vllm version: 0.10.2
[kvcached][INFO][2025-10-15 23:01:39][version_utils.py:189] Detected vllm version: 0.10.2
[kvcached][INFO][2025-10-15 23:01:39][version_utils.py:189] Detected vllm version: 0.10.2
[kvcached][INFO][2025-10-15 23:01:39][patch_base.py:178] Successfully patched vllm: elastic_block_pool, engine_core, gpu_model_runner, gpu_worker, kv_cache_coordinator
```

Another way to verify is to check the memory consumption using `nvidia-smi`. With kvcached, you should see that the memory usage is closer to model weight size when there are no requests.

## 5. Building the image locally (optional)

If you have modified the source code or want to build for a different base CUDA version you can rebuild the image yourself:

```bash
# Build vLLM image
docker build -f docker/Dockerfile.vllm -t vllm-[version]-kvcached .

# Build SGLang image
docker build -f docker/Dockerfile.sglang -t sglang-[version]-kvcached .

# Build development image
docker build -f docker/Dockerfile.dev -t kvcached-dev .
```
