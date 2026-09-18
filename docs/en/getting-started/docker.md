# Docker Deployment

Refer to the [Installation guide](installation.md#using-docker) for pulling images and basic container setup.

## Available Images

| Engine | Image | Tag |
|--------|-------|-----|
| vLLM | `ghcr.io/ovg-project/kvcached-vllm` | `latest` |
| SGLang | `ghcr.io/ovg-project/kvcached-sglang` | `latest` |
| Development (both) | `ghcr.io/ovg-project/kvcached-dev` | `latest` |

## Running a Benchmark

```bash
docker exec -it kvcached-vllm bash

export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1

vllm serve meta-llama/Llama-3.2-1B --no-enable-prefix-caching --port=12346
vllm bench serve --model meta-llama/Llama-3.2-1B --request-rate 10 --num-prompts 1000 --port 12346
```

## Building Custom Images

```bash
# Build vLLM image
docker build -f docker/Dockerfile.vllm -t vllm-custom-kvcached .

# Build SGLang image
docker build -f docker/Dockerfile.sglang -t sglang-custom-kvcached .

# Build development image
docker build -f docker/Dockerfile.dev -t kvcached-dev .
```
