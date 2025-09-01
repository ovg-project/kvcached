# Try kvcached Using Docker Images

This directory contains Dockerfiles for running **kvcached** together with either [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/lm-sys/sglang). Both images are built upon existing SGLang or vLLM images, and installed with needed patched and kvcached.

We also provide a development image that has both vLLM and SGLang.

---

## 1. Image names & tags

| Engine | Public image | Default tag |
| ------ | ------------ | ----------- |
| vLLM   | `ghcr.io/ovg-project/kvcached-vllm`     | `latest` |
| SGLang | `ghcr.io/ovg-project/kvcached-sglang`   | `latest` |
| vLLM+SGLang | `ghcr.io/ovg-project/kvcached-dev`   | `latest` |

## 2. Pulling a pre-built image

```bash
# vLLM engine
docker pull ghcr.io/ovg-project/kvcached-vllm:latest

# SGLang engine
docker pull ghcr.io/ovg-project/kvcached-sglang:latest

# vLLM+SGLang kvcached development
docker pull ghcr.io/ovg-project/kvcached-dev:latest
```

## 3. Running the containers

We use development as an example.

```bash
docker run -itd \
  --shm-size 32g \
  --gpus all \
  --env "HF_TOKEN=<secret>" \
  -v /dev/shm:/shm \
  --ipc=host \
  --network=host \
  --privileged \
  --name kvcached-dev \
  ghcr.io/ovg-project/kvcached-dev \
  bash
```

## 4 Performing a benchmark

Attach to the container first.

```bash
docker exec -it kvcached-dev bash
```

Then, you can run the `engine_integration/benchmark` as usual for the development image.

```bash
cd engine_integration/benchmark
./start_server.sh [sgl|vllm]
# Wait until LLM server is ready
./start_client.sh [sgl|vllm]
```

For vLLM and SGLang image, need to specify development mode as prod before running the benchmark.

```bash
sed -i 's/^DEFAULT_MODE="dev"$/DEFAULT_MODE="prod"/' \
    engine_integration/benchmark/start_server.sh

sed -i 's/^DEFAULT_MODE="dev"$/DEFAULT_MODE="prod"/' \
    engine_integration/benchmark/start_client.sh
```

## 5. Building the image locally (optional)

If you have modified the source code or want to build for a different base CUDA version you can rebuild the image yourself:

```bash
# Build vLLM image
docker build -f docker/Dockerfile.vllm -t kvcached-vllm .

# Build SGLang image
docker build -f docker/Dockerfile.sglang -t kvcached-sglang .

# Build development image
docker build -f docker/Dockerfile.dev -t kvcached-dev .
```
