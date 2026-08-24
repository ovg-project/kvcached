# Two-H20 CPU offload validation

This runbook validates the CPU-offload implementation independently of any GPU
provider. It uses one checkout for the core VMM implementation and a sibling
checkout for the vLLM integration.

## Prepare before the reservation

```bash
git clone --branch zixuan/cpu-offload-control-plane \
  https://github.com/Lanoxia/kvcached.git kvcached-cpu-offload
git clone --branch zixuan/vllm-cpu-offload \
  https://github.com/Lanoxia/kvcached.git kvcached-vllm-offload
```

Install the intended CUDA-enabled PyTorch and vLLM environment before starting
the timed experiment. Pre-download the selected model when possible. The
default model is `Qwen/Qwen2.5-1.5B-Instruct`.

## Run

Choose an absolute directory on persistent storage. Do not point
`ARTIFACT_ROOT` only at `/tmp`.

```bash
cd kvcached-cpu-offload
ARTIFACT_ROOT=/persistent/kvcached-results \
VLLM_REPO_DIR="$PWD/../kvcached-vllm-offload" \
MODEL=Qwen/Qwen2.5-1.5B-Instruct \
bash tools/run_two_gpu_cpu_offload_campaign.sh
```

If time and engine environments permit, the same archive can also include the
latest two-GPU NIXL regression and a targeted real-GPU model matrix:

```bash
ARTIFACT_ROOT=/persistent/kvcached-results \
VLLM_REPO_DIR="$PWD/../kvcached-vllm-offload" \
GPU_CI_REPO_DIR="$PWD/../kvcached-gpu-ci" \
MODEL_MATRIX_REPO_DIR="$PWD/../kvcached-model-matrix" \
RUN_NIXL=1 RUN_MODEL_MATRIX=1 \
bash tools/run_two_gpu_cpu_offload_campaign.sh
```

The optional model matrix defaults to the Qwen3 family across both engines and
both layouts. Use `MODEL_COMPAT_MODEL`, `MODEL_COMPAT_MODEL_OVERRIDE`,
`MODEL_COMPAT_ENGINE`, and `MODEL_COMPAT_LAYOUT` to narrow or change it.

The campaign performs these checks in order:

1. Verify that `nvidia-smi` and PyTorch can see two GPUs.
2. Run VMM offload/restore correctness, memory reclamation, and transfer
   benchmarks independently on both GPUs.
3. Run VMM round trips concurrently on both GPUs to detect device-selection or
   shared-state errors.
4. Run a normal vLLM server on GPU 1 and a kvcached CPU-offload server on GPU 0.
   Compare deterministic output and require non-zero GPU-to-CPU store and
   CPU-to-GPU replay metrics.

Every phase has a separate status and log. Independent phases continue after a
failure. The exit trap always writes a manifest, a compressed archive, and the
archive SHA-256 checksum.

## Before releasing the instance

1. Confirm that `campaign-status.txt`, `MANIFEST.sha256`, the `.tar.gz` archive,
   and its `.sha256` file exist.
2. Copy the archive and checksum to durable local storage.
3. Run `sha256sum -c <archive>.sha256` on the copied files.
4. Only release the GPU instance after the local checksum passes.
