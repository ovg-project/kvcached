# GPU CI runner setup

The workflow in `.github/workflows/gpu-tests.yml` expects a Linux x86-64
self-hosted GitHub Actions runner with these labels:

```text
self-hosted, linux, x64, gpu, kvcached
```

The machine must provide:

- an NVIDIA driver and `nvidia-smi`;
- a Python environment with a CUDA-enabled PyTorch build;
- GCC 9 or newer and the CUDA build tools required by kvcached;
- vLLM and NIXL when the `nixl` profile is used;
- access to the selected Hugging Face model.

With `GPU_CI_INSTALL=1`, the entry point installs `pytest`, `setuptools`,
`packaging`, and `wheel` before building kvcached. System compiler and CUDA
toolkit installation remain runner-image responsibilities; their versions are
captured in the artifact environment snapshot.

Set repository variable `KVCACHED_GPU_PYTHON` when the desired interpreter is
not named `python`. Add secret `HF_TOKEN` for gated models.

## Trigger policy

- Pushes to `main` run the `core` profile.
- Same-repository pull requests run only after a maintainer adds the `gpu-ci`
  label. Fork pull requests are intentionally excluded because they must not
  execute untrusted code on a persistent lab machine.
- Maintainers can start either `core` or `nixl` from **Actions > GPU Tests >
  Run workflow**.

The `nixl` profile requires two visible GPUs. The other profiles use one GPU:

- `core`: allocator and extension tests;
- `vllm` or `sglang`: core plus one real engine correctness request;
- `engines`: vLLM and SGLang smoke tests sequentially;

The scheduled workflow runs `engines` daily. Logs are uploaded as a workflow
artifact even when the test fails.

The runner script also:

- acquires a host-wide `flock` so CI and local jobs cannot overlap;
- refuses to start on a busy GPU by default;
- records the driver, GPU model, PyTorch/CUDA and compiler versions, package
  snapshot, commit, `/dev/shm` capacity, and compute processes;
- supports repeat counts from 1 to 10 for flaky-test detection;
- writes `summary.json` and post-run GPU process state on success or failure.

Set repository variable `KVCACHED_GPU_REQUIRE_IDLE=0` only when the machine has
external GPU scheduling that already guarantees isolation.

For a short reserved GPU window, run `core` once and set
`GPU_CI_SKIP_CORE=1 GPU_CI_INSTALL=0` on later profiles. This avoids rebuilding
the extension and repeating allocator tests before every engine or benchmark
run.

Before registering the runner, the script can be checked on any machine:

```bash
CHECK_ONLY=1 GPU_CI_PROFILE=core bash tools/run_gpu_ci.sh
```
