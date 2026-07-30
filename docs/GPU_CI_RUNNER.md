# GPU CI runner setup

For a concise implementation and deployment summary, see
[GPU CI first delivery](GPU_CI_DELIVERY.md).

The workflow in `.github/workflows/gpu-tests.yml` expects a Linux x86-64
self-hosted GitHub Actions runner with these labels:

```text
self-hosted, linux, x64, gpu, kvcached
```

The machine only needs to provide the standard host-level GPU toolchain:

- an NVIDIA driver and `nvidia-smi`;
- Python 3.11 with `venv`;
- GCC 9 or newer and the CUDA build tools required by kvcached;
- outbound access to GitHub, PyPI, and the selected Hugging Face model.

The repository provisions the Python dependencies itself. Run:

```bash
bash tools/bootstrap_gpu_ci_envs.sh
```

The bootstrap creates separate core, vLLM, and SGLang environments, builds
kvcached as a separate installed wheel in each one, runs `pip check`, and
prints the three repository
variables to configure:

```text
KVCACHED_GPU_PYTHON
KVCACHED_VLLM_PYTHON
KVCACHED_SGLANG_PYTHON
```

The default engine versions match the public kvcached engine images. Override
`CORE_TORCH_SPEC`, `VLLM_SPEC`, or `SGLANG_SPEC` when testing another upstream
version. Add secret `HF_TOKEN` only when the selected model is gated.

## Trigger policy

- Pushes to `main` run the `core` profile.
- Same-repository pull requests run only after a maintainer adds the `gpu-ci`
  label. Fork pull requests are intentionally excluded because they must not
  execute untrusted code on a persistent lab machine.
- Maintainers can start every profile from **Actions > GPU Tests > Run
  workflow**.

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
- rebuilds the current commit into each selected isolated environment, so
  vLLM and SGLang never share an extension binary or dependency set.

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

The only deployment-specific inputs are the runner registration and schedule.
Register the machine with the labels shown above, configure the three printed
Python variables, and adjust the UTC cron expression in
`.github/workflows/gpu-tests.yml` if the default `09:41 UTC` is unsuitable.
