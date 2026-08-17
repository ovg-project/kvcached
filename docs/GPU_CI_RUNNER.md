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
prints the three Python repository variables to configure:

```text
KVCACHED_GPU_PYTHON
KVCACHED_VLLM_PYTHON
KVCACHED_SGLANG_PYTHON
```

Separately, set repository variable `KVCACHED_GPU_VISIBLE_DEVICES` to the
physical GPU index or UUID assigned to this runner, for example `0`. A manual
`nixl` run must select exactly two devices through the workflow's `devices`
input, for example `0,1`. If an external scheduler already exports
`CUDA_VISIBLE_DEVICES`, the script accepts that value instead. All subprocesses
then see only the selected devices, renumbered as logical `cuda:0` (and
`cuda:1` for NIXL).

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

The `nixl` profile requires exactly two selected GPUs. The other profiles use
exactly one GPU:

- `core`: allocator and extension tests;
- `vllm` or `sglang`: core plus one real engine correctness request;
- `engines`: vLLM and SGLang smoke tests sequentially;
- `compat`: the manually selected model, engine, and layout compatibility cells;

The scheduled workflow runs `engines` daily. Logs are uploaded as a workflow
artifact even when the test fails.

## Model compatibility matrix

The separate **Model Compatibility Matrix** workflow is manual-only because a
full run starts 20 engine/model/layout combinations. It runs the cells
sequentially on one selected GPU so the Hugging Face cache is reused and one
failed cell does not prevent later cells from being measured.

Inputs can select all cells or one engine, model architecture, and layout. When
the default model is too large for the runner, select one architecture and set
`model_override` to a smaller model ID from the same family. The model manifest
is stored in `tools/model_compatibility_matrix.json`; Qwen3.5 automatically uses
the required 4 MiB kvcached page size.

Each cell starts the engine with kvcached, sends a deterministic short request,
and records one of `pass`, `crash-at-startup`, or `garbled-output`. JSON,
Markdown, server/client logs, and the runner environment are uploaded for 30
days. By default the workflow reports all results without failing early. Enable
`fail_on_non_pass` when using a selected set of expected-pass cells as a
release gate.

The runner script also:

- acquires a host-wide `flock` so CI and local jobs cannot overlap;
- refuses to start when one of the selected GPUs is busy by default;
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
CHECK_ONLY=1 GPU_CI_PROFILE=core \
  KVCACHED_GPU_VISIBLE_DEVICES=0 bash tools/run_gpu_ci.sh
```

The only deployment-specific inputs are the runner registration, selected GPU,
and schedule. Register the machine with the labels shown above, configure the
three printed Python variables and `KVCACHED_GPU_VISIBLE_DEVICES`, then adjust
the UTC cron expression in
`.github/workflows/gpu-tests.yml` if the default `09:41 UTC` is unsuitable.
