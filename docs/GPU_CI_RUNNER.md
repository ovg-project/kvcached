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
- Python 3.11 or newer with `venv`;
- GCC 9 or newer and the CUDA build tools required by kvcached;
- outbound access to GitHub, PyPI, and the selected Hugging Face model.

## Provisioning

The repository provisions the Python dependencies itself. Run:

```bash
bash tools/bootstrap_gpu_ci_envs.sh
```

The bootstrap creates separate core, vLLM, and SGLang environments, builds
kvcached as a separate installed wheel in each one, and runs `pip check`. A
host that only serves the `core` profile can skip the two engine environments,
each of which pulls its own PyTorch:

```bash
GPU_CI_ENVS=core bash tools/bootstrap_gpu_ci_envs.sh
```

The default engine versions match the versions kvcached documents support for.
Override `CORE_TORCH_SPEC`, `VLLM_SPEC`, or `SGLANG_SPEC` when testing another
upstream version. Add secret `HF_TOKEN` only when the selected model is gated.

## Host configuration

Which interpreters exist and which GPUs they may use are properties of the
machine, not of this repository, so they live in the runner's own `.env` file
(usually `~/actions-runner/.env`) rather than in repository variables. That is
also the only arrangement that works for the case the GPU run exists for:
GitHub delivers neither secrets nor repository variables to a workflow
triggered by a pull request from a fork.

The bootstrap prints the interpreter lines to paste. Add a
`CUDA_VISIBLE_DEVICES` line naming the physical GPU indices or UUIDs this
runner may use, then restart the runner service — `.env` is read once at
startup.

```text
PYTHON=/home/<user>/.cache/kvcached/gpu-ci-envs/core/bin/python
VLLM_PYTHON=/home/<user>/.cache/kvcached/gpu-ci-envs/vllm/bin/python
SGLANG_PYTHON=/home/<user>/.cache/kvcached/gpu-ci-envs/sglang/bin/python
CUDA_VISIBLE_DEVICES=0
```

```bash
sudo ~/actions-runner/svc.sh stop && sudo ~/actions-runner/svc.sh start
```

`run_gpu_ci.sh` exports the selection to every subprocess, which therefore see
only the named devices, renumbered as logical `cuda:0` (and `cuda:1` for
NIXL). The `nixl` and `all` profiles need exactly two devices; every other
profile needs exactly one, and the script refuses to start on a mismatch.

Set `GPU_CI_REQUIRE_IDLE=0` in the same `.env` only when the machine has
external GPU scheduling that already guarantees isolation. By default the run
refuses to start when another process is resident on a selected GPU.

## Trigger policy

- **Pull request label.** Applying `gpu-ci` to a pull request runs the `core`
  profile. `gpu-ci-vllm`, `gpu-ci-sglang`, `gpu-ci-engines`, `gpu-ci-nixl`,
  and `gpu-ci-all` run the corresponding profile, and beat a plain `gpu-ci`
  carried alongside them. An unlabelled pull request does not run.

  The label is the only gate, deliberately including forks: reviewing a
  contributor's change before merging it is what this run is for, and the
  contributors here work from forks. GitHub withholds secrets and issues a
  read-only token for a fork's `pull_request`, so applying the label accepts
  code execution on the registered host and nothing wider. Applying it is
  restricted to accounts with write or triage permission.

- **Schedule.** The daily cron slot is only a *chance* to run. The plan job
  spends it on the `all` profile only when at least `MIN_MERGES` commits (5 by
  default, overridable with repository variable
  `KVCACHED_GPU_SCHEDULED_MIN_MERGES`) have landed on the default branch since
  the last successful scheduled pass. Otherwise it declines and says why.

- **Manual.** **Actions > GPU Tests > Run workflow** starts any profile, with
  an optional model and a repeat count.

Pushes to `main` do not trigger a run.

## Profiles

The GPU pytest set is `tests/manifests/gpu.txt`, read at run time rather than
listed in the script, so a newly classified GPU test is covered automatically.
Every profile except `nixl` runs it first.

| Profile | GPUs | What it adds to the GPU pytest set |
| --- | --- | --- |
| `core` | 1 | nothing |
| `vllm` | 1 | a kvcached-backed vLLM correctness request, then the vLLM elasticity check |
| `sglang` | 1 | the same two against SGLang |
| `engines` | 1 | both engines' smoke and elasticity checks, sequentially |
| `nixl` | 2 | the vLLM + NIXL prefill/decode disaggregation smoke test |
| `all` | 2 | everything above |

The smoke test proves the engine boots on kvcached and answers correctly. The
elasticity check (`tests/test_elastic_serving*.py`) is the one that puts the KV
cache under real pressure: it drives a 128-prompt batch, watches the mapped
footprint in the `/dev/shm` segment grow and then fall as requests drain, cuts
the limit through the same path `kvctl` uses, and finally re-runs a greedy probe
to confirm the output is unchanged.

Logs are uploaded as a workflow artifact even when the run fails.

The runner script also:

- acquires a host-wide `flock` so CI and local jobs cannot overlap;
- records the driver, GPU model, PyTorch/CUDA and compiler versions, package
  snapshot, commit, `/dev/shm` capacity, and compute processes;
- prints which `kvcached` and which `vmm_ops` the tests actually imported;
- supports repeat counts from 1 to 10 for flaky-test detection;
- writes `summary.json` and post-run GPU process state on success or failure;
- rebuilds the current commit into each selected isolated environment, so
  vLLM and SGLang never share an extension binary or dependency set.

For a short reserved GPU window, run `core` once and set
`GPU_CI_SKIP_CORE=1 GPU_CI_INSTALL=0` on later profiles. This avoids rebuilding
the extension and repeating allocator tests before every engine or benchmark
run.

## Checking the setup without a GPU

```bash
CHECK_ONLY=1 GPU_CI_PROFILE=core \
  CUDA_VISIBLE_DEVICES=0 bash tools/run_gpu_ci.sh
```

The only deployment-specific inputs are the runner registration, the `.env`
above, and the schedule. Register the machine with the labels shown above,
write the `.env`, then adjust the UTC cron expression in
`.github/workflows/gpu-tests.yml` if the default `09:41 UTC` is unsuitable.
