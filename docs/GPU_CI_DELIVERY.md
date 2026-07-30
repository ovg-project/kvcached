# GPU CI first delivery

This delivery turns kvcached GPU testing into a repository-owned workflow.
It can run on any persistent Linux x86-64 NVIDIA GPU host and does not depend
on a particular cloud, university cluster, scheduler, or private service.

## Included

- Daily and manually dispatched GitHub Actions workflow.
- Safe opt-in GPU testing for same-repository pull requests.
- Core allocator, vLLM, SGLang, combined-engine, and two-GPU NIXL profiles.
- Separate Python environments and extension builds for core, vLLM, and
  SGLang.
- One-command environment provisioning with dependency checks.
- Host-wide locking and idle-GPU protection.
- Deterministic correctness requests for both serving engines.
- Repeated-run support for detecting flaky allocator behavior.
- Environment snapshots, server logs, client responses, post-run process
  state, and machine-readable summaries.
- Artifact upload on both success and failure.

## Validated

The core profile was validated on one NVIDIA H20:

- 36 tests passed per run;
- five complete runs passed consecutively;
- 180 consecutive test executions passed;
- the resize regression was restored and passed;
- a container linker issue was converted into a portable CUDA-stub fix.

The public vLLM/NIXL prefill-decode harness was separately validated on two
NVIDIA H20 GPUs, both with and without kvcached.

## Deploy

On the persistent GPU host:

```bash
git clone --branch zixuan/gpu-ci-runner \
  https://github.com/Lanoxia/kvcached.git
cd kvcached
bash tools/bootstrap_gpu_ci_envs.sh
```

After this delivery is merged, use the OVG repository's default branch
instead.

Register the host as a GitHub Actions runner with these labels:

```text
self-hosted, linux, x64, gpu, kvcached
```

Set the three repository variables printed by the bootstrap command, then
start **Actions > GPU Tests > Run workflow** with the `core` profile. The
daily `engines` profile runs automatically at the configured UTC cron time.

The deployment owner only chooses:

1. which persistent GPU host to register;
2. whether one or two GPUs are exposed to the runner;
3. the preferred daily UTC execution time;
4. an optional Hugging Face token when a gated model is selected.
