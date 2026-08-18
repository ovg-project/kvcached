# GPU CI overview

kvcached GPU testing is a repository-owned workflow. It can run on any
persistent Linux x86-64 NVIDIA GPU host and does not depend on a particular
cloud, university cluster, scheduler, or private service. For setup, see
[GPU CI runner setup](GPU_CI_RUNNER.md).

## Included

- Scheduled and manually dispatched GitHub Actions workflow.
- Opt-in GPU testing for any pull request, including from a fork, gated on a
  label only someone with write or triage permission can apply.
- Core allocator, vLLM, SGLang, combined-engine, two-GPU NIXL, and `all`
  profiles.
- The GPU pytest set read from `tests/manifests/gpu.txt` at run time, so a
  newly classified GPU test is covered without editing the script.
- Separate Python environments and extension builds for core, vLLM, and
  SGLang, each optional at provisioning time.
- One-command environment provisioning with dependency checks.
- Host-wide locking and idle-GPU protection.
- Deterministic correctness requests for both serving engines.
- Repeated-run support for detecting flaky allocator behavior.
- Environment snapshots, server logs, client responses, post-run process
  state, and machine-readable summaries.
- Artifact upload on both success and failure.

## Validated

The core profile was validated by the original author on one NVIDIA H20 over
five consecutive runs, and the public vLLM/NIXL prefill-decode harness on two
NVIDIA H20 GPUs, both with and without kvcached.

The label-triggered path was validated end to end on an NVIDIA A100: a labelled
fork pull request planned the `core` profile and ran the full GPU pytest set on
the registered host.

## Deployment choices

The deployment owner only chooses:

1. which persistent GPU host to register;
2. whether one or two GPUs are exposed to the runner;
3. the preferred UTC execution time for the scheduled slot, and how many
   merges must accumulate before that slot is spent;
4. an optional Hugging Face token when a gated model is selected.
