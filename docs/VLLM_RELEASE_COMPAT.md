# vLLM release compatibility workflow

This workflow detects stable vLLM releases, prepares a bounded compatibility
repair, validates it on a separate GPU host, and publishes a PR for human review.
It does not merge changes or certify every supported model and GPU topology.

## Execution flow

1. An hourly poll selects the oldest unseen stable release at or above the
   configured starting version. Manual dispatch can select an exact tag.
2. The detector pins the upstream commit and claims the release in a tracker
   issue. Successful, failed, interrupted and no-change runs remain recorded.
   The candidate starts at the PR target's pinned default branch, not the branch
   used to dispatch the controller workflow.
3. A CPU job runs immutable behavioral probes and, when needed, invokes the
   bounded repair runner. Passing CPU probes still produces a candidate for GPU
   validation; it is not a reason to skip that stage.
4. A dedicated GPU runner independently reconstructs the candidate, builds it,
   and runs the GPU probe. It has no Codex credentials.
5. A behavioral GPU failure can trigger one additional CPU/GPU round. The CPU
   job verifies that the report belongs to this candidate and upstream commit.
   Infrastructure failures stop the run instead of asking the agent to fix code.
6. A passing candidate runs the complete CPU suite, MyPy on Python 3.9 through
   3.13, pre-commit, and GPU-free C++ tests in fresh hosted workers.
7. A separate publisher reconstructs exactly the tested commit and creates a PR.
   Existing human branch changes are never overwritten. Unchanged candidates
   produce a successful report without an empty PR. Publication is complete only
   after the PR identity, mergeability and remote checks for that SHA are verified.

Each CPU round permits one Codex attempt. There are at most two CPU/GPU rounds.
GitHub's rerun button is deliberately disabled for this workflow: start a new
manual dispatch with a tag and `retry=true` to obtain a new explicit budget.
An interrupted claim is retained until that explicit retry; polling does not
silently spend another repair budget.

The next hourly poll discovers releases missed while another run was executing.
The `release` event on this repository cannot observe releases in the separate
vLLM repository, so it is not used here.

## Enable the workflow

Enable only after an operator has validated the chosen execution environments.
No registration, credential copying or upstream setting changes happen merely
by merging these files.

| Setting | Purpose |
| --- | --- |
| `VLLM_RELEASE_COMPAT_ENABLED` | Set to `true` to enable discovery. Unset by default. |
| `VLLM_RELEASE_TRACKER_ISSUE` | Number of a dedicated tracker issue in the workflow repository. |
| `VLLM_COMPAT_FIRST_VERSION` | Oldest release to process; defaults to `0.28.0`. |
| `VLLM_COMPAT_PUBLISH_REPOSITORY` | Branch destination, normally an automation fork. |
| `VLLM_COMPAT_PR_REPOSITORY` | Repository receiving the PR. Defaults to the workflow repository. |
| `VLLM_COMPAT_IMAGE` | Optional operator-maintained runtime image; its installed vLLM version must match the selected release. |
| `ENGINE_COMPAT_CODEX_API_KEY` | Secret used only by the CPU repair job. |
| `ENGINE_COMPAT_PUBLISH_TOKEN` | Dedicated publication credential, never supplied to candidate execution. |

The tracker issue body must contain this exact marker:

```text
<!-- kvcached-vllm-release-tracker -->
```

Use a token that can update the selected fork and open the target PR. The normal
workflow `GITHUB_TOKEN` is intentionally not used for publication: its generated
events do not generally start another workflow. Keep the published PR's normal
CI and review requirements enabled. The pipeline performs the full applicable
validation before publication; remote PR checks still confirm the published SHA.
If the target branch advances during validation, the publisher stops and asks
for a new validated run rather than silently rebasing the candidate.

Register an isolated Linux x64 GPU runner with the `engine-compat-gpu` label.
It needs Python 3.9+, Git, Docker with GPU support and a compatible host driver.
Do not register a shared production inference host or a developer's everyday
workstation under that label. A dedicated/ephemeral runner is the intended
deployment, and it must be able to reach GitHub and obtain the runtime image.

The default image is `vllm/vllm-openai:<release-tag>`; its resolved image ID is
recorded for the run. Missing images, incompatible drivers and occupied GPUs are
infrastructure failures. A custom image is useful for an approved CUDA variant,
but does not allow testing one vLLM version while reporting another.

## CPU and GPU separation

Codex calls its model service from the CPU execution host. The GPU host only
receives source, the immutable check program and the selected release. It does
not need to install or log in to Codex. Use a legally supported and reachable
Codex execution environment; changing authentication methods is not a remedy
for service access restrictions.

For local development, run the CPU stage on a machine where Codex is already
usable and provide `--gpu-command` to the GPU stage. This option accepts an
operator-owned JSON argv array, not a shell command from an issue or a candidate.
It receives `ENGINE_COMPAT_SOURCE`, `ENGINE_COMPAT_OUTPUT` and
`ENGINE_COMPAT_VERSION`. Return 0 for success, 1 for a behavioral failure, and 2
or another nonzero code for an infrastructure failure. The command must use a
bounded remote timeout and clean up its own processes. It is deliberately not
an input accepted by the public workflow.

Run `python tools/engine_compat_stage.py --help` for stage arguments. Both local
and Actions handoffs use the same `candidate.json` envelope and reconstruction
checks. Local replay verifies those stages, but is not evidence that a hosted
Action's authentication and runner configuration have been exercised.

## Trust and validation boundaries

- The controller, allowlist and GPU probes come from the workflow revision,
  never from the generated candidate. Check and candidate directories are separate.
- Only exact paths in `.github/engine-compat/vllm-allow.json` may change. This
  initial policy covers the vLLM Python adapter and its profiling regressions;
  unsupported C++, dependency or workflow changes require a human-owned task.
- The envelope records the original base, tag, file bytes, executable modes and
  reconstructed commit SHA. Every receiving job validates it again. The fixed
  automation commit identity/date makes reconstruction deterministic; original
  contributor commits are not rewritten.
- After CPU CI, the controller verifies HEAD, the index tree and actual tracked
  file contents/modes against that envelope. A clean `git status` alone is not
  accepted as proof that validation left the candidate unchanged.
- Hashes detect transfer errors and accidental candidate changes. They are not
  authentication on their own. Actions downloads artifacts by exact name from
  the current run; never accept an arbitrary public artifact URL as evidence.
- GPU source and check mounts are read-only. Only runtime results are writable;
  the candidate cannot modify the supervisor's handoff or success/failure report.
- Candidate execution has no publishing credentials. The publisher does not
  execute candidate code. Allowlist checks are not an OS sandbox; dedicated
  disposable runners and normal human review remain necessary.
- The single-GPU probe covers a small deterministic model and CUDA allocation
  failure/recovery, verifies a partially mapped virtual KV pool, and explicitly
  shuts down the embedded engine. It cannot establish TP/PP, MPS, hybrid/Mamba, multimodal,
  multi-instance fairness or large-model performance. Those remain explicit
  additional acceptance gates for relevant changes.

## Relationship to engine-fork synchronization

The bounded repair component is described in [ENGINE_COMPAT_REPAIR.md](ENGINE_COMPAT_REPAIR.md).
The separate engine-fork synchronization proposal (#423) maintains downstream
vLLM/SGLang patch stacks. It is useful when such a fork is the deployment target,
but synchronizing a fork is not required merely to select an official release.

This workflow targets the pinned official vLLM release and adapts KVCached. It
does not silently rebase or publish an engine fork as a side effect. If a maintained
engine fork is selected later, its prepared commit and matching runtime must be
passed through the same checks before a compatibility result is claimed.
