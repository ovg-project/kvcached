---
name: ovg-engine-upstream-sync
description: >-
  Synchronize OVG-maintained vLLM or SGLang forks with official upstream,
  diagnose merge and API compatibility failures, repair narrowly scoped
  kvcached integration patches, run CPU and GPU validation gates, and prepare
  a reviewable pull request. Use for scheduled engine updates, failed Engine
  Upstream Sync workflows, compatibility drift after an upstream release, or
  requests to update the OVG engine forks.
---

# OVG Engine Upstream Sync

Preserve upstream history while keeping the smallest possible OVG delta.
Never push or open a pull request unless the user or calling workflow
explicitly authorizes it.

## Prepare

1. Read `references/compatibility-surfaces.md`.
2. Inspect the kvcached and target-engine worktree status. Preserve unrelated
   changes.
3. Record the target repository, official upstream repository, base branch,
   upstream branch, and engine.
4. Use a new `automation/upstream-<engine>-<date>` branch.

## Attempt the sync

Run the repository tool in dry-run mode first:

```bash
python tools/sync_engine_upstream.py \
  --engine <vllm|sglang> \
  --target-repository <target-url> \
  --upstream-repository <upstream-url> \
  --sync-branch <branch> \
  --check "python <kvcached-root>/tools/check_engine_compat.py --engine <engine> --repo ." \
  --result-json <result.json> \
  --report <report.md>
```

For an expected conflict, create a dedicated temporary directory and add
`--workdir <directory> --keep-conflicts`. The cloned engine worktree remains at
`<directory>/engine`.

## Diagnose

- `up-to-date`: stop without creating a branch.
- `synced`: inspect the API fingerprint and run the validation gates.
- `conflict`: inspect only the files listed in the JSON report and retain both
  upstream behavior and the OVG integration intent.
- `check-failed`: determine whether the failure is an upstream API move, a
  changed signature, a removed compatibility shim, or an unrelated engine
  failure.

Do not resolve a conflict by deleting OVG behavior, pinning an old upstream
commit, weakening a test, or widening a supported-version range without
evidence.

## Repair

1. Resolve conflict markers in the persistent engine worktree.
2. Compare changed symbols with the compatibility surfaces reference.
3. Prefer adapting kvcached's integration layer when the official engine API
   has legitimately changed.
4. Keep engine-fork changes narrow and mark every intentional divergence.
5. Add a regression test for every changed patch point or signature.
6. Stage resolved files and complete the merge commit only after checks pass.

## Validate

Run these gates in order:

1. `tools/check_engine_compat.py` against the merged engine source.
2. Engine import or focused upstream tests for touched modules.
3. `tools/run_cpu_tests.sh` in kvcached.
4. GPU `core` profile.
5. GPU `nixl` profile when transfer or KV-layout code changed.
6. Repeat GPU tests at least three times when concurrency, cleanup, mapping, or
   process lifecycle changed.

Treat a GPU test without a model response correctness check as incomplete.

## Deliver

Prepare a pull request that includes:

- upstream and resulting commits;
- conflict files and repair rationale;
- compatibility report;
- CPU/GPU commands and exact results;
- supported engine version change, if any;
- residual risks and rollback instructions.

Leave unresolved or ambiguous API changes for human review. Never represent a
static contract check as proof of runtime compatibility.
