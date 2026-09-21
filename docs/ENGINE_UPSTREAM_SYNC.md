# OVG engine fork synchronization

`.github/workflows/engine-upstream-sync.yml` is the automated synchronization
layer for OVG-maintained vLLM and SGLang forks.

Configure repository variables:

```text
OVG_VLLM_REPOSITORY=ovg-project/vllm
OVG_SGLANG_REPOSITORY=ovg-project/sglang
```

Each target repository should contain a `kvcached-main` branch created from
official upstream with the small OVG integration patch stack on top. The
workflow leaves the target repository's `main` branch available as a clean
upstream mirror and opens synchronization pull requests against
`kvcached-main`. A different integration branch can be selected with the
optional `OVG_VLLM_BASE_BRANCH` and `OVG_SGLANG_BASE_BRANCH` variables.

Until a target variable is configured, the scheduled job reports a notice and
skips that engine. This lets the synchronization code land before the matching
OVG repository is created without producing a failing daily workflow.

Add `OVG_SYNC_TOKEN` as a repository secret. The token needs permission to push
branches and create pull requests in both target repositories.

The daily workflow:

1. fetches the OVG fork and official upstream;
2. checks whether the upstream commit is already present;
3. merges upstream into the integration history, retaining merge-only repairs;
4. runs compatibility checks;
5. pushes the branch and opens a pull request;
6. uploads JSON and Markdown reports.

Each engine uses one stable automation-owned branch. A later daily run updates
that branch with `force-with-lease` and refreshes the existing pull request
body, so an unmerged update never creates duplicate pull requests.
The workflow checks the existing PR base before pushing and again before
publication. Runs are serialized without cancelling an in-progress update.
An unchanged pending result keeps its SHA and can recover a failed PR creation.
The managed marker records the tree and parent as well as the input revisions;
appended or amended repairs stop automation without overwriting the branch.
Markers created by the older workflow lack this metadata and also stop safely:
merge or otherwise resolve that pending branch before starting a new sync.

## One-time repository bootstrap

Before enabling the schedule for an engine:

1. fork the official engine into the OVG organization;
2. create `kvcached-main` from the supported upstream commit;
3. add and review the minimal kvcached integration patch stack;
4. protect `kvcached-main` and require the engine's CPU and GPU checks;
5. configure the repository variable and `OVG_SYNC_TOKEN` in kvcached.

The scheduled workflow owns only `automation/upstream-vllm` and
`automation/upstream-sglang`. It never pushes directly to the protected
integration branch.

The automation uses `merge`. Plain rebase can silently discard changes made
only while resolving a merge. The CLI retains `--strategy rebase` for linear
patch stacks, but refuses it when the integration-only history contains merges.
All checks run against the final marker commit; changes to that commit or tracked
files during validation prevent publication. The remote SHA is checked after push.

Conflicts and failed checks intentionally stop before push. The JSON report
contains conflict paths, commits, commands, return codes, and log tails. That
report is the stable handoff format for an agent skill that can inspect API
changes, propose patch updates, and rerun the same checks.

The repository includes that workflow as
`.agents/skills/ovg-engine-upstream-sync`. It instructs an agent to preserve a
conflicted worktree, inspect kvcached's vLLM/SGLang compatibility surfaces,
repair only the affected integration, add regression coverage, and require
the appropriate GPU gate before proposing a pull request.

`tools/check_engine_compat.py` scans selected module, class, and method contracts,
including vLLM worker initialization and memory profiling and SGLang allocator
methods. Its JSON report includes method signature fingerprints, but does not
yet validate argument compatibility or every patch point. Passing this scan is
only a preliminary structural check, not a claim of runtime compatibility.

The Git operation can also be exercised locally without pushing:

```bash
python tools/sync_engine_upstream.py \
  --engine vllm \
  --target-repository https://github.com/ovg-project/vllm.git \
  --upstream-repository https://github.com/vllm-project/vllm.git \
  --sync-branch automation/local-vllm-sync \
  --strategy merge \
  --check "python -m compileall -q ." \
  --result-json /tmp/vllm-sync.json \
  --report /tmp/vllm-sync.md
```
