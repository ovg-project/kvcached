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
3. rebases the OVG patch stack onto the new upstream commit;
4. runs compatibility checks;
5. pushes the branch and opens a pull request;
6. uploads JSON and Markdown reports.

Each engine uses one stable automation-owned branch. A later daily run updates
that branch with `force-with-lease` and refreshes the existing pull request
body, so an unmerged update never creates duplicate pull requests.

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

The automation uses `rebase` so the target branch remains the official engine
history followed by a small OVG-owned patch stack. The command-line tool also
supports `--strategy merge` for an existing fork whose history cannot yet be
cleanly rebased.

Conflicts and failed checks intentionally stop before push. The JSON report
contains conflict paths, commits, commands, return codes, and log tails. That
report is the stable handoff format for an agent skill that can inspect API
changes, propose patch updates, and rerun the same checks.

The repository includes that workflow as
`.agents/skills/ovg-engine-upstream-sync`. It instructs an agent to preserve a
conflicted worktree, inspect kvcached's vLLM/SGLang compatibility surfaces,
repair only the affected integration, add regression coverage, and require
the appropriate GPU gate before proposing a pull request.

`tools/check_engine_compat.py` scans the merged engine source for every module,
class, and required allocator method currently patched by kvcached. Its JSON
report includes method signature fingerprints. This catches common upstream
drift before a GPU is occupied, but does not replace runtime tests.

The Git operation can also be exercised locally without pushing:

```bash
python tools/sync_engine_upstream.py \
  --engine vllm \
  --target-repository https://github.com/ovg-project/vllm.git \
  --upstream-repository https://github.com/vllm-project/vllm.git \
  --sync-branch automation/local-vllm-sync \
  --strategy rebase \
  --check "python -m compileall -q ." \
  --result-json /tmp/vllm-sync.json \
  --report /tmp/vllm-sync.md
```
