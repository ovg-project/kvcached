# Bounded engine compatibility repair

`tools/repair_engine_compat.py` runs behavior checks, asks Codex to repair a
failure, and checks the resulting candidate independently. It produces local
files for review, never commits, pushes, opens a PR, or merges one.

This complements the engine-fork synchronization proposed in #423. That job
updates the engine fork; this tool repairs **KVCached's integration** when an
intentional upstream API change breaks it. They need not change the same repo.

## Prepare the inputs

Use a disposable, clean Git checkout for the candidate. Keep the engine source,
trusted checks, task description and output outside it. Pin the KVCached and
engine revisions, install the required test dependencies, and verify which
installation the tests import. Do not run against an active development tree.

The trusted checks directory contains `checks.json` and the tests/scripts it
invokes. Choose probes that exercise actual behavior, not only symbol names or
import success. A static compatibility scan can pass despite broken runtime
accounting. Include the applicable CPU suite and real-device checks in
`validation`; agent-written tests are additional evidence, not substitutes.

Example `checks.json` (the operator supplies both scripts):

```json
{
  "probes": [
    {
      "name": "worker-contract",
      "argv": ["python", "{checks}/probe_worker.py"]
    }
  ],
  "validation": [
    {
      "name": "cpu-suite",
      "argv": ["python", "{checks}/validate_cpu.py"]
    },
    {
      "name": "gpu-faults",
      "argv": ["python", "{checks}/validate_gpu.py", "{source}"]
    }
  ]
}
```

Commands are argument arrays, not shell strings. `{source}`, `{engine}`,
`{checks}` and `{output}` expand to absolute paths. Commands run from the checks
directory. `ENGINE_COMPAT_SOURCE` and `ENGINE_COMPAT_ENGINE` are also available
to scripts. Scripts must explicitly load the candidate and selected engine;
a previously installed package is not a valid substitute.

Exit 0 means a check passed. By default, exit 1 is a behavior failure that the
agent may repair. Other exit codes, including timeouts and missing executables,
stop the run as infrastructure errors. A check can explicitly declare other
expected `failure_codes`; do not classify a missing dependency or GPU as a code
defect. Pytest's no-tests-collected and import/collection errors are not passes.

## Run

Install and authenticate the [Codex CLI](https://learn.chatgpt.com/docs/non-interactive-mode)
in the repair environment. The tool uses `codex exec --sandbox workspace-write`
and preserves the operator's configured model. `--codex` accepts an alternative
executable path; it does not install or log in to Codex.

```bash
python tools/repair_engine_compat.py \
  --source /work/candidate \
  --engine-source /work/engine \
  --checks /work/trusted-checks \
  --task /work/task.md \
  --allow kvcached/integration/vllm/patches.py \
  --allow tests/test_vllm_virtual_kv_capacity.py \
  --output /work/results/run-001
```

Only exact files under `kvcached/` and `tests/` may change. Task text should
identify the broken contract, required behavior and scope, not tell the agent
to weaken validation until it passes. Defaults are two repair attempts, eight
minutes per agent attempt, and ten minutes per check. All are configurable.
Leave candidate edits unstaged; changes to the Git index or refs stop the run.
Ignored files such as native extensions are fingerprinted too. Only `.git`,
`__pycache__` and `.pytest_cache` directories are excluded. Install environments
and keep build outputs outside the candidate; validation must not rebuild its
native dependencies in place.

The output directory must not already exist. Each attempt has separate logs.
`result.json` includes the baseline SHA, input hashes, check outcomes and candidate
file hashes. On success, `candidate.patch` contains tracked changes and
`candidate-files/` contains every changed file, including new tests.

| Status | Meaning |
|---|---|
| `no-repair-needed` | The selected probes passed; no agent ran. Not full compatibility certification. |
| `validated-candidate` | A repair passed the configured checks. Human review is still required. |
| `no-change` | The agent returned without a repair. |
| `agent-failed` | The agent timed out or failed to execute successfully. |
| `attempt-limit-reached` | Checks still failed after the last permitted attempt. |
| `blocked` | Invalid inputs, modified protected files, scope violation, or a check infrastructure error. |

## Integration and review

Run behavioral probes even if upstream rebase and structural checks succeed.
Provide the selected engine checkout from #423 as `--engine-source`, and a
separate disposable KVCached checkout as `--source`. Do not invoke repair inside
the synchronization job that holds cross-repository write credentials.

A follow-up workflow should retain failure reports and pass a validated
candidate to a separate review/publishing step. Use one stable repair branch
per work item after review, and preserve existing human changes. This tool
deliberately has no unattended scheduler or publication credentials.

The checks' coverage determines the strength of the result. A single-GPU
profiling test does not establish TP/PP, MPS, hybrid-cache or concurrent-serving
correctness. Scope review is still necessary even when tests pass: a repair
could unintentionally extend another backend's behavior inside an allowed file.

## Execution trust boundary

The input hashes and change allowlist detect mistakes; they are **not a security
sandbox**. Candidate code can execute during validation. The task, command
configuration, tests and Codex configuration must be operator-controlled.

Use disposable, isolated workers for automation, without deployment/SSH/GitHub
write credentials or unrelated data. Configure the Codex sandbox and network
policy, and keep trusted validation inputs outside the writable checkout. The
tool removes common GitHub-token and SSH-agent environment variables, but cannot
remove credentials stored in a home directory or contain a malicious child
process by itself. Public automation must separate repair, validation and
publication privileges; a successful local run does not establish that isolation.
