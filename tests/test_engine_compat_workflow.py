# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Execution-boundary contracts for the release workflow (no GitHub writes)."""

import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    spec = importlib.util.spec_from_file_location(
        "compat_stage", ROOT / "tools/engine_compat_stage.py"
    )
    assert spec is not None and spec.loader is not None
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
finally:
    sys.path.pop(0)


def workflow(name):
    return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)


def test_release_has_explicit_enable_and_bounded_replay():
    config = workflow("vllm-release-compat.yml")
    assert set(config["on"]) == {"schedule", "workflow_dispatch"}
    assert config["concurrency"]["cancel-in-progress"] == "false"
    jobs = config["jobs"]
    assert "VLLM_RELEASE_COMPAT_ENABLED" in jobs["discover"]["if"]
    assert "run_attempt == 1" in jobs["discover"]["if"]
    assert jobs["second"]["needs"] == ["discover", "first"]
    assert "needs.first.outputs.status == 'failed'" in jobs["second"]["if"]
    assert "needs.verify.result == 'success'" in jobs["publish"]["if"]
    assert jobs["publish"]["needs"] == ["discover", "select", "verify"]
    for job in jobs.values():
        assert "run_attempt == 1" in job["if"]
    attempts = [job for job in jobs.values() if "engine-compat-attempt.yml" in job.get("uses", "")]
    assert len(attempts) == 2


def python_block(step):
    return step["run"].split("python - <<'PY'\n", 1)[1].split("\nPY", 1)[0]


def test_baseline_outputs_are_pinned_before_claim_and_forwarded():
    jobs = workflow("vllm-release-compat.yml")["jobs"]
    steps = jobs["discover"]["steps"]
    baseline = next(step for step in steps if step.get("id") == "baseline")
    plan = next(step for step in steps if step.get("id") == "plan")
    assert steps.index(baseline) < steps.index(plan)
    assert baseline["env"]["PR_REPO"] == (
        "${{ vars.VLLM_COMPAT_PR_REPOSITORY || github.repository }}"
    )
    assert "rev-parse" not in plan["run"]
    assert plan["run"].index('Path("baseline.json")') < plan["run"].index(".py claim")
    for name in ("base_repository", "base_branch", "base_sha"):
        assert jobs["discover"]["outputs"][name] == "${{ steps.baseline.outputs." + name + " }}"
        for job in ("first", "second", "verify"):
            assert jobs[job]["with"][name] == "${{ needs.discover.outputs." + name + " }}"
    publication = next(
        step for step in jobs["publish"]["steps"] if "PR_REPO" in step.get("env", {})
    )
    assert publication["env"]["PR_REPO"] == "${{ needs.discover.outputs.base_repository }}"
    assert publication["env"]["BASE_BRANCH"] == "${{ needs.discover.outputs.base_branch }}"
    assert publication["env"]["BASE_SHA"] == "${{ needs.discover.outputs.base_sha }}"
    assert '--pr-repository "$PR_REPO"' in publication["run"]
    assert '--base-branch "$BASE_BRANCH"' in publication["run"]


@pytest.fixture
def baseline_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PR_REPO", "target/engine")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    responses = [
        {"full_name": "Target/Engine", "default_branch": "release/stable"},
        {"ref": "refs/heads/release/stable", "object": {"type": "commit", "sha": "a" * 40}},
    ]
    calls = []

    def gh(argv, *, text):
        assert argv[:4] == ["gh", "api", "--hostname", "github.com"] and text is True
        calls.append(argv[4])
        response = responses[len(calls) - 1]
        if isinstance(response, Exception):
            raise response
        return json.dumps(response)

    monkeypatch.setattr(subprocess, "check_output", gh)
    steps = workflow("vllm-release-compat.yml")["jobs"]["discover"]["steps"]
    code = python_block(next(step for step in steps if step.get("id") == "baseline"))
    return SimpleNamespace(root=tmp_path, responses=responses, calls=calls, code=code)


def test_baseline_uses_target_default_branch_api_not_dispatch_checkout(baseline_case):
    case = baseline_case
    exec(compile(case.code, "<baseline workflow>", "exec"), {})
    expected = dict(
        base_repository="Target/Engine", base_branch="release/stable", base_sha="a" * 40
    )
    assert json.loads((case.root / "baseline.json").read_text()) == expected
    assert (case.root / "outputs").read_text().splitlines() == [
        f"{key}={value}" for key, value in expected.items()
    ]
    assert case.calls == [
        "repos/target/engine",
        "repos/Target/Engine/git/ref/heads/release%2Fstable",
    ]


@pytest.mark.parametrize("fault", ["repository", "branch", "ref", "sha", "type", "api"])
def test_baseline_fault_does_not_emit_partial_pin(baseline_case, fault):
    case = baseline_case
    if fault == "repository":
        case.responses[0]["full_name"] = "other/engine"
    elif fault == "branch":
        case.responses[0]["default_branch"] = "main\ninjected=value"
    elif fault == "ref":
        case.responses[1]["ref"] = "refs/heads/other"
    elif fault == "sha":
        case.responses[1]["object"]["sha"] = "main"
    elif fault == "type":
        case.responses[1]["object"]["type"] = "tag"
    else:
        case.responses[1] = subprocess.CalledProcessError(1, ["gh", "api"])
    with pytest.raises((SystemExit, subprocess.CalledProcessError)):
        exec(compile(case.code, "<baseline workflow>", "exec"), {})
    assert not (case.root / "baseline.json").exists()
    assert not (case.root / "outputs").exists()


@pytest.mark.parametrize("status", ["idle", "pending"])
def test_release_plan_preserves_pinned_baseline(tmp_path, monkeypatch, status):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROFILE", "vllm")
    baseline = dict(base_repository="target/engine", base_branch="main", base_sha="b" * 40)
    (tmp_path / "baseline.json").write_text(json.dumps(baseline))
    (tmp_path / "plan.json").write_text(json.dumps(dict(status=status, tag="v0.28.0")))
    steps = workflow("vllm-release-compat.yml")["jobs"]["discover"]["steps"]
    code = python_block(next(step for step in steps if step.get("id") == "plan"))
    exec(compile(code, "<plan workflow>", "exec"), {})
    assert json.loads((tmp_path / "plan.json").read_text()) == {
        "status": status,
        "tag": "v0.28.0",
        **baseline,
    }


@pytest.mark.parametrize("name", ["engine-compat-attempt.yml", "engine-compat-ci.yml"])
def test_reusable_candidates_use_pinned_repo_and_sha_with_separate_controller(name):
    config = workflow(name)
    for key in ("base_repository", "base_branch", "base_sha"):
        assert config["on"]["workflow_call"]["inputs"][key] == {
            "type": "string",
            "required": "true",
        }
    for job in config["jobs"].values():
        checkouts = [
            step["with"] for step in job["steps"] if step.get("uses") == "actions/checkout@v4"
        ]
        candidate = next(checkout for checkout in checkouts if checkout["path"] == "candidate")
        assert candidate["repository"] == "${{ inputs.base_repository }}"
        assert candidate["ref"] == "${{ inputs.base_sha }}"
        assert candidate["persist-credentials"] == "false"
        controller = next(checkout for checkout in checkouts if checkout["path"] == "controller")
        assert controller["repository"] == "${{ github.repository }}"
        assert controller["ref"] == "${{ github.sha }}"
    if name == "engine-compat-attempt.yml":
        engine = next(
            checkout
            for checkout in config["jobs"]["repair"]["steps"]
            if checkout.get("with", {}).get("path") == "engine"
        )
        assert engine["with"] == {
            "repository": "vllm-project/vllm",
            "ref": "${{ inputs.engine_sha }}",
            "path": "engine",
            "persist-credentials": "false",
        }


def test_publish_checkout_uses_the_same_pinned_pr_target():
    steps = workflow("vllm-release-compat.yml")["jobs"]["publish"]["steps"]
    candidate = next(
        step["with"] for step in steps if step.get("with", {}).get("path") == "candidate"
    )
    assert candidate["repository"] == "${{ needs.discover.outputs.base_repository }}"
    assert candidate["ref"] == "${{ needs.discover.outputs.base_sha }}"


def test_authentication_is_not_sent_to_gpu_or_cpu_validation():
    attempt = workflow("engine-compat-attempt.yml")
    gpu = attempt["jobs"]["gpu"]
    assert "engine-compat-gpu" in gpu["runs-on"]
    assert "CODEX_API_KEY" not in json.dumps(gpu)
    assert "PUBLISH_TOKEN" not in json.dumps(attempt)
    ci = workflow("engine-compat-ci.yml")
    assert "secrets" not in json.dumps(ci)
    shell = (ROOT / "tools/engine_compat_gpu.sh").read_text()
    assert '-v "$OUTPUT/runtime:/results"' in shell
    assert '-v "$OUTPUT:/results"' not in shell
    assert "docker.sock" not in shell


def test_full_candidate_ci_is_not_just_the_repair_probe():
    ci = workflow("engine-compat-ci.yml")["jobs"]["verify"]
    assert ci["strategy"]["matrix"]["python"] == ["3.9", "3.10", "3.11", "3.12", "3.13"]
    text = json.dumps(ci)
    for command in (
        "run_cpu_tests.sh",
        "mypy-",
        "tests/cpp/run_tests.sh",
        "pre-commit run --all-files",
        "controller/tools/engine_compat_artifact.py verify",
    ):
        assert command in text
    gate = ci["steps"][-1]
    assert "if" not in gate and "working-directory" not in gate
    assert gate["env"] == {"BASE_SHA": "${{ inputs.base_sha }}", "RELEASE_TAG": "${{ inputs.tag }}",
                           "PROFILE": "${{ inputs.profile }}"}
    assert shlex.split(gate["run"].replace("\\\n", "")) == [
        "python",
        "controller/tools/engine_compat_artifact.py",
        "verify",
        "--source",
        "candidate",
        "--base",
        "$BASE_SHA",
        "--tag",
        "$RELEASE_TAG",
        "--allow-list",
        "controller/.github/engine-compat/$PROFILE-allow.json",
        "--artifact",
        "handoff/candidate.json",
    ]
    assert "git status" not in gate["run"]


@pytest.fixture
def case(tmp_path, monkeypatch):
    source, prior, output = [tmp_path / name for name in ("source", "prior", "output")]
    for root in (source, prior, output):
        root.mkdir()
    payload = dict(candidate_head="a" * 40, digest="b" * 64, files={})
    stage.write_json(prior / "candidate.json", payload)
    stage.write_json(
        prior / "stage.json",
        dict(status="candidate", digest=payload["digest"], engine_sha="c" * 40,
             candidate_head=payload["candidate_head"], profile=stage.identity(stage.load_profile("vllm"))),
    )
    monkeypatch.setattr(stage, "materialize", lambda *a: payload["candidate_head"])
    monkeypatch.setattr(stage.subprocess, "run", mock.Mock())
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    return SimpleNamespace(
        source=source,
        prior=prior,
        output=output,
        base="d" * 40,
        tag="v0.28.0",
        engine_sha="c" * 40,
        image="",
        gpu_command=None,
        profile="vllm",
        mode="repair",
    )


@pytest.mark.parametrize(
    "code,status",
    [(0, "passed"), (1, "failed"), (2, "blocked"), (124, "blocked"), (137, "blocked")],
)
def test_gpu_failure_classification_and_owned_cleanup(case, monkeypatch, code, status):
    def execute(argv, cwd, log, timeout, env):
        log.write_text("actual GPU evidence")
        assert env["COMPAT_CONTAINER_NAME"].startswith("kvcached-compat-")
        profile = stage.load_profile(case.profile)
        stage.write_json(case.output / "runtime/contracts/checks.json", dict(
            profile=stage.identity(profile), exit_code=0, trusted=True,
            checks=[dict(test=name) for name in profile["gpu_tests"]],
        ))
        stage.write_json(case.output / "runtime/probe/matrix.json", dict(
            profile=stage.identity(profile), candidate_head="a" * 40, version="0.28.0",
            cases=[dict(layout=layout, runner=profile["runner"], exit_code=0)
                   for layout in profile["layouts"]], exit_code=0,
        ))
        return {"exit_code": code}

    monkeypatch.setattr(stage, "run_command", execute)
    stage.gpu(case)
    value = json.loads((case.output / "stage.json").read_text())
    assert value["status"] == status
    assert (case.output / "failure.json").exists() == (status == "failed")
    if status == "failed":
        report = json.loads((case.output / "failure.json").read_text())
        assert report["source_head"] == value["candidate_head"]
        assert report["summary"] == "actual GPU evidence"
    cleanup = stage.subprocess.run.call_args.args[0]
    assert cleanup[:3] == ["docker", "rm", "-f"]
    assert cleanup[3].startswith("kvcached-compat-")


def test_stale_engine_evidence_does_not_run_candidate(case, monkeypatch):
    case.engine_sha = "e" * 40
    execute = mock.Mock()
    monkeypatch.setattr(stage, "run_command", execute)
    with pytest.raises(ValueError, match="expected CPU stage"):
        stage.gpu(case)
    execute.assert_not_called()


@pytest.mark.parametrize("change", ["profile", "policy", "head"])
def test_gpu_rejects_cross_task_or_changed_policy_evidence(case, monkeypatch, change):
    path = case.prior / "stage.json"
    previous = json.loads(path.read_text())
    if change == "profile":
        previous["profile"]["name"] = "allocation"
    elif change == "policy":
        previous["profile"]["policy_digest"] = "0" * 64
    else:
        previous["candidate_head"] = "0" * 40
    stage.write_json(path, previous)
    execute = mock.Mock()
    monkeypatch.setattr(stage, "run_command", execute)
    with pytest.raises(ValueError, match="expected CPU stage"):
        stage.gpu(case)
    execute.assert_not_called()


def test_zero_exit_without_independent_contract_evidence_cannot_pass(case, monkeypatch):
    monkeypatch.setattr(stage, "run_command", lambda *args: {"exit_code": 0})
    with pytest.raises(FileNotFoundError):
        stage.gpu(case)
    assert not (case.output / "stage.json").exists()


def test_cpu_check_reports_are_outside_protected_inputs(case, monkeypatch):
    case.prior = None
    case.codex = "codex"
    case.engine = case.source
    commands = []

    def execute(argv, cwd, log, timeout, env):
        checks_root = Path(argv[argv.index("--checks") + 1])
        checks = json.loads((checks_root / "checks.json").read_text())
        for category in ("probes", "validation"):
            command = checks[category][0]["argv"]
            destination = Path(command[command.index("--output") + 1])
            assert not destination.is_relative_to(checks_root)
            commands.append(command)
        return {"exit_code": 0}

    monkeypatch.setattr(stage, "run_command", execute)
    monkeypatch.setattr(stage, "pack", lambda *args: dict(candidate_head="a" * 40, digest="b" * 64))
    stage.cpu(case)
    assert len(commands) == 2


@pytest.mark.parametrize("code,status", [(0, "candidate"), (1, "failed"), (2, "blocked")])
def test_validation_only_never_invokes_agent_or_changes_candidate(case, monkeypatch, code, status):
    case.mode = "validate"
    case.prior = None
    payload = dict(candidate_head=case.base, digest="b" * 64, files={})
    monkeypatch.setattr(stage, "pack", lambda *args: payload)
    checks = mock.Mock(return_value=code)
    monkeypatch.setattr(stage, "run_checks", checks)
    execute = mock.Mock(side_effect=AssertionError("Must not invoke Codex"))
    monkeypatch.setattr(stage, "run_command", execute)
    stage.cpu(case)
    assert json.loads((case.output / "stage.json").read_text())["status"] == status
    execute.assert_not_called()
    assert checks.call_args.kwargs["trusted"] is True
    assert (case.output / "candidate.json").exists() == (code == 0)


def test_validation_only_rejects_source_mutation(case, monkeypatch):
    case.mode = "validate"
    case.prior = None
    before = dict(candidate_head=case.base, digest="b" * 64, files={})
    after = dict(before, files={"kvcached/integration/vllm/patches.py": "changed"})
    monkeypatch.setattr(stage, "pack", mock.Mock(side_effect=[before, after]))
    monkeypatch.setattr(stage, "run_checks", lambda *a, **k: 0)
    with pytest.raises(ValueError, match="changed"):
        stage.cpu(case)
    assert not (case.output / "candidate.json").exists()


def test_unqualified_release_stops_before_spending_a_repair_attempt(case, monkeypatch):
    case.profile = "native-layout"
    case.tag = "v0.30.0"
    execute = mock.Mock()
    monkeypatch.setattr(stage, "run_command", execute)
    with pytest.raises(ValueError, match="no trusted contracts"):
        stage.cpu(case)
    execute.assert_not_called()


def test_profile_is_forwarded_to_every_candidate_stage():
    jobs = workflow("vllm-release-compat.yml")["jobs"]
    for name in ("first", "second", "verify"):
        assert jobs[name]["with"]["profile"] == "${{ needs.discover.outputs.profile }}"
    for name in ("engine-compat-attempt.yml", "engine-compat-ci.yml"):
        config = workflow(name)
        assert config["on"]["workflow_call"]["inputs"]["profile"]["required"] == "true"
    steps = jobs["publish"]["steps"]
    publish = next(step for step in steps if "PROFILE" in step.get("env", {}))
    assert '--profile "$PROFILE"' in publish["run"]


def test_validation_only_workflow_cannot_retry_or_publish():
    release = workflow("vllm-release-compat.yml")
    assert release["on"]["workflow_dispatch"]["inputs"]["mode"]["default"] == "validate"
    jobs = release["jobs"]
    assert "mode == 'repair'" in jobs["second"]["if"]
    assert "mode == 'repair'" in jobs["publish"]["if"]
    for name in ("first", "second"):
        assert jobs[name]["with"]["mode"] == "${{ needs.discover.outputs.mode }}"
    attempt = workflow("engine-compat-attempt.yml")
    assert attempt["on"]["workflow_call"]["secrets"]["CODEX_API_KEY"]["required"] == "false"
    install = next(step for step in attempt["jobs"]["repair"]["steps"]
                   if "npm install" in step.get("run", ""))
    assert install["if"] == "inputs.mode == 'repair'"


def test_validation_receipt_cannot_be_used_for_publication(case):
    receipt = json.loads((case.prior / "stage.json").read_text())
    receipt.update(status="passed", mode="validate")
    stage.write_json(case.prior / "stage.json", receipt)
    with pytest.raises(ValueError, match="cannot authorize publication"):
        stage.publish(case)


def test_publication_rejects_nonpassing_gpu_stage(case):
    with pytest.raises(ValueError, match="successful GPU evidence"):
        stage.publish(case)


def test_unchanged_candidate_does_not_create_a_pr(case, monkeypatch):
    stage.write_json(
        case.prior / "stage.json",
        dict(status="passed", digest="b" * 64, engine_sha=case.engine_sha, candidate_head="a" * 40,
             profile=stage.identity(stage.load_profile("vllm"))),
    )
    execute = mock.Mock()
    monkeypatch.setattr(stage.subprocess, "check_output", execute)
    stage.publish(case)
    execute.assert_not_called()
    assert json.loads((case.output / "stage.json").read_text())["status"] == "passed"


def test_expected_target_checks_match_current_repository_ci():
    expected: set[str] = set()
    for filename in ("cpu-tests.yml", "pre-commit.yml"):
        for job_id, job in workflow(filename)["jobs"].items():
            template = job.get("name", job_id)
            versions = job.get("strategy", {}).get("matrix", {}).get("python-version", [""])
            expected.update(template.replace("${{ matrix.python-version }}", v) for v in versions)
    assert stage.EXPECTED_TARGET_CHECKS == expected
