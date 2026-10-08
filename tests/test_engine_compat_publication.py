# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Publication gates with all Git and GitHub operations mocked."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    spec = importlib.util.spec_from_file_location(
        "publication_stage", ROOT / "tools/engine_compat_stage.py"
    )
    assert spec is not None and spec.loader is not None
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
finally:
    sys.path.pop(0)


@pytest.fixture
def case(tmp_path, monkeypatch):
    args = SimpleNamespace(
        source=tmp_path / "source",
        prior=tmp_path / "prior",
        output=tmp_path / "output",
        base="b" * 40,
        tag="v0.28.0",
        engine_sha="e" * 40,
        profile="vllm",
        base_branch="main",
        publish_repository="operator/fork",
        pr_repository="upstream/project",
        run_url="https://github.com/operator/fork/actions/runs/1",
    )
    for path in (args.source, args.prior, args.output):
        path.mkdir()
    head = "a" * 40
    payload = dict(candidate_head=head, digest="d" * 64, files={"allowed.py": {}})
    stage.write_json(args.prior / "candidate.json", payload)
    stage.write_json(
        args.prior / "stage.json",
        dict(
            status="passed",
            candidate_head=head,
            digest=payload["digest"],
            engine_sha=args.engine_sha,
            profile=stage.identity(stage.load_profile(args.profile)),
        ),
    )
    materialize = mock.Mock(return_value=head)
    monkeypatch.setattr(stage, "materialize", materialize)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setenv("GH_TOKEN", "publisher-only-test-token")
    pr = dict(
        number=7,
        state="open",
        merged=False,
        merged_at=None,
        mergeable=True,
        mergeable_state="clean",
        base=dict(repo=dict(full_name=args.pr_repository), ref="main", sha=args.base),
        head=dict(
            repo=dict(full_name=args.publish_repository), ref="automation/vllm-v0.28.0", sha=head
        ),
    )
    state = SimpleNamespace(
        args=args,
        head=head,
        payload=payload,
        pr=pr,
        pulls=[],
        calls=[],
        remote=None,
        base=args.base,
        now=0,
        materialize=materialize,
        reads=0,
        checks=[
            dict(name=name, head_sha=head, status="completed", conclusion="success")
            for name in sorted(stage.EXPECTED_TARGET_CHECKS)
        ],
        target_checks=None,
        statuses=[],
        status_sha=head,
        before=None,
    )

    def execute(argv, **kwargs):
        state.calls.append(argv)
        assert 0 < kwargs["timeout"] <= 60
        assert kwargs.get("env", os.environ)["GH_TOKEN"] == "publisher-only-test-token"
        if state.before:
            state.before(argv)
        if argv[0] == "git":
            if "ls-remote" in argv:
                sha = state.base if argv[-1] == "refs/heads/main" else state.remote
                return f"{sha}\t{argv[-1]}\n" if sha else ""
            assert "push" in argv
            assert f"--force-with-lease=refs/heads/automation/vllm-{args.tag}:" in argv
            state.remote = head
            return ""
        assert argv[:5] == ["gh", "api", "--hostname", "github.com", "--method"]
        method, endpoint = argv[5:7]
        if endpoint.endswith("/pulls"):
            if method == "POST":
                assert "head_repo=fork" in argv
                return json.dumps(state.pr)
            assert all(
                flag in argv
                for flag in (
                    "--paginate",
                    "--slurp",
                    "head=operator:automation/vllm-v0.28.0",
                    "state=all",
                )
            )
            return json.dumps(state.pulls)
        if endpoint.endswith("/pulls/7"):
            state.reads += 1
            return json.dumps(state.pr)
        assert f"/commits/{head}/" in endpoint
        assert "--paginate" in argv and "--slurp" in argv
        if endpoint.endswith("/check-runs"):
            checks = state.checks
            if (
                endpoint.startswith(f"repos/{args.pr_repository}/")
                and state.target_checks is not None
            ):
                checks = state.target_checks
            return json.dumps([dict(check_runs=checks)])
        assert endpoint.endswith("/status")
        return json.dumps([dict(sha=state.status_sha, statuses=state.statuses)])

    def sleep(seconds):
        state.now += seconds

    monkeypatch.setattr(stage.subprocess, "check_output", execute)
    monkeypatch.setattr(stage.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(stage.time, "sleep", sleep)
    monkeypatch.setattr(stage, "PUBLICATION_TIMEOUT", 20)
    return state


def published(case):
    return json.loads((case.args.output / "stage.json").read_text())


def writes(case):
    return [argv for argv in case.calls if "push" in argv or "POST" in argv]


def test_create_pr_verifies_exact_commit_and_preserves_publisher_auth(case):
    stage.publish(case.args)
    assert published(case)["status"] == "passed"
    assert published(case)["candidate_head"] == case.head
    assert published(case)["pr_url"] == "https://github.com/upstream/project/pull/7"
    assert len(writes(case)) == 2
    assert case.reads == 2
    assert case.materialize.call_args.args[2:4] == (case.args.base, case.args.tag)


def test_existing_pr_found_on_later_page_is_verified_without_writes(case):
    case.pulls = [[], [case.pr]]
    case.remote = case.head
    stage.publish(case.args)
    assert published(case)["status"] == "passed"
    assert not writes(case)


@pytest.mark.parametrize("partial", [False, True])
def test_successful_fork_checks_do_not_replace_missing_target_ci(case, partial):
    case.target_checks = case.checks[:1] if partial else []
    with pytest.raises(ValueError, match="timed out"):
        stage.publish(case.args)
    assert not (case.args.output / "stage.json").exists()


@pytest.mark.parametrize("base", [None, "f" * 40])
def test_target_base_must_match_before_any_write(case, base):
    case.base = base
    with pytest.raises(ValueError, match="Target base"):
        stage.publish(case.args)
    assert not writes(case)


def test_changed_remote_branch_is_not_overwritten(case):
    case.remote = "f" * 40
    with pytest.raises(ValueError, match="Existing branch changed"):
        stage.publish(case.args)
    assert not writes(case)


@pytest.mark.parametrize(
    "side,field,value",
    [
        ("base", "ref", "other"),
        ("base", "sha", "f" * 40),
        ("base", "repo", {"full_name": "wrong/project"}),
        ("head", "ref", "other"),
        ("head", "sha", "f" * 40),
        ("head", "repo", {"full_name": "wrong/fork"}),
        ("head", "repo", None),
    ],
)
@pytest.mark.parametrize("existing", [True, False])
def test_pr_identity_mismatch_is_never_accepted(case, side, field, value, existing):
    case.pr[side][field] = value
    if existing:
        case.pulls = [[case.pr]]
    with pytest.raises(ValueError, match="PR .* mismatch"):
        stage.publish(case.args)
    assert not (case.args.output / "stage.json").exists()
    if existing:
        assert not writes(case)


@pytest.mark.parametrize(
    "change", [dict(state="closed"), dict(merged=True), dict(merged_at="today")]
)
def test_closed_or_merged_pr_is_not_reused(case, change):
    case.pr.update(change)
    case.pulls = [[case.pr]]
    with pytest.raises(ValueError, match="Closed or merged"):
        stage.publish(case.args)
    assert not writes(case)


def test_ambiguous_prs_require_operator(case):
    case.pulls = [[case.pr], [copy.deepcopy(case.pr)]]
    with pytest.raises(ValueError, match="Multiple matching"):
        stage.publish(case.args)
    assert not writes(case)


@pytest.mark.parametrize("mergeable", [False, None])
def test_conflicts_or_unknown_mergeability_never_pass(case, mergeable):
    case.pr["mergeable"] = mergeable
    with pytest.raises(ValueError, match="conflicts|timed out"):
        stage.publish(case.args)
    assert not (case.args.output / "stage.json").exists()
    assert case.now <= 20


@pytest.mark.parametrize(
    "checks,statuses",
    [
        ([], []),
        ([dict(status="in_progress")], []),
        ([dict(status="completed", conclusion="failure")], []),
        ([dict(status="completed", conclusion="skipped")], []),
        ([], [dict(state="pending")]),
        ([], [dict(state="failure")]),
    ],
)
def test_absent_pending_or_failed_checks_never_pass(case, checks, statuses):
    case.checks = [dict(head_sha=case.head, **check) for check in checks]
    case.statuses = statuses
    with pytest.raises(ValueError, match="timed out|did not succeed"):
        stage.publish(case.args)
    assert not (case.args.output / "stage.json").exists()
    assert case.now <= 20


def test_pending_checks_and_mergeability_are_polled(case):
    case.checks[0]["status"] = "in_progress"
    case.pr["mergeable"] = None

    def advance(argv):
        if case.now:
            case.checks[0]["status"] = "completed"
            case.pr["mergeable"] = True

    case.before = advance
    stage.publish(case.args)
    assert published(case)["status"] == "passed"
    assert case.now == 10
    assert len(writes(case)) == 2


@pytest.mark.parametrize("kind", ["check", "status"])
def test_checks_from_another_commit_are_rejected(case, kind):
    if kind == "check":
        case.checks[0]["head_sha"] = "f" * 40
    else:
        case.status_sha = "f" * 40
    with pytest.raises(ValueError, match="SHA mismatch"):
        stage.publish(case.args)


@pytest.mark.parametrize("kind", ["base", "remote", "pr"])
def test_concurrent_identity_change_is_rejected(case, kind):
    def drift(argv):
        if case.reads and argv[0] == "gh" and argv[6].endswith("/status"):
            if kind == "base":
                case.base = "f" * 40
            elif kind == "remote":
                case.remote = "f" * 40
            else:
                case.pr["base"]["ref"] = "other"

    case.before = drift
    with pytest.raises(ValueError, match="Target base|SHA mismatch|PR base"):
        stage.publish(case.args)
    assert not (case.args.output / "stage.json").exists()


def test_api_timeout_is_not_retried_or_reported_as_pass(case):
    def timeout(argv):
        if argv[0] == "gh":
            raise subprocess.TimeoutExpired(argv, 60)

    case.before = timeout
    with pytest.raises(subprocess.TimeoutExpired):
        stage.publish(case.args)
    assert not writes(case)
    assert not (case.args.output / "stage.json").exists()


@pytest.mark.parametrize("change", [dict(mergeable=False), dict(number=8), dict(mergeable=None)])
def test_final_pr_read_must_still_be_mergeable_and_identical(case, change):
    def change_final_read(argv):
        if case.reads and argv[0] == "gh" and argv[6].endswith("/pulls/7"):
            case.pr.update(change)

    case.before = change_final_read
    with pytest.raises(ValueError, match="conflicts|number changed|timed out"):
        stage.publish(case.args)
    assert not (case.args.output / "stage.json").exists()


def test_successful_commit_status_does_not_replace_expected_target_checks(case):
    case.checks = []
    case.statuses = [dict(state="success")]
    with pytest.raises(ValueError, match="timed out"):
        stage.publish(case.args)
    assert not (case.args.output / "stage.json").exists()


def test_command_timeout_records_blocked_in_main(case, monkeypatch):
    def timeout(args):
        raise subprocess.TimeoutExpired("gh", 60)

    monkeypatch.setattr(stage, "publish", timeout)
    output = case.args.output / "timeout"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "engine_compat_stage.py",
            "publish",
            "--source",
            str(case.args.source),
            "--base",
            case.args.base,
            "--tag",
            case.args.tag,
            "--engine-sha",
            case.args.engine_sha,
            "--output",
            str(output),
        ],
    )
    assert stage.main() == 2
    assert json.loads((output / "stage.json").read_text())["status"] == "blocked"
