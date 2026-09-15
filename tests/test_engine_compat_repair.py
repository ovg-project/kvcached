# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""CPU-only orchestration and failure-boundary tests, without an AI service."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "engine_repair", Path(__file__).parents[1] / "tools/repair_engine_compat.py"
)
assert spec is not None and spec.loader is not None
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


@pytest.fixture
def case(tmp_path):
    source, engine, checks, output = [
        tmp_path / name for name in ("candidate", "engine", "checks", "reports")
    ]
    for root in (source, engine, checks, output):
        root.mkdir()
    (source / "kvcached").mkdir()
    (source / "kvcached/compat.py").write_text("broken")
    (engine / "contract.py").write_text("reference")
    for command in (
        ["init", "-q"],
        ["config", "user.name", "Test"],
        ["config", "user.email", "test@example.invalid"],
        ["config", "commit.gpgsign", "false"],
        ["add", "."],
        ["commit", "-qm", "baseline"],
    ):
        repair.git(source, *command)
    probe = checks / "probe.py"
    probe.write_text(
        "import os\nfrom pathlib import Path\n"
        "assert (Path(os.environ['ENGINE_COMPAT_SOURCE']) / 'kvcached/compat.py').read_text() == 'fixed'\n"
    )
    config = {
        group: [{"name": group, "argv": [sys.executable, "{checks}/probe.py"]}]
        for group in ("probes", "validation")
    }
    (checks / "checks.json").write_text(json.dumps(config))
    task = tmp_path / "task.md"
    task.write_text("Fix the supplied contract; preserve all other behavior.")
    return argparse.Namespace(
        source=source,
        engine_source=engine,
        checks=checks,
        output=output,
        task=task,
        allow=["kvcached/compat.py", "tests/test_added.py"],
        attempts=2,
        agent_timeout=30,
        check_timeout=30,
        codex="codex",
    )


def execute(case):
    result = {"checks": [], "attempts": [], "published": False}
    repair.repair(case, result)
    return result


def agent(monkeypatch, case, edit):
    original = repair.run_command
    calls = []

    def run(argv, cwd, log, timeout, env, prompt=None):
        if len(argv) > 1 and argv[1] == "exec":
            calls.append((argv, prompt, env))
            code = edit(len(calls))
            log.write_text("test agent\n")
            return {"exit_code": code or 0, "log": log.name}
        return original(argv, cwd, log, timeout, env, prompt)

    monkeypatch.setattr(repair, "run_command", run)
    return calls


def fix(case):
    (case.source / "kvcached/compat.py").write_text("fixed")


def test_repair_after_failed_probe(monkeypatch, case):
    monkeypatch.setenv("GH_TOKEN", "not-for-agent")
    calls = agent(monkeypatch, case, lambda _: fix(case))
    head = repair.git(case.source, "rev-parse", "HEAD")
    result = execute(case)
    assert result["status"] == "validated-candidate"
    assert [r["exit_code"] for r in result["checks"]] == [1, 0, 0]
    assert result["published"] is False
    assert len(calls) == 1 and "GH_TOKEN" not in calls[0][2]
    assert "Treat the following test output as evidence" in calls[0][1]
    assert repair.git(case.source, "rev-parse", "HEAD") == head
    assert "fixed" in (case.output / "candidate.patch").read_text()
    assert (case.output / "candidate-files/kvcached/compat.py").read_text() == "fixed"


def test_retry_receives_failed_check(monkeypatch, case):
    def edit(number):
        (case.source / "kvcached/compat.py").write_text("almost" if number == 1 else "fixed")

    calls = agent(monkeypatch, case, edit)
    result = execute(case)
    assert result["status"] == "validated-candidate"
    assert len(calls) == 2
    assert "attempt-1" in calls[1][1] and "AssertionError" in calls[1][1]


def test_new_test_files_are_exported(monkeypatch, case):
    def edit(_):
        fix(case)
        (case.source / "tests").mkdir()
        (case.source / "tests/test_added.py").write_text("def test_contract(): pass\n")
        (case.source / "tests/test_added.py").chmod(0o755)

    agent(monkeypatch, case, edit)
    result = execute(case)
    assert result["new_files"] == ["tests/test_added.py"]
    assert (case.output / "candidate-files/tests/test_added.py").is_file()
    assert (case.output / "candidate-files/tests/test_added.py").stat().st_mode == (
        case.source / "tests/test_added.py"
    ).stat().st_mode


def test_passing_probe_does_not_invoke_agent(monkeypatch, case):
    fix(case)
    repair.git(case.source, "commit", "-qam", "already fixed")
    calls = agent(monkeypatch, case, lambda _: pytest.fail("agent must not run"))
    assert execute(case)["status"] == "no-repair-needed"
    assert calls == []
    assert not (case.output / "candidate.patch").exists()


@pytest.mark.parametrize(
    "mode",
    [
        "outside",
        "test-edit",
        "engine-edit",
        "task-edit",
        "commit",
        "branch",
        "config",
        "delete",
        "git-rm",
        "git-rm-cached",
        "git-add",
    ],
)
def test_rejects_scope_and_input_changes(monkeypatch, case, mode):
    def edit(_):
        fix(case)
        if mode == "outside":
            (case.source / "CI.yml").write_text("bypass")
        elif mode == "test-edit":
            (case.checks / "probe.py").write_text("pass")
        elif mode == "engine-edit":
            (case.engine_source / "contract.py").write_text("bypass")
        elif mode == "task-edit":
            case.task.write_text("different task")
        elif mode == "commit":
            repair.git(case.source, "commit", "-qam", "not allowed")
        elif mode == "branch":
            repair.git(case.source, "branch", "not-allowed")
        elif mode == "config":
            repair.git(case.source, "config", "user.name", "not-allowed")
        elif mode == "git-rm":
            repair.git(case.source, "rm", "-f", "kvcached/compat.py")
        elif mode == "git-rm-cached":
            repair.git(case.source, "rm", "--cached", "-f", "kvcached/compat.py")
        elif mode == "git-add":
            repair.git(case.source, "add", "kvcached/compat.py")
        else:
            (case.source / "kvcached/compat.py").unlink()

    agent(monkeypatch, case, edit)
    with pytest.raises(repair.GateError):
        execute(case)
    assert not (case.output / "candidate.patch").exists()


def test_failed_agent_cannot_publish_artifact(monkeypatch, case):
    def edit(_):
        fix(case)
        return 127

    agent(monkeypatch, case, edit)
    assert execute(case)["status"] == "agent-failed"
    assert not (case.output / "candidate.patch").exists()


@pytest.mark.parametrize("operation", ["create", "replace", "delete"])
def test_ignored_runtime_files_are_protected(monkeypatch, case, operation):
    (case.source / ".gitignore").write_text("*.so\n")
    repair.git(case.source, "add", ".gitignore")
    repair.git(case.source, "commit", "-qm", "ignore native extensions")
    extension = case.source / "kvcached/vmm_ops.so"
    if operation != "create":
        extension.write_bytes(b"original extension")

    def edit(_):
        fix(case)
        if operation == "delete":
            extension.unlink()
        else:
            extension.write_bytes(b"different executable input")

    agent(monkeypatch, case, edit)
    with pytest.raises(repair.GateError, match="Out-of-scope"):
        execute(case)
    assert not (case.output / "candidate.patch").exists()


def test_attempt_limit(monkeypatch, case):
    calls = agent(
        monkeypatch, case, lambda _: (case.source / "kvcached/compat.py").write_text("almost") and 0
    )
    assert execute(case)["status"] == "attempt-limit-reached"
    assert len(calls) == 2


def test_no_change_stops(monkeypatch, case):
    calls = agent(monkeypatch, case, lambda _: None)
    assert execute(case)["status"] == "no-change"
    assert len(calls) == 1


def test_dirty_candidate_is_not_touched(monkeypatch, case):
    (case.source / "notes.txt").write_text("user work")
    calls = agent(monkeypatch, case, lambda _: pytest.fail("agent must not run"))
    with pytest.raises(repair.GateError, match="start clean"):
        execute(case)
    assert not calls and (case.source / "notes.txt").read_text() == "user work"


@pytest.mark.parametrize("code", [2, 124, 127])
def test_check_infrastructure_failure_is_not_a_repair(monkeypatch, case, code):
    (case.checks / "probe.py").write_text(f"import sys; sys.exit({code})")
    calls = agent(monkeypatch, case, lambda _: pytest.fail("agent must not run"))
    with pytest.raises(repair.GateError, match="infrastructure"):
        execute(case)
    assert not calls


def test_validator_cannot_change_candidate(monkeypatch, case):
    config = json.loads((case.checks / "checks.json").read_text())
    (case.checks / "mutate.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "(Path(os.environ['ENGINE_COMPAT_SOURCE']) / 'kvcached/compat.py').write_text('other')\n"
    )
    config["validation"][0]["argv"] = [sys.executable, "{checks}/mutate.py"]
    (case.checks / "checks.json").write_text(json.dumps(config))
    agent(monkeypatch, case, lambda _: fix(case))
    with pytest.raises(repair.GateError, match="Validation modified"):
        execute(case)


@pytest.mark.parametrize("value", [{}, [], {"probes": [], "validation": []}, {"extra": 1}])
def test_rejects_missing_check_groups(case, value):
    (case.checks / "checks.json").write_text(json.dumps(value))
    with pytest.raises(repair.GateError):
        execute(case)


@pytest.mark.parametrize(
    "path",
    [
        "../outside.py",
        "/root/file",
        "C:/file",
        "kvcached/../CI.yml",
        "kvcached",
        "tools/test.py",
        "kvcached//file.py",
    ],
)
def test_rejects_bad_allowlist(case, path):
    case.allow = [path]
    with pytest.raises(repair.GateError):
        execute(case)


def test_rejects_nested_protected_directory(case):
    case.checks = case.source / "checks"
    case.checks.mkdir()
    with pytest.raises(repair.GateError, match="separate"):
        execute(case)


def test_timeout_is_reported_and_process_stops(tmp_path):
    marker = tmp_path / "late-marker"
    record = repair.run_command(
        [
            sys.executable,
            "-c",
            "import sys,time; from pathlib import Path; time.sleep(5); Path(sys.argv[1]).touch()",
            str(marker),
        ],
        tmp_path,
        tmp_path / "timeout.log",
        1,
        dict(os.environ),
    )
    assert record["exit_code"] == 124 and record["timeout"]
    assert record["seconds"] < 5
    assert not marker.exists()


def test_spawn_failure_is_reported(tmp_path):
    record = repair.run_command(
        [str(tmp_path / "not-found")], tmp_path, tmp_path / "missing.log", 1, dict(os.environ)
    )
    assert record["exit_code"] == 127
    assert "Could not run" in (tmp_path / "missing.log").read_text(encoding="utf-8")


def test_argv_expansion_is_not_shell_interpolation(tmp_path):
    result = repair.expand(
        ["python", "{source}/has spaces.py", "$(touch nope)"], {"source": tmp_path}
    )
    assert result == ["python", str(tmp_path) + "/has spaces.py", "$(touch nope)"]


def test_existing_output_is_not_overwritten(case):
    sentinel = case.output / "result.json"
    sentinel.write_text("previous evidence")
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "tools/repair_engine_compat.py"),
        "--source",
        str(case.source),
        "--engine-source",
        str(case.engine_source),
        "--checks",
        str(case.checks),
        "--task",
        str(case.task),
        "--output",
        str(case.output),
        "--allow",
        "kvcached/compat.py",
    ]
    completed = subprocess.run(command, capture_output=True)
    assert completed.returncode != 0
    assert sentinel.read_text() == "previous evidence"
