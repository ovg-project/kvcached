# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Deterministic deadlines and resumable read-only agent stages."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    spec = importlib.util.spec_from_file_location("compat_agent", ROOT / "tools/engine_compat_agent.py")
    assert spec is not None and spec.loader is not None
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
finally:
    sys.path.pop(0)

SESSION = "11111111-2222-4333-8444-555555555555"


def event(kind="item.completed"):
    return json.dumps(dict(type=kind, item=dict(type="command_execution", status="completed")))


def test_recent_progress_is_not_killed_at_the_old_twenty_minute_boundary():
    budget = agent.ProgressBudget(0)
    for now in range(300, 1501, 300):
        budget.feed(event(), now)
    assert budget.reason(1501) is None
    assert budget.reason(1800) == "work_timeout"


def test_network_retries_do_not_count_as_progress_or_grant_unbounded_time():
    budget = agent.ProgressBudget(0)
    for now in range(100, 901, 100):
        budget.feed(json.dumps(dict(type="error", message="Reconnecting... stream disconnected")), now)
    assert budget.reason(899) is None
    assert budget.reason(900) == "idle_timeout"
    assert budget.report(900)["recovery_seconds"] == 300
    assert budget.report(900)["progress_events"] == 0


def test_recovery_interval_ends_on_actual_progress_and_is_cumulative():
    budget = agent.ProgressBudget(0)
    budget.feed(json.dumps(dict(type="error", message="Reconnecting...")), 100)
    budget.feed(event(), 125)
    budget.feed(json.dumps(dict(type="error", message="Reconnecting...")), 200)
    budget.feed(event(), 351)
    assert budget.report(400)["recovery_seconds"] == 176
    assert budget.report(400)["work_seconds"] == 224
    assert budget.reason(1000) == "idle_timeout"


def test_activity_does_not_remove_wall_limit_and_noise_does_not_reset_idle():
    budget = agent.ProgressBudget(0, work=10000)
    for now in range(300, 2701, 300):
        budget.feed(event(), now)
    assert budget.reason(2700) == "wall_timeout"
    idle = agent.ProgressBudget(0)
    idle.feed("unrelated verbose stderr", 590)
    idle.feed(json.dumps(dict(type="error", message="invalid schema")), 591)
    assert idle.reason(600) == "idle_timeout"


def test_transport_error_items_and_malformed_events_are_not_work():
    budget = agent.ProgressBudget(0)
    for now in (100, 300, 599):
        budget.feed(json.dumps(dict(type="item.completed", item=dict(
            type="error", message="Falling back from WebSockets to HTTPS"))), now)
        budget.feed(json.dumps(dict(type={})), now)
    assert budget.events == 0
    assert budget.reason(600) == "idle_timeout"


def test_split_log_lines_and_session_identity(tmp_path):
    log = tmp_path / "agent.log"
    budget = agent.ProgressBudget(0)
    value = json.dumps(dict(type="thread.started", thread_id=SESSION))
    log.write_text(value[:20])
    budget.poll(log, 1)
    assert budget.session is None
    with log.open("a") as stream:
        stream.write(value[20:] + "\n" + event() + "\n")
    budget.poll(log, 2)
    budget.poll(log, 3)
    assert budget.session == SESSION
    assert budget.report(3)["progress_events"] == 1
    with pytest.raises(ValueError, match="session"):
        budget.feed(json.dumps(dict(type="thread.started", thread_id="--last")), 4)


def test_controller_watchdog_stops_a_silent_real_process(tmp_path):
    from repair_engine_compat import run_command

    budget = agent.ProgressBudget(0, idle=0.15)
    result = run_command([sys.executable, "-c", "import time; time.sleep(20)"], tmp_path,
                         tmp_path / "log", 5, dict(os.environ), watch=budget)
    assert result["exit_code"] == 124
    assert result["timeout_reason"] == "idle_timeout"
    assert result["seconds"] < 10


def test_invalid_event_terminates_process_without_a_resumable_timeout(tmp_path):
    from repair_engine_compat import run_command

    code = ('import time; print(\'{"type":"thread.started","thread_id":"--last"}\', '
            'flush=True); time.sleep(20)')
    result = run_command([sys.executable, "-c", code], tmp_path, tmp_path / "log", 5,
                         dict(os.environ), watch=agent.ProgressBudget(0))
    assert result["exit_code"] == 2
    assert not result["timeout"]
    assert result["timeout_reason"] == "protocol_error"
    assert result["seconds"] < 10


@pytest.fixture
def stage(tmp_path, monkeypatch):
    inputs = dict(prompt="Read supplied sources", schema={"type": "object"},
                  sources={"candidate": {"head": "a" * 40}}, artifacts={"inputs": "b" * 64})
    calls = []

    def run(argv, cwd, log, timeout, env, prompt, watch):
        calls.append(argv)
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        assert "--output-schema" in argv
        assert not {"GH_TOKEN", "GITHUB_TOKEN", "SSH_AUTH_SOCK"} & env.keys()
        reply = Path(argv[argv.index("--output-last-message") + 1])
        reply.write_text('{"ok": true}')
        watch.feed(json.dumps(dict(type="thread.started", thread_id=SESSION)), 0)
        return dict(exit_code=0, timeout=False, timeout_reason=None, seconds=1,
                    activity=watch.report(1))

    monkeypatch.setattr(agent, "run_command", run)
    monkeypatch.setenv("GH_TOKEN", "private")
    return tmp_path, inputs, calls


def invoke(stage, resume=False):
    root, inputs, _ = stage
    return agent.execute_stage("codex", root, root, "contracts", inputs, resume=resume)


def test_completed_stage_is_reused_only_with_matching_identity(stage):
    assert invoke(stage) == {"ok": True}
    assert invoke(stage, resume=True) == {"ok": True}
    assert len(stage[2]) == 1
    stage[1]["artifacts"]["inputs"] = "c" * 64
    with pytest.raises(ValueError, match="identity"):
        invoke(stage, resume=True)
    assert len(stage[2]) == 1


def test_existing_stage_requires_explicit_resume_and_checks_reply_digest(stage):
    invoke(stage)
    with pytest.raises(ValueError, match="resume"):
        invoke(stage)
    (stage[0] / "contracts.json").write_text('{"ok": false}')
    with pytest.raises(ValueError, match="changed"):
        invoke(stage, resume=True)


def test_interrupted_stage_resumes_exact_session_and_preserves_attempts(stage, monkeypatch):
    original = agent.run_command

    def interrupted(*args, **kwargs):
        result = original(*args, **kwargs)
        result.update(exit_code=124, timeout=True, timeout_reason="work_timeout")
        return result

    monkeypatch.setattr(agent, "run_command", interrupted)
    with pytest.raises(ValueError, match="interrupted"):
        invoke(stage)
    assert not (stage[0] / "contracts.json").exists()
    monkeypatch.setattr(agent, "run_command", original)
    assert invoke(stage, resume=True) == {"ok": True}
    command = stage[2][-1]
    assert command[command.index("resume") + 1] == SESSION
    assert "--last" not in command
    assert len(list((stage[0] / "contracts-attempts").glob("*.json"))) == 2


def test_failed_nontransport_stage_cannot_be_retried_as_network_error(stage, monkeypatch):
    original = agent.run_command

    def failed(*args, **kwargs):
        result = original(*args, **kwargs)
        result.update(exit_code=1)
        return result

    monkeypatch.setattr(agent, "run_command", failed)
    with pytest.raises(ValueError, match="failed"):
        invoke(stage)
    with pytest.raises(ValueError, match="not resumable"):
        invoke(stage, resume=True)


def test_checkpoint_implementation_changes_invalidate_resume(stage, monkeypatch):
    invoke(stage)
    monkeypatch.setattr(agent, "implementation_digest", lambda: "new controller")
    with pytest.raises(ValueError, match="identity"):
        invoke(stage, resume=True)


def test_controller_change_during_execution_invalidates_output(stage, monkeypatch):
    calls = iter(["before", "after"])
    monkeypatch.setattr(agent, "implementation_digest", lambda: next(calls))
    with pytest.raises(ValueError, match="controller changed"):
        invoke(stage)
    assert not (stage[0] / "contracts.json").exists()
    assert agent.read(stage[0] / "contracts-state.json")["status"] == "failed"


def test_existing_lock_is_not_stolen(stage):
    lock = stage[0] / "contracts.lock"
    lock.write_text("another invocation")
    with pytest.raises(FileExistsError):
        invoke(stage, resume=True)
    assert lock.read_text() == "another invocation"
    assert not stage[2]
