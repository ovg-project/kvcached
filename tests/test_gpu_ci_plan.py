# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for tools/resolve_gpu_ci_plan.sh.

The script decides which GPU profile a trigger asks for, which devices it
gets, and whether a scheduled slot is spent at all. That decision used to be
a chain of GitHub expressions inside the workflow, where the only way to
learn it was wrong was to push a workflow and label a pull request, and where
every failure was silent: the wrong profile simply ran.

These tests drive the script directly with a synthetic environment. The
scheduled path needs `gh` and a git history, so it is covered only up to the
point where it would query the workflow run history.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "resolve_gpu_ci_plan.sh"


def resolve(
    tmp_path: Path,
    event: str,
    labels: str = "",
    dispatch_profile: str = "",
    dispatch_devices: str = "",
    single_devices: str = "0",
    dual_devices: str = "0,1",
) -> Dict[str, str]:
    """Run the script and return what it wrote to GITHUB_OUTPUT."""
    output = tmp_path / "github_output"
    output.touch()
    env = os.environ.copy()
    env.update(
        {
            "EVENT": event,
            "LABELS": labels,
            "DISPATCH_PROFILE": dispatch_profile,
            "DISPATCH_DEVICES": dispatch_devices,
            "SINGLE_DEVICES": single_devices,
            "DUAL_DEVICES": dual_devices,
            "GITHUB_OUTPUT": str(output),
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    parsed: Dict[str, str] = {}
    for line in output.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            parsed[key] = value
    parsed["_returncode"] = str(completed.returncode)
    parsed["_stdout"] = completed.stdout
    return parsed


@pytest.mark.parametrize(
    ("labels", "profile"),
    [
        ("gpu-ci", "core"),
        ("gpu-ci-vllm", "vllm"),
        ("gpu-ci-sglang", "sglang"),
        ("gpu-ci-engines", "engines"),
        ("gpu-ci-nixl", "nixl"),
        ("gpu-ci-all", "all"),
    ],
)
def test_each_label_selects_its_profile(tmp_path, labels, profile):
    plan = resolve(tmp_path, "pull_request", labels=labels)
    assert plan["_returncode"] == "0"
    assert plan["run"] == "true"
    assert plan["profile"] == profile


@pytest.mark.parametrize(
    ("labels", "profile"),
    [
        ("gpu-ci gpu-ci-vllm", "vllm"),
        ("gpu-ci gpu-ci-all", "all"),
        ("gpu-ci gpu-ci-vllm gpu-ci-nixl", "nixl"),
    ],
)
def test_specific_label_beats_the_plain_one(tmp_path, labels, profile):
    """Carrying both must not silently downgrade the run to core."""
    assert resolve(tmp_path, "pull_request", labels=labels)["profile"] == profile


def test_unrelated_labels_do_not_run(tmp_path):
    plan = resolve(tmp_path, "pull_request", labels="needs-review gpu")
    assert plan["run"] == "false"
    assert "profile" not in plan


def test_no_labels_do_not_run(tmp_path):
    assert resolve(tmp_path, "pull_request")["run"] == "false"


def test_a_labelled_pull_request_runs_whoever_it_came_from(tmp_path):
    """The label is the gate, including for a fork.

    Reviewing a contributor's change before merging it is the main reason
    the GPU run exists, and contributors here work from forks. GitHub
    withholds secrets and issues a read-only token for a fork's
    pull_request, so applying the label accepts code execution on the
    runner and nothing wider.
    """
    plan = resolve(tmp_path, "pull_request", labels="gpu-ci-all")
    assert plan["run"] == "true"
    assert plan["profile"] == "all"


def test_dispatch_uses_its_input(tmp_path):
    plan = resolve(tmp_path, "workflow_dispatch", dispatch_profile="sglang")
    assert plan["profile"] == "sglang"


def test_dispatch_without_an_input_falls_back_to_core(tmp_path):
    assert resolve(tmp_path, "workflow_dispatch")["profile"] == "core"


def test_dispatch_devices_override_the_repository_variables(tmp_path):
    plan = resolve(
        tmp_path,
        "workflow_dispatch",
        dispatch_profile="nixl",
        dispatch_devices="4,5",
    )
    assert plan["devices"] == "4,5"


@pytest.mark.parametrize(
    ("labels", "devices"),
    [
        ("gpu-ci", "0"),
        ("gpu-ci-vllm", "0"),
        ("gpu-ci-engines", "0"),
        ("gpu-ci-nixl", "0,1"),
        ("gpu-ci-all", "0,1"),
    ],
)
def test_two_gpu_profiles_take_the_dual_device_variable(
    tmp_path, labels, devices
):
    assert resolve(tmp_path, "pull_request", labels=labels)["devices"] == devices


@pytest.mark.parametrize("labels", ["gpu-ci-nixl", "gpu-ci-all"])
def test_missing_dual_device_variable_fails_before_the_gpu_is_taken(
    tmp_path, labels
):
    plan = resolve(tmp_path, "pull_request", labels=labels, dual_devices="")
    assert plan["_returncode"] == "1"
    assert "KVCACHED_GPU_DUAL_DEVICES" in plan["_stdout"]
    assert plan.get("run") != "true"


def test_unsupported_event_is_rejected(tmp_path):
    plan = resolve(tmp_path, "issue_comment", labels="gpu-ci")
    assert plan["_returncode"] == "2"
    assert "unsupported event" in plan["_stdout"]


def _workflow() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(
        (ROOT / ".github" / "workflows" / "gpu-tests.yml").read_text("utf-8")
    )


def test_the_plan_job_checks_out_on_every_event():
    """The plan step runs a script from this repository.

    A first attempt gated the checkout on the scheduled event, because the
    step used to be an inline shell block that needed no working tree. A
    pull request then reached `bash tools/resolve_gpu_ci_plan.sh` with
    nothing checked out and the job died with exit 127.
    """
    steps = _workflow()["jobs"]["plan"]["steps"]
    checkouts = [s for s in steps if "actions/checkout" in s.get("uses", "")]
    assert checkouts, "the plan job must check out this repository"
    for step in checkouts:
        assert "if" not in step, (
            "the plan job's checkout must not be conditional: the plan step "
            "runs a script from the working tree on every event"
        )
    assert steps.index(checkouts[0]) < next(
        i for i, s in enumerate(steps) if s.get("id") == "plan"
    )


def test_the_plan_script_the_workflow_names_exists():
    steps = _workflow()["jobs"]["plan"]["steps"]
    run = next(s["run"] for s in steps if s.get("id") == "plan")
    named = run.split()[-1]
    assert (ROOT / named).is_file(), f"{named} is not in the repository"
