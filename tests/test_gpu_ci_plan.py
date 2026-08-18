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
from typing import Dict, Optional

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "resolve_gpu_ci_plan.sh"


def resolve(
    tmp_path: Path,
    event: str,
    labels: str = "",
    same_repo: str = "true",
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
            "SAME_REPO": same_repo,
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


def test_fork_pull_request_never_reaches_the_runner(tmp_path):
    plan = resolve(
        tmp_path, "pull_request", labels="gpu-ci-all", same_repo="false"
    )
    assert plan["run"] == "false"
    assert "fork pull request" in plan["_stdout"]


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
