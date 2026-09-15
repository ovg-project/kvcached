# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "model-compatibility.yml"


def test_workflow_is_manual_only_and_uses_gpu_runner():
    payload = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert set(payload["on"]) == {"workflow_dispatch"}
    job = payload["jobs"]["matrix"]
    assert job["runs-on"] == ["self-hosted", "linux", "x64", "gpu", "kvcached"]
    assert job["timeout-minutes"] == "720"


def test_workflow_exposes_filters_and_always_uploads_results():
    payload = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    inputs = payload["on"]["workflow_dispatch"]["inputs"]

    assert set(inputs) == {
        "engine",
        "model",
        "model_override",
        "layout",
        "devices",
        "fail_on_non_pass",
    }
    steps = payload["jobs"]["matrix"]["steps"]
    run_step = next(step for step in steps if "run" in step)
    assert run_step["env"]["GPU_CI_PROFILE"] == "compat"
    assert run_step["run"] == "bash tools/run_gpu_ci.sh"
    upload = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v4")
    assert upload["if"] == "always()"
