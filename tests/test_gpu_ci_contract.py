# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "run_gpu_ci.sh"


def run_preflight(
    tmp_path: Path,
    profile: str,
    repeat: str = "1",
    skip_core: str = "0",
):
    env = os.environ.copy()
    env.update(
        {
            "CHECK_ONLY": "1",
            "GPU_CI_ARTIFACT_DIR": str(tmp_path / "artifacts"),
            "GPU_CI_PROFILE": profile,
            "GPU_CI_REPEAT": repeat,
            "GPU_CI_SKIP_CORE": skip_core,
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_supported_profiles_pass_cpu_only_preflight(tmp_path):
    for profile in ("core", "vllm", "sglang", "engines", "nixl"):
        completed = run_preflight(tmp_path, profile)
        assert completed.returncode == 0
        assert f"profile={profile}" in completed.stdout


def test_unknown_profile_is_rejected(tmp_path):
    completed = run_preflight(tmp_path, "unknown")
    assert completed.returncode == 2
    assert "Unknown GPU_CI_PROFILE" in completed.stdout


def test_repeat_count_is_bounded(tmp_path):
    for repeat in ("0", "11", "not-a-number"):
        completed = run_preflight(tmp_path, "core", repeat)
        assert completed.returncode == 2
        assert "integer from 1 to 10" in completed.stdout


def test_skip_core_is_boolean(tmp_path):
    completed = run_preflight(tmp_path, "nixl", skip_core="yes")
    assert completed.returncode == 2
    assert "GPU_CI_SKIP_CORE must be 0 or 1" in completed.stdout
