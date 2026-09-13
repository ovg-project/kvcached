# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "bootstrap_gpu_ci_envs.sh"


def test_bootstrap_uses_three_isolated_environments():
    source = SCRIPT.read_text(encoding="utf-8")

    for name in ("core", "vllm", "sglang"):
        assert f"create_env {name} " in source
    # The emitted names must be the ones run_gpu_ci.sh reads, because a host
    # pastes them straight into the runner's .env.
    assert "PYTHON=" in source
    assert "VLLM_PYTHON=" in source
    assert "SGLANG_PYTHON=" in source
    assert "pip check" in source


def test_bootstrap_can_provision_a_subset(tmp_path):
    """A host that only runs the core profile should not have to download the
    two engine environments, which pull their own PyTorch each.

    The preflight demands a GPU toolchain, which a hosted runner has none of,
    so stand in for the three commands it looks for. Skipping instead would
    leave this untested exactly where it runs.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name, body in (
        ("nvidia-smi", "#!/bin/sh\nexit 0\n"),
        ("nvcc", "#!/bin/sh\nexit 0\n"),
        ("c++", '#!/bin/sh\n[ "$1" = -dumpversion ] && echo 13 || exit 0\n'),
    ):
        path = fake_bin / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "CHECK_ONLY": "1",
        "GPU_CI_ENVS": "core",
        "BASE_PYTHON": sys.executable,
        "CXX": str(fake_bin / "c++"),
        "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
    })
    completed = subprocess.run(
        ["bash", str(SCRIPT)], cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout
    assert "envs=core" in completed.stdout


def test_bootstrap_rejects_an_unknown_environment(tmp_path):
    env = os.environ.copy()
    env.update({
        "CHECK_ONLY": "1",
        "GPU_CI_ENVS": "core tensorrt",
        "BASE_PYTHON": sys.executable,
    })
    completed = subprocess.run(
        ["bash", str(SCRIPT)], cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 2
    assert "Unknown environment 'tensorrt'" in completed.stdout


def test_bootstrap_has_hardware_and_toolchain_preflight():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "command -v nvidia-smi" in source
    assert "command -v nvcc" in source
    assert "GCC 9 or newer is required" in source
    assert 'CHECK_ONLY="${CHECK_ONLY:-0}"' in source
