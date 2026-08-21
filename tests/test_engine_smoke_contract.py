# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "run_engine_smoke.sh"
ACTIVATION_CHECK = ROOT / "tools" / "check_engine_activation.py"


def run_preflight(tmp_path: Path, engine: str, port: str = "12346"):
    env = os.environ.copy()
    env.update(
        {
            "CHECK_ONLY": "1",
            "ENGINE": engine,
            "LOG_DIR": str(tmp_path / "logs"),
            "PORT": port,
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


def test_vllm_and_sglang_preflight(tmp_path):
    for engine in ("vllm", "sglang"):
        completed = run_preflight(tmp_path, engine)
        assert completed.returncode == 0
        assert f"engine={engine}" in completed.stdout


def test_unknown_engine_is_rejected(tmp_path):
    completed = run_preflight(tmp_path, "unknown")
    assert completed.returncode == 2
    assert "must be 'vllm' or 'sglang'" in completed.stdout


def test_invalid_port_is_rejected(tmp_path):
    completed = run_preflight(tmp_path, "vllm", port="70000")
    assert completed.returncode == 2
    assert "PORT must be an integer" in completed.stdout


def run_activation_check(tmp_path: Path, engine: str, log_text: str):
    log_path = tmp_path / f"{engine}.log"
    log_path.write_text(log_text, encoding="utf-8")
    return subprocess.run(
        [
            "python",
            str(ACTIVATION_CHECK),
            "--engine",
            engine,
            "--log",
            str(log_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_activation_check_requires_allocator_initialization_marker(tmp_path):
    completed = run_activation_check(
        tmp_path,
        "vllm",
        "kvcached autopatch discovered\nengine response was correct\n",
    )

    assert completed.returncode == 1
    assert "missing successful kvcached allocator initialization marker" in completed.stdout


def test_activation_check_accepts_exact_engine_marker(tmp_path):
    completed = run_activation_check(
        tmp_path,
        "sglang",
        "INFO KVCACHED_ENGINE_INTEGRATION_READY engine=sglang device=cuda:0\n",
    )

    assert completed.returncode == 0
    assert "KVCACHED_ACTIVATION_VERIFIED engine=sglang" in completed.stdout


def test_engine_server_uses_and_cleans_a_dedicated_process_group():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "os.setsid()" in source
    assert 'kill -TERM -- "-${SERVER_PID}"' in source
    assert 'kill -KILL -- "-${SERVER_PID}"' in source
    assert "introduced_gpu_pids" in source


def test_the_server_is_launched_against_the_installed_kvcached():
    """`python -m` from the repository root shadows the installed package.

    The source tree has no compiled extension, so both engines died at
    `ModuleNotFoundError: No module named 'kvcached.vmm_ops'` before this was
    set -- a failure that only shows up when the smoke test is actually run.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "export PYTHONSAFEPATH=1" in source
