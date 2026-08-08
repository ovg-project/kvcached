# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "run_engine_smoke.sh"


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
