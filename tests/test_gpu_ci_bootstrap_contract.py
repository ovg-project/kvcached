# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "bootstrap_gpu_ci_envs.sh"


def test_bootstrap_uses_three_isolated_environments():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "create_env core" in source
    assert "create_env vllm" in source
    assert "create_env sglang" in source
    assert "KVCACHED_GPU_PYTHON=" in source
    assert "KVCACHED_VLLM_PYTHON=" in source
    assert "KVCACHED_SGLANG_PYTHON=" in source
    assert "pip check" in source


def test_bootstrap_has_hardware_and_toolchain_preflight():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "command -v nvidia-smi" in source
    assert "command -v nvcc" in source
    assert "GCC 9 or newer is required" in source
    assert 'CHECK_ONLY="${CHECK_ONLY:-0}"' in source
