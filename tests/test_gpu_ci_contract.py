# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "run_gpu_ci.sh"


def run_preflight(
    tmp_path: Path,
    profile: str,
    repeat: str = "1",
    skip_core: str = "0",
    visible_devices: Optional[str] = None,
    cuda_visible_devices: Optional[str] = None,
    vllm_python: Optional[str] = None,
    sglang_python: Optional[str] = None,
    cxx: Optional[str] = None,
):
    env = os.environ.copy()
    env.pop("KVCACHED_GPU_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(
        {
            "CHECK_ONLY": "1",
            "GPU_CI_ARTIFACT_DIR": str(tmp_path / "artifacts"),
            "GPU_CI_PROFILE": profile,
            "GPU_CI_REPEAT": repeat,
            "GPU_CI_SKIP_CORE": skip_core,
        }
    )
    if visible_devices is None:
        visible_devices = "0,1" if profile in ("nixl", "all") else "0"
    env["KVCACHED_GPU_VISIBLE_DEVICES"] = visible_devices
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if vllm_python is not None:
        env["VLLM_PYTHON"] = vllm_python
    if sglang_python is not None:
        env["SGLANG_PYTHON"] = sglang_python
    if cxx is not None:
        env["CXX"] = cxx
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_supported_profiles_pass_cpu_only_preflight(tmp_path):
    for profile in ("core", "vllm", "sglang", "engines", "nixl", "all"):
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


def test_profiles_select_required_gpus_from_runner_pool(tmp_path):
    completed = run_preflight(tmp_path, "core", visible_devices="0,1")
    assert completed.returncode == 0
    assert "devices=0" in completed.stdout

    completed = run_preflight(tmp_path, "all", visible_devices="0,1,2")
    assert completed.returncode == 0
    assert "devices=0,1" in completed.stdout

    for profile in ("nixl", "all"):
        completed = run_preflight(tmp_path, profile, visible_devices="0")
        assert completed.returncode == 2
        assert "requires at least 2 selected GPU" in completed.stdout


def test_device_selection_rejects_invalid_or_duplicate_ids(tmp_path):
    for devices in ("", "0, 1", "0,,1", "0,0"):
        completed = run_preflight(tmp_path, "nixl", visible_devices=devices)
        assert completed.returncode == 2


def test_cuda_visible_devices_is_accepted_from_external_scheduler(tmp_path):
    completed = run_preflight(
        tmp_path,
        "core",
        visible_devices="",
        cuda_visible_devices="4",
    )
    assert completed.returncode == 0
    assert "devices=4" in completed.stdout


def test_compiler_runtime_is_added_to_the_environment(tmp_path):
    runtime_dir = tmp_path / "compiler" / "lib"
    runtime_dir.mkdir(parents=True)
    runtime = runtime_dir / "libstdc++.so.6"
    runtime.touch()
    compiler = tmp_path / "fake-cxx"
    compiler.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"${FAKE_CXX_RUNTIME}\"\n",
        encoding="utf-8",
    )
    compiler.chmod(0o755)

    env_runtime = os.environ.get("FAKE_CXX_RUNTIME")
    os.environ["FAKE_CXX_RUNTIME"] = str(runtime)
    try:
        completed = run_preflight(tmp_path, "core", cxx=str(compiler))
    finally:
        if env_runtime is None:
            os.environ.pop("FAKE_CXX_RUNTIME", None)
        else:
            os.environ["FAKE_CXX_RUNTIME"] = env_runtime

    assert completed.returncode == 0
    assert f"cxx_runtime={runtime_dir}" in completed.stdout


def test_engine_profiles_validate_their_isolated_python(tmp_path):
    completed = run_preflight(
        tmp_path,
        "vllm",
        vllm_python="missing-vllm-python",
    )
    assert completed.returncode == 2
    assert "vLLM Python command not found" in completed.stdout

    completed = run_preflight(
        tmp_path,
        "sglang",
        sglang_python="missing-sglang-python",
    )
    assert completed.returncode == 2
    assert "SGLang Python command not found" in completed.stdout
