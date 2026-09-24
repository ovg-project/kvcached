# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""A task may expand acceptance, never silently replace it with smoke tests."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    import engine_compat_profile as profiles
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("name", ["vllm", "native-layout", "allocation"])
def test_reviewed_profiles_have_exact_scopes_and_independent_checks(name):
    profile = profiles.load_profile(name)
    assert len(profile["policy_digest"]) == 64
    assert profile["cpu_tests"] and profile["gpu_tests"]
    assert all(path.startswith(("kvcached/", "tests/")) for path in profile["allow"])
    if name == "native-layout":
        assert profile["allow"] == ["kvcached/integration/vllm/model_runner_v2.py",
                                    "tests/test_vllm_model_runner_v2.py"]
        assert "tests/test_vllm_model_runner_v2.py" in profile["gpu_tests"]


@pytest.mark.parametrize("name", ["../vllm", "vllm;true", "vllm\n", "tp2", "pp2", "hybrid", "unknown"])
def test_unknown_or_unqualified_profiles_fail_closed(name):
    with pytest.raises(ValueError):
        profiles.load_profile(name)


@pytest.mark.parametrize("name", ["native-layout", "allocation"])
def test_native_contracts_are_only_dispatched_for_the_reviewed_release(name):
    profile = profiles.load_profile(name)
    profiles.validate_release(profile, "v0.29.0")
    for tag in ("v0.28.0", "v0.30.0", "main", "v0.29.0rc1"):
        with pytest.raises(ValueError):
            profiles.validate_release(profile, tag)


def test_profile_and_trusted_test_changes_invalidate_evidence(tmp_path, monkeypatch):
    root = tmp_path / "controller"
    policy = root / ".github/engine-compat"
    policy.mkdir(parents=True)
    profile = profiles.load_profile("vllm")
    for name in ("vllm.json", "vllm-allow.json"):
        (policy / name).write_bytes((profiles.PROFILES / name).read_bytes())
    for test in profile["cpu_tests"] + ["tools/engine_compat_gpu_probe.py"]:
        target = root / test
        target.parent.mkdir(exist_ok=True)
        target.write_bytes((ROOT / test).read_bytes())
    monkeypatch.setattr(profiles, "ROOT", root)
    monkeypatch.setattr(profiles, "PROFILES", policy)
    before = profiles.load_profile("vllm")["policy_digest"]
    (root / profile["cpu_tests"][0]).write_text("def test_changed(): pass\n")
    assert profiles.load_profile("vllm")["policy_digest"] != before
    before = profiles.load_profile("vllm")["policy_digest"]
    (root / "tools/engine_compat_gpu_probe.py").write_text("# different probe\n")
    assert profiles.load_profile("vllm")["policy_digest"] != before


@pytest.mark.parametrize("body,status", [
    ("def test_ok(): assert True\n", 0),
    ("def test_bad(): assert False\n", 1),
    ("import pytest\ndef test_skip(): pytest.skip('no GPU')\n", 2),
    ("# Empty required contract\n", 2),
    ("raise ImportError('missing runtime')\n", 2),
])
def test_required_checks_record_counts_and_never_pass_skips(tmp_path, monkeypatch, body, status):
    controller, candidate = tmp_path / "controller", tmp_path / "candidate"
    for root in (controller, candidate):
        (root / "tests").mkdir(parents=True)
    name = "tests/test_contract.py"
    (controller / name).write_text(body)
    (candidate / name).write_text("def test_faked_success(): pass\n")
    monkeypatch.setattr(profiles, "ROOT", controller)
    profile = dict(name="test", policy_digest="a" * 64, cpu_tests=[name], gpu_tests=[name])
    output = tmp_path / "evidence"
    assert profiles.run_checks(profile, candidate, output) == status
    receipt = json.loads((output / "checks.json").read_text())
    assert receipt["trusted"] is True
    assert receipt["checks"][0]["command"][-2] == str(controller / name)
    assert receipt["exit_code"] == status


def test_gpu_profile_is_checked_inside_the_isolated_container():
    shell = (ROOT / "tools/engine_compat_gpu.sh").read_text()
    assert shell.index("--gpu") < shell.index("--probe")
    assert '"$ENGINE_COMPAT_PROFILE"' in shell
    assert "/controller:ro" in shell


@pytest.mark.parametrize("runner", ["v1", "v2"])
def test_028_attention_matrix_is_validation_only(runner):
    profile = profiles.load_profile(f"attention-{runner}-028")
    assert profile["runner"] == runner
    assert profile["layouts"] == ["non-contiguous", "contiguous"]
    profiles.validate_release(profile, "v0.28.0")
    profiles.validate_mode(profile, "validate")
    with pytest.raises(ValueError, match="validation-only"):
        profiles.validate_mode(profile, "repair")
    with pytest.raises(ValueError, match="no trusted contracts"):
        profiles.validate_release(profile, "v0.29.0")


@pytest.mark.parametrize("fault", [None, "missing", "wrong-runner", "wrong-sha", "failed", "blocked"])
def test_probe_matrix_requires_every_cell_and_matching_identity(tmp_path, monkeypatch, fault):
    profile = profiles.load_profile("attention-v1-028")
    calls = []

    def execute(command, **kwargs):
        from types import SimpleNamespace
        destination = Path(command[command.index("--output") + 1])
        calls.append(destination.name)
        destination.mkdir(parents=True)
        value = dict(status="passed", comparison="passed", candidate_sha="a" * 40,
                     expected_vllm_version="0.28.0", requested_runner="v1", layout=destination.name)
        code = 0
        if destination.name == "contiguous":
            if fault == "missing":
                return SimpleNamespace(returncode=0)
            if fault == "wrong-runner":
                value["requested_runner"] = "v2"
            if fault == "wrong-sha":
                value["candidate_sha"] = "b" * 40
            if fault in ("failed", "blocked"):
                code = 1 if fault == "failed" else 2
        (destination / "result.json").write_text(json.dumps(value))
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(profiles.subprocess, "run", execute)
    status = profiles.run_probe_matrix(profile, tmp_path, tmp_path / "matrix", "0.28.0", "a" * 40)
    assert status == (0 if fault is None else 1 if fault == "failed" else 2)
    assert calls == profile["layouts"]


@pytest.mark.parametrize("candidate_state,status", [("valid", 0), ("missing", 2), ("broken", 2)])
def test_ownership_contract_loads_candidate_not_controller(tmp_path, candidate_state, status):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    implementation = "kvcached/integration/vllm/native_block_pool.py"
    target = candidate / implementation
    if candidate_state != "missing":
        target.parent.mkdir(parents=True)
        target.write_text((ROOT / implementation).read_text() if candidate_state == "valid"
                          else "raise RuntimeError('broken candidate implementation')\n")
    profile = profiles.load_profile("allocation")
    profile["cpu_tests"] = ["tests/test_vllm_native_block_lifetimes.py"]
    output = tmp_path / "result"
    assert profiles.run_checks(profile, candidate, output) == status
    receipt = json.loads((output / "checks.json").read_text())
    assert receipt["checks"][0]["counts"]["tests"] > 0
