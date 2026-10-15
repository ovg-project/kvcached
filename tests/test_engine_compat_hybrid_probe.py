# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed hybrid evidence checks; CPU tests are not GPU qualification."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    import engine_compat_hybrid_probe as probe
    import engine_compat_profile as profiles
finally:
    sys.path.pop(0)


def case(runner="v1", stage="sync-fault"):
    rows = [
        dict(input=token, length=length, repeat=repeat, tokens=[17] * 32)
        for token in (17, 23)
        for length in (48, 64, 783, 784, 785, 1567)
        for repeat in (0, 1)
    ]
    data = dict(
        status="passed",
        explicit_shutdown_completed=True,
        requests=24,
        output_tokens=768,
        rows=rows,
        worker_evidence=[
            dict(
                runner_module=probe.RUNNERS[runner],
                prefix_match_unit=16,
                groups=[dict(type="MambaSpec", block_size=784)],
                async_scheduling=stage.startswith("async-"),
            )
        ],
    )
    markers = [
        dict(event="partial_hit", hash_block_size=16, allocation_block_size=784),
        dict(event="copy_submitted", runner=runner, count=1),
    ]
    if stage.endswith("fault"):
        markers.extend(dict(event="cow_admission_miss", hit=i) for i in (1, 2))
    return data, markers


@pytest.mark.parametrize("runner", ["v1", "v2"])
@pytest.mark.parametrize("stage", probe.STAGES)
def test_complete_case_requires_real_partial_hit_copy_and_injection(runner, stage):
    data, markers = case(runner, stage)
    vectors, counts = probe.validate_case(data, markers, "", stage, runner)
    assert len(vectors) == 24
    assert counts["partial_hit"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "status",
        "shutdown",
        "requests",
        "tokens",
        "truncated",
        "duplicate",
        "runner",
        "schedule",
        "hash",
        "no-partial",
        "partial-hash",
        "no-copy",
        "copy-runner",
        "copy-empty",
        "no-injection",
        "duplicate-injection",
    ],
)
def test_incomplete_or_wrong_evidence_cannot_pass(fault):
    data, markers = case()
    if fault == "status":
        data["status"] = "blocked"
    elif fault in ("shutdown", "requests", "tokens"):
        key = {
            "shutdown": "explicit_shutdown_completed",
            "requests": "requests",
            "tokens": "output_tokens",
        }[fault]
        data[key] = 0
    elif fault == "truncated":
        data["rows"][0]["tokens"].pop()
    elif fault == "duplicate":
        data["rows"][0] = data["rows"][1]
    elif fault == "runner":
        data["worker_evidence"][0]["runner_module"] = probe.RUNNERS["v2"]
    elif fault == "schedule":
        data["worker_evidence"][0]["async_scheduling"] = True
    elif fault == "hash":
        data["worker_evidence"][0]["prefix_match_unit"] = 784
    elif fault == "no-partial":
        markers.pop(0)
    elif fault == "partial-hash":
        markers[0]["hash_block_size"] = 784
    elif fault == "no-copy":
        markers.pop(1)
    elif fault == "copy-runner":
        markers[1]["runner"] = "v2"
    elif fault == "copy-empty":
        markers[1]["count"] = 0
    elif fault == "no-injection":
        markers.pop()
    elif fault == "duplicate-injection":
        markers[-1]["hit"] = 1
    with pytest.raises(AssertionError):
        probe.validate_case(data, markers, "", "sync-fault", "v1")


@pytest.mark.parametrize(
    "log",
    [
        "CUDA out of memory",
        "CUDA illegal instruction",
        "CUDA error: invalid resource handle",
        "EngineDeadError",
        "illegal memory access",
        "unknown CUDA driver error",
        "EngineCore failed",
        "Page 2 is not mapped",
        "Cannot get 1 free blocks",
        "Traceback (most recent call last):",
        "ERROR [base.py:42] Traceback",
    ],
)
def test_fatal_or_unexpected_traceback_cannot_pass(log):
    data, markers = case()
    with pytest.raises(AssertionError):
        probe.validate_case(data, markers, log, "sync-fault", "v1")


def test_known_skip_tokenizer_warmup_warning_is_not_hidden():
    data, markers = case()
    log = (
        "WARNING 00:00 [base.py:42] Traceback (most recent call last):\n"
        "Chat template warmup failed\nTokenizer not available when `skip_tokenizer_init=True`"
    )
    probe.validate_case(data, markers, log, "sync-fault", "v1")
    with pytest.raises(AssertionError):
        probe.validate_case(data, markers, log + "\nTraceback: another error", "sync-fault", "v1")


@pytest.mark.parametrize(
    "name,version,runner", [("hybrid-v1-028", "0.28.0", "v1"), ("hybrid-v2-029", "0.29.0", "v2")]
)
def test_hybrid_profiles_have_no_repair_or_wrong_version_escape(name, version, runner):
    profile = profiles.load_profile(name)
    assert profile["runner"] == runner and profile["probe"] == "hybrid"
    assert profile["layouts"] == ["non-contiguous"]
    profiles.validate_release(profile, "v" + version)
    profiles.validate_mode(profile, "validate")
    with pytest.raises(ValueError, match="validation-only"):
        profiles.validate_mode(profile, "repair")
    with pytest.raises(ValueError, match="no trusted contracts"):
        profiles.validate_release(profile, "v0.30.0")


@pytest.mark.parametrize("changed", profiles.PROBES["hybrid"])
def test_model_hooks_and_probe_changes_invalidate_profile(tmp_path, monkeypatch, changed):
    name = "hybrid-v1-028"
    profile = profiles.load_profile(name)
    for path in [
        f".github/engine-compat/{name}.json",
        f".github/engine-compat/{name}-allow.json",
        *profile["cpu_tests"],
        *profiles.PROBES["hybrid"],
    ]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / path).read_bytes())
    monkeypatch.setattr(profiles, "ROOT", tmp_path)
    monkeypatch.setattr(profiles, "PROFILES", tmp_path / ".github/engine-compat")
    before = profiles.load_profile(name)["policy_digest"]
    path = tmp_path / changed
    path.write_text(path.read_text() + "\n")
    assert profiles.load_profile(name)["policy_digest"] != before


@pytest.mark.parametrize(
    "change",
    [
        {"probe": "../arbitrary.py"},
        {"repair": True},
        {"layouts": ["contiguous"]},
        {"releases": ["0.29"]},
    ],
)
def test_unqualified_hybrid_matrix_is_rejected(tmp_path, monkeypatch, change):
    name = "hybrid-v1-028"
    value = json.loads((profiles.PROFILES / f"{name}.json").read_text())
    value.update(change)
    (tmp_path / f"{name}.json").write_text(json.dumps(value))
    monkeypatch.setattr(profiles, "PROFILES", tmp_path)
    with pytest.raises(ValueError):
        profiles.load_profile(name)


def test_matrix_selects_the_hybrid_probe_not_attention_smoke(tmp_path, monkeypatch):
    profile = profiles.load_profile("hybrid-v1-028")
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(profiles.subprocess, "run", execute)
    assert profiles.run_probe_matrix(profile, tmp_path, tmp_path / "out", "0.28.0", "a" * 40) == 2
    assert len(commands) == 1
    assert commands[0][1].endswith("engine_compat_hybrid_probe.py")


def test_fixture_is_dense_hybrid_and_has_no_remote_model_reference():
    config = json.loads(probe.FIXTURE.read_text())
    assert config["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    text = config["text_config"]
    assert text["layer_types"].count("linear_attention") == 3
    assert text["layer_types"].count("full_attention") == 1
    assert text["num_hidden_layers"] == 4
    assert not any("expert" in key for key in text)
    assert "_name_or_path" not in config


@pytest.mark.parametrize("fault", ["no-mamba", "bad-unit", "wrong-boundary"])
def test_missing_hybrid_geometry_or_boundary_cannot_pass(fault):
    data, markers = case()
    group = data["worker_evidence"][0]["groups"][0]
    if fault == "no-mamba":
        group["type"] = "FullAttentionSpec"
    elif fault == "bad-unit":
        group["block_size"] = 16
    else:
        data["rows"][0]["length"] = 50
    with pytest.raises(AssertionError):
        probe.validate_case(data, markers, "", "sync-fault", "v1")


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "candidate_sha",
        "expected_vllm_version",
        "requested_runner",
        "layout",
        "model_config_sha256",
        "missing-result",
        "blocked",
        "timeout",
    ],
)
def test_child_identity_cleanup_and_infrastructure_gate(tmp_path, monkeypatch, fault):
    args = SimpleNamespace(
        source=tmp_path,
        output=tmp_path,
        version="0.28.0",
        runner="v1",
        candidate_sha="a" * 40,
        timeout=1,
        layout="non-contiguous",
    )
    data, markers = case()
    data.update(
        candidate_sha=args.candidate_sha,
        expected_vllm_version=args.version,
        requested_runner=args.runner,
        layout=args.layout,
        model_config_sha256=hashlib.sha256(probe.FIXTURE.read_bytes()).hexdigest(),
    )
    if fault in data:
        data[fault] = "wrong"
    if fault != "missing-result":
        (tmp_path / "sync-fault.json").write_text(json.dumps(data))
    (tmp_path / "sync-fault.jsonl").write_text("\n".join(map(json.dumps, markers)))
    cleaned: list[SimpleNamespace] = []
    environments: list[dict[str, str]] = []

    def wait(timeout):
        if fault == "timeout":
            raise subprocess.TimeoutExpired("probe", timeout)

    process = SimpleNamespace(returncode=2 if fault == "blocked" else 0, wait=wait)
    monkeypatch.setattr(probe, "environment", lambda *_: {})
    def spawn(*args, **kwargs):
        environments.append(kwargs["env"])
        assert Path(kwargs["env"]["TMPDIR"]).is_dir()
        return process

    monkeypatch.setattr(probe.subprocess, "Popen", spawn)
    monkeypatch.setattr(probe, "cleanup", cleaned.append)
    if fault == "none":
        _, vectors, _ = probe.run_case(args, "sync-fault")
        assert len(vectors) == 24
    else:
        with pytest.raises(
            (AssertionError, FileNotFoundError, probe.Blocked, subprocess.TimeoutExpired)
        ):
            probe.run_case(args, "sync-fault")
    assert cleaned == [process]
    ipc_dir = Path(environments[0]["TMPDIR"])
    assert ipc_dir.name.startswith("kch-")
    assert not ipc_dir.exists()
    assert not ipc_dir.is_relative_to(tmp_path)
