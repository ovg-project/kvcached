# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""CPU-only supervisor contracts; these mocks provide no GPU execution evidence."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location(
    "engine_gpu_probe", Path(__file__).parents[1] / "tools/engine_compat_gpu_probe.py"
)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
SHA = "ab" * 20


@pytest.fixture
def args(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "candidate.py").write_text("value = 1\n", encoding="utf-8")
    return SimpleNamespace(
        source=source, output=tmp_path / "output", version="0.28.0", sha=SHA, timeout=1, model=None,
        runner="auto", layout="contiguous",
    )


@pytest.mark.parametrize("runner", ["v1", "v2"])
def test_runner_selection_is_verified_from_the_loaded_class(runner):
    module = "vllm.v1.worker.gpu_model_runner" if runner == "v1" else "vllm.v1.worker.gpu.model_runner"
    model_runner = type("GPUModelRunner", (), {"__module__": module})()
    value = SimpleNamespace(model_runner=model_runner)
    for name in ("worker", "driver_worker", "model_executor", "engine_core", "engine_core", "llm_engine"):
        value = SimpleNamespace(**{name: value})
    assert probe.runner_identity(value, runner)["runner"] == runner
    with pytest.raises(AssertionError, match="Expected"):
        probe.runner_identity(value, "v2" if runner == "v1" else "v1")


@pytest.mark.parametrize("runner,flag", [("v1", "0"), ("v2", "1")])
def test_probe_pins_both_runner_and_elastic_layout(args, runner, flag):
    args.runner, args.layout = runner, "non-contiguous"
    env = probe.environment(args, "patched")
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == flag
    assert env["KVCACHED_CONTIGUOUS_LAYOUT"] == "false"


def cli(monkeypatch, args, *extra):
    monkeypatch.setattr(
        probe.sys,
        "argv",
        [
            "probe",
            "--source",
            str(args.source),
            "--output",
            str(args.output),
            "--version",
            args.version,
            "--candidate-sha",
            SHA,
            *extra,
        ],
    )
    monkeypatch.setattr(probe.signal, "signal", Mock())


@pytest.mark.parametrize("actual", ["0.28.0", "0.28.0+cu129"])
def test_exact_public_release_accepts_build_metadata(monkeypatch, actual):
    monkeypatch.setattr(probe.importlib.metadata, "version", Mock(return_value=actual))
    result: dict = {}
    probe.version("0.28.0", result)
    assert result["actual_vllm_version"] == actual


@pytest.mark.parametrize(
    "actual",
    [
        "0.28.0rc1",
        "0.28.0rc1+cu129",
        "0.28.0.dev1",
        "0.28.0.post1",
        "0.28.1",
        "0.29.0",
    ],
)
def test_other_public_versions_are_blocked(monkeypatch, actual):
    monkeypatch.setattr(probe.importlib.metadata, "version", Mock(return_value=actual))
    with pytest.raises(probe.Blocked, match="Expected exact vLLM"):
        probe.version("0.28.0", {})


def test_missing_vllm_is_blocked(monkeypatch):
    missing = probe.importlib.metadata.PackageNotFoundError("vllm")
    monkeypatch.setattr(probe.importlib.metadata, "version", Mock(side_effect=missing))
    with pytest.raises(probe.Blocked, match="not installed"):
        probe.version("0.28.0", {})


def test_fingerprint_ignores_caches_but_detects_content_names_and_deletions(args):
    args.output = args.source / "evidence"
    baseline = probe.fingerprint(args.source, args.output)
    for name in probe.IGNORED | {"evidence"}:
        directory = args.source / name
        directory.mkdir()
        (directory / "generated").write_bytes(b"ignored")
    for suffix in (".pyc", ".pyo"):
        (args.source / ("cached" + suffix)).write_bytes(b"ignored")
    assert probe.fingerprint(args.source, args.output) == baseline
    path = args.source / "candidate.py"
    path.write_bytes(b"value = 2\n")
    changed = probe.fingerprint(args.source, args.output)
    assert changed != baseline
    renamed = path.rename(args.source / "renamed.py")
    assert probe.fingerprint(args.source, args.output) != changed
    renamed.unlink()
    assert probe.fingerprint(args.source, args.output)["files"] == 0


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("passed", "passed"),
        ("blocked", "blocked"),
        ("failed", "failed"),
        ("mixed", "failed"),
        ("mismatch", "failed"),
        ("mutated", "failed"),
    ],
)
def test_supervisor_provenance_and_classification(monkeypatch, args, scenario, expected):
    cli(monkeypatch, args)
    platform = Mock(wraps=probe.os)
    platform.name = "posix"
    monkeypatch.setattr(probe, "os", platform)
    monkeypatch.setattr(probe, "version", Mock())
    git = Mock(side_effect=AssertionError("Supplied SHA must not invoke Git"))
    monkeypatch.setattr(probe.subprocess, "check_output", git)

    def stage(options, name):
        status = scenario if name == "native" and scenario in ("blocked", "failed") else "passed"
        if scenario == "mixed" and name in ("native", "patched"):
            status = "blocked" if name == "native" else "failed"
        if scenario == "mutated" and name == "oom":
            (args.source / "candidate.py").write_bytes(b"changed\n")
        token = 2 if scenario == "mismatch" and name == "patched" else 1
        return dict(status=status, result=dict(prompt_token_ids=[[1]], token_ids=[[token]]))

    stages = Mock(side_effect=stage)
    monkeypatch.setattr(probe, "run", stages)
    assert probe.main() == probe.EXIT[expected]
    result = json.loads((args.output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == expected
    assert result["candidate_sha"] == SHA
    assert result["candidate_sha_provenance"] == "supplied_by_trusted_host"
    assert result["expected_vllm_version"] == args.version
    assert [call.args[1] for call in stages.call_args_list] == [
        "prepare",
        "native",
        "patched",
        "oom",
    ]
    assert (result["source_fingerprint_before"] == result["source_fingerprint_after"]) == (
        scenario != "mutated"
    )
    if scenario == "mismatch":
        assert result["comparison"] == "failed"
    git.assert_not_called()


@pytest.mark.parametrize("sha", ["abc", "g" * 40, "a" * 41])
def test_invalid_supplied_sha_is_rejected(monkeypatch, args, sha):
    cli(monkeypatch, args, "--candidate-sha", sha)
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2
    assert not args.output.exists()


@pytest.mark.parametrize(
    "error,status",
    [
        (probe.Blocked("CUDA unavailable"), "blocked"),
        (ModuleNotFoundError("vllm.old_helper"), "failed"),
        (AssertionError("incomplete output"), "failed"),
        (RuntimeError("worker crashed"), "failed"),
    ],
)
def test_worker_error_classification_preserves_identity(monkeypatch, args, error, status):
    cli(monkeypatch, args, "--_stage", "native")
    args.output.mkdir()
    monkeypatch.setattr(probe, "worker", Mock(side_effect=error))
    assert probe.main() == probe.EXIT[status]
    result = json.loads((args.output / "native.json").read_text(encoding="utf-8"))
    assert result["status"] == status
    assert result["candidate_sha"] == SHA
    assert result["expected_vllm_version"] == args.version
    assert type(error).__name__ in result["traceback"]


def test_missing_runtime_torch_is_explicitly_blocked(monkeypatch, args):
    monkeypatch.setattr(probe, "version", Mock())
    monkeypatch.setitem(probe.sys.modules, "torch", None)
    with pytest.raises(probe.Blocked, match="PyTorch"):
        probe.worker(args, {})


@pytest.mark.parametrize(
    "case,status,code",
    [
        ("normal", "passed", 0),
        ("normal", "blocked", 2),
        ("normal", "failed", 1),
        ("crash", "passed", 9),
        ("timeout", "passed", 0),
        ("missing", "passed", 0),
        ("candidate_sha", "passed", 0),
        ("expected_vllm_version", "passed", 0),
    ],
)
def test_parent_checks_exit_timeout_and_child_identity(monkeypatch, args, case, status, code):
    args.output.mkdir()
    detail = dict(status=status, candidate_sha=SHA, expected_vllm_version=args.version)
    if case in detail:
        detail[case] = "wrong"
    if case != "missing":
        probe.save(args.output / "native.json", detail)
    process = Mock(returncode=code)
    if case == "timeout":
        process.wait.side_effect = probe.subprocess.TimeoutExpired("child", args.timeout)
    launch = Mock(return_value=process)
    cleanup = Mock()
    monkeypatch.setattr(probe.subprocess, "Popen", launch)
    monkeypatch.setattr(probe, "cleanup", cleanup)
    if case in ("candidate_sha", "expected_vllm_version"):
        with pytest.raises(AssertionError, match="differs"):
            probe.run(args, "native")
    else:
        result = probe.run(args, "native")
        expected = status if case == "normal" else "blocked" if case == "timeout" else "failed"
        assert result["status"] == expected
        assert result["timed_out"] == (case == "timeout")
        assert result["returncode"] == code
    cleanup.assert_called_once_with(process)
    assert launch.call_args.kwargs["start_new_session"] is True
    assert launch.call_args.kwargs["cwd"] == args.output
    if case == "timeout":
        assert "TimeoutExpired" in (args.output / "native.log").read_text(encoding="utf-8")


def test_environment_filters_credentials_and_isolates_modes(monkeypatch, args):
    secrets = (
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "CODEX_HOME",
        "HF_TOKEN",
        "AWS_SESSION_TOKEN",
        "SSH_AUTH_SOCK",
        "HTTPS_PROXY",
        "LD_PRELOAD",
    )
    for name in secrets:
        monkeypatch.setenv(name, "must-not-inherit")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-selected,GPU-other")
    monkeypatch.setenv("PYTHONPATH", "must-not-inherit")
    environments = {
        stage: probe.environment(args, stage) for stage in ("prepare", "native", "patched")
    }
    for stage, env in environments.items():
        assert not set(secrets) & env.keys()
        assert env["PYTHONPATH"] == str(args.source)
        assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
        assert env["CUDA_VISIBLE_DEVICES"] == ("" if stage == "prepare" else "GPU-selected")
        assert env["ENABLE_KVCACHED"] == ("true" if stage == "patched" else "false")
        assert env["KVCACHED_AUTOPATCH"] == ("1" if stage == "patched" else "0")
        for key in ("HOME", "TMPDIR", "HF_HOME", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR"):
            assert Path(env[key]).is_relative_to(args.output)
    assert len({env["KVCACHED_IPC_NAME"] for env in environments.values()}) == 3
