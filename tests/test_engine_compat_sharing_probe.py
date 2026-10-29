# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed Gemma probe contracts, not GPU execution evidence."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    import engine_compat_profile as profiles
    import engine_compat_sharing_probe as sharing
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("present", [False, True])
def test_fixture_supplies_only_missing_unused_borrower_norms(present):
    owners = {f"model.layers.{idx}.self_attn.k_norm.weight": Mock() for idx in (0, 1)}
    state = dict(owners)
    if present:
        state.update({f"model.layers.{idx}.self_attn.k_norm.weight": Mock() for idx in (2, 3)})
    before = dict(state)
    result = sharing.fixture_state_dict(Mock(state_dict=lambda: state))
    for borrower, owner in ((2, 0), (3, 1)):
        key = f"model.layers.{borrower}.self_attn.k_norm.weight"
        source = owners[f"model.layers.{owner}.self_attn.k_norm.weight"]
        assert result[key] is (before[key] if present else source.clone.return_value)
        if present:
            source.clone.assert_not_called()
        else:
            source.clone.assert_called_once_with()
    assert all(result[key] is value for key, value in before.items())


@pytest.mark.parametrize("fault", [None, "pointer", "shape", "stride", "dtype"])
def test_borrower_must_alias_its_exact_owner_view(fault):
    left = SimpleNamespace(data_ptr=lambda: 16, shape=(4, 2), stride=lambda: (2, 1), dtype="half")
    right = SimpleNamespace(**vars(left))
    if fault == "pointer":
        right.data_ptr = lambda: 32
    elif fault == "shape":
        right.shape = (2, 4)
    elif fault == "stride":
        right.stride = lambda: (1, 4)
    elif fault == "dtype":
        right.dtype = "int16"
    if fault:
        with pytest.raises(AssertionError):
            sharing.validate_alias(left, right)
    else:
        sharing.validate_alias(left, right)


@pytest.mark.parametrize("fault", [None, "missing", "unfinished", "early-stop", "truncated",
                                   "prompt", "duplicate-completion", "no-logprobs", "nan", "inf"])
def test_http_or_request_success_cannot_hide_bad_generation(fault):
    prompts = [{"prompt_token_ids": [17] * 64}]
    completion = SimpleNamespace(token_ids=[1] * 32, finish_reason="length",
                                 logprobs=[{1: SimpleNamespace(logprob=-0.1)} for _ in range(32)])
    output = SimpleNamespace(prompt_token_ids=[17] * 64, finished=True, outputs=[completion])
    outputs = [output]
    if fault == "missing":
        outputs = []
    elif fault == "unfinished":
        output.finished = False
    elif fault == "early-stop":
        completion.finish_reason = "stop"
    elif fault == "truncated":
        completion.token_ids.pop()
    elif fault == "prompt":
        output.prompt_token_ids[0] = 23
    elif fault == "duplicate-completion":
        output.outputs.append(completion)
    elif fault == "no-logprobs":
        completion.logprobs = None
    elif fault in ("nan", "inf"):
        completion.logprobs[0][1].logprob = float(fault)
    if fault:
        with pytest.raises(AssertionError):
            sharing.collect_outputs(prompts, outputs)
    else:
        assert sharing.collect_outputs(prompts, outputs) == [[1] * 32]


def test_fixture_rejects_vacuous_all_identical_token_comparison():
    with pytest.raises(AssertionError, match="Degenerate fixture"):
        sharing.validate_diversity([[0] * 32] * 28)
    sharing.validate_diversity([[17] * 32, [23] * 32])


@pytest.mark.parametrize("version,runner,model", [("0.28.0", "v2", None),
                                                  ("0.29.0", "v1", None),
                                                  ("0.29.0", "v2", Path("other"))])
def test_probe_does_not_silently_accept_another_model_or_runner(version, runner, model):
    args = SimpleNamespace(version=version, runner=runner, model=model)
    with pytest.raises(sharing.probe.Blocked, match="fixed tiny fixture"):
        sharing.worker(args, {})


@pytest.mark.parametrize("stage", ["native", "patched"])
@pytest.mark.parametrize("layout,expected", [("contiguous", "BLNHC"), ("non-contiguous", "LBNHC")])
def test_native_and_patched_pin_the_same_layout(monkeypatch, stage, layout, expected):
    args = SimpleNamespace(version="0.29.0", runner="v2", model=None, layout=layout, _stage=stage)
    # Stop before importing CUDA/vLLM; the environment must already be pinned.
    monkeypatch.setattr(sharing.probe, "version", Mock(side_effect=RuntimeError("stop")))
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "old")
    with pytest.raises(RuntimeError, match="stop"):
        sharing.worker(args, {})
    assert sharing.os.environ["VLLM_KV_CACHE_LAYOUT"] == expected


def test_sharing_profile_is_validation_only_and_requires_both_layouts():
    profile = profiles.load_profile("sharing-v2-029")
    assert profile["probe"] == "sharing" and profile["runner"] == "v2"
    assert profile["layouts"] == ["non-contiguous", "contiguous"]
    profiles.validate_release(profile, "v0.29.0")
    with pytest.raises(ValueError, match="validation-only"):
        profiles.validate_mode(profile, "repair")
    with pytest.raises(ValueError):
        profiles.validate_release(profile, "v0.28.0")


@pytest.mark.parametrize("field,value", [("layouts", ["non-contiguous"]),
                                         ("repair", True), ("runner", "v1"),
                                         ("releases", ["*"])])
def test_profile_cannot_skip_failing_layout_or_expand_qualification(tmp_path, monkeypatch, field, value):
    profile = json.loads((profiles.PROFILES / "sharing-v2-029.json").read_text())
    profile[field] = value
    (tmp_path / "sharing-v2-029.json").write_text(json.dumps(profile))
    monkeypatch.setattr(profiles, "PROFILES", tmp_path)
    with pytest.raises(ValueError, match="Sharing acceptance"):
        profiles.load_profile("sharing-v2-029")


@pytest.mark.parametrize("changed", profiles.PROBES["sharing"])
def test_sharing_probe_changes_invalidate_trusted_policy(tmp_path, monkeypatch, changed):
    name = "sharing-v2-029"
    profile = profiles.load_profile(name)
    paths = [f".github/engine-compat/{name}.json", f".github/engine-compat/{name}-allow.json",
             *profile["cpu_tests"], *profile["gpu_tests"], *profiles.PROBES["sharing"]]
    for name in paths:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(profiles, "ROOT", tmp_path)
    monkeypatch.setattr(profiles, "PROFILES", tmp_path / ".github/engine-compat")
    before = profiles.load_profile("sharing-v2-029")["policy_digest"]
    path = tmp_path / changed
    path.write_text(path.read_text() + "\n# changed\n")
    assert profiles.load_profile("sharing-v2-029")["policy_digest"] != before
