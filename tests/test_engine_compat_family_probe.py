# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Trusted fixed-fixture and matrix contracts; not GPU execution evidence."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    import engine_compat_family_probe as family_probe
    import engine_compat_profile as profiles
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("family", ["gptoss", "mla"])
@pytest.mark.parametrize("version,runner,model", [("0.28.0", "v2", None),
                                                  ("0.29.0", "v1", None),
                                                  ("0.29.0", "v2", Path("other"))])
def test_probe_rejects_unreviewed_runtime_and_model(family, version, runner, model):
    args = SimpleNamespace(version=version, runner=runner, model=model)
    with pytest.raises(family_probe.probe.Blocked, match="fixed tiny fixture"):
        family_probe.worker(family, args, {})


def test_unknown_family_is_not_silently_treated_as_mla():
    with pytest.raises(family_probe.probe.Blocked, match="Unknown"):
        family_probe.worker("typo", None, {})


@pytest.mark.parametrize("throws", [False, True])
def test_admission_hook_counts_real_calls_and_restores_original(throws):
    class Manager:
        def alloc(self, size):
            return list(range(size))

    original = Manager.alloc
    manager = Manager()
    try:
        with family_probe.reject_two_allocations(Manager) as counts:
            assert manager.alloc(0) == []
            assert counts["injected"] == 0
            assert manager.alloc(3) is None
            assert manager.alloc(3) is None
            assert manager.alloc(3) == [0, 1, 2]
            assert counts == {"injected": 2, "subsequent_successes": 1}
            if throws:
                raise RuntimeError("injected test exception")
    except RuntimeError:
        assert throws
    assert Manager.alloc is original


@pytest.mark.parametrize("family", ["gptoss", "mla"])
def test_fixture_keeps_real_attention_geometry_and_cpu_only_preparation(monkeypatch, tmp_path, family):
    torch = Mock(float16="half")
    torch.cuda.is_initialized.return_value = False
    model = Mock()
    model.to.return_value = model
    config = Mock()
    cls = Mock(return_value=model)
    transformers = SimpleNamespace(GptOssConfig=config, DeepseekV2Config=config,
                                   GptOssForCausalLM=cls, DeepseekV2ForCausalLM=cls)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(family_probe.probe, "version", lambda *args: None)
    args = SimpleNamespace(version="0.29.0", runner="v2", model=None,
                           layout="non-contiguous", _stage="prepare", output=tmp_path)
    family_probe.worker(family, args, {})
    values = config.call_args.kwargs
    if family == "gptoss":
        assert values["layer_types"] == ["sliding_attention", "full_attention"] * 2
        assert values["num_local_experts"] == 4
        assert values["num_experts_per_tok"] == 2
        assert "rope_parameters" not in values  # Keep the native YaRN fields.
    else:
        assert values["kv_lora_rank"] == 512
        assert values["qk_rope_head_dim"] == 64
        assert values["first_k_dense_replace"] == values["num_hidden_layers"]
        assert isinstance(values["n_routed_experts"], int)
    torch.manual_seed.assert_called_once_with(20260926)
    model.to.assert_called_once_with(device="cpu", dtype="half")
    model.save_pretrained.assert_called_once_with(tmp_path / "model", safe_serialization=True)


@pytest.mark.parametrize("family", ["gptoss", "mla"])
def test_model_profiles_are_validation_only_and_require_both_layouts(family):
    profile = profiles.load_profile(f"{family}-v2-029")
    assert profile["probe"] == family
    assert profile["layouts"] == ["non-contiguous", "contiguous"]
    with pytest.raises(ValueError, match="validation-only"):
        profiles.validate_mode(profile, "repair")
    with pytest.raises(ValueError):
        profiles.validate_release(profile, "v0.28.0")


@pytest.mark.parametrize("family", ["gptoss", "mla"])
@pytest.mark.parametrize("field,value", [("layouts", ["non-contiguous"]), ("repair", True),
                                         ("runner", "v1"), ("releases", ["*"])])
def test_cannot_waive_layout_or_expand_qualification(tmp_path, monkeypatch, family, field, value):
    name = f"{family}-v2-029"
    value_json = json.loads((profiles.PROFILES / f"{name}.json").read_text())
    value_json[field] = value
    (tmp_path / f"{name}.json").write_text(json.dumps(value_json))
    monkeypatch.setattr(profiles, "PROFILES", tmp_path)
    with pytest.raises(ValueError, match="Model-family acceptance"):
        profiles.load_profile(name)


@pytest.mark.parametrize("family,changed", [(f, p) for f in ("gptoss", "mla")
                                           for p in profiles.PROBES[f]])
def test_all_probe_dependencies_are_part_of_trusted_digest(tmp_path, monkeypatch, family, changed):
    name = f"{family}-v2-029"
    profile = profiles.load_profile(name)
    paths = [f".github/engine-compat/{name}.json", f".github/engine-compat/{name}-allow.json",
             *profile["cpu_tests"], *profile["gpu_tests"], *profiles.PROBES[family]]
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / path).read_bytes())
    monkeypatch.setattr(profiles, "ROOT", tmp_path)
    monkeypatch.setattr(profiles, "PROFILES", tmp_path / ".github/engine-compat")
    before = profiles.load_profile(name)["policy_digest"]
    path = tmp_path / changed
    path.write_text(path.read_text() + "\n# changed\n")
    assert profiles.load_profile(name)["policy_digest"] != before
