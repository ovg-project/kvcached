# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Release analysis policy and handoff tests; no agent, GPU or network required."""

import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    spec = importlib.util.spec_from_file_location("release_analysis", ROOT / "tools/engine_release_analysis.py")
    assert spec is not None and spec.loader is not None
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
finally:
    sys.path.pop(0)


def example_plan():
    return dict(summary="New worker contract", coverage={area: dict(
        disposition="repair" if area == "startup" else "unchanged",
        reason="Compared both release sources", references=[],
    ) for area in analysis.AREAS}, tasks=[dict(
        id="worker-contract", title="Preserve the worker contract", problem="New return contract",
        allow=["kvcached/integration/vllm/patches.py", "tests/test_new_contract.py"],
        evidence=[dict(tree="new", path="worker.py", line=2, explanation="Changed contract")],
        acceptance=["Fail before the fix; preserve the return contract afterwards"],
        hardware="single-gpu",
    )])


@pytest.fixture
def trees(tmp_path):
    result = {}
    for name in ("old", "new", "candidate"):
        root = tmp_path / name
        root.mkdir()
        (root / "worker.py").write_text("def f():\n    return 1\n")
        result[name] = root
    return result


def test_concrete_task_scope_comes_from_release_analysis_not_a_fixed_profile(trees):
    plan = example_plan()
    assert analysis.validate_plan(plan, "vllm", trees) == sorted([
        *plan["tasks"][0]["allow"], "tests/manifests/cpu.txt"])


def test_agent_schema_exposes_the_same_reference_and_task_id_restrictions():
    fields = analysis.PLAN_SCHEMA["properties"]
    coverage = fields["coverage"]
    assert set(coverage["properties"]) == set(coverage["required"]) == set(analysis.AREAS)
    assert coverage["additionalProperties"] is False
    reference = coverage["properties"]["startup"]["properties"]["references"]["items"]
    task = fields["tasks"]["items"]["properties"]
    assert re.fullmatch(reference["pattern"], "https://github.com/ovg-project/kvcached/pull/515")
    assert not re.fullmatch(reference["pattern"], "new:worker.py:12")
    assert re.fullmatch(task["id"]["pattern"], "worker-contract")
    assert not re.fullmatch(task["id"]["pattern"], "R1")
    assert task["hardware"]["enum"] == ["single-gpu"]


@pytest.mark.parametrize("field,value", [("id", "R1"), ("reference", "new:worker.py:12")])
def test_structured_schema_rejects_live_replay_format_errors(field, value):
    plan = example_plan()
    if field == "id":
        plan["tasks"][0]["id"] = value
    else:
        plan["coverage"]["startup"]["references"] = [value]
    with pytest.raises(ValueError, match="structured analysis"):
        analysis.validate_schema(plan, analysis.PLAN_SCHEMA)


@pytest.mark.parametrize("path", [
    "../config.toml", "kvcached/../tools/repair.py", "tests/../../token", "/tmp/test.py",
    "C:/secret.py", "kvcached\\module.py", "tools/engine_compat_stage.py",
    ".github/workflows/ci.yml", "kvcached/integration/sglang/patches.py", "tests/conftest.py",
])
def test_analysis_cannot_grant_arbitrary_write_scope(trees, path):
    plan = example_plan()
    plan["tasks"][0]["allow"] = [path]
    with pytest.raises(ValueError):
        analysis.validate_plan(plan, "vllm", trees)


@pytest.mark.parametrize("fault", ["area", "duplicate-area", "line", "source", "task", "acceptance",
                                  "duplicate-task", "reference", "no-task", "excess-tasks"])
def test_invalid_or_incomplete_analysis_stops_before_repair(trees, fault):
    plan = example_plan()
    if fault == "area":
        plan["coverage"].pop("startup")
    elif fault == "duplicate-area":
        plan["coverage"]["startup-duplicate"] = copy.deepcopy(plan["coverage"]["startup"])
    elif fault == "line":
        plan["tasks"][0]["evidence"][0]["line"] = 99
    elif fault == "source":
        plan["tasks"][0]["evidence"][0]["path"] = "missing.py"
    elif fault == "task":
        plan["tasks"][0]["id"] = "not a task"
    elif fault == "acceptance":
        plan["tasks"][0]["acceptance"] = []
    elif fault == "duplicate-task":
        plan["tasks"].append(copy.deepcopy(plan["tasks"][0]))
    elif fault == "reference":
        plan["coverage"]["startup"]["references"] = ["https://example.com/instructions"]
    elif fault == "no-task":
        plan["tasks"] = []
    else:
        plan["tasks"] *= 9
    with pytest.raises(ValueError):
        analysis.validate_plan(plan, "vllm", trees)


@pytest.fixture
def case(tmp_path, trees, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    args = SimpleNamespace(source=trees["candidate"], old_engine=trees["old"],
                           engine_source=trees["new"], output=output, engine="vllm",
                           tag="v0.29.0", old_tag="v0.28.0", codex="codex", pending_prs=None)
    inputs = dict(engine="vllm", pins=dict(candidate="a" * 40, old="b" * 40, new="c" * 40))
    monkeypatch.setattr(analysis, "context", lambda args: (trees, inputs))
    replies = [example_plan(), dict(task_ids=["worker-contract"],
               cpu_test="def test_worker_contract__cpu():\n    assert True\n",
               gpu_test="def test_gpu():\n    assert True\n", rationale="fixture only"),
               dict(approved=True, task_ids=["worker-contract"], findings=[])]
    calls = []

    def agent(*args):
        calls.append(args)
        return replies[len(calls) - 1]

    monkeypatch.setattr(analysis, "agent", agent)
    return args, replies, calls


def test_analysis_contract_author_and_reviewer_are_separate_frozen_stages(case):
    args, _, calls = case
    manifest = analysis.analyze(args)
    assert [c[3] for c in calls] == ["analysis", "contracts", "contract-review"]
    assert "no operator-authored gap list" in calls[0][5].lower()
    bundle = args.output / "bundle"
    assert analysis.load_bundle(bundle, manifest["digest"]) == manifest
    assert manifest["base_sha"] == "a" * 40
    assert manifest["engine_sha"] == "c" * 40


def test_unavailable_hardware_remains_reported_without_blocking_independent_task(case):
    args, replies, calls = case
    distributed = replies[0]["coverage"]["distributed"]
    distributed.update(disposition="blocked", reason="Requires two GPUs for acceptance")
    manifest = analysis.analyze(args)
    assert manifest["plan"]["coverage"]["distributed"]["disposition"] == "blocked"
    assert len(manifest["plan"]["tasks"]) == 1
    assert "coverage as blocked" in calls[0][5]


def test_partial_area_repair_does_not_clear_its_other_blockers(case):
    args, replies, _ = case
    replies[0]["coverage"]["startup"].update(
        disposition="blocked", reason="Worker fix is actionable; another startup gap needs two GPUs")
    manifest = analysis.analyze(args)
    assert manifest["plan"]["tasks"][0]["id"] == "worker-contract"
    assert manifest["plan"]["coverage"]["startup"]["disposition"] == "blocked"


@pytest.mark.parametrize("fault", ["missing-task", "review-rejected", "review-findings", "no-assert",
                                  "no-test", "multi-gpu"])
def test_contract_gate_cannot_silently_waive_evidence(case, fault):
    args, replies, _ = case
    if fault == "missing-task":
        replies[1]["task_ids"] = []
    elif fault == "review-rejected":
        replies[2]["approved"] = False
    elif fault == "review-findings":
        replies[2]["findings"] = ["Tautological test"]
    elif fault == "no-assert":
        replies[1]["cpu_test"] = "def test_cpu():\n    pass\n"
    elif fault == "no-test":
        replies[1]["gpu_test"] = "assert True\n"
    else:
        replies[0]["tasks"][0]["hardware"] = "multi-gpu"
    with pytest.raises(ValueError):
        analysis.analyze(args)
    assert not (args.output / "bundle/manifest.json").exists()


@pytest.mark.parametrize("fault", ["contract", "manifest", "allow", "digest"])
def test_downstream_refuses_changed_analysis_or_acceptance(case, fault):
    args, _, _ = case
    manifest = analysis.analyze(args)
    bundle = args.output / "bundle"
    expected = manifest["digest"]
    if fault == "contract":
        (bundle / "test_gpu.py").write_text("assert True")
    elif fault == "manifest":
        manifest["base_sha"] = "e" * 40
        analysis.save(bundle / "manifest.json", manifest)
    elif fault == "allow":
        analysis.save(bundle / "allow.json", ["kvcached/extra.py"])
    else:
        expected = "0" * 64
    with pytest.raises(ValueError):
        analysis.load_bundle(bundle, expected)


def test_agent_runs_read_only_without_publication_credentials(tmp_path, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.setenv("GH_TOKEN", "not-for-agent")
    monkeypatch.setenv("GITHUB_TOKEN", "not-for-agent")
    monkeypatch.setenv("SSH_AUTH_SOCK", "not-for-agent")
    monkeypatch.setattr(analysis, "git", lambda *args: "a" * 40)
    monkeypatch.setattr(analysis, "fingerprint", lambda *args: {"source": "unchanged"})

    def run(codex, cwd, destination, name, inputs, resume):
        assert inputs["prompt"] == "analyze"
        assert inputs["schema"] == analysis.REVIEW_SCHEMA
        assert inputs["sources"][str(tmp_path)][0] == "a" * 40
        assert not resume
        return dict(approved=True, task_ids=[], findings=[])

    monkeypatch.setattr(analysis, "execute_stage", run)
    assert analysis.agent("codex", tmp_path, output, "review", analysis.REVIEW_SCHEMA,
                          "analyze", [tmp_path])["approved"]


def test_bundle_digest_is_content_bound(case):
    args, _, _ = case
    manifest = analysis.analyze(args)
    digest = manifest.pop("digest")
    assert digest == hashlib.sha256(analysis.canonical(manifest)).hexdigest()
    assert json.loads((args.output / "bundle/allow.json").read_text()) == manifest["allow"]


def test_resume_refuses_changed_release_inventory_before_agent_runs(case):
    args, _, calls = case
    args.resume = True
    analysis.save(args.output / "inputs.json", {"different": "source"})
    with pytest.raises(ValueError, match="inputs changed"):
        analysis.analyze(args)
    assert calls == []


def test_source_mutation_on_interrupted_stage_invalidates_checkpoint(tmp_path, monkeypatch):
    state = {"source": "original"}
    monkeypatch.setattr(analysis, "git", lambda *args: "a" * 40)
    monkeypatch.setattr(analysis, "fingerprint", lambda *args: state.copy())

    def execute(*args, **kwargs):
        analysis.save(tmp_path / "review-state.json", {"status": "interrupted"})
        state["source"] = "changed"
        raise ValueError("interrupted")

    monkeypatch.setattr(analysis, "execute_stage", execute)
    with pytest.raises(ValueError, match="protected checkout"):
        analysis.agent("codex", tmp_path, tmp_path, "review", analysis.REVIEW_SCHEMA,
                       "inspect", [tmp_path])
    assert analysis.read_json(tmp_path / "review-state.json")["status"] == "failed"


@pytest.mark.parametrize("mutation", ["replace", "delete", "invalid-json"])
def test_agent_cannot_change_its_task_input_during_execution(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(analysis, "git", lambda *args: "a" * 40)
    monkeypatch.setattr(analysis, "fingerprint", lambda *args: {"source": "unchanged"})
    analysis.save(tmp_path / "inputs.json", {"original": True})

    def execute(*args, **kwargs):
        analysis.save(tmp_path / "analysis-state.json", {"status": "complete"})
        if mutation == "delete":
            (tmp_path / "inputs.json").unlink()
        elif mutation == "invalid-json":
            (tmp_path / "inputs.json").write_text("{")
        else:
            analysis.save(tmp_path / "inputs.json", {"original": False})
        return example_plan()

    monkeypatch.setattr(analysis, "execute_stage", execute)
    with pytest.raises(ValueError, match="analysis input"):
        analysis.agent("codex", tmp_path, tmp_path, "analysis", analysis.PLAN_SCHEMA,
                       "inspect", [tmp_path])
    assert analysis.read_json(tmp_path / "analysis-state.json")["status"] == "failed"


def test_reference_checkout_keeps_symlinks_as_inert_source_text(tmp_path):
    root = tmp_path / "reference"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "config", "core.symlinks", "true"], check=True)
    blob = subprocess.check_output(["git", "-C", str(root), "hash-object", "-w", "--stdin"],
                                   input=b"../outside").decode().strip()
    subprocess.run(["git", "-C", str(root), "update-index", "--add", "--cacheinfo",
                    f"120000,{blob},link"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-m", "symlink fixture"],
                   check=True, capture_output=True)
    env = dict(os.environ, GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="core.symlinks",
               GIT_CONFIG_VALUE_0="false")
    subprocess.run(["git", "-C", str(root), "checkout-index", "--all"], env=env, check=True)
    assert not (root / "link").is_symlink()
    assert (root / "link").read_text() == "../outside"
    assert subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"])
    subprocess.run(["git", "-C", str(root), "config", "core.symlinks", "false"], check=True)
    assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"])
    assert "link" in analysis.fingerprint(root)
