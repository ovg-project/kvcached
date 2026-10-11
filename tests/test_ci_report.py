# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""GPU-free checks of tools/ci/report.py, which writes the GPU CI results issue."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

_SPEC = importlib.util.spec_from_file_location(
    "ci_report", Path(__file__).resolve().parents[1] / "tools" / "ci" / "report.py")
assert _SPEC and _SPEC.loader
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)

SETUP: dict[str, Any] = {
    "profile": "nightly", "kvcached_sha": "a" * 40,
    "gpus": [{"name": "NVIDIA L4", "memory": "23034 MiB", "driver": "580.178.04"}],
    "engines": {"vllm": {"image": "vllm/vllm-openai:v0.30.0",
                         "digest": "vllm/vllm-openai@sha256:0123456789abcdef", "engine": "0.30.0",
                         "kvcached": "0.1.6", "torch": "2.13.0", "cuda": "13.0",
                         "env": {"VLLM_USE_V2_MODEL_RUNNER": "1"}, "server": "vllm serve <model>"}},
    "model_revisions": {"Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775"},
    "workload": ["gen_seq: 9 prompts"], "checks": {"server_ready": "answers /health"},
    "cases": [
        {"name": "vllm-qwen05b-kv_c1", "engine": "vllm", "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
         "layout": "c1", "tp": 1, "kv_cache_dtype": None, "model_args": ["--dtype", "bfloat16"],
         "extra_args": [],
         "env": {"VLLM_USE_V2_MODEL_RUNNER": "1", "ENABLE_KVCACHED": "true",
                 "KVCACHED_CONTIGUOUS_LAYOUT": "true"}},
        {"name": "vllm-qwen05b-kv_c0", "engine": "vllm", "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
         "layout": "c0", "tp": 1, "kv_cache_dtype": None, "model_args": ["--dtype", "bfloat16"],
         "extra_args": [],
         "env": {"VLLM_USE_V2_MODEL_RUNNER": "1", "ENABLE_KVCACHED": "true",
                 "KVCACHED_CONTIGUOUS_LAYOUT": "false"}},
        {"name": "vllm-qwen05b-native", "engine": "vllm", "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
         "layout": None, "tp": 1, "kv_cache_dtype": None, "model_args": ["--dtype", "bfloat16"],
         "extra_args": ["--block-size", "1024"],
         "env": {"VLLM_USE_V2_MODEL_RUNNER": "1", "ENABLE_KVCACHED": "false"}},
    ],
    "elastic": [],
}


def args(**kw):
    base = dict(run_url="https://github.com/ovg-project/kvcached/actions/runs/1", sha="b" * 40,
                subject="fix: something", prev="", artifact_url="", pr=None)
    base.update(kw)
    return argparse.Namespace(**base)


def write(path, cases, setup=SETUP):
    path.mkdir(parents=True, exist_ok=True)
    (path / "setup.json").write_text(json.dumps(setup))
    (path / "summary.json").write_text(json.dumps({"cases": cases}))
    return path


def sglang_part() -> dict[str, Any]:
    """The setup of the SGLang half of a nightly run, on the second GPU."""
    setup = json.loads(json.dumps(SETUP))
    setup["engines"] = {"sglang": {"image": "lmsysorg/sglang:v0.5.20", "engine": "0.5.20",
                                   "env": {}, "server": "python3 -m sglang.launch_server"}}
    setup["cases"] = [dict(c, name=c["name"].replace("vllm", "sglang"), engine="sglang")
                      for c in SETUP["cases"]]
    return setup


def test_splice_keeps_other_text_and_orders_sections():
    body = report.splice("Notes by hand.\n", "perf", "## Perf\n")
    body = report.splice(body, "nightly", "## Nightly v1\n")
    body = report.splice(body, "weekly", "## Weekly\n")
    body = report.splice(body, "nightly", "## Nightly v2\n")
    assert body.startswith("Notes by hand.")
    assert "Nightly v1" not in body
    assert body.index("Nightly v2") < body.index("## Weekly") < body.index("## Perf")
    assert body.count(report.marker("nightly")) == 1


def test_correctness_comment_lists_failed_cases_with_setup(tmp_path):
    write(tmp_path, [
        {"name": "vllm-qwen05b-kv_c1", "ok": True, "checks": [{"name": "server_ready", "ok": True}]},
        {"name": "vllm-qwen05b-native", "ok": False,
         "checks": [{"name": "gen_seq_ok", "ok": False, "detail": "500 boom | x"}]},
    ])
    text = report.correctness_comment("nightly", [tmp_path], args(prev=f"{'c' * 40}:success"))
    assert "FAIL (1 of 2 cases)" in text
    assert "| vllm-qwen05b-native |" in text and "| vllm-qwen05b-kv_c1 |" not in text
    assert "native, TP 1, KV model dtype, `--block-size 1024`" in text
    assert "500 boom \\| x" in text
    assert f"compare/{'c' * 40}...{'b' * 40}" in text and "(PASS)" in text


def test_correctness_comment_when_nothing_ran(tmp_path):
    text = report.correctness_comment("nightly", [tmp_path / "all"], args())
    assert "DID NOT RUN" in text and "The tests did not finish" in text


def test_parts_on_two_gpus_make_one_comment_and_setup(tmp_path):
    ok = [{"name": "vllm-qwen05b-kv_c1", "ok": True, "checks": []}]
    vllm = write(tmp_path / "vllm", ok)
    sglang = write(tmp_path / "sglang", [dict(ok[0], name="sglang-qwen05b-kv_c1")], sglang_part())
    text = report.correctness_comment("nightly", [sglang, vllm], args())
    assert "· PASS" in text and "All 2 cases passed." in text
    assert "2x NVIDIA L4" in text and "SGLang and vLLM cases run at the same time" in text
    setup = report.merge_setups([("vLLM", SETUP), ("SGLang", sglang_part())])
    section = report.correctness_section("nightly", setup, args())
    assert "| vllm-qwen05b-kv_c1 |" in section and "| sglang-qwen05b-kv_c1 |" in section
    assert "`lmsysorg/sglang:v0.5.20`" in section and "`vllm/vllm-openai:v0.30.0`" in section


def test_a_part_that_did_not_finish_fails_the_run(tmp_path):
    vllm = write(tmp_path / "vllm", [{"name": "vllm-qwen05b-kv_c1", "ok": True, "checks": []}])
    text = report.correctness_comment("nightly", [vllm, tmp_path / "sglang"], args())
    assert "FAIL (0 of 1 cases; the SGLang cases did not finish)" in text
    assert "The SGLang cases did not finish" in text


def test_pull_request_comment_carries_its_setup(tmp_path):
    run = write(tmp_path / "all", [{"name": "vllm-qwen05b-kv_c1", "ok": True, "checks": []}])
    text = report.correctness_comment("weekly", [run], args(pr=12))
    assert "<details><summary>Setup of every case</summary>" in text
    assert "On request in a pull request" in text and "from the pull request merged into main" in text
    # No issue description to point at: the workload and checks are spelled out.
    assert "| check | passes when |" in text and "#nightly-correctness" not in text
    assert "\n## Weekly correctness" not in text


def test_correctness_section_describes_every_case(tmp_path):
    # An older setup.json stores the engine env as [key, value] pairs.
    setup = json.loads(json.dumps(SETUP))
    setup["engines"]["vllm"]["env"] = [["VLLM_USE_V2_MODEL_RUNNER", "1"]]
    assert report.correctness_section("nightly", setup, args()) == \
        report.correctness_section("nightly", SETUP, args())
    section = report.correctness_section("nightly", SETUP, args())
    assert "Daily at 08:00 UTC" in section and "1x NVIDIA L4" in section
    assert "`Qwen/Qwen2.5-0.5B-Instruct` @ [7ae5576]" in section
    for case in SETUP["cases"]:
        assert f"| {case['name']} |" in section
    # The env shared by kvcached cases, and the one that varies by layout.
    assert "kvcached cases add `ENABLE_KVCACHED=true`" in section
    assert "`KVCACHED_CONTIGUOUS_LAYOUT` follows the KV layout" in section
    assert "native cases add `ENABLE_KVCACHED=false`" in section
    # Other profiles point at the nightly section's workload and checks.
    weekly = report.correctness_section("weekly", SETUP, args())
    assert "| check | passes when |" not in weekly and "(#nightly-correctness)" in weekly


def test_perf_comment(tmp_path):
    (tmp_path / "perf").mkdir()
    table = [{"model": "m", "concurrency": 1, "metric": metric, "native": 100.0,
              "kvcached": 90.0, "ratio": 0.9}
             for metric in ("output_throughput", "mean_tpot_ms", "p99_tpot_ms", "mean_ttft_ms")]
    runs = [{"model": "m", "arm": arm, "run": i, "idle_gpu_mib": mib}
            for i, (arm, mib) in enumerate((("native", 70), ("kvcached", 50)))]
    (tmp_path / "perf" / "summary.json").write_text(
        json.dumps({"table": table, "runs": runs, "errors": []}))
    text = report.perf_comment(tmp_path, args())
    assert "| m | 1 | 100 | 90 | 0.90 | 0.90 | 0.90 | 0.90 |" in text
    assert "native 70 MiB, kvcached 50 MiB" in text
    # The Hopper correctness results are missing, so the run did not pass.
    assert "· FAIL" in text and "Hopper correctness: no results." in text


def test_case_flags_show_the_override_once():
    s = {"model_args": ["--dtype", "bfloat16", "--max-num-seqs", "32", "--trust-remote-code"],
         "extra_args": ["--max-num-seqs", "128"]}
    assert report.model_flags(s) == "`--dtype bfloat16 --trust-remote-code --max-num-seqs 128`"
