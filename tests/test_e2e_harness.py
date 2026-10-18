# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""GPU-free checks of the e2e driver's parsing and comparison logic."""

import sys
import time
import types

import pytest
from e2e import client as cl
from e2e.matrix import ENGINES, PROFILES, Case, Group
from e2e.run import CaseResult, compare_with_native, ipc_name, parse_log

VLLM_LOG = """\
[kvcached][INFO] Successfully patched vllm: nixl_connector_compat, elastic_block_pool, \
model_runner_v2, kv_layout_v2, gpu_worker
Using kvcached process-local KV capacity: budget=1 bytes, weights=2 bytes, available=4096 bytes
GPU KV cache size: 1,142,198 tokens
Setting attention block size to 1024 tokens (was 784) so the KV unit (4194304 bytes) tiles the \
4194304-byte kvcached page
"""

SGLANG_LOG = """\
[kvcached][INFO] Successfully patched sglang: elastic_allocator, elastic_memory_pool
KV Cache is allocated. #tokens: 5454455, K size: 39.61 GB, V size: 39.61 GB
max_total_num_tokens=5454455
Traceback (most recent call last):
"""


def test_parse_vllm_log():
    info = parse_log(VLLM_LOG, ENGINES["vllm"])
    assert info["patched"] == ["nixl_connector_compat", "elastic_block_pool", "model_runner_v2",
                               "kv_layout_v2", "gpu_worker"]
    assert info["capacity"] == 4096
    assert info["tokens"] == [1142198]
    assert info["kvcached_lines"] == 1
    assert info["errors"] == []
    assert info["block_size"] == (1024, 784)


def test_parse_sglang_log_reports_errors():
    info = parse_log(SGLANG_LOG, ENGINES["sglang"])
    assert info["capacity"] == 5454455
    assert info["tokens"] == [5454455]
    assert info["errors"] == [r"Traceback \(most recent call last\)"]


def test_kvcached_cases_run_before_native():
    # On vLLM the native case takes the attention block size kvcached chose.
    assert [c.layout for c in Group("vllm", "qwen05b").cases()] == ["c1", "c0", None]
    assert [c.layout for c in Group("sglang", "qwen05b").cases()] == ["c1", "c0", None]
    assert Group("vllm", "qwen05b").cases()[-1].name == "vllm-qwen05b-native"
    assert Case("vllm", "qwen05b", "c1", tag="elastic").name == "elastic-vllm-qwen05b-kv_c1"


def test_case_arguments_for_tp_and_fp8_kv():
    case = Case("sglang", "qwen35_9b", "c0", tp=2, fp8_kv=True)
    assert case.name == "sglang-qwen35_9b-kv_c0-tp2-fp8kv"
    assert case.args()[-4:] == ("--tp-size", "2", "--kv-cache-dtype", "fp8_e4m3")
    assert Case("vllm", "qwen05b", None, tp=2).args()[-2:] == ("--tensor-parallel-size", "2")
    # Models an engine cannot serve get no group, and profiles list their engines.
    weekly = PROFILES["weekly"]
    assert ("sglang", "gemma4_12b") not in {(g.engine, g.model) for g in weekly.groups}
    assert weekly.engines == ("vllm", "sglang")


def test_ipc_name_is_shm_safe():
    assert ipc_name("vllm-qwen05b-kv_c1", "1007") == "e2e_1007_vllm_qwen05b_kv_c1"


def test_first_divergence():
    assert cl.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert cl.first_divergence([1, 2, 3], [1, 5, 3]) == {"index": 1, "a": 2, "b": 5}
    assert cl.first_divergence([1, 2], [1, 2, 3]) == {"index": 2, "len_a": 2, "len_b": 3}


def _case(name, tokens, cached):
    res = CaseResult(name)
    for phase in ("gen_seq", "apc_0", "apc_1", "apc_2"):
        res.phases[phase] = cl.Phase(phase, 1, [
            cl.Result(200, list(tokens), "", 10, cached, 0.1)])
    return res


def test_compare_with_native():
    kv = _case("kv", [1, 2, 3], 1024)
    assert all(c.ok for c in compare_with_native(kv, _case("native", [1, 2, 3], 1024)))

    failed = {c.name for c in compare_with_native(kv, _case("native", [1, 9, 3], 1280))
              if not c.ok}
    assert failed == {"gen_seq_tokens_equal", "apc_0_tokens_equal",
                      "apc_1_tokens_equal", "apc_2_tokens_equal", "apc_0_cached_tokens_equal",
                      "apc_1_cached_tokens_equal", "apc_2_cached_tokens_equal"}


def test_tokens_differing_in_fewer_than_half_of_the_prompts_pass():
    def case(name, differing):
        res = CaseResult(name)
        res.phases["gen_seq"] = cl.Phase("gen_seq", 1, [
            cl.Result(200, [1, 9] if i < differing else [1, 2], "", 10, 0, 0.1)
            for i in range(9)])
        return res

    native = case("native", 0)
    check = compare_with_native(case("kv", 4), native)[0]
    assert check.ok and check.detail.startswith("4 of 9 prompts differ: prompt 0:")
    assert not compare_with_native(case("kv", 5), native)[0].ok


def test_read_segments_reads_only_the_servers_segments(tmp_path):
    from e2e.elastic import SEGMENT, read_segments

    (tmp_path / "e2e_a").write_bytes(SEGMENT.pack(100, 7, 0) + b"KVCOFD01")  # since #546
    (tmp_path / "e2e_a_g1000").write_bytes(SEGMENT.pack(100, 2, 0))
    (tmp_path / "e2e_ab").write_bytes(SEGMENT.pack(100, 5, 0))  # another server
    (tmp_path / "e2e_b").write_bytes(SEGMENT.pack(0, 3, 0))  # no limit
    assert read_segments(["e2e_a", "e2e_b"], str(tmp_path)) == {"e2e_a": 7, "e2e_a_g1000": 2}


def test_sglang_flush_retries_while_finished_requests_are_pending(monkeypatch):
    statuses = [400, 400, 200]
    monkeypatch.setattr(cl, "_post", lambda *args: (statuses.pop(0), {}, 0.0))
    monkeypatch.setattr(cl.time, "sleep", lambda seconds: None)
    assert cl.SGLangClient("127.0.0.1", 1).reset_prefix_cache()
    assert statuses == []
    monkeypatch.setattr(cl, "_post", lambda *args: (500, {}, 0.0))
    assert not cl.SGLangClient("127.0.0.1", 1).reset_prefix_cache()


def test_servers_start_offline():
    from e2e.run import server_env

    for engine in ENGINES:
        for layout in ("c1", None):
            assert server_env(Case(engine, "qwen05b", layout), "e2e_x")["HF_HUB_OFFLINE"] == "1"


def test_fetch_models_falls_back_to_the_cache(monkeypatch, capsys):

    from e2e.run import FETCH_MODELS

    cached = {"cached/model"}

    def snapshot_download(hf_id, local_files_only=False):
        if hf_id == "online/model":
            return "/hub/online"
        if local_files_only and hf_id in cached:
            return "/hub/cached"
        raise OSError("429 Too Many Requests")

    sleeps: list[float] = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(sys, "argv", ["-c", "online/model", "cached/model", "missing/model"])
    with pytest.raises(SystemExit) as exit_info:
        exec(FETCH_MODELS, {})
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert "online/model at /hub/online" in out and "cached/model cached at /hub/cached" in out
    assert out.strip().endswith("not fetched: missing/model")
    assert sleeps == [60, 120, 180]  # only the model that is not cached is retried
