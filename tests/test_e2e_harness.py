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


def test_elastic_pair_with_a_tp_server():
    from e2e.matrix import ElasticPair

    pair = ElasticPair("sglang", "qwen35_4b", "vllm", "qwen35_4b", tp_a=2, gpu_b=1)
    assert pair.name == "elastic-sglang-qwen35_4b-tp2+vllm-qwen35_4b-gpu1"
    # Weekly's one elastic case: a starved TP rank, and two engines on a GPU.
    assert PROFILES["weekly"].elastic == (pair,)
    assert PROFILES["weekly"].engines == ("vllm", "sglang")
    assert (ElasticPair("vllm", "qwen05b", "sglang", "qwen35_4b").name
            == "elastic-vllm-qwen05b+sglang-qwen35_4b")
    case = Case(pair.engine_a, pair.model_a, "c1", tag="elastic", tp=pair.tp_a)
    assert case.name == "elastic-sglang-qwen35_4b-kv_c1-tp2"
    assert case.args()[-2:] == ("--tp-size", "2")


def test_host_gpu_is_the_runs_gpu_at_the_container_index(tmp_path):
    from e2e.run import LocalHost

    assert LocalHost(ENGINES["sglang"], "x", "all", tmp_path).host_gpu(1) == "1"
    assert LocalHost(ENGINES["sglang"], "x", "1,0", tmp_path).host_gpu(1) == "0"


def test_servers_are_pinned_to_the_selected_gpus(tmp_path):
    from e2e.run import Container, LocalHost, launch

    # A local run with --gpus 2,3: B goes to host GPU 3, A's ranks to 2 and 3,
    # whatever CUDA_VISIBLE_DEVICES the run inherited.
    local = LocalHost(ENGINES["vllm"], "x", "2,3", tmp_path)
    captured: dict[str, str] = {}
    local.start_server = lambda case_dir, env, argv: captured.update(env) or "1"  # type: ignore[method-assign]
    launch(local, Case("vllm", "qwen35_4b", "c1", tag="elastic"), "1", 1,
           extra_env=local.gpu_env(1))
    assert captured["CUDA_VISIBLE_DEVICES"] == local.host_gpu(1) == "3"
    assert local.gpu_env() == {"CUDA_VISIBLE_DEVICES": "2,3"}
    assert LocalHost(ENGINES["vllm"], "x", "all", tmp_path).gpu_env() == {}
    assert LocalHost(ENGINES["vllm"], "x", "all", tmp_path).gpu_env(1) == {
        "CUDA_VISIBLE_DEVICES": "1"}
    # A container sees the run's GPUs as 0, 1, ...
    docker = Container.__new__(Container)
    docker.gpus = "2,3"
    assert docker.gpu_env(1) == {"CUDA_VISIBLE_DEVICES": "1"}
    assert docker.gpu_env() == {}


TP_MISS_LOG = """\
[kvcached][WARNING][2026-10-14 10:00:00][patches.py:2680] kvcached could not back a prefill \
batch of 2 requests; returned them to the waiting queue and capped the next batch at 1 \
requests (1 misses so far): ElasticTokenToKVPoolAllocator cannot back 3521 tokens; available=1100
[kvcached][WARNING][2026-10-14 10:00:00][patches.py:2680] kvcached could not back a prefill \
batch of 2 requests; returned them to the waiting queue and capped the next batch at 1 \
requests (1 misses so far): a peer rank sharing this waiting queue could not back the batch
[2026-10-14 10:00:12 TP1] KV cache pool is full. Retract requests. #retracted_reqs: 1, \
#new_tokens_gained: 559, #mamba_num_gained: 3, #new_token_ratio: 0.1883 -> 0.5175
[2026-10-14 10:00:12 TP0] KV cache pool is full. Retract requests. #retracted_reqs: 1, \
#new_tokens_gained: 559, #mamba_num_gained: 3, #new_token_ratio: 0.1883 -> 0.5175
[kvcached][WARNING][2026-10-14 10:00:15][patches.py:2680] kvcached could not back a prefill \
batch of 1 requests; returned them to the waiting queue and capped the next batch at 1 \
requests (4 misses so far): a peer rank sharing this waiting queue could not back the batch
"""


def _failed_tp_miss_checks(text, tp=2):
    from e2e.elastic import tp_miss_checks

    return {c.name for c in tp_miss_checks(text, tp) if not c.ok}


def test_tp_miss_checks_read_both_ranks():
    failed = _failed_tp_miss_checks
    assert failed(TP_MISS_LOG) == set()
    lines = TP_MISS_LOG.splitlines(keepends=True)
    # Only one rank logged the first miss, and only one rank retracted.
    assert failed(lines[0] + lines[2]) == {"a_miss_agreed", "a_miss_on_one_rank"}
    # Both ranks logged their first miss, but not for the same batch.
    assert failed(lines[0] + lines[1].replace("batch of 2", "batch of 1")) == {"a_miss_agreed"}
    # One rank retracted once more than the other.
    assert failed(TP_MISS_LOG + lines[2]) == {"a_miss_agreed"}
    # Both ranks missed on their own: no rank followed a peer's verdict.
    assert failed(lines[0] * 2) == {"a_miss_on_one_rank"}
    # Nothing ran short.
    assert failed("[2026-10-14 10:00:00 TP0] Prefill batch, #new-seq: 1\n") == {"a_pool_short"}


def test_tp_miss_checks_take_the_lowest_lifetime_count_as_the_first_miss():
    failed = _failed_tp_miss_checks
    # The ranks missed three times before this part of the log; their fourth
    # miss is the first one here, and they agree on it.
    later = TP_MISS_LOG.replace("(1 misses so far)", "(3 misses so far)")
    assert failed(later) == set()
    # One rank's count is ahead of the other's: they did not agree on every
    # miss before, and the lines with the lowest count are one rank's.
    lines = later.splitlines(keepends=True)
    assert failed(lines[0] + lines[1].replace("(3 misses", "(5 misses")) == {"a_miss_agreed"}


def test_tp_miss_checks_read_only_the_contention_phase(tmp_path):
    from e2e.elastic import phase_log

    lines = TP_MISS_LOG.splitlines(keepends=True)
    # Calibration and A's solo load retracted once on each rank; nothing ran
    # short once B's load landed.
    log = tmp_path / "serve.log"
    log.write_text(lines[2] + lines[3])
    contention_start = log.stat().st_size
    log.write_text(lines[2] + lines[3] + "[2026-10-14 10:05:00 TP0] Decode batch, #running-req: 2\n")
    assert _failed_tp_miss_checks(log.read_text()) == set()  # the whole log passes
    assert _failed_tp_miss_checks(phase_log(log, contention_start)) == {"a_pool_short"}
    # The same misses after the boundary count.
    log.write_text(lines[2] + lines[3] + TP_MISS_LOG)
    assert _failed_tp_miss_checks(phase_log(log, contention_start)) == set()


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


def test_priority_pair_in_weekly():
    from e2e.matrix import PriorityPair

    pair = PriorityPair("vllm", "qwen05b", "vllm", "qwen35_4b")
    assert pair.name == "priority-vllm-qwen05b+vllm-qwen35_4b"
    assert (PriorityPair("sglang", "qwen05b", "vllm", "qwen05b", gpu=1).name
            == "priority-sglang-qwen05b+vllm-qwen05b-gpu1")
    # Weekly's priority case: an idle lower priority full of prefix cache
    # gives it back to a higher priority's burst, on one GPU (#585).
    assert PROFILES["weekly"].priority == (pair,)
    assert PROFILES["weekly"].engines == ("vllm", "sglang")
    case = Case(pair.engine_l, pair.model_l, "c1", tag="priority-lo")
    assert case.name == "priority-lo-vllm-qwen05b-kv_c1"


# What `kvctl list --json` prints next to the servers: one entry per
# instance, with the ledger state of each of its GPUs (#605, #608).
LEDGER_JSON = """\
[
 {"ipc": "e2e_1_lo", "limit_bytes": 4096, "used_bytes": 700,
  "gpus": [{"charged_bytes": 600, "watermark_bytes": 256, "shortfall_bytes": 0},
           {"charged_bytes": 100, "watermark_bytes": 64}]},
 {"ipc": "e2e_1_hi", "limit_bytes": 4096, "used_bytes": 50,
  "gpus": [{"charged_bytes": 50, "watermark_bytes": 0, "shortfall_bytes": 320}]},
 {"ipc": "another_server", "limit_bytes": 64, "used_bytes": 1,
  "gpus": [{"charged_bytes": 1, "shortfall_bytes": 999}]}
]
"""


def test_ledger_stats_reads_only_the_cases_instances():
    from e2e.priority import ledger_stats

    ipcs = ("e2e_1_lo", "e2e_1_hi")
    stats = ledger_stats(LEDGER_JSON, ipcs)
    # Fields are summed over the instance's GPUs; a missing field counts as 0.
    assert stats["e2e_1_lo"] == {"charged_bytes": 700, "watermark_bytes": 320,
                                 "shortfall_bytes": 0}
    assert stats["e2e_1_hi"] == {"charged_bytes": 50, "watermark_bytes": 0,
                                 "shortfall_bytes": 320}
    assert set(stats) == set(ipcs)
    # A failed kvctl run, an old kvcached or a dead server yield no stats, so
    # the ledger checks fail on missing evidence instead of crashing.
    assert ledger_stats("", ipcs) == {}
    assert ledger_stats("kvctl: boom", ipcs) == {}
    assert ledger_stats("[]", ipcs) == {}
    assert ledger_stats('{"ipc": "e2e_1_lo"}', ipcs) == {}
    assert ledger_stats('[{"ipc": "e2e_1_lo"}]', ipcs) == {
        "e2e_1_lo": {"charged_bytes": 0, "watermark_bytes": 0, "shortfall_bytes": 0}}


def test_set_priority_requires_an_accepted_publish():
    import subprocess

    from e2e.priority import set_priority

    def ct(code, out=b"", err=b""):
        return types.SimpleNamespace(
            exec=lambda cmd, timeout=None: subprocess.CompletedProcess(cmd, code, out, err))

    # kvctl prints the publish result and exits 0 even when it was rejected.
    accepted = b'{"accepted": true, "priority": 1, "status": "accepted"}\n'
    assert set_priority(ct(0, accepted), "e2e_1_hi", 1) == ""
    rejected = b'{"accepted": false, "status": "rejected"}'
    assert "rejected" in set_priority(ct(0, rejected), "e2e_1_hi", 1)
    assert "incomplete" in set_priority(ct(0, b'{"status": "incomplete"}'), "e2e_1_hi", 1)
    assert set_priority(ct(0, b""), "e2e_1_hi", 1) == "no kvctl output"
    assert set_priority(ct(0, b"Traceback (most recent call last)"), "e2e_1_hi", 1)
    assert "boom" in set_priority(ct(1, b"", b"boom"), "e2e_1_hi", 1)
    assert set_priority(ct(1), "e2e_1_hi", 1) == "kvctl failed"


def _failed_priority_checks(mem, shortfall_h, watermark_l):
    from e2e.priority import priority_checks

    mib = 1 << 20
    return {c.name for c in priority_checks({k: v * mib for k, v in mem.items()},
                                            shortfall_h * mib, watermark_l * mib) if not c.ok}


def test_priority_checks_need_the_ledger_and_the_release():
    failed = _failed_priority_checks
    # L grew 6000 MiB of cache and held it to the burst, H posted a shortfall,
    # L kept 300 MiB, and H grew past the 2500 MiB that were free.
    mem = {"l_idle": 100, "l_growth": 6000, "l_at_burst": 6100, "l_after": 400,
           "h_growth": 5000, "free_at_burst": 2500}
    assert failed(mem, 800, 320) == set()
    # The ledger evidence is required on both sides: H must have posted a
    # shortfall and L must have seen a watermark.
    assert failed(mem, 0, 320) == {"h_shortfall_posted"}
    assert failed(mem, 800, 0) == {"l_watermark_raised"}
    # L kept its cache: more than half of what it grew is still charged.
    assert failed(dict(mem, l_after=3200), 800, 320) == {"l_released_for_h"}
    # H grew into memory that was free anyway, not into what L gave back.
    assert failed(dict(mem, h_growth=2400), 800, 320) == {"h_reused_l_memory"}
    # L reached the burst with less than half of what it grew, e.g. with its
    # retention cap still on: the case must fail its pressure precondition.
    assert failed(dict(mem, l_at_burst=2900, l_after=200), 800, 320) == {"l_held_at_burst"}
