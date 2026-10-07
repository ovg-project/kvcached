# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Parent-owned cleanup survives scheduler exits that skip Python teardown."""

import os
import sys
import types
from typing import Any
from unittest.mock import Mock

import pytest

from kvcached.integration.sglang import shutdown
from kvcached.pool_registry import (
    clear_registered_kv_cache_pools,
    register_kv_cache_pool,
)


@pytest.fixture(params=[None, 60], ids=["legacy", "0.5.20"])
def serving(monkeypatch, tmp_path, request):
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    monkeypatch.setattr(shutdown, "SHM_DIR", str(tmp_path))
    clear_registered_kv_cache_pools()
    pools = [Mock(ipc_name="serve"), Mock(ipc_name="serve_g1000")]
    for pool in pools:
        register_kv_cache_pool(pool, integration="sglang")
        (tmp_path / pool.ipc_name).write_bytes(b"control")
    foreign = Mock(ipc_name="foreign")
    register_kv_cache_pool(foreign, integration="vllm")
    (tmp_path / "foreign").write_bytes(b"foreign")

    processes = {pid: Mock(pid=pid) for pid in (101, 102)}
    for process in processes.values():
        process.is_running.return_value = True
        process.status.return_value = "running"
    psutil: Any = types.ModuleType("psutil")
    psutil.Process = Mock(side_effect=lambda pid: processes[pid])
    psutil.STATUS_ZOMBIE = "zombie"
    psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    state = types.SimpleNamespace(calls=[], error=None, hold=False, launches=[], exits=[])

    def kill_process_tree(parent_pid, include_parent=True, skip_pid=None,
                          wait_timeout=request.param):
        state.calls.append((parent_pid, include_parent, skip_pid, wait_timeout))
        if include_parent:
            state.exits.append(sorted(p.name for p in tmp_path.iterdir()))
        if not state.hold:
            for pid, process in processes.items():
                if pid != skip_pid:
                    process.is_running.return_value = False
        if state.error is not None:
            raise state.error
        return "upstream result"

    class Scheduler:
        def get_init_info(self):
            return {"status": "ready", "max_req_input_len": 512}

    class Engine:
        @classmethod
        def _launch_subprocesses(cls, *args, **kwargs):
            state.launches.append((cls, args, kwargs))
            return (None, None, None, types.SimpleNamespace(
                scheduler_infos=[Scheduler().get_init_info()],
                all_child_pids=list(processes),
            ), None)

    engine = types.SimpleNamespace(Engine=Engine, kill_process_tree=kill_process_tree)
    scheduler = types.SimpleNamespace(Scheduler=Scheduler)
    tokenizer = types.SimpleNamespace(kill_process_tree=kill_process_tree)
    patch = shutdown.SGLangShutdownPatch()
    assert patch.patch_shutdown(engine, scheduler, tokenizer)
    assert patch.patch_shutdown(engine, scheduler, tokenizer)
    state.engine = engine
    state.scheduler = scheduler
    state.tokenizer = tokenizer
    state.processes = processes
    state.directory = tmp_path
    state.default_wait = request.param
    state.psutil = psutil
    yield state
    clear_registered_kv_cache_pools()


def test_readiness_reports_only_sglang_pools_and_preserves_launch_result(serving):
    info = serving.scheduler.Scheduler().get_init_info()
    assert info[shutdown._SEGMENTS_KEY] == ["serve", "serve_g1000"]

    BaseEngine = serving.engine.Engine

    class Subclass(BaseEngine):  # type: ignore[misc, valid-type]
        pass

    result = Subclass._launch_subprocesses("args", custom=True)
    assert serving.launches == [(Subclass, ("args",), {"custom": True})]
    assert result[3].scheduler_infos == [{"status": "ready", "max_req_input_len": 512}]


def test_engine_shutdown_releases_all_pools_after_workers_exit(serving):
    serving.engine.Engine._launch_subprocesses()
    for _ in range(2):
        result = serving.engine.kill_process_tree(os.getpid(), include_parent=False,
                                                  wait_timeout=60)
        assert result == "upstream result"
        assert sorted(p.name for p in serving.directory.iterdir()) == ["foreign"]
    assert serving.calls == [(os.getpid(), False, None, 60)] * 2


def test_http_self_exit_waits_for_workers_and_cleans_before_parent_exit(serving):
    serving.engine.Engine._launch_subprocesses()
    assert serving.tokenizer.kill_process_tree(os.getpid()) == "upstream result"
    assert serving.calls == [
        (os.getpid(), False, None, 60),
        (os.getpid(), True, None, serving.default_wait),
    ]
    assert serving.exits == [["foreign"]]
    assert not (serving.directory / "serve").exists()
    assert not (serving.directory / "serve_g1000").exists()


def test_live_rank_keeps_every_segment_until_retry(serving):
    serving.engine.Engine._launch_subprocesses()
    serving.engine.kill_process_tree(os.getpid(), False, skip_pid=102)
    assert (serving.directory / "serve").exists()
    assert (serving.directory / "serve_g1000").exists()
    serving.engine.kill_process_tree(os.getpid(), False)
    assert not (serving.directory / "serve").exists()
    assert not (serving.directory / "serve_g1000").exists()


def test_unconfirmed_exit_keeps_segments(serving):
    serving.engine.Engine._launch_subprocesses()
    serving.hold = True
    serving.processes[101].status.side_effect = PermissionError("denied")
    serving.engine.kill_process_tree(os.getpid(), False)
    assert (serving.directory / "serve").exists()
    serving.hold = False
    serving.engine.kill_process_tree(os.getpid(), False)
    assert not (serving.directory / "serve").exists()


def test_same_name_replacement_is_preserved(serving):
    serving.engine.Engine._launch_subprocesses()
    segment = serving.directory / "serve"
    segment.unlink()
    segment.write_bytes(b"replacement")
    serving.engine.kill_process_tree(os.getpid(), False)
    assert segment.read_bytes() == b"replacement"
    assert not (serving.directory / "serve_g1000").exists()


def test_file_absent_at_ready_is_not_claimed_later(serving):
    segment = serving.directory / "serve"
    segment.unlink()
    serving.engine.Engine._launch_subprocesses()
    segment.write_bytes(b"later instance")
    serving.engine.kill_process_tree(os.getpid(), False)
    assert segment.read_bytes() == b"later instance"


def test_failed_unlink_is_retried_without_recapturing_files(serving, monkeypatch):
    serving.engine.Engine._launch_subprocesses()
    original_unlink = os.unlink

    def fail_group(path):
        if str(path).endswith("_g1000"):
            raise PermissionError("denied")
        original_unlink(path)

    monkeypatch.setattr(os, "unlink", fail_group)
    serving.engine.kill_process_tree(os.getpid(), False)
    assert not (serving.directory / "serve").exists()
    assert (serving.directory / "serve_g1000").exists()
    (serving.directory / "serve").write_bytes(b"replacement")
    monkeypatch.setattr(os, "unlink", original_unlink)
    serving.engine.kill_process_tree(os.getpid(), False)
    assert (serving.directory / "serve").read_bytes() == b"replacement"
    assert not (serving.directory / "serve_g1000").exists()


def test_shutdown_error_is_preserved_and_live_workers_protect_segments(serving):
    serving.engine.Engine._launch_subprocesses()
    serving.error = RuntimeError("upstream timeout")
    serving.hold = True
    with pytest.raises(RuntimeError, match="upstream timeout"):
        serving.engine.kill_process_tree(os.getpid(), False)
    assert (serving.directory / "serve").exists()
    serving.hold = False
    with pytest.raises(RuntimeError, match="upstream timeout"):
        serving.engine.kill_process_tree(os.getpid(), False)
    assert not (serving.directory / "serve").exists()


@pytest.mark.parametrize("case", ["disabled", "unowned", "other-parent", "forked"])
def test_unowned_calls_keep_native_behavior(serving, monkeypatch, case):
    if case != "unowned":
        serving.engine.Engine._launch_subprocesses()
    pid = os.getpid()
    if case == "disabled":
        monkeypatch.setenv("ENABLE_KVCACHED", "false")
    elif case == "other-parent":
        pid += 1
    elif case == "forked":
        monkeypatch.setattr(os, "getpid", lambda: pid + 1)
    serving.tokenizer.kill_process_tree(pid)
    assert serving.calls == [(pid, True, None, serving.default_wait)]
    assert (serving.directory / "serve").exists()


def test_capture_failure_does_not_break_startup_or_claim_files(serving, monkeypatch):
    monkeypatch.setattr(shutdown, "IPCSegmentCleanup", Mock(side_effect=PermissionError("denied")))
    serving.engine.Engine._launch_subprocesses()
    serving.tokenizer.kill_process_tree(os.getpid())
    assert serving.calls == [(os.getpid(), True, None, serving.default_wait)]
    assert (serving.directory / "serve").exists()


def test_cleanup_error_does_not_prevent_original_parent_exit(serving, monkeypatch):
    serving.engine.Engine._launch_subprocesses()
    monkeypatch.setattr(shutdown._ShutdownOwner, "cleanup", Mock(side_effect=OSError("failed")))
    assert serving.tokenizer.kill_process_tree(os.getpid()) == "upstream result"
    assert [call[1] for call in serving.calls] == [False, True]
    assert (serving.directory / "serve").exists()


def test_child_timeout_still_preserves_original_parent_exit(serving):
    serving.engine.Engine._launch_subprocesses()
    serving.hold = True
    serving.error = RuntimeError("timeout")
    with pytest.raises(RuntimeError, match="timeout"):
        serving.tokenizer.kill_process_tree(os.getpid())
    assert [call[1] for call in serving.calls] == [False, True]
    assert (serving.directory / "serve").exists()


def test_none_parent_keeps_children_only_semantics(serving):
    serving.engine.Engine._launch_subprocesses()
    serving.engine.kill_process_tree(None)
    assert serving.calls == [(None, True, None, serving.default_wait)]
    assert not (serving.directory / "serve").exists()


def test_no_local_children_does_not_claim_segments(serving):
    serving.processes.clear()
    serving.engine.Engine._launch_subprocesses()
    serving.tokenizer.kill_process_tree(os.getpid())
    assert serving.calls == [(os.getpid(), True, None, serving.default_wait)]
    assert (serving.directory / "serve").exists()


def test_disabled_startup_does_not_publish_or_capture_segments(serving, monkeypatch):
    monkeypatch.setenv("ENABLE_KVCACHED", "false")
    assert shutdown._SEGMENTS_KEY not in serving.scheduler.Scheduler().get_init_info()
    serving.engine.Engine._launch_subprocesses()
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    serving.tokenizer.kill_process_tree(os.getpid())
    assert serving.calls == [(os.getpid(), True, None, serving.default_wait)]
    assert (serving.directory / "serve").exists()
