# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Parent-owned cleanup survives scheduler exits that skip Python teardown."""

import os
import pickle
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import Mock

import pytest

from kvcached.integration.sglang import shutdown
from kvcached.pool_registry import (
    clear_registered_kv_cache_pools,
    register_kv_cache_pool,
)


@pytest.fixture(params=[False, True], ids=["direct", "dp-controller"])
def dp_mode(request):
    return request.param


@pytest.fixture(params=[None, 60], ids=["legacy", "0.5.20"])
def serving(monkeypatch, tmp_path, request, dp_mode):
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    monkeypatch.setattr(shutdown, "register_owner_sigterm_cleanup", lambda callback: True)
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
    state = types.SimpleNamespace(
        calls=[], error=None, hold=False, launches=[], exits=[], worker_infos=None,
        tp_size=1, received=[], sent=[],
    )

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

    # The controller consumes worker readiness and builds a new message in
    # both 0.5.12.post1 and 0.5.20. Do not forward arbitrary worker fields here:
    # bypassing this boundary hid the DP cleanup regression in the old tests.
    controller: Any = types.ModuleType("sglang.srt.managers.data_parallel_controller")
    def pipe(**kwargs):
        def recv():
            info = (Scheduler().get_init_info() if state.worker_infos is None
                    else next(state.worker_infos))
            state.received.append(info)
            return info

        return Mock(recv=recv), Mock()

    controller.mp = types.SimpleNamespace(Pipe=pipe)
    original_mp = controller.mp

    def run_data_parallel_controller_process(server_args, port_args, pipe_writer,
                                             run_scheduler_process_func=None):
        def launch_group(_rank):
            return [controller.mp.Pipe(duplex=False)[0].recv() for _ in range(state.tp_size)]

        # Upstream starts one thread per DP group and waits for all groups.
        with ThreadPoolExecutor(max_workers=2) as executor:
            groups = list(executor.map(launch_group, range(2)))
        info = {
            "status": "ready",
            "max_req_input_len": groups[0][0]["max_req_input_len"],
            "scheduler_pids": list(processes),
        }
        if request.param == 60:
            info["startup_time"] = {"model_load": 1.5}
        state.sent.append(info)
        pipe_writer.send(info)
        return "controller result"

    controller.run_data_parallel_controller_process = run_data_parallel_controller_process
    for module_name in ("sglang", "sglang.srt", "sglang.srt.managers"):
        module = types.ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(sys.modules, controller.__name__, controller)

    class SchedulerInitResult:
        def __init__(self, scheduler_infos, all_child_pids):
            self.scheduler_infos = scheduler_infos
            self.all_child_pids = all_child_pids
            self.wait_for_ready = lambda: None

    class Engine:
        @classmethod
        def _launch_subprocesses(cls, *args, **kwargs):
            state.launches.append((cls, args, kwargs))
            if dp_mode:
                writer = Mock()
                engine.run_data_parallel_controller_process(None, None, writer)
                infos = [writer.send.call_args.args[0]]
            else:
                infos = [Scheduler().get_init_info()]
            result = SchedulerInitResult(infos, list(processes))
            result.wait_for_ready()
            return (None, None, None, result, None)

    engine = types.SimpleNamespace(
        Engine=Engine, SchedulerInitResult=SchedulerInitResult,
        kill_process_tree=kill_process_tree,
        run_data_parallel_controller_process=run_data_parallel_controller_process,
    )
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
    state.dp_mode = dp_mode
    state.controller = controller
    state.original_mp = original_mp
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
    expected: dict[str, Any] = {"status": "ready", "max_req_input_len": 512}
    if serving.dp_mode:
        expected["scheduler_pids"] = [101, 102]
        if serving.default_wait == 60:
            expected["startup_time"] = {"model_load": 1.5}
    assert result[3].scheduler_infos == [expected]


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
        (os.getpid(), False, None, serving.default_wait),
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


@pytest.mark.parametrize("dp_mode", [True], indirect=True)
class TestDPControllerReadiness:
    def test_all_dp_tp_ranks_and_pools_are_forwarded_without_mutating_messages(self, serving):
        serving.tp_size = 2
        infos = [
            {"status": "ready", "max_req_input_len": 512, "_kvcached_ipc_names": names}
            for names in [
                ["serve", "serve_g1000", "dp0_tp0"],
                ["serve", "dp0_tp1"],
                ["serve_g1000", "dp1_tp0"],
                ["dp1_tp1", "dp1_mamba"],
            ]
        ]
        serving.worker_infos = iter(infos)
        expected_names = [
            "dp0_tp0", "dp0_tp1", "dp1_mamba", "dp1_tp0", "dp1_tp1", "serve", "serve_g1000",
        ]
        for name in expected_names:
            (serving.directory / name).write_bytes(b"control")

        # Both DP threads reach recv together, exercising the shared collector.
        barrier = threading.Barrier(2, timeout=5)
        original_pipe = serving.controller.mp.Pipe

        def pipe(**kwargs):
            reader, writer = original_pipe(**kwargs)
            original_recv = reader.recv

            def recv():
                barrier.wait()
                return original_recv()

            reader.recv = recv
            return reader, writer

        serving.controller.mp.Pipe = pipe
        writer = Mock()
        assert serving.engine.run_data_parallel_controller_process(
            None, None, writer) == "controller result"
        forwarded = writer.send.call_args.args[0]
        assert forwarded["_kvcached_ipc_names"] == expected_names
        assert {key: value for key, value in forwarded.items()
                if key != "_kvcached_ipc_names"} == serving.sent[0]
        assert "_kvcached_ipc_names" not in serving.sent[0]
        assert all("_kvcached_ipc_names" in info for info in infos)
        assert serving.controller.mp is serving.original_mp

        serving.worker_infos = iter(infos)
        serving.engine.Engine._launch_subprocesses()
        serving.engine.kill_process_tree(os.getpid(), False, skip_pid=102)
        assert all((serving.directory / name).exists() for name in expected_names)
        serving.engine.kill_process_tree(os.getpid(), False)
        assert sorted(p.name for p in serving.directory.iterdir()) == ["foreign"]

    def test_spawn_target_keeps_importable_identity(self, serving):
        target = serving.engine.run_data_parallel_controller_process
        assert pickle.loads(pickle.dumps(target)) is target

    @pytest.mark.parametrize("stage", ["receive", "send", "entrypoint"])
    def test_controller_error_restores_module_and_global_pipe(self, serving, monkeypatch, stage):
        import multiprocessing

        native_pipe = multiprocessing.Pipe
        writer = Mock()
        error = RuntimeError("controller startup failed")
        if stage == "receive":
            serving.controller.mp.Pipe = Mock(return_value=(Mock(recv=Mock(side_effect=error)), Mock()))
        elif stage == "send":
            writer.send.side_effect = error
        else:
            original = serving.controller.run_data_parallel_controller_process

            def fail(server_args, port_args, pipe_writer, run_scheduler_process_func=None):
                raise error

            monkeypatch.setattr(serving.controller, "run_data_parallel_controller_process", fail)
            assert original is not fail

        with pytest.raises(RuntimeError, match="controller startup failed"):
            serving.engine.run_data_parallel_controller_process(None, None, writer)
        assert serving.controller.mp is serving.original_mp
        assert multiprocessing.Pipe is native_pipe
        assert (serving.directory / "serve").exists()

    def test_disabled_controller_preserves_arguments_and_native_objects(self, serving, monkeypatch):
        monkeypatch.setenv("ENABLE_KVCACHED", "false")
        original = Mock(return_value="native result")
        monkeypatch.setattr(serving.controller, "run_data_parallel_controller_process", original)
        writer, callback = object(), object()
        assert serving.engine.run_data_parallel_controller_process(
            "server", "ports", pipe_writer=writer,
            run_scheduler_process_func=callback) == "native result"
        original.assert_called_once_with("server", "ports", pipe_writer=writer,
                                         run_scheduler_process_func=callback)
        assert serving.controller.mp is serving.original_mp

    def test_missing_worker_metadata_does_not_claim_unreported_files(self, serving):
        serving.worker_infos = iter([
            {"status": "ready", "max_req_input_len": 512},
            {"status": "ready", "max_req_input_len": 512, "_kvcached_ipc_names": []},
        ])
        serving.engine.Engine._launch_subprocesses()
        serving.engine.kill_process_tree(os.getpid(), False)
        assert sorted(p.name for p in serving.directory.iterdir()) == [
            "foreign", "serve", "serve_g1000",
        ]

    def test_successive_controllers_have_independent_segment_sets(self, serving):
        for names in (["first"], ["second", "second_mamba"]):
            serving.worker_infos = iter([
                {"status": "ready", "max_req_input_len": 512, "_kvcached_ipc_names": names},
                {"status": "ready", "max_req_input_len": 512},
            ])
            writer = Mock()
            serving.engine.run_data_parallel_controller_process(None, None, writer)
            assert writer.send.call_args.args[0]["_kvcached_ipc_names"] == names
            assert serving.controller.mp is serving.original_mp


@pytest.mark.parametrize("entrypoint", ["sglang.launch_server", "sglang.cli.serve", "__main__",
                                        "sglang.srt.entrypoints.http_server"])
@pytest.mark.parametrize("include_parent", [False, True])
def test_http_alias_bound_before_launch_is_cleaned(serving, monkeypatch, entrypoint, include_parent):
    # Model the actual frontends: from-import before load_plugins(), then a
    # classmethod launch without ever constructing Engine, finally a local alias.
    original = serving.engine.kill_process_tree.__wrapped__
    module: Any = types.ModuleType(entrypoint)
    module.kill_process_tree = original
    module.__spec__ = types.SimpleNamespace(name="sglang.launch_server")
    monkeypatch.setitem(sys.modules, entrypoint, module)
    serving.engine.Engine._launch_subprocesses()
    wrapped = module.kill_process_tree
    serving.engine.Engine._launch_subprocesses()
    assert module.kill_process_tree is wrapped
    module.kill_process_tree(os.getpid(), include_parent=include_parent)
    assert sorted(p.name for p in serving.directory.iterdir()) == ["foreign"]
    expected = [(os.getpid(), False, None, serving.default_wait)]
    if include_parent:
        expected.append((os.getpid(), True, None, serving.default_wait))
        assert serving.exits == [["foreign"]]
    assert serving.calls == expected


def test_unrelated_main_and_process_utility_are_not_patched(serving, monkeypatch):
    original = serving.engine.kill_process_tree.__wrapped__
    main: Any = types.ModuleType("__main__")
    main.__spec__ = types.SimpleNamespace(name="unrelated.application")
    main.kill_process_tree = original
    utility: Any = types.ModuleType("sglang.srt.utils")
    utility.kill_process_tree = original
    monkeypatch.setitem(sys.modules, "__main__", main)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utility)
    serving.engine.Engine._launch_subprocesses()
    assert main.kill_process_tree is original
    assert utility.kill_process_tree is original


def test_dead_worker_at_capture_does_not_claim_replacement(serving):
    serving.processes[101].is_running.return_value = False
    serving.engine.Engine._launch_subprocesses()
    serving.engine.kill_process_tree(os.getpid(), include_parent=False)
    assert (serving.directory / "serve").exists()


def test_worker_exit_during_capture_preserves_segment(serving, monkeypatch):
    original = shutdown.IPCSegmentCleanup
    def capture(path):
        segment = original(path)
        serving.processes[101].is_running.return_value = False
        return segment
    monkeypatch.setattr(shutdown, "IPCSegmentCleanup", capture)
    serving.engine.Engine._launch_subprocesses()
    serving.engine.kill_process_tree(os.getpid(), include_parent=False)
    assert (serving.directory / "serve").exists()


def test_readiness_captures_before_launch_returns_or_nonzero_node_blocks(serving):
    # Both old and new upstreams wait inside _launch_subprocesses on nonzero
    # nodes. Simulate readiness without letting that function return at all.
    info = serving.scheduler.Scheduler().get_init_info()
    result = serving.engine.SchedulerInitResult([info], list(serving.processes))
    result.wait_for_ready()
    serving.engine.kill_process_tree(os.getpid(), include_parent=False)
    assert sorted(p.name for p in serving.directory.iterdir()) == ["foreign"]
