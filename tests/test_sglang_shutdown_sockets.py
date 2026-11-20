# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""The launching process removes the TP worker sockets its SIGKILLed
schedulers left behind (issue #548): each scheduler reports the sockets its
listeners bound. The owner removes only those socket nodes once every recorded
child is gone; control files remain untouched."""

import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import types
from typing import Any
from unittest.mock import Mock

import pytest

import kvcached.utils as kv_utils
from kvcached.integration.sglang import shutdown
from kvcached.pool_registry import (
    clear_registered_kv_cache_pools,
    register_kv_cache_pool,
)


def _bind_worker_socket(path):
    """A listening socket at *path*, as a live scheduler binds it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen()
    return server


def _replace_worker_socket(path):
    """A same-name replacement with a different inode, created beside the
    old node and renamed into place like a restart's unlink and bind."""
    replacement = _bind_worker_socket(path + ".new")
    os.rename(path + ".new", path)
    return replacement


def _accepts_connection(path):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(5)
        try:
            probe.connect(path)
        except ConnectionRefusedError:
            return False
    return True


@pytest.fixture
def serving(monkeypatch, tmp_path):
    """One launching process with two TP=2 schedulers whose IPC listeners
    bound w0.sock and w1.sock before READY and reported them. Closing a
    listening socket is what a SIGKILLed scheduler leaves: the node stays,
    nobody accepts."""
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    monkeypatch.setattr(shutdown, "register_owner_sigterm_cleanup", lambda callback: True)
    # Not tmp_path: unix socket paths are limited to ~104 characters on macOS.
    root = tempfile.mkdtemp(prefix="kvcached-test-", dir="/tmp")
    monkeypatch.setattr(kv_utils, "TP_SOCKET_DIR_ROOT", root)
    clear_registered_kv_cache_pools()
    pools = [Mock(ipc_name="serve"), Mock(ipc_name="serve_g1000")]
    for pool in pools:
        register_kv_cache_pool(pool, integration="sglang")
        (tmp_path / pool.ipc_name).write_bytes(b"control")

    processes = {pid: Mock(pid=pid) for pid in (101, 102)}
    for process in processes.values():
        process.is_running.return_value = True
        process.status.return_value = "running"
    psutil: Any = types.ModuleType("psutil")
    psutil.Process = Mock(side_effect=lambda pid: processes[pid])
    psutil.STATUS_ZOMBIE = "zombie"
    psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    monkeypatch.setitem(sys.modules, "psutil", psutil)

    socket_dir = kv_utils.get_tp_socket_dir("serve")
    paths = [os.path.join(socket_dir, f"w{rank}.sock") for rank in range(2)]
    state = types.SimpleNamespace(
        calls=[], exits=[], hold=False, workers={}, processes=processes,
        socket_dir=socket_dir, paths=paths, directory=tmp_path,
    )
    # What the schedulers' listeners report having bound, at readiness.
    state.report = lambda: sorted(state.workers)
    state.listener_report = shutdown._bound_worker_sockets
    monkeypatch.setattr(shutdown, "_bound_worker_sockets", lambda: state.report())

    def bind_workers():
        for path in paths:
            state.workers[path] = _bind_worker_socket(path)

    def kill_workers(*which):
        for path in which or tuple(paths):
            state.workers.pop(path).close()

    state.bind_workers = bind_workers
    state.kill_workers = kill_workers

    def kill_process_tree(parent_pid, include_parent=True, skip_pid=None,
                          wait_timeout=60):
        state.calls.append((parent_pid, include_parent, skip_pid, wait_timeout))
        if include_parent:
            state.exits.append(sorted(os.listdir(socket_dir))
                               if os.path.isdir(socket_dir) else None)
        if not state.hold:
            for pid, process in processes.items():
                if pid != skip_pid:
                    process.is_running.return_value = False
        return "upstream result"

    class Scheduler:
        def get_init_info(self):
            return {"status": "ready", "max_req_input_len": 512}

    class SchedulerInitResult:
        def __init__(self, scheduler_infos, all_child_pids):
            self.scheduler_infos = scheduler_infos
            self.all_child_pids = all_child_pids
            self.wait_for_ready = lambda: None

    class Engine:
        @classmethod
        def _launch_subprocesses(cls, *args, **kwargs):
            # Readiness is awaited inside the launch, as upstream does.
            result = SchedulerInitResult([Scheduler().get_init_info()], list(processes))
            result.wait_for_ready()
            return (None, None, None, result, None)

    # launch_server.py's own binding of the name, made before the patch ran.
    launch_server: Any = types.ModuleType("sglang.launch_server")
    launch_server.kill_process_tree = kill_process_tree
    launch_server.__spec__ = types.SimpleNamespace(name="sglang.launch_server")
    monkeypatch.setitem(sys.modules, "sglang.launch_server", launch_server)

    engine = types.SimpleNamespace(
        Engine=Engine, SchedulerInitResult=SchedulerInitResult,
        kill_process_tree=kill_process_tree,
    )
    scheduler = types.SimpleNamespace(Scheduler=Scheduler)
    tokenizer = types.SimpleNamespace(kill_process_tree=kill_process_tree)
    patch = shutdown.SGLangShutdownPatch()
    assert patch.patch_shutdown(engine, scheduler, tokenizer)
    assert patch.patch_shutdown(engine, scheduler, tokenizer)
    state.engine = engine
    state.scheduler = scheduler
    state.tokenizer = tokenizer
    state.launch_server = launch_server
    state.original_kill = kill_process_tree
    state.launch = engine.Engine._launch_subprocesses
    try:
        yield state
    finally:
        for server in state.workers.values():
            server.close()
        shutil.rmtree(root, ignore_errors=True)
        clear_registered_kv_cache_pools()


def test_readiness_reports_the_sockets_the_listeners_bound(serving, monkeypatch):
    state = serving
    state.bind_workers()
    listeners: Any = types.ModuleType("kvcached.tp_ipc_util")
    listeners.worker_socket_paths = lambda: list(state.paths)
    monkeypatch.setattr(shutdown, "_bound_worker_sockets", state.listener_report)
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "kvcached.tp_ipc_util", listeners)
        info = state.scheduler.Scheduler().get_init_info()
    assert info[shutdown._SOCKETS_KEY] == state.paths
    assert "_kvcached_ipc_names" not in info

    with monkeypatch.context() as context:
        context.setitem(sys.modules, "kvcached.tp_ipc_util", None)  # no extension here
        info = state.scheduler.Scheduler().get_init_info()
    assert info[shutdown._SOCKETS_KEY] == []


def test_sigint_exit_removes_the_sockets_killed_schedulers_left(serving):
    """The SIGINT shape from issue #548: launch_server.py ends with its own
    kill_process_tree(os.getpid(), include_parent=False), and the
    schedulers die before their atexit hook."""
    state = serving
    state.bind_workers()
    state.launch()
    state.kill_workers()
    from_entrypoint = state.launch_server.kill_process_tree
    assert from_entrypoint is not state.original_kill

    assert from_entrypoint(os.getpid(), include_parent=False) == "upstream result"

    for path in state.paths:
        assert not os.path.exists(path), "a killed scheduler's socket was left behind"
    assert not os.path.exists(state.socket_dir)
    assert sorted(p.name for p in state.directory.iterdir()) == ["serve", "serve_g1000"]
    assert state.calls == [(os.getpid(), False, None, 60)]


def test_engine_shutdown_removes_the_sockets_as_well(serving):
    state = serving
    state.bind_workers()
    state.launch()
    state.kill_workers()
    state.engine.kill_process_tree(os.getpid(), include_parent=False, wait_timeout=60)
    for path in state.paths:
        assert not os.path.exists(path)
    assert not os.path.exists(state.socket_dir)


def test_pp_stage_sockets_and_their_directories_go_too(serving):
    state = serving
    stage = [os.path.join(state.socket_dir, "pp1", f"w{rank}.sock") for rank in range(2)]
    state.bind_workers()
    for path in stage:
        state.workers[path] = _bind_worker_socket(path)
    state.launch()
    state.kill_workers(*state.paths, *stage)
    state.engine.kill_process_tree(os.getpid(), include_parent=False)
    for path in state.paths + stage:
        assert not os.path.exists(path)
    assert not os.path.exists(os.path.join(state.socket_dir, "pp1"))
    assert not os.path.exists(state.socket_dir)


def test_self_exit_removes_the_sockets_before_the_parent_exits(serving):
    """running_phase_sigquit_handler kills the parent too: the sockets go
    between the reap of the workers and the original self-kill."""
    state = serving
    state.bind_workers()
    state.launch()
    state.kill_workers()
    assert state.tokenizer.kill_process_tree(os.getpid()) == "upstream result"
    assert [call[1] for call in state.calls] == [False, True]
    assert state.exits == [None], "the directory was still there at the self-kill"


def test_live_schedulers_keep_their_sockets_until_the_retry(serving):
    state = serving
    state.bind_workers()
    state.launch()
    state.kill_workers()
    state.engine.kill_process_tree(os.getpid(), False, skip_pid=102)
    for path in state.paths:
        assert os.path.exists(path)
    assert (state.directory / "serve").exists()

    state.engine.kill_process_tree(os.getpid(), False)
    for path in state.paths:
        assert not os.path.exists(path)
    assert (state.directory / "serve").read_bytes() == b"control"


def test_served_socket_is_kept_and_the_owner_retried_in_bounded_time(serving):
    """Child exit is checked per process and per socket: a socket that still
    accepts after the recorded children are gone is kept, the call returns
    in bounded time, and the owner stays queued for the next kill call."""
    state = serving
    state.bind_workers()
    state.launch()
    state.kill_workers(state.paths[0])

    started = time.monotonic()
    state.engine.kill_process_tree(os.getpid(), False)
    assert time.monotonic() - started < 5

    assert not os.path.exists(state.paths[0])
    assert os.path.exists(state.paths[1])
    assert _accepts_connection(state.paths[1])
    assert os.path.isdir(state.socket_dir)
    assert (state.directory / "serve").read_bytes() == b"control"

    state.kill_workers(state.paths[1])
    state.engine.kill_process_tree(os.getpid(), False)
    assert not os.path.exists(state.paths[1])
    assert not os.path.exists(state.socket_dir)


@pytest.mark.parametrize("replacement_alive", [True, False])
def test_same_name_replacement_socket_is_never_claimed(serving, replacement_alive):
    state = serving
    state.bind_workers()
    state.launch()
    state.kill_workers()
    replacement = _replace_worker_socket(state.paths[0])
    try:
        if not replacement_alive:
            replacement.close()
        state.engine.kill_process_tree(os.getpid(), False)
        assert not os.path.exists(state.paths[1])
        assert os.path.exists(state.paths[0]), "the replacement's socket was removed"
        assert _accepts_connection(state.paths[0]) is replacement_alive
        assert os.path.isdir(state.socket_dir)
    finally:
        replacement.close()


def test_reported_sockets_absent_at_ready_are_never_claimed(serving):
    state = serving
    state.report = lambda: list(state.paths)
    state.launch()
    state.bind_workers()
    state.kill_workers()
    state.engine.kill_process_tree(os.getpid(), False)
    assert (state.directory / "serve").read_bytes() == b"control"
    for path in state.paths:
        assert os.path.exists(path)


def test_repeated_kill_calls_leave_a_replacement_directory(serving):
    """After the cleanup completed, a same-name replacement paused between
    its makedirs() and bind() keeps the directory it created."""
    state = serving
    state.bind_workers()
    state.launch()
    state.kill_workers()
    state.engine.kill_process_tree(os.getpid(), False)
    assert not os.path.exists(state.socket_dir)

    os.makedirs(state.socket_dir)
    state.engine.kill_process_tree(os.getpid(), False)
    state.tokenizer.kill_process_tree(os.getpid(), False)

    assert os.path.isdir(state.socket_dir)
    state.bind_workers()


def test_unreported_sockets_are_left_alone(serving):
    """A scheduler without a listener (TP=1) or without this report leaves
    nothing to remember; control files remain untouched."""
    state = serving
    state.bind_workers()
    state.report = list
    state.launch()
    state.kill_workers()
    state.engine.kill_process_tree(os.getpid(), False)
    assert (state.directory / "serve").read_bytes() == b"control"
    for path in state.paths:
        assert os.path.exists(path)


@pytest.mark.parametrize("reported", [
    "relative/w0.sock", "/tmp/kvcached-tp-serve-x/other.sock", "/tmp/elsewhere/w0.sock",
    "/tmp/kvcached-tp-serve-x/pp1/pp2/w0.sock",
], ids=["relative", "name", "directory", "stage"])
def test_a_reported_path_outside_the_listener_layout_claims_nothing(serving, reported):
    state = serving
    state.bind_workers()
    state.report = lambda: sorted(state.workers) + [reported]
    state.launch()
    state.kill_workers()
    state.engine.kill_process_tree(os.getpid(), False)
    assert (state.directory / "serve").read_bytes() == b"control"
    for path in state.paths:
        assert os.path.exists(path)


def test_dp_controller_forwards_the_reported_sockets_of_every_rank():
    """DP rebuilds readiness, forwarding sockets but no control metadata."""
    collected: set[str] = set()
    lock = threading.Lock()
    paths = ["/tmp/kvcached-tp-serve-x/w1.sock", "/tmp/kvcached-tp-serve-x/w0.sock"]
    for path in paths:
        info = {"status": "ready", shutdown._SOCKETS_KEY: [path]}
        shutdown._DPReadinessConnection(Mock(recv=Mock(return_value=info)), collected, lock).recv()
    writer = Mock()
    rebuilt = {"status": "ready", "scheduler_pids": [101, 102]}
    shutdown._DPReadinessConnection(writer, collected, lock).send(rebuilt)
    assert writer.send.call_args.args[0] == {
        **rebuilt, shutdown._SOCKETS_KEY: sorted(paths),
    }
    assert rebuilt == {"status": "ready", "scheduler_pids": [101, 102]}
    shutdown._DPReadinessConnection(writer, set(), lock).send(dict(rebuilt))
    assert writer.send.call_args.args[0] == rebuilt


def test_socket_released_after_the_exit_report_goes_in_the_final_call(serving):
    """shipiyouniao's Linux spawn finding on PR #553: a SIGKILLed scheduler
    is reported as a zombie while its remaining threads are still torn
    down, and its listener accepts for a few more milliseconds. The final
    teardown of launch_server.py makes one kill call, so that call has to
    wait, bounded, for the release instead of leaving the owner queued."""
    state = serving
    state.bind_workers()
    state.launch()
    threading.Timer(0.1, state.kill_workers).start()

    state.launch_server.kill_process_tree(os.getpid(), include_parent=False)

    for path in state.paths:
        assert not os.path.exists(path), "the socket released after the exit report was left behind"
    assert not os.path.exists(state.socket_dir)
    assert sorted(p.name for p in state.directory.iterdir()) == ["serve", "serve_g1000"]
    assert state.calls == [(os.getpid(), False, None, 60)]


def test_release_wait_keeps_checking_the_identity(serving):
    """A replacement bound at the same path during the wait is not ours:
    the wait ends with it untouched and the directory kept."""
    state = serving
    state.bind_workers()
    state.launch()
    state.kill_workers(state.paths[1])
    replacements = []

    def restart():
        state.kill_workers(state.paths[0])
        replacements.append(_replace_worker_socket(state.paths[0]))

    threading.Timer(0.1, restart).start()
    try:
        started = time.monotonic()
        state.engine.kill_process_tree(os.getpid(), include_parent=False)
        assert time.monotonic() - started < 5
        assert os.path.exists(state.paths[0]), "the replacement's socket was removed"
        assert _accepts_connection(state.paths[0])
        assert not os.path.exists(state.paths[1])
        assert os.path.isdir(state.socket_dir)
    finally:
        for replacement in replacements:
            replacement.close()


def test_disabled_kvcached_leaves_the_sockets_alone(serving, monkeypatch):
    state = serving
    state.bind_workers()
    monkeypatch.setenv("ENABLE_KVCACHED", "false")
    state.launch()
    state.kill_workers()
    state.launch_server.kill_process_tree(os.getpid(), include_parent=False)
    for path in state.paths:
        assert os.path.exists(path)
    assert (state.directory / "serve").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
