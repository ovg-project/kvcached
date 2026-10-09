# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""The launching process removes the TP worker sockets its SIGKILLed
schedulers left behind (issue #548), from the same owner that removes the
control segments and only once every recorded child is gone."""

import os
import shutil
import socket
import sys
import tempfile
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
    bound w0.sock and w1.sock before READY. Closing a listening socket is
    what a SIGKILLed scheduler leaves: the node stays, nobody accepts."""
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    monkeypatch.setattr(shutdown, "SHM_DIR", str(tmp_path))
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
        server_args=types.SimpleNamespace(tp_size=2, pp_size=1),
    )

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

    class Engine:
        @classmethod
        def _launch_subprocesses(cls, *args, **kwargs):
            return (None, None, None, types.SimpleNamespace(
                scheduler_infos=[Scheduler().get_init_info()],
                all_child_pids=list(processes),
            ), None, [])

    engine = types.SimpleNamespace(Engine=Engine, kill_process_tree=kill_process_tree)
    scheduler = types.SimpleNamespace(Scheduler=Scheduler)
    tokenizer = types.SimpleNamespace(kill_process_tree=kill_process_tree)
    # The package launch_server.py and cli/serve.py bind their name from.
    utils = types.SimpleNamespace(kill_process_tree=kill_process_tree)
    patch = shutdown.SGLangShutdownPatch()
    assert patch.patch_shutdown(engine, scheduler, tokenizer, utils)
    assert patch.patch_shutdown(engine, scheduler, tokenizer, utils)
    state.engine = engine
    state.tokenizer = tokenizer
    state.utils = utils
    state.original_kill = kill_process_tree

    def launch(server_args=state.server_args):
        return engine.Engine._launch_subprocesses(server_args=server_args)

    state.launch = launch
    try:
        yield state
    finally:
        for server in state.workers.values():
            server.close()
        shutil.rmtree(root, ignore_errors=True)
        clear_registered_kv_cache_pools()


def test_sigint_exit_removes_the_sockets_killed_schedulers_left(serving):
    """The SIGINT shape from issue #548: launch_server.py ends with its own
    kill_process_tree(os.getpid(), include_parent=False), bound from
    sglang.srt.utils, and the schedulers die before their atexit hook."""
    state = serving
    state.bind_workers()
    state.launch()
    state.kill_workers()
    from_package = state.utils.kill_process_tree  # launch_server.py's binding
    assert from_package is not state.original_kill

    assert from_package(os.getpid(), include_parent=False) == "upstream result"

    for path in state.paths:
        assert not os.path.exists(path), "a killed scheduler's socket was left behind"
    assert not os.path.exists(state.socket_dir)
    assert sorted(p.name for p in state.directory.iterdir()) == []
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
    assert not (state.directory / "serve").exists()


def test_delayed_scheduler_keeps_its_socket_until_it_stops_serving(serving):
    """Child exit is checked per process and per socket: a scheduler that
    still accepts on its socket keeps it, and the owner stays queued for
    the next kill call."""
    state = serving
    state.bind_workers()
    state.launch()
    state.kill_workers(state.paths[0])

    state.engine.kill_process_tree(os.getpid(), False)

    assert not os.path.exists(state.paths[0])
    assert os.path.exists(state.paths[1])
    assert _accepts_connection(state.paths[1])
    assert os.path.isdir(state.socket_dir)
    assert not (state.directory / "serve").exists()

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


def test_sockets_absent_at_ready_are_never_claimed(serving):
    state = serving
    state.launch()
    state.bind_workers()
    state.kill_workers()
    state.engine.kill_process_tree(os.getpid(), False)
    assert not (state.directory / "serve").exists()
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


def test_without_a_parallel_config_the_sockets_are_left_alone(serving):
    state = serving
    state.bind_workers()
    state.launch(server_args=None)
    state.kill_workers()
    state.engine.kill_process_tree(os.getpid(), False)
    assert not (state.directory / "serve").exists()
    for path in state.paths:
        assert os.path.exists(path)


def test_socket_directory_follows_the_reported_pool_name(serving, monkeypatch):
    """The scheduler derives its socket directory from its own default IPC
    name, which is the name of its group-0 pool. The parent's derivation
    may differ, for instance after a late import, and must not be used
    when the reported names agree."""
    state = serving
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "serve_1")
    state.bind_workers()
    state.launch()
    state.kill_workers()
    state.engine.kill_process_tree(os.getpid(), False)
    for path in state.paths:
        assert not os.path.exists(path)
    assert not os.path.exists(state.socket_dir)


def test_unrelated_pool_names_fall_back_to_the_parent_directory(serving, monkeypatch):
    state = serving
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "parent_name")
    clear_registered_kv_cache_pools()
    pools = [Mock(ipc_name=name) for name in ("alpha", "beta")]  # kept alive: weak registry
    for pool in pools:
        register_kv_cache_pool(pool, integration="sglang")
        (state.directory / pool.ipc_name).write_bytes(b"control")
    path = os.path.join(kv_utils.get_tp_socket_dir(), "w0.sock")
    _bind_worker_socket(path).close()
    state.launch()
    state.engine.kill_process_tree(os.getpid(), False)
    assert not os.path.exists(path)
    assert not (state.directory / "alpha").exists()


def test_disabled_kvcached_leaves_the_sockets_alone(serving, monkeypatch):
    state = serving
    state.bind_workers()
    monkeypatch.setenv("ENABLE_KVCACHED", "false")
    state.launch()
    state.kill_workers()
    state.utils.kill_process_tree(os.getpid(), include_parent=False)
    for path in state.paths:
        assert os.path.exists(path)
    assert (state.directory / "serve").exists()


def test_entry_point_bindings_are_wrapped_once(serving):
    """launch_server.py, cli/serve.py and http_server bind the name from
    sglang.srt.utils after the patch ran, so they hold the wrapper; a
    second patch leaves a wrapped binding as it is."""
    state = serving
    wrapper = state.utils.kill_process_tree
    assert getattr(wrapper, shutdown._KILL_WRAPPER_MARK)
    entrypoint = types.SimpleNamespace(kill_process_tree=state.utils.kill_process_tree)

    class Engine:
        @classmethod
        def _launch_subprocesses(cls, *args, **kwargs):
            return (None, None, None, types.SimpleNamespace(
                scheduler_infos=[], all_child_pids=[]), None)

    class Scheduler:
        def get_init_info(self):
            return {}

    engine = types.SimpleNamespace(Engine=Engine, kill_process_tree=state.original_kill)
    tokenizer = types.SimpleNamespace(kill_process_tree=wrapper)
    assert shutdown.SGLangShutdownPatch().patch_shutdown(
        engine, types.SimpleNamespace(Scheduler=Scheduler), tokenizer, entrypoint)

    assert entrypoint.kill_process_tree is wrapper
    assert tokenizer.kill_process_tree is wrapper
    assert engine.kill_process_tree is not state.original_kill
    assert getattr(engine.kill_process_tree, shutdown._KILL_WRAPPER_MARK)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
