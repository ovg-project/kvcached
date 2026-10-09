# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Real SIGKILLed scheduler processes holding live IPC listeners (issue #548):
the launching process removes their sockets once psutil reports them gone,
through the kill_process_tree binding launch_server.py uses."""

import multiprocessing as mp
import os
import shutil
import signal
import socket
import tempfile
import types
from typing import Any

import pytest

import kvcached.utils as kv_utils
from kvcached.integration.sglang import shutdown
from kvcached.pool_registry import (
    clear_registered_kv_cache_pools,
    register_kv_cache_pool,
)

psutil = pytest.importorskip("psutil")


class _Pool:
    ipc_name = "serve"


def _scheduler_process(path, ready, stop):
    """Bind the worker socket like tp_ipc_util's listener and serve until
    stopped; a SIGKILL ends it before any teardown runs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen()
    ready.send(os.getpid())
    ready.close()
    stop.recv()
    server.close()


@pytest.fixture(params=[method for method in ("spawn", "fork")
                        if method in mp.get_all_start_methods()])
def launched(monkeypatch, tmp_path, request):
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    monkeypatch.setattr(shutdown, "SHM_DIR", str(tmp_path))
    root = tempfile.mkdtemp(prefix="kvcached-test-", dir="/tmp")
    monkeypatch.setattr(kv_utils, "TP_SOCKET_DIR_ROOT", root)
    clear_registered_kv_cache_pools()
    pool = _Pool()  # the registry holds pools weakly; kept alive here
    register_kv_cache_pool(pool, integration="sglang")
    (tmp_path / "serve").write_bytes(b"control")
    context = mp.get_context(request.param)
    socket_dir = kv_utils.get_tp_socket_dir("serve")
    paths = [os.path.join(socket_dir, f"w{rank}.sock") for rank in range(2)]
    workers: list[Any] = []
    stops = []

    class Scheduler:
        def get_init_info(self):
            return {"status": "ready"}

    class Engine:
        @classmethod
        def _launch_subprocesses(cls, server_args):
            for path in paths:
                ready_r, ready_w = context.Pipe(duplex=False)
                stop_r, stop_w = context.Pipe(duplex=False)
                process = context.Process(target=_scheduler_process,
                                          args=(path, ready_w, stop_r))
                process.start()
                ready_w.close()
                assert ready_r.poll(30), "scheduler readiness timed out"
                assert ready_r.recv() == process.pid
                ready_r.close()
                workers.append(process)
                stops.append(stop_w)
            return (None, None, None, types.SimpleNamespace(
                scheduler_infos=[Scheduler().get_init_info()],
                all_child_pids=[process.pid for process in workers],
            ), None)

    calls = []

    def kill_process_tree(parent_pid, include_parent=True, skip_pid=None,
                          wait_timeout=60):
        """SGLang's kill: SIGKILL every child, then poll until none holds
        resources any more. The children are not reaped here."""
        calls.append((parent_pid, include_parent, skip_pid, wait_timeout))
        assert parent_pid == os.getpid() and not include_parent
        for process in workers:
            if process.pid != skip_pid:
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for process in workers:
            if process.pid != skip_pid:
                for _ in range(200):
                    try:
                        if psutil.Process(process.pid).status() == psutil.STATUS_ZOMBIE:
                            break
                    except psutil.NoSuchProcess:
                        break
                    mp.connection.wait([], 0.05)
        return "upstream result"

    engine = types.SimpleNamespace(Engine=Engine, kill_process_tree=kill_process_tree)
    utils = types.SimpleNamespace(kill_process_tree=kill_process_tree)
    assert shutdown.SGLangShutdownPatch().patch_shutdown(
        engine, types.SimpleNamespace(Scheduler=Scheduler),
        types.SimpleNamespace(kill_process_tree=kill_process_tree), utils)
    state = types.SimpleNamespace(
        engine=engine, utils=utils, paths=paths, socket_dir=socket_dir,
        workers=workers, directory=tmp_path, calls=calls,
        server_args=types.SimpleNamespace(tp_size=2, pp_size=1), pool=pool,
    )
    try:
        yield state
    finally:
        for process in workers:
            if process.is_alive():
                process.kill()
            process.join(10)
        for stop in stops:
            stop.close()
        shutil.rmtree(root, ignore_errors=True)
        clear_registered_kv_cache_pools()


def _accepts_connection(path):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(5)
        try:
            probe.connect(path)
        except ConnectionRefusedError:
            return False
    return True


def test_sigint_exit_removes_the_sockets_of_sigkilled_schedulers(launched):
    state = launched
    state.engine.Engine._launch_subprocesses(server_args=state.server_args)
    for path in state.paths:
        assert _accepts_connection(path)

    # launch_server.py's finally: the binding from sglang.srt.utils.
    assert state.utils.kill_process_tree(os.getpid(), include_parent=False) == "upstream result"

    for path in state.paths:
        assert not os.path.exists(path), "a SIGKILLed scheduler's socket was left behind"
    assert not os.path.exists(state.socket_dir)
    assert not (state.directory / "serve").exists()
    for process in state.workers:
        process.join(10)
        assert process.exitcode == -signal.SIGKILL


def test_live_scheduler_keeps_its_socket_until_the_retry(launched):
    state = launched
    state.engine.Engine._launch_subprocesses(server_args=state.server_args)
    survivor = state.workers[1]

    state.engine.kill_process_tree(os.getpid(), include_parent=False, skip_pid=survivor.pid)

    assert os.path.exists(state.paths[1])
    assert _accepts_connection(state.paths[1])
    assert os.path.exists(state.paths[0]), "exit of every child is the gate"
    assert (state.directory / "serve").exists()

    state.engine.kill_process_tree(os.getpid(), include_parent=False)
    for path in state.paths:
        assert not os.path.exists(path)
    assert not os.path.exists(state.socket_dir)
    assert not (state.directory / "serve").exists()
