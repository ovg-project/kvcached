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
import sys
import tempfile
import textwrap
import threading
import time
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


def _scheduler_process(path, ready, stop, report_sockets=False):
    """Bind the worker socket and serve it from a thread, like tp_ipc_util's
    listener, while the main thread waits; a SIGKILL ends both before any
    teardown runs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen()

    def serve():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    ready.send({"status": "ready", "pid": os.getpid(),
                "_kvcached_worker_sockets": [path]} if report_sockets else os.getpid())
    ready.close()
    stop.recv()
    server.close()


@pytest.fixture(params=[method for method in ("spawn", "fork")
                        if method in mp.get_all_start_methods()])
def launched(monkeypatch, tmp_path, request):
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    monkeypatch.setattr(shutdown, "register_owner_sigterm_cleanup", lambda callback: True)
    root = tempfile.mkdtemp(prefix="kvcached-test-", dir="/tmp")
    monkeypatch.setattr(kv_utils, "TP_SOCKET_DIR_ROOT", root)
    clear_registered_kv_cache_pools()
    pool = _Pool()  # the registry holds pools weakly; kept alive here
    register_kv_cache_pool(pool, integration="sglang")
    (tmp_path / "serve").write_bytes(b"control")
    context = mp.get_context(request.param)
    socket_dir = kv_utils.get_tp_socket_dir("serve")
    paths = [os.path.join(socket_dir, f"w{rank}.sock") for rank in range(2)]
    # The schedulers' readiness reports the sockets their listeners bound.
    monkeypatch.setattr(shutdown, "_bound_worker_sockets", lambda: list(paths))
    workers: list[Any] = []
    stops = []

    class Scheduler:
        def get_init_info(self):
            return {"status": "ready"}

    class SchedulerInitResult:
        def __init__(self, scheduler_infos, all_child_pids):
            self.scheduler_infos = scheduler_infos
            self.all_child_pids = all_child_pids
            self.wait_for_ready = lambda: None

    class Engine:
        @classmethod
        def _launch_subprocesses(cls):
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
            result = SchedulerInitResult([Scheduler().get_init_info()],
                                         [process.pid for process in workers])
            result.wait_for_ready()
            return (None, None, None, result, None)

    calls = []

    def kill_process_tree(parent_pid, include_parent=True, skip_pid=None,
                          wait_timeout=60):
        """SGLang's kill: SIGKILL every child, then poll until none holds
        resources any more, which its 0.5.20 wait takes to be the zombie
        status. The children are not reaped here."""
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

    # launch_server.py's own binding of the name, made before the patch ran.
    launch_server: Any = types.ModuleType("sglang.launch_server")
    launch_server.kill_process_tree = kill_process_tree
    launch_server.__spec__ = types.SimpleNamespace(name="sglang.launch_server")
    monkeypatch.setitem(sys.modules, "sglang.launch_server", launch_server)

    engine = types.SimpleNamespace(
        Engine=Engine, SchedulerInitResult=SchedulerInitResult,
        kill_process_tree=kill_process_tree,
    )
    assert shutdown.SGLangShutdownPatch().patch_shutdown(
        engine, types.SimpleNamespace(Scheduler=Scheduler),
        types.SimpleNamespace(kill_process_tree=kill_process_tree))
    assert launch_server.kill_process_tree is not kill_process_tree
    state = types.SimpleNamespace(
        engine=engine, launch_server=launch_server, paths=paths, socket_dir=socket_dir,
        workers=workers, directory=tmp_path, calls=calls, pool=pool,
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
    state.engine.Engine._launch_subprocesses()
    for path in state.paths:
        assert _accepts_connection(path)

    # launch_server.py's finally: the one and only kill call of the teardown.
    assert state.launch_server.kill_process_tree(
        os.getpid(), include_parent=False) == "upstream result"

    for path in state.paths:
        assert not os.path.exists(path), "a SIGKILLed scheduler's socket was left behind"
    assert not os.path.exists(state.socket_dir)
    assert (state.directory / "serve").read_bytes() == b"control"
    for process in state.workers:
        process.join(10)
        assert process.exitcode == -signal.SIGKILL


def test_live_scheduler_keeps_its_socket_until_the_retry(launched):
    state = launched
    state.engine.Engine._launch_subprocesses()
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
    assert (state.directory / "serve").read_bytes() == b"control"


def test_spawn_dp_controller_forwards_sockets_to_owner_cleanup(monkeypatch, tmp_path):
    """Run the actual patched spawn target across the rebuilding DP boundary.

    Only SGLang's controller is a stand-in: its schedulers, pipes, socket
    listeners, SIGKILL and owner process observations are real.
    """
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    package = tmp_path / "sglang"
    for part in (package, package / "srt", package / "srt" / "managers"):
        part.mkdir()
        (part / "__init__.py").write_text("")
    controller_file = package / "srt" / "managers" / "data_parallel_controller.py"
    controller_file.write_text(textwrap.dedent(f"""
        import multiprocessing as mp
        from {__name__} import _scheduler_process

        def run_data_parallel_controller_process(server_args, port_args, pipe_writer):
            context = mp.get_context("spawn")
            workers, stops, infos = [], [], []
            try:
                for path in port_args:
                    reader, writer = mp.Pipe(duplex=False)
                    stop_reader, stop_writer = context.Pipe(duplex=False)
                    worker = context.Process(target=_scheduler_process,
                        args=(path, writer, stop_reader, True))
                    worker.start()
                    workers.append(worker)
                    stops.append(stop_writer)
                    writer.close()
                    stop_reader.close()
                    assert reader.poll(30)
                    infos.append(reader.recv())
                    reader.close()
                # Deliberately rebuild, as SGLang does; forwarding the original
                # worker dictionaries here would bypass the boundary under test.
                pipe_writer.send({{"status": "ready",
                    "scheduler_pids": [info["pid"] for info in infos]}})
                server_args.recv()
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.kill()
                    worker.join(10)
                for stop in stops:
                    stop.close()
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    control = tmp_path / "control"
    control.write_bytes(b"control")
    root = tempfile.mkdtemp(prefix="kvcached-tp-dp-", dir="/tmp")
    paths = [os.path.join(root, "w0.sock"), os.path.join(root, "pp1", "w1.sock")]
    context = mp.get_context("spawn")
    ready_reader, ready_writer = context.Pipe(duplex=False)
    stop_reader, stop_writer = context.Pipe(duplex=False)
    controller = context.Process(target=shutdown._run_data_parallel_controller_process,
                                 args=(stop_reader, paths, ready_writer))
    controller.start()
    ready_writer.close()
    stop_reader.close()
    try:
        assert ready_reader.poll(60), "DP controller readiness timed out"
        info = ready_reader.recv()
        assert info[shutdown._SOCKETS_KEY] == sorted(paths)
        assert "_kvcached_ipc_names" not in info
        assert all(_accepts_connection(path) for path in paths)
        result = types.SimpleNamespace(scheduler_infos=[info],
                                       all_child_pids=[controller.pid, *info["scheduler_pids"]])
        owner = shutdown._capture_owner(result, lambda *args, **kwargs: None)
        assert owner is not None
        assert shutdown._SOCKETS_KEY not in info
        assert not owner.cleanup(), "live controller and schedulers must keep their sockets"
        stop_writer.send("stop")  # Controller SIGKILLs and reaps both schedulers.
        controller.join(30)
        assert controller.exitcode == 0
        assert owner.cleanup(time.monotonic() + 1.0)
        assert not os.path.exists(root)
        assert control.read_bytes() == b"control"
    finally:
        if controller.is_alive():
            # Ask the controller to stop its own children before terminating it.
            stop_writer.send("stop")
            controller.join(30)
        if controller.is_alive():
            controller.kill()
            controller.join(10)
        ready_reader.close()
        stop_writer.close()
        shutil.rmtree(root, ignore_errors=True)
