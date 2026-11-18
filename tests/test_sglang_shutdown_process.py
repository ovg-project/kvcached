# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Real SGLang-shaped owner/worker signals, without control-file unlink."""

import multiprocessing as mp
import os
import signal
import sys
import types
from pathlib import Path
from typing import Any

import pytest

psutil = pytest.importorskip("psutil")


def _signal_worker(writer, directory):
    import time

    Path(directory, "owned").write_bytes(b"control")
    writer.send(os.getpid())
    writer.close()
    while True:
        time.sleep(0.1)


def _signal_owner(writer, directory, entrypoint, replace, delivery, handler):
    """A real parent finally and child SIGKILL, with frontend imports bound
    before patch installation. Only the external engine APIs are doubles."""
    import signal

    from kvcached.integration.sglang import shutdown as module

    os.environ["ENABLE_KVCACHED"] = "true"
    if delivery == "group":
        os.setsid()
    context = mp.get_context("spawn")
    reader, child_writer = context.Pipe(duplex=False)
    worker = context.Process(target=_signal_worker, args=(child_writer, directory))
    worker.start()
    child_writer.close()
    assert reader.poll(20)
    pid = reader.recv()
    reader.close()

    def native_kill(parent_pid, include_parent=True, skip_pid=None, wait_timeout=None):
        assert parent_pid == os.getpid() and not include_parent
        worker.kill()  # Legacy kill returns before reaping: exercise confirmation.

    frontend: Any = types.ModuleType(entrypoint)
    frontend.kill_process_tree = native_kill
    frontend.__spec__ = types.SimpleNamespace(name="sglang.launch_server")
    # Do not replace spawn's __main__: only the binding lookup needs this copy.
    sys.modules[entrypoint] = frontend

    class Scheduler:
        def get_init_info(self):
            return {"status": "ready"}

    class InitResult:
        def __init__(self):
            self.scheduler_infos = [{"status": "ready", "_kvcached_ipc_names": ["owned"]}]
            self.all_child_pids = [pid]
            self.wait_for_ready = lambda: None

    engine = types.SimpleNamespace(
        Engine=type("Engine", (), {}), SchedulerInitResult=InitResult, kill_process_tree=native_kill
    )
    assert module.SGLangShutdownPatch().patch_shutdown(
        engine,
        types.SimpleNamespace(Scheduler=Scheduler),
        types.SimpleNamespace(kill_process_tree=native_kill),
    )
    result = InitResult()
    result.wait_for_ready()
    if replace:
        Path(directory, "replacement").write_bytes(b"replacement")
        Path(directory, "replacement").replace(Path(directory, "owned"))

    def interrupt(signum, frame):
        raise SystemExit(0)

    if handler == "framework":
        signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        # Publishing readiness lets the parent signal immediately. Install the
        # finally first so SIGINT cannot land between ready and cleanup scope.
        writer.send(pid)
        writer.close()
        signal.pause()
    finally:
        frontend.kill_process_tree(os.getpid(), include_parent=False)
        worker.join(5)


@pytest.mark.parametrize("entrypoint", ["sglang.launch_server", "sglang.cli.serve", "__main__"])
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("delivery", ["parent", "group"])
@pytest.mark.parametrize("handler", ["framework", "default"])
def test_frontend_signal_finally_cleans_only_exited_workers(
    tmp_path, entrypoint, signum, replace, delivery, handler
):
    context = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    parent = context.Process(
        target=_signal_owner, args=(writer, str(tmp_path), entrypoint, replace, delivery, handler)
    )
    parent.start()
    writer.close()
    worker_pid = None
    try:
        assert reader.poll(30), "owner readiness timed out"
        worker_pid = reader.recv()
        assert parent.pid is not None
        if delivery == "group":
            os.killpg(parent.pid, signum)
        else:
            os.kill(parent.pid, signum)
        parent.join(15)
        expected = -signal.SIGTERM if handler == "default" and signum == signal.SIGTERM else 0
        assert parent.exitcode == expected
        assert (
            not psutil.pid_exists(worker_pid)
            or psutil.Process(worker_pid).status() == psutil.STATUS_ZOMBIE
        )
        path = tmp_path / "owned"
        assert path.exists()
        if replace:
            assert path.read_bytes() == b"replacement"
    finally:
        reader.close()
        if parent.is_alive():
            parent.kill()
            parent.join(5)
        if worker_pid is not None:
            try:
                psutil.Process(worker_pid).kill()
            except psutil.NoSuchProcess:
                pass
