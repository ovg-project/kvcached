# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Exercise DP cleanup across real controller/scheduler processes without GPUs."""

import multiprocessing as mp
import os
import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest

from kvcached.integration.sglang import shutdown

psutil = pytest.importorskip("psutil")


def _scheduler_process(writer, stop, directory, rank):
    names = ["shared", f"dp{rank}_attention", f"dp{rank}_mamba"]
    for name in names:
        Path(directory, name).write_bytes(b"control")
    writer.send({
        "status": "ready", "max_req_input_len": 512,
        "_kvcached_ipc_names": names,
    })
    writer.close()
    if not stop.poll(45):
        raise TimeoutError("test scheduler was not stopped")
    stop.recv()
    stop.close()


def _native_controller(server_args, port_args, pipe_writer,
                       run_scheduler_process_func=_scheduler_process):
    """Use upstream's threaded Pipe/Process/recv/rebuilt-send boundary."""
    from sglang.srt.managers import data_parallel_controller as controller

    workers: list[Any] = [None, None]
    errors = []

    def launch_group(rank):
        reader, writer = controller.mp.Pipe(duplex=False)
        process = controller.mp.Process(
            target=run_scheduler_process_func,
            args=(writer, server_args.stops[rank], server_args.directory, rank),
        )
        process.start()
        writer.close()
        try:
            assert reader.poll(30), "scheduler readiness timed out"
            workers[rank] = (process, reader.recv())
        except Exception as error:
            errors.append(error)
        finally:
            reader.close()

    threads = [threading.Thread(target=launch_group, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(35)
        assert not thread.is_alive()
    assert not errors, errors
    pipe_writer.send({
        "status": "ready",
        "max_req_input_len": workers[0][1]["max_req_input_len"],
        "scheduler_pids": [process.pid for process, _ in workers],
    })
    pipe_writer.close()
    for process, _ in workers:
        process.join(45)
        assert process.exitcode == 0


def _controller_process(target, writer, stops, directory, start_method):
    # SGLang is the external boundary here; the pipes, serialization, process
    # identities and control files are real. Install its stand-in in the fresh
    # interpreter so the test also works with spawn, not only inherited state.
    mp.set_start_method(start_method, force=True)
    for name in ("sglang", "sglang.srt", "sglang.srt.managers"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    controller: Any = types.ModuleType("sglang.srt.managers.data_parallel_controller")
    controller.mp = mp
    controller.run_data_parallel_controller_process = _native_controller
    sys.modules[controller.__name__] = controller
    native_pipe = mp.Pipe
    target(types.SimpleNamespace(stops=stops, directory=directory), None, writer)
    assert controller.mp is mp
    assert mp.Pipe is native_pipe


@pytest.fixture(params=[method for method in ("spawn", "fork")
                        if method in mp.get_all_start_methods()])
def process_engine(monkeypatch, tmp_path, request):
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    monkeypatch.setattr(shutdown, "SHM_DIR", str(tmp_path))
    context = mp.get_context(request.param)
    stop_pipes = [context.Pipe(duplex=False), context.Pipe(duplex=False)]
    controllers: list[Any] = []
    scheduler_pids: list[int] = []
    stopped: set[int] = set()

    def stop_rank(rank):
        if rank not in stopped:
            stop_pipes[rank][1].send("stop")
            stopped.add(rank)

    class Scheduler:
        def get_init_info(self):
            return {"status": "ready"}

    class Engine:
        @classmethod
        def _launch_subprocesses(cls):
            reader, writer = context.Pipe(duplex=False)
            process = context.Process(
                target=_controller_process,
                args=(engine.run_data_parallel_controller_process, writer,
                      [reader for reader, _ in stop_pipes],
                      str(tmp_path), request.param),
            )
            controllers.append(process)
            process.start()
            writer.close()
            try:
                assert reader.poll(40), f"controller readiness timed out: {process.exitcode}"
                info = reader.recv()
            finally:
                reader.close()
            scheduler_pids.extend(info["scheduler_pids"])
            return (None, None, None, types.SimpleNamespace(
                scheduler_infos=[info], all_child_pids=[process.pid, *scheduler_pids],
            ), None)

        def shutdown(self, skip_pid=None):
            return engine.kill_process_tree(os.getpid(), include_parent=False,
                                            skip_pid=skip_pid, wait_timeout=20)

    def kill_process_tree(parent_pid, include_parent=True, skip_pid=None, wait_timeout=None):
        assert parent_pid == os.getpid() and not include_parent
        for rank, pid in enumerate(scheduler_pids):
            if pid != skip_pid:
                stop_rank(rank)
        if skip_pid is None:
            for process in controllers:
                process.join(wait_timeout)
                assert process.exitcode == 0

    engine = types.SimpleNamespace(
        Engine=Engine, kill_process_tree=kill_process_tree,
        run_data_parallel_controller_process=_native_controller,
    )
    assert shutdown.SGLangShutdownPatch().patch_shutdown(
        engine, types.SimpleNamespace(Scheduler=Scheduler),
        types.SimpleNamespace(kill_process_tree=kill_process_tree),
    )
    try:
        yield engine.Engine(), tmp_path
    finally:
        for rank in range(2):
            stop_rank(rank)
        for process in controllers:
            process.join(10)
            if process.is_alive():
                for child in psutil.Process(process.pid).children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.join(5)
        for reader, writer in stop_pipes:
            reader.close()
            writer.close()


@pytest.mark.parametrize("replace_file", [False, True], ids=["cleanup", "replacement"])
def test_dp_shutdown_waits_for_real_workers_and_preserves_replacements(process_engine, replace_file):
    engine, directory = process_engine
    result = engine._launch_subprocesses()
    info = result[3].scheduler_infos[0]
    assert "_kvcached_ipc_names" not in info
    names = ["dp0_attention", "dp0_mamba", "dp1_attention", "dp1_mamba", "shared"]
    assert sorted(path.name for path in directory.iterdir()) == names
    processes = [psutil.Process(pid) for pid in result[3].all_child_pids]

    replacement = directory / "dp1_mamba"
    if replace_file:
        identity = replacement.stat().st_ino
        new_file = directory / "replacement"
        new_file.write_bytes(b"replacement")
        new_file.replace(replacement)
        assert replacement.stat().st_ino != identity

    engine.shutdown(skip_pid=info["scheduler_pids"][1])
    assert sorted(path.name for path in directory.iterdir()) == names
    engine.shutdown()
    engine.shutdown()
    assert all(not process.is_running() for process in processes)
    assert sorted(path.name for path in directory.iterdir()) == (
        ["dp1_mamba"] if replace_file else [])
    if replace_file:
        assert replacement.read_bytes() == b"replacement"
