# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Release control segments after SGLang's owning child processes exit."""

import functools
import inspect
import os
import threading
import types
from typing import Any

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.version_utils import VersionAwarePatch
from kvcached.utils import SHM_DIR, IPCSegmentCleanup, get_kvcached_logger

logger = get_kvcached_logger()
_SEGMENTS_KEY = "_kvcached_ipc_names"


class _DPReadinessConnection:
    """Carry scheduler metadata across the controller's rebuilt ready message."""

    def __init__(self, connection, names, lock):
        self.connection = connection
        self.names = names
        self.lock = lock

    def recv(self):
        info = self.connection.recv()
        if isinstance(info, dict) and info.get("status") == "ready":
            with self.lock:
                self.names.update(info.get(_SEGMENTS_KEY, ()))
        return info

    def send(self, info):
        if isinstance(info, dict) and info.get("status") == "ready":
            with self.lock:
                info = {**info, _SEGMENTS_KEY: sorted(
                    self.names.union(info.get(_SEGMENTS_KEY, ())))}
        return self.connection.send(info)

    def __getattr__(self, name):
        return getattr(self.connection, name)


def _run_data_parallel_controller_process(*args, **kwargs):
    # Keep this target at module scope (without wraps) so multiprocessing spawn
    # can resolve it. The controller's original entrypoint remains untouched.
    from sglang.srt.managers import data_parallel_controller as controller

    original = controller.run_data_parallel_controller_process
    if not enable_kvcached():
        return original(*args, **kwargs)

    bound = inspect.signature(original).bind(*args, **kwargs)
    names: set[str] = set()
    lock = threading.Lock()
    original_mp = controller.mp

    def pipe(*pipe_args, **pipe_kwargs):
        reader, writer = original_mp.Pipe(*pipe_args, **pipe_kwargs)
        # Only the reader stays in the controller; leave the writer passed to
        # scheduler subprocesses unchanged and picklable.
        return _DPReadinessConnection(reader, names, lock), writer

    bound.arguments["pipe_writer"] = _DPReadinessConnection(
        bound.arguments["pipe_writer"], names, lock)
    # Each DP group receives readiness in its own thread. Observe only pipes
    # created by this controller module, never global multiprocessing.Pipe.
    controller.mp = types.SimpleNamespace(**vars(original_mp))
    controller.mp.Pipe = pipe
    try:
        return original(*bound.args, **bound.kwargs)
    finally:
        controller.mp = original_mp


class _ShutdownOwner:
    def __init__(self, processes, segments):
        self.pid = os.getpid()
        self.processes = processes
        self.segments = segments

    def cleanup(self):
        import psutil

        if self.pid != os.getpid():
            return False
        try:
            for process in self.processes:
                try:
                    if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                        return False
                except psutil.NoSuchProcess:
                    pass
        except Exception as e:
            logger.warning("Cannot confirm SGLang worker exit; keeping KV segments: %s", e)
            return False
        done = True
        for segment in self.segments:
            done &= segment.unlink()
        return done


class SGLangShutdownPatch(VersionAwarePatch, BasePatch):
    library = "sglang"
    target_module = "sglang.srt.entrypoints.engine"
    target_class = "Engine"
    patch_name = "sglang_shutdown"

    def apply(self, engine_mod) -> bool:
        if not self.initialize_version_info():
            return False
        from sglang.srt.managers import scheduler, tokenizer_manager

        return self.patch_shutdown(engine_mod, scheduler, tokenizer_manager)

    def patch_shutdown(self, engine_mod, scheduler_mod, tokenizer_mod) -> bool:
        Engine = self._get_target_class(engine_mod)
        if Engine is None:
            return False
        if self._is_already_patched(Engine):
            return True
        Scheduler = scheduler_mod.Scheduler
        original_info = Scheduler.get_init_info
        original_launch = Engine._launch_subprocesses.__func__
        owners: list[_ShutdownOwner] = []
        lock = threading.RLock()

        @functools.wraps(original_info)
        def get_init_info(self, *args, **kwargs):
            info = original_info(self, *args, **kwargs)
            if enable_kvcached():
                from kvcached.pool_registry import get_registered_kv_cache_pools

                info[_SEGMENTS_KEY] = sorted({
                    pool.ipc_name for pool, _ in
                    get_registered_kv_cache_pools(integration="sglang")
                })
            return info

        @functools.wraps(original_launch)
        def launch(cls, *args, **kwargs):
            result = original_launch(cls, *args, **kwargs)
            if enable_kvcached():
                # _launch_subprocesses returns the readiness records and local
                # child PIDs in its fourth item on all supported releases.
                owner = _capture_owner(result[3])
                if owner is not None:
                    with lock:
                        owners.append(owner)
            return result

        def wrap_kill(original):
            signature = inspect.signature(original)

            @functools.wraps(original)
            def kill(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                params = bound.arguments
                with lock:
                    owned = any(owner.pid == os.getpid() for owner in owners)
                if (not enable_kvcached() or not owned
                        or params["parent_pid"] not in (None, os.getpid())):
                    return original(*args, **kwargs)
                kill_self = (params["parent_pid"] == os.getpid()
                             and params["include_parent"])
                if kill_self:
                    # Older HTTP shutdown kills the parent before waiting for
                    # workers, bypassing finally/atexit. Reap children first,
                    # then preserve SGLang's original parent-exit behavior.
                    params["include_parent"] = False
                    if params["wait_timeout"] is None:
                        params["wait_timeout"] = 60
                try:
                    return original(**params)
                finally:
                    try:
                        with lock:
                            owners[:] = [owner for owner in owners if not owner.cleanup()]
                    except Exception as e:
                        logger.warning("Failed to clean up SGLang KV segments: %s", e)
                    finally:
                        if kill_self:
                            original(*args, **kwargs)

            return kill

        Scheduler.get_init_info = get_init_info
        Engine._launch_subprocesses = classmethod(launch)
        engine_mod.run_data_parallel_controller_process = _run_data_parallel_controller_process
        # Keep the hook on these owning entrypoints, not SGLang's global
        # process utility, which is also used by workers and unrelated tools.
        engine_mod.kill_process_tree = wrap_kill(engine_mod.kill_process_tree)
        tokenizer_mod.kill_process_tree = wrap_kill(tokenizer_mod.kill_process_tree)
        self._mark_as_patched(Engine)
        return True


def _capture_owner(init_result):
    import psutil

    segments: list[Any] = []
    try:
        names = {name for info in init_result.scheduler_infos
                 for name in info.pop(_SEGMENTS_KEY, [])}
        if not names or not init_result.all_child_pids:
            return None
        # psutil retains each PID's creation time, so PID reuse cannot turn a
        # stopped worker into an unrelated process during the exit check.
        processes = tuple(psutil.Process(pid) for pid in init_result.all_child_pids)
        for name in sorted(names):
            segments.append(IPCSegmentCleanup(os.path.join(SHM_DIR, name)))
        return _ShutdownOwner(processes, segments)
    except Exception as e:
        for segment in segments:
            segment.close()
        logger.warning("Cannot capture SGLang KV segments; leaving cleanup to kvctl: %s", e)
        return None
