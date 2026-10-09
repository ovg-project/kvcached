# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Clean up SGLang worker sockets while preserving owner SIGTERM handling.

Schedulers report their bound sockets through DP readiness. The owner captures
socket identities while children are alive, then removes matching unserved nodes
only after all recorded children exit. Control-segment lifetime and deletion stay
in the common OFD tracker and explicit kvctl deletion; no control names or file
descriptors are retained here.

Engine, tokenizer, launch_server and cli.serve bind kill_process_tree separately.
Keep these entrypoint and readiness contracts covered when adding engine versions.
"""

import functools
import inspect
import os
import re
import signal
import sys
import threading
import time
import types
from typing import Any, Callable

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.version_utils import VersionAwarePatch
from kvcached.utils import (
    WorkerSocketCleanup,
    get_kvcached_logger,
    get_tp_socket_dir_of,
    register_owner_sigterm_cleanup,
    register_owner_sigterm_loop,
)

logger = get_kvcached_logger()
_SOCKETS_KEY = "_kvcached_worker_sockets"
_SOCKET_NAME = re.compile(r"w\d+\.sock")


class _DPReadinessConnection:
    """Carry scheduler metadata across the controller's rebuilt ready message."""

    def __init__(self, connection, sockets, lock):
        self.connection = connection
        self.sockets = sockets
        self.lock = lock

    def recv(self):
        info = self.connection.recv()
        if isinstance(info, dict) and info.get("status") == "ready":
            with self.lock:
                self.sockets.update(info.get(_SOCKETS_KEY, ()))
        return info

    def send(self, info):
        if isinstance(info, dict) and info.get("status") == "ready":
            with self.lock:
                if self.sockets or _SOCKETS_KEY in info:
                    info = {**info, _SOCKETS_KEY: sorted(
                        self.sockets.union(info.get(_SOCKETS_KEY, ())))}
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
    sockets: set[str] = set()
    lock = threading.Lock()
    original_mp = controller.mp

    def pipe(*pipe_args, **pipe_kwargs):
        reader, writer = original_mp.Pipe(*pipe_args, **pipe_kwargs)
        # Only the reader stays in the controller; leave the writer passed to
        # scheduler subprocesses unchanged and picklable.
        return _DPReadinessConnection(reader, sockets, lock), writer

    bound.arguments["pipe_writer"] = _DPReadinessConnection(
        bound.arguments["pipe_writer"], sockets, lock)
    # Each DP group receives readiness in its own thread. Observe only pipes
    # created by this controller module, never global multiprocessing.Pipe.
    controller.mp = types.SimpleNamespace(**vars(original_mp))
    controller.mp.Pipe = pipe
    try:
        return original(*bound.args, **bound.kwargs)
    finally:
        controller.mp = original_mp


class _ShutdownOwner:
    def __init__(self, processes, stop_children, worker_sockets=None):
        self.pid = os.getpid()
        self.processes = processes
        self.stop_children = stop_children
        self.worker_sockets = worker_sockets

    def has_live_children(self):
        import psutil

        # psutil retains process creation time, so a reused pid cannot revive a
        # completed owner and authorize stopping a later engine's children.
        for process in self.processes:
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def shutdown(self):
        if self.pid != os.getpid():
            return
        if self.has_live_children():
            self.stop_children(os.getpid(), include_parent=False)
        else:
            self.cleanup(time.monotonic() + 1.0)

    def cleanup(self, deadline=0.0):
        import psutil

        if self.pid != os.getpid():
            return False
        try:
            for process in self.processes:
                try:
                    while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                        if time.monotonic() >= deadline:
                            return False
                        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
                except psutil.NoSuchProcess:
                    pass
        except Exception as e:
            logger.warning("Cannot confirm SGLang worker exit; keeping worker sockets: %s", e)
            return False
        # Zombie status can precede the last listener thread releasing its
        # socket. The final entrypoint kill has no subsequent call to retry it.
        while True:
            done = self.worker_sockets is None or self.worker_sockets.unlink()
            if done or time.monotonic() >= deadline:
                return done
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


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
        InitResult = getattr(engine_mod, "SchedulerInitResult", None)
        if InitResult is None:
            self.logger.warning("No SchedulerInitResult; owner cleanup cannot be installed")
            return False
        original_result_init = InitResult.__init__
        owners: list[_ShutdownOwner] = []
        lock = threading.RLock()

        @functools.wraps(original_info)
        def get_init_info(self, *args, **kwargs):
            info = original_info(self, *args, **kwargs)
            if enable_kvcached():
                info[_SOCKETS_KEY] = _bound_worker_sockets()
            return info

        @functools.wraps(original_result_init)
        def result_init(result, *args, **kwargs):
            original_result_init(result, *args, **kwargs)
            # HTTP does not instantiate Engine. Nonzero nodes may never return
            # from _launch_subprocesses. The common readiness callback precedes
            # both the HTTP dispatch and the nonzero-node wait-for-exit loop.
            # Bind known frontend aliases now, after their imports but before
            # readiness; keep upstream's callback result/exception unchanged.
            bind_exit_entrypoints()
            original_ready = result.wait_for_ready

            @functools.wraps(original_ready)
            def ready(*ready_args, **ready_kwargs):
                value = original_ready(*ready_args, **ready_kwargs)
                if enable_kvcached():
                    with lock:
                        owners[:] = [owner for owner in owners
                                     if owner.pid == os.getpid()
                                     and (owner.has_live_children() or not owner.cleanup())]
                        owner = _capture_owner(result, engine_mod.kill_process_tree)
                        if owner is not None:
                            owners.append(owner)
                            register_owner_sigterm_cleanup(owner.shutdown)
                return value

            result.wait_for_ready = ready

        wrappers: dict[Callable[..., Any], Callable[..., Any]] = {}

        def wrap_kill(original):
            if original in wrappers:
                return wrappers[original]
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
                    # The upstream function SIGKILLs itself before its wait.
                    # A finally block cannot run after that. Defer ONLY the
                    # self-kill until the same child-stop call and cleanup have
                    # completed, then invoke the original self-exit unchanged.
                    # Do not replace upstream's wait_timeout or skipped PID.
                    params["include_parent"] = False
                try:
                    return original(**params)
                finally:
                    try:
                        with lock:
                            # Legacy include_parent=False can return immediately
                            # after kill(), and a child that a wait saw become
                            # a zombie can hold its socket a moment longer.
                            # Bound confirmation and release across ALL owners
                            # to one second, as the vLLM fallback does. An
                            # explicitly skipped live child must keep its files.
                            deadline = time.monotonic()
                            if params["skip_pid"] is None:
                                deadline += 1.0
                            owners[:] = [owner for owner in owners
                                         if not owner.cleanup(deadline)]
                    except Exception as e:
                        logger.warning("Failed to clean up SGLang worker sockets: %s", e)
                    finally:
                        if kill_self:
                            original(*args, **kwargs)

            wrappers[original] = kill
            wrappers[kill] = kill
            return kill

        def bind_exit_entrypoints():
            # Deliberately enumerate owning entrypoints, never scan/rewrite
            # arbitrary sys.modules globals or SGLang's shared process utility.
            # HTTP aliases were identified in PR #553 (74bda5f). Bind existing
            # attributes rather than relying on imports from the global utility.
            # Run at each readiness-object construction: imports may follow installation.
            modules = [engine_mod, tokenizer_mod]
            for name in ("sglang.launch_server", "sglang.cli.serve",
                         "sglang.srt.entrypoints.http_server"):
                module = sys.modules.get(name)
                if module is not None:
                    modules.append(module)
            main = sys.modules.get("__main__")
            if getattr(getattr(main, "__spec__", None), "name", None) == "sglang.launch_server":
                modules.append(main)
            for entrypoint in modules:
                original = getattr(entrypoint, "kill_process_tree", None)
                if callable(original):
                    entrypoint.kill_process_tree = wrap_kill(original)

        TokenizerManager = getattr(tokenizer_mod, "TokenizerManager", None)
        original_loop = getattr(TokenizerManager, "auto_create_handle_loop", None)
        if TokenizerManager is not None and callable(original_loop):
            @functools.wraps(original_loop)
            def create_handle_loop(manager, *args, **kwargs):
                creating = getattr(manager, "event_loop", None) is None
                previous = signal.getsignal(signal.SIGTERM)
                result = original_loop(manager, *args, **kwargs)
                loop = getattr(manager, "event_loop", None)
                if creating and loop is not None and enable_kvcached():
                    # Sync Engine requests stop this loop before returning. The
                    # upstream asyncio handler otherwise silently queues TERM
                    # forever between calls. HTTP/async requests retain native
                    # delivery while their loop runs; custom policy is skipped.
                    register_owner_sigterm_loop(loop, previous)
                return result

            TokenizerManager.auto_create_handle_loop = create_handle_loop

        Scheduler.get_init_info = get_init_info
        InitResult.__init__ = result_init
        engine_mod.run_data_parallel_controller_process = _run_data_parallel_controller_process
        # Keep the hook on these owning entrypoints, not SGLang's global
        # process utility, which is also used by workers and unrelated tools.
        bind_exit_entrypoints()
        self._mark_as_patched(Engine)
        return True


def _bound_worker_sockets():
    """The sockets this scheduler's IPC listeners bound (issue #548). None
    where the listener module cannot load, as in a process without the
    extension, or where listing them fails."""
    try:
        from kvcached.tp_ipc_util import worker_socket_paths
    except Exception:
        return []
    try:
        return worker_socket_paths()
    except Exception as e:
        logger.warning("Cannot list this scheduler's IPC listener sockets: %s", e)
        return []


def _capture_owner(init_result, stop_children):
    import psutil

    try:
        sockets = {path for info in init_result.scheduler_infos
                   for path in info.pop(_SOCKETS_KEY, [])}
        if not init_result.all_child_pids:
            return None
        # psutil retains creation time so PID reuse cannot claim later workers.
        processes = tuple(psutil.Process(pid) for pid in init_result.all_child_pids)

        def all_live():
            return all(process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                       for process in processes)

        if not all_live():
            return None
        worker_sockets = _capture_worker_sockets(sockets)
        # A restart may replace pathnames after these workers exit. Only claim
        # sockets while all original workers remain alive on both sides.
        if not all_live():
            return None
        return _ShutdownOwner(processes, stop_children, worker_sockets)
    except Exception as error:
        logger.warning("Cannot establish SGLang child ownership: %s", error)
        return None


def _capture_worker_sockets(paths):
    """Remember the sockets the schedulers reported binding (issue #548):
    w<rank>.sock in a kvcached-tp-* deployment directory or one of its pp<k>
    stage directories, the layout tp_ipc_util binds. Anything else leaves the
    sockets to explicit kvctl deletion; child ownership is still recorded."""
    if not paths:
        return None
    try:
        for path in paths:
            directory = get_tp_socket_dir_of(path)
            if (not os.path.isabs(path)
                    or not _SOCKET_NAME.fullmatch(os.path.basename(path))
                    or not os.path.basename(directory).startswith("kvcached-tp-")):
                raise ValueError(f"unexpected worker socket path {path!r}")
        return WorkerSocketCleanup.for_paths(sorted(paths))
    except Exception as e:
        logger.warning("Cannot remember the SGLang workers' sockets; leaving the "
                       "socket directory to kvctl: %s", e)
        return None
