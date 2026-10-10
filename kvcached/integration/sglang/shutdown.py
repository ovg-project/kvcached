# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Owner-side fallback for SGLang's normal engine and HTTP shutdown.

The scheduler owns the process-local pool registry. Its ready message carries
exact names through the DP controller to the launching process, which retains
file identities while those workers are alive. A pathname/inode match prevents
replacement deletion; only the separately checked worker lifetimes authorize
unlinking. There is no directory sweep or claim about unrelated segment users.

Upstream contracts: get_init_info returns a dict, DP rebuilds its ready dict via
module-local mp.Pipe, and SchedulerInitResult.wait_for_ready fills scheduler_infos
and all_child_pids before returning. Engine, tokenizer, launch_server and cli.serve retain separate imports
of kill_process_tree. Keep these contracts covered when adding engine versions;
patching the utility alone misses aliases bound before plugin loading.
"""

import functools
import inspect
import os
import signal
import sys
import threading
import time
import types
from typing import Any, Callable

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.version_utils import VersionAwarePatch
from kvcached.utils import (
    SHM_DIR,
    IPCSegmentCleanup,
    get_kvcached_logger,
    register_owner_sigterm_cleanup,
    register_owner_sigterm_loop,
)

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
        self.stop_children: Any = None

    def shutdown(self):
        # The ordinary upstream owner operation already defines which children
        # to stop and how long to wait. Its wrapped finally performs cleanup.
        if self.stop_children is not None:
            self.stop_children(os.getpid(), include_parent=False)

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
                from kvcached.pool_registry import get_registered_kv_cache_pools

                info[_SEGMENTS_KEY] = sorted({
                    pool.ipc_name for pool, _ in
                    get_registered_kv_cache_pools(integration="sglang")
                })
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
                    owner = _capture_owner(result)
                    if owner is not None:
                        with lock:
                            owners.append(owner)
                        owner.stop_children = engine_mod.kill_process_tree
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
                            # after kill(). Bound confirmation across ALL owners
                            # to one second, as the vLLM fallback does. An
                            # explicitly skipped live child must keep its files.
                            deadline = time.monotonic()
                            if params["wait_timeout"] is None and params["skip_pid"] is None:
                                deadline += 1.0
                            owners[:] = [owner for owner in owners
                                         if not owner.cleanup(deadline)]
                    except Exception as e:
                        logger.warning("Failed to clean up SGLang KV segments: %s", e)
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

        def all_live():
            return all(process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                       for process in processes)

        if not all_live():
            return None
        for name in sorted(names):
            if not isinstance(name, str) or not name or os.path.basename(name) != name:
                raise ValueError("invalid scheduler control segment name")
            segments.append(IPCSegmentCleanup(os.path.join(SHM_DIR, name)))
        # Do not claim a pathname after its original worker exited and a
        # same-name deployment may have taken over. This is still a lifecycle
        # assumption, not a cross-deployment global ownership protocol.
        if not all_live():
            for segment in segments:
                segment.close()
            return None
        return _ShutdownOwner(processes, segments)
    except Exception as e:
        for segment in segments:
            segment.close()
        logger.warning("Cannot capture SGLang KV segments; leaving cleanup to kvctl: %s", e)
        return None
