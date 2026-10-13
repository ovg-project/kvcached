# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Preserve owner SIGTERM/child shutdown, independently of control files.

Control usage lives in the common tracker. No scheduler/DP readiness metadata,
control identity capture, unlink wrappers, or cleanup waits belong here. The
remaining readiness boundary records child ownership for the existing signal
bridge, including synchronous Engine loops stopped between requests.
"""

import functools
import os
import signal

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.version_utils import VersionAwarePatch
from kvcached.utils import (
    get_kvcached_logger,
    register_owner_sigterm_cleanup,
    register_owner_sigterm_loop,
)

logger = get_kvcached_logger()


class _ShutdownOwner:
    def __init__(self, processes, stop_children):
        self.pid = os.getpid()
        self.processes = processes
        self.stop_children = stop_children

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
        if self.pid == os.getpid() and self.has_live_children():
            self.stop_children(os.getpid(), include_parent=False)


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
        InitResult = getattr(engine_mod, "SchedulerInitResult", None)
        if InitResult is None:
            self.logger.warning("No SchedulerInitResult; owner signal bridge unavailable")
            return False
        original_init = InitResult.__init__
        # The signal registry keeps weak callbacks. Retain the owners at this
        # process's readiness boundary, without retaining any control-file fd.
        owners: list[_ShutdownOwner] = []

        @functools.wraps(original_init)
        def result_init(result, *args, **kwargs):
            original_init(result, *args, **kwargs)
            original_ready = result.wait_for_ready

            @functools.wraps(original_ready)
            def ready(*ready_args, **ready_kwargs):
                value = original_ready(*ready_args, **ready_kwargs)
                if enable_kvcached():
                    owners[:] = [
                        owner
                        for owner in owners
                        if owner.pid == os.getpid() and owner.has_live_children()
                    ]
                    owner = _capture_owner(result, engine_mod.kill_process_tree)
                    if owner is not None:
                        owners.append(owner)
                        register_owner_sigterm_cleanup(owner.shutdown)
                return value

            result.wait_for_ready = ready

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
                    register_owner_sigterm_loop(loop, previous)
                return result

            TokenizerManager.auto_create_handle_loop = create_handle_loop

        InitResult.__init__ = result_init
        self._mark_as_patched(Engine)
        return True


def _capture_owner(init_result, stop_children):
    import psutil

    try:
        if not init_result.all_child_pids:
            return None
        processes = tuple(psutil.Process(pid) for pid in init_result.all_child_pids)
        if any(
            not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
            for process in processes
        ):
            return None
        return _ShutdownOwner(processes, stop_children)
    except Exception as error:
        logger.warning("Cannot establish SGLang child ownership: %s", error)
        return None
