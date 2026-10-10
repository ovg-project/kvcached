# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import signal
from typing import List, Optional

import posix_ipc

from kvcached.cli.utils import (
    MemInfoStruct,
    RwLockedShm,
    get_ipc_name,
    get_ipc_path,
    init_kv_cache_limit,
)
from kvcached.utils import DEFAULT_IPC_NAME

# Process-wide registry.  Per-tracker signal handlers would clobber each other
# (signal.signal replaces, not chains), leaving segments from all but the last
# tracker orphaned on Ctrl-C / SIGTERM.  One handler, many trackers.
_active_trackers: List["MemInfoTracker"] = []
_cleanup_installed: bool = False


def _cleanup_all(*args):
    for tracker in list(_active_trackers):
        tracker._unlink_segment()
    _active_trackers.clear()
    if args and isinstance(args[0], int):
        signum = args[0]
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


def _install_cleanup_handlers():
    global _cleanup_installed
    if _cleanup_installed:
        return
    atexit.register(_cleanup_all)
    for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        try:
            signal.signal(_sig, _cleanup_all)
        except Exception:
            pass
    _cleanup_installed = True


class MemInfoTracker:
    """Tracks memory usage information through shared memory."""

    def __init__(self, total_mem_size: int, group_id: int = 0):
        """
        Args:
            total_mem_size: Total memory size to initialize shared memory with
            group_id: KV cache group id.  group_id=0 uses DEFAULT_IPC_NAME
                unchanged; non-zero groups get a "_g<id>" suffix so multiple
                pools in one process don't share a segment.
        """
        base = DEFAULT_IPC_NAME if group_id == 0 else f"{DEFAULT_IPC_NAME}_g{group_id}"
        self.ipc_name = get_ipc_name(base)
        init_kv_cache_limit(self.ipc_name, total_mem_size)
        _active_trackers.append(self)
        _install_cleanup_handlers()

    def check_and_get_resize_target(self, current_mem_size: int,
                                    num_layers: int,
                                    num_kv_buffers: int = 2) -> Optional[int]:
        """
        Check if memory size has changed and return new target size if needed.

        Returns:
            New memory size if resize is needed, None otherwise
        """
        with RwLockedShm(self.ipc_name, MemInfoStruct.SHM_SIZE,
                         RwLockedShm.RLOCK) as mm:
            mem_info = MemInfoStruct.from_buffer(mm)
            new_mem_size = mem_info.total_size // num_layers // num_kv_buffers
            if new_mem_size != current_mem_size:
                return new_mem_size
        return None

    def update_memory_usage(self, used_size: int, prealloc_size: int):
        """Update the memory usage information in shared memory."""
        with RwLockedShm(self.ipc_name, MemInfoStruct.SHM_SIZE,
                         RwLockedShm.WLOCK) as mm:
            mem_info = MemInfoStruct.from_buffer(mm)
            mem_info.used_size = used_size
            mem_info.prealloc_size = prealloc_size
            mem_info.write_to_buffer(mm)

    def _unlink_segment(self):
        """Remove the POSIX shared-memory segment and its backing file."""
        try:
            posix_ipc.unlink_shared_memory(self.ipc_name)
        except Exception:
            pass
        try:
            os.unlink(get_ipc_path(self.ipc_name))
        except FileNotFoundError:
            pass
