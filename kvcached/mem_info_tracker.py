# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

from kvcached import control_segment
from kvcached.cli.utils import MemInfoStruct, RwLockedShm, get_ipc_name
from kvcached.utils import DEFAULT_IPC_NAME


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
        self._fd: Optional[int] = None
        shm = RwLockedShm(self.ipc_name, MemInfoStruct.SHM_SIZE,
                          RwLockedShm.WLOCK, retain_usage=True)
        with shm as mm:
            fd = os.dup(shm.file.fileno())
            try:
                MemInfoStruct(total_mem_size, 0, 0).write_to_buffer(mm)
                control_segment.publish(fd)
            except BaseException:
                os.close(fd)
                raise
            self._fd = fd

    def close(self):
        """Release only this reference after users stop; leave removal to kvctl."""
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)

    def __del__(self):
        if getattr(self, "_fd", None) is not None:
            self.close()

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
