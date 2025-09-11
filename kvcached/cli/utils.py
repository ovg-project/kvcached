# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import fcntl
import mmap
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import posix_ipc

from kvcached.utils import SHM_DIR


def get_ipc_path(ipc_name: str) -> str:
    """Convert IPC name to full path in /dev/shm."""
    if ipc_name.startswith('/'):
        return ipc_name
    return os.path.join(SHM_DIR, ipc_name)


def get_ipc_name(ipc_path: str) -> str:
    """Convert full path to IPC name for posix_ipc."""
    return os.path.basename(ipc_path)


@dataclass
class MemInfoStruct:
    total_size: int
    used_size: int
    prealloc_size: int

    DTYPE = np.int64
    N_FIELDS = 3
    SHM_SIZE = np.dtype(DTYPE).itemsize * N_FIELDS

    @classmethod
    def _view(cls, buf: mmap.mmap) -> np.ndarray:
        """Return a live NumPy view onto the shared buffer."""
        return np.ndarray((cls.N_FIELDS, ), dtype=cls.DTYPE, buffer=buf)

    @classmethod
    def from_buffer(cls, buf: mmap.mmap) -> "MemInfoStruct":
        arr = cls._view(buf)
        return cls(int(arr[0]), int(arr[1]),
                   int(arr[2]))  # total, used, prealloc

    def write_to_buffer(self, buf: mmap.mmap) -> None:
        arr = self._view(buf)
        arr[:] = (self.total_size, self.used_size, self.prealloc_size)


class RwLockedShm:
    RLOCK = fcntl.LOCK_SH
    WLOCK = fcntl.LOCK_EX

    def __init__(self, file_path: str, size: int, lock_type: int):
        self.file_path = get_ipc_path(file_path)
        # Always use r+b mode for memory mapping
        self.mode = "r+b"
        self.size = size
        self.lock_type = lock_type

    def __enter__(self):
        """Open the shared-memory file with the requested lock.

        If the file does not yet exist *and* we are taking a write lock, the
        file is created with the requested size so mapping succeeds.  For
        read-only access when the segment is missing we propagate
        FileNotFoundError so the caller can decide what to do (usually treat
        as "no limit set yet").
        """
        try:
            self.file = open(self.file_path, self.mode)
        except FileNotFoundError:
            if self.lock_type != RwLockedShm.WLOCK:
                raise
            # Create the file and pre-size it
            self.file = open(self.file_path, "w+b")
            self.file.truncate(self.size)

        # Ensure the file is large enough for the mapping size
        stat_info = os.fstat(self.file.fileno())
        if stat_info.st_size < self.size and self.lock_type == RwLockedShm.WLOCK:
            self.file.truncate(self.size)

        fcntl.flock(self.file, self.lock_type)
        access = mmap.ACCESS_READ if self.lock_type == fcntl.LOCK_SH else mmap.ACCESS_WRITE
        self.mm = mmap.mmap(self.file.fileno(), self.size, access=access)
        return self.mm

    def __exit__(self, exc_type, exc_value, traceback):
        self.mm.close()
        fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()


def init_kv_cache_limit(ipc_name: str, kv_cache_limit: int):
    """
    Set the kv cache limit for the current process.
    Creates a persistent shared memory file that remains even after the script exits.
    """
    shm = posix_ipc.SharedMemory(get_ipc_name(ipc_name),
                                 posix_ipc.O_CREAT,
                                 size=MemInfoStruct.SHM_SIZE,
                                 mode=0o666)
    shm.close_fd()

    # Now we can safely memory map and write the values
    with RwLockedShm(get_ipc_name(ipc_name), MemInfoStruct.SHM_SIZE,
                     RwLockedShm.WLOCK) as mm:
        mem_info = MemInfoStruct(kv_cache_limit, 0, 0)
        mem_info.write_to_buffer(mm)
        return mem_info


def get_kv_cache_limit(ipc_name: str) -> Optional[MemInfoStruct]:
    """
    Get the kv cache limit for the current process.
    """
    try:
        with RwLockedShm(get_ipc_name(ipc_name), MemInfoStruct.SHM_SIZE,
                         RwLockedShm.RLOCK) as mm:
            return MemInfoStruct.from_buffer(mm)
    except FileNotFoundError:
        return None


def update_kv_cache_limit(ipc_name: str,
                          kv_cache_limit: int) -> Optional[MemInfoStruct]:
    """
    Update the kv cache limit for the current process.
    """
    try:
        with RwLockedShm(get_ipc_name(ipc_name), MemInfoStruct.SHM_SIZE,
                         RwLockedShm.WLOCK) as mm:
            mem_info = MemInfoStruct.from_buffer(mm)
            delta = kv_cache_limit - mem_info.total_size
            if delta < 0:
                if mem_info.used_size > kv_cache_limit:
                    print(
                        f"No enough free space to decrease for the new kv_cache_limit for {ipc_name}"
                    )
            mem_info.total_size = kv_cache_limit
            mem_info.write_to_buffer(mm)
            human_limit = _format_size(kv_cache_limit)
            print(
                f"Updated kv cache limit for {ipc_name} to {human_limit} ({kv_cache_limit} bytes)"
            )
            return mem_info
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# IPC cleanup helpers
# ---------------------------------------------------------------------------


def delete_kv_cache_segment(ipc_name: str) -> bool:
    """Remove the shared-memory segment and backing file for *ipc_name*.

    Returns True if the segment existed and was removed, False if it was not
    found. Any other exception is propagated so callers can handle unexpected
    errors.
    """
    shm_name = get_ipc_name(ipc_name)

    removed = False
    try:
        posix_ipc.unlink_shared_memory(shm_name)
        removed = True
    except posix_ipc.ExistentialError:
        # Segment did not exist.
        pass

    # Best-effort removal of the backing file created by RwLockedShm.
    try:
        os.unlink(get_ipc_path(shm_name))
        removed = True or removed
    except FileNotFoundError:
        pass

    return removed


def get_total_gpu_memory() -> int:
    """Return total memory of CUDA device 0 or 0 if CUDA unavailable."""
    try:
        import torch  # imported lazily to avoid heavy import cost when not needed

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
    except Exception:  # pragma: no cover – best-effort helper
        pass
    return 0


def _format_size(num_bytes: int) -> str:
    """Return human-readable size (B, KB, MB, GB, TB).
    """

    size: float = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024

    raise RuntimeError("_format_size fell through all units")
