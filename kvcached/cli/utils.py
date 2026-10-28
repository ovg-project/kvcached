# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import fcntl
import mmap
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from kvcached import control_segment
from kvcached.utils import SHM_DIR, get_tp_socket_dir


def get_ipc_path(ipc_name: str) -> str:
    """Convert IPC name to full path in /dev/shm."""
    if ipc_name.startswith('/'):
        return ipc_name
    return os.path.join(SHM_DIR, ipc_name)


def get_ipc_name(ipc_path: str) -> str:
    """Convert full path to IPC name for posix_ipc."""
    return os.path.basename(ipc_path)


def get_segment_control(ipc_name: str, *, observe: bool = True):
    """Resolve an advertised SGLang group segment to its instance ledger."""
    from kvcached.control import _get_instance_memory_limit, get_instance_memory_limit

    def query(name: str):
        if observe:
            return get_instance_memory_limit(name)
        return _get_instance_memory_limit(name, observe=False)

    name = get_ipc_name(ipc_name)
    state = query(name)
    # Prefer an exact instance, including user names that end in _g<digits>.
    base, separator, group = name.rpartition("_g")
    if not state["gpus"] and separator and base and group.isdecimal() and int(group) > 0:
        owner = query(base)
        if owner["gpus"]:
            return owner
    return state


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

    def __init__(self, file_path: str, size: int, lock_type: int, *, retain_usage: bool = False, create: bool = True):
        self.file_path = get_ipc_path(file_path)
        # Readers open read-only so a segment owned by another user (e.g. root
        # in a container sharing /dev/shm) stays readable; flock(LOCK_SH)
        # works on a read-only fd.
        self.mode = "rb" if lock_type == self.RLOCK else "r+b"
        self.size = size
        self.lock_type = lock_type
        self.retain_usage = retain_usage
        self.create = create

    def __enter__(self):
        flags = (os.O_RDONLY if self.mode == "rb" else os.O_RDWR) | os.O_CLOEXEC
        if self.lock_type == self.WLOCK and self.create:
            flags |= os.O_CREAT
        if self.retain_usage:
            flags |= getattr(os, "O_NOFOLLOW", 0)
        for _ in range(4):
            self.file = os.fdopen(os.open(self.file_path, flags, 0o666), self.mode)
            try:
                if self.retain_usage and sys.platform.startswith("linux"):
                    control_segment.lock(self.file.fileno())
                fcntl.flock(self.file, self.lock_type)
                if self.retain_usage and not control_segment.is_current(self.file.fileno(), self.file_path):
                    self.file.close()
                    continue
                if os.fstat(self.file.fileno()).st_size < self.size and self.lock_type == self.WLOCK:
                    self.file.truncate(self.size)
                access = mmap.ACCESS_READ if self.lock_type == self.RLOCK else mmap.ACCESS_WRITE
                self.mm = mmap.mmap(self.file.fileno(), self.size, access=access)
                return self.mm
            except BaseException:
                self.file.close()
                raise
        raise OSError("control segment changed during initialization")

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.mm.close()
        finally:
            # Only the data flock is unlocked. The OFD lease is close-only, so
            # a retained duplicate or forked child keeps its usage protection.
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()


def init_kv_cache_limit(ipc_name: str, kv_cache_limit: int):
    """
    Set the kv cache limit for the current process.
    Creates a persistent shared memory file that remains even after the script exits.
    """
    shm = RwLockedShm(ipc_name, MemInfoStruct.SHM_SIZE,
                      RwLockedShm.WLOCK, retain_usage=True)
    with shm as mm:
        mem_info = MemInfoStruct(kv_cache_limit, 0, 0)
        mem_info.write_to_buffer(mm)
        control_segment.publish(shm.file.fileno())
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


def update_kv_cache_limit(ipc_name: str, kv_cache_limit: int):
    """Publish through the same native path as the Python control frontend."""
    import json

    from kvcached.control import set_instance_memory_limit

    result = set_instance_memory_limit(kv_cache_limit, instance_name=get_segment_control(ipc_name, observe=False)["instance"])
    print(json.dumps(result, sort_keys=True))
    return result


def update_kv_cache_controls(ipc_name: str, *, min_bytes: Optional[int] = None,
                             priority: Optional[int] = None,
                             headroom_bytes: Optional[int] = None):
    """Publish min, priority or headroom like update_kv_cache_limit()."""
    import json

    from kvcached.control import set_instance_controls

    result = set_instance_controls(
        min_bytes=min_bytes, priority=priority, headroom_bytes=headroom_bytes,
        instance_name=get_segment_control(ipc_name, observe=False)["instance"])
    print(json.dumps(result, sort_keys=True))
    return result


# ---------------------------------------------------------------------------
# IPC cleanup helpers
# ---------------------------------------------------------------------------


def delete_kv_cache_segment(ipc_name: str) -> bool:
    """Remove the shared-memory segment and backing file for *ipc_name*.

    Returns True if the segment existed and was removed, False if it was not
    found. Any other exception is propagated so callers can handle unexpected
    errors.
    """
    return control_segment.delete(get_ipc_path(get_ipc_name(ipc_name)))


def delete_tp_socket_dir(ipc_name: str) -> bool:
    """Remove the TP worker socket directory an engine instance using
    *ipc_name* created under /tmp (see kvcached.tp_ipc_util).

    Returns True if the directory existed and was removed, False otherwise.
    """
    socket_dir = get_tp_socket_dir(get_ipc_name(ipc_name))
    if not os.path.isdir(socket_dir):
        return False
    shutil.rmtree(socket_dir)
    return True


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
