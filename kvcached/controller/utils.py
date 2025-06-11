import fcntl
import logging
import mmap
import os

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_IPC_NAME = "kvcached_mem_info"
SHM_DIR = "/dev/shm"


def get_ipc_path(ipc_name: str) -> str:
    """Convert IPC name to full path in /dev/shm."""
    if ipc_name.startswith('/'):
        return ipc_name
    return os.path.join(SHM_DIR, ipc_name)


def get_ipc_name(ipc_path: str) -> str:
    """Convert full path to IPC name for posix_ipc."""
    return os.path.basename(ipc_path)


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
        self.file = open(self.file_path, self.mode)
        fcntl.flock(self.file, self.lock_type)
        access = (mmap.ACCESS_READ
                  if self.lock_type == fcntl.LOCK_SH else mmap.ACCESS_WRITE)
        self.mm = mmap.mmap(self.file.fileno(), self.size, access=access)
        return self.mm

    def __exit__(self, exc_type, exc_value, traceback):
        self.mm.close()
        fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()


def get_kv_cache_limit(ipc_name: str) -> np.ndarray:
    """
    Get the kv cache limit for the current process.
    """
    try:
        with RwLockedShm(get_ipc_name(ipc_name),
                         np.int64().itemsize * 2, RwLockedShm.RLOCK) as mm:
            mem_info = np.ndarray((2, ), dtype=np.int64, buffer=mm).copy()
            return mem_info
    except FileNotFoundError:
        return None
