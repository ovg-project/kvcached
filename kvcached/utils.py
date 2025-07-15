import fcntl
import logging
import mmap
import os

import numpy as np
import posix_ipc

PAGE_SIZE = 2 * 1024 * 1024  # 2MB


def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def align_up_to_page(n_cells: int, cell_size: int) -> int:
    n_cells_per_page = PAGE_SIZE // cell_size
    aligned_n_cells = align_to(n_cells, n_cells_per_page)
    return aligned_n_cells


def get_log_level():
    import os  # noqa: E501

    level = os.getenv("KVCACHED_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level, logging.INFO)


def get_kvcached_logger(name: str = "kvcached") -> logging.Logger:
    logger = logging.getLogger(name)

    # Only add handler if none exists (prevents duplicate handlers)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"[{name}][%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(get_log_level())
        # Prevent propagation to inference engines; avoid duplicate messages
        logger.propagate = False

    return logger


# Allow overriding the shared-memory segment name via env var so multiple
# kvcached deployments on one machine can coexist without collision.
DEFAULT_IPC_NAME = os.getenv("KVCACHED_IPC_NAME", "kvcached_mem_info")
SHM_DIR = "/dev/shm"
SHM_SIZE = np.int64().itemsize * 2


def get_ipc_path(ipc_name: str) -> str:
    """Convert IPC name to full path in /dev/shm."""
    if ipc_name.startswith('/'):
        return ipc_name
    return os.path.join(SHM_DIR, ipc_name)


def get_ipc_name(ipc_path: str) -> str:
    """Convert full path to IPC name for posix_ipc."""
    return os.path.basename(ipc_path)


class MemInfoStruct:

    def __init__(self, total_size: int, used_size: int):
        self.total_size = total_size
        self.used_size = used_size

    def to_np_array(self) -> np.ndarray:
        return np.array([self.total_size, self.used_size], dtype=np.int64)

    @staticmethod
    def from_np_array(data: np.ndarray) -> "MemInfoStruct":
        return MemInfoStruct(data[0], data[1])


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


def get_kv_cache_limit(ipc_name: str) -> np.ndarray:
    """
    Get the kv cache limit for the current process.
    """
    try:
        with RwLockedShm(get_ipc_name(ipc_name), SHM_SIZE,
                         RwLockedShm.RLOCK) as mm:
            mem_info = np.ndarray((2, ), dtype=np.int64, buffer=mm).copy()
            return mem_info
    except FileNotFoundError:
        return None


def init_kv_cache_limit(ipc_name: str, kv_cache_limit: int):
    """
    Set the kv cache limit for the current process.
    Creates a persistent shared memory file that remains even after the script exits.
    """
    shm = posix_ipc.SharedMemory(get_ipc_name(ipc_name),
                                 posix_ipc.O_CREAT,
                                 size=SHM_SIZE,
                                 mode=0o666)
    shm.close_fd()

    # Now we can safely memory map and write the values
    with RwLockedShm(get_ipc_name(ipc_name), SHM_SIZE,
                     RwLockedShm.WLOCK) as mm:
        mem_info = np.ndarray((2, ), dtype=np.int64, buffer=mm)
        mem_info[:] = [kv_cache_limit, 0]
        return mem_info
