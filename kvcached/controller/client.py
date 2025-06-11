import logging

import numpy as np

from kvcached.controller.utils import (DEFAULT_IPC_NAME, RwLockedShm,
                                       get_kv_cache_limit)

logger = logging.getLogger(__name__)


class MemoryInfoClient:

    def __init__(self, ipc_name: str = DEFAULT_IPC_NAME):
        self.ipc_name = ipc_name

    def charge(self, size: int) -> bool:
        with RwLockedShm(self.ipc_name,
                         np.int64().itemsize * 2, RwLockedShm.WLOCK) as mm:
            mem_info = np.ndarray((2, ), dtype=np.int64, buffer=mm)
            if mem_info[1] + size > mem_info[0]:
                logger.warning(f"Not enough memory available for {size} bytes")
                return False
            mem_info[1] += size
            return True

    def uncharge(self, size: int) -> bool:
        with RwLockedShm(self.ipc_name,
                         np.int64().itemsize * 2, RwLockedShm.WLOCK) as mm:
            mem_info = np.ndarray((2, ), dtype=np.int64, buffer=mm)
            mem_info[1] -= size
            return True

    def get_available_memory_in_bytes(self) -> int:
        info = get_kv_cache_limit(self.ipc_name)
        if info is None:
            logger.warning(f"No memory info found for {self.ipc_name}")
            return 0
        return info[0] - info[1]

    def get_available_memory_in_mb(self) -> float:
        return self.get_available_memory_in_bytes() / 1024 / 1024

    def get_available_memory_in_gb(self) -> float:
        return self.get_available_memory_in_bytes() / 1024 / 1024 / 1024
