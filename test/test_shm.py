import time
from multiprocessing import shared_memory

import numpy as np


class MemoryUsageReader:

    def __init__(self, ipc_name: str, create_timeout=60):

        start_time = time.perf_counter()
        while True:
            try:
                self.shm = shared_memory.SharedMemory(name=ipc_name)
                break
            except FileNotFoundError as e:
                if time.perf_counter() - start_time > create_timeout:
                    raise e
                time.sleep(0.5)
        self._memory_in_use = np.ndarray((2, ),
                                         dtype=np.int64,
                                         buffer=self.shm.buf)

    def get_memory_usage_in_mb(self):
        return self._memory_in_use[0] / 1024 / 1024

    def get_memory_usage_in_gb(self):
        return self._memory_in_use[0] / 1024 / 1024 / 1024

    def get_memory_usage_in_bytes(self):
        return self._memory_in_use[0]

    def __del__(self):
        self.shm.close()


if __name__ == "__main__":
    reader = MemoryUsageReader("kvcached_mem_info")
    print(reader.get_memory_usage_in_mb())
