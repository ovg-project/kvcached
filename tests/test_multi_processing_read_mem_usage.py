import argparse
import time

import numpy as np
import torch

from kvcached.integration.sglang.interfaces import (alloc_kv_cache,
                                                    init_kvcached,
                                                    shutdown_kvcached)
from kvcached.kv_cache_manager import KVCacheManager

kvcache_shape = (643009, 8, 64)
dtype = torch.bfloat16
device = "cuda"
num_layers = 16
page_size = 1
attention_type = "MHA"
kv_layout = "NHD"

head_num = 4
head_dim = 128

num_blocks = 643009
block_size = 1


def run_kv_cached(i):
    """
    The original purpose is to test allocation, freeing, and trimming of PageAllocator,
    as well as testing shm write performance during above operations.
    Comment: 1. Update API usage according to the latest version.
             2. The latest version doesn't have a helper Timer class (Seen in Yifan's repo, e7519f6).
                Thus, I may either measure time using time module, or skip measuring time.
    """

    torch.set_default_device("cuda:0")
    print(f"Initializing kvcached in process {i}")
    init_kvcached()
    print(f"Initialized kvcached in process {i}")
    k_buffer, v_buffer = alloc_kv_cache(
        kvcache_shape,
        dtype,
        device,
        num_layers,
        page_size,
        attention_type,
        kv_layout,
    )

    cell_size = head_num * head_dim * dtype.itemsize

    kv_allocator = KVCacheManager(
        num_blocks,
        block_size,
        cell_size,
        num_layers,
    )

    allocated_indices: list[int] = []

    for i in range(10):
        allocated = kv_allocator.alloc(500)
        if allocated is None:
            raise RuntimeError("Failed to allocate kv cache")
        allocated_indices.extend(allocated)
        time.sleep(1)
    kv_allocator.free(allocated_indices)
    time.sleep(5)

    kv_allocator.trim()
    time.sleep(5)

    print("originally do:")
    print(
        "average setting memory usage time 'np.average(kv_allocator.page_allocator.write_shm_times * 1e6'us"
    )
    print("but this write_shm_times is not available in the latest version")
    shutdown_kvcached()
    return


def read_kv_cached_memory_usage():
    """
    The original purpose is to test the overhead of reading from shm.
    Comment: 1. shm management data structure has changed. Memory usage probing API also changed.
             2. Adapt/Comment out the old API calls for now, for CI error fix.
             3. For updated code: Use MemInfoTracker for initialization,
             4. Use MemInfoStruct and RwLockedShm to get mem info (learn from kvtop).
    """
    from kvcached.cli.utils import MemInfoStruct, RwLockedShm, get_ipc_name
    from kvcached.mem_info_tracker import MemInfoTracker
    from kvcached.utils import DEFAULT_IPC_NAME

    reader = MemInfoTracker(total_mem_size=1024 * 1024 * 1024)
    reader.update_memory_usage(used_size=0, prealloc_size=0)

    times = []
    for i in range(20):
        t0 = time.perf_counter()
        with RwLockedShm(get_ipc_name(DEFAULT_IPC_NAME),
                         MemInfoStruct.SHM_SIZE, RwLockedShm.RLOCK) as mm:
            mem_info = MemInfoStruct.from_buffer(mm)
            mem_in_mb = int(mem_info.used_size) / (1024 * 1024)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"Memory in use: {mem_in_mb}MB")
        time.sleep(1)
    print(f"average reading memory usage time {np.average(times) * 1e6}us")


parser = argparse.ArgumentParser()
parser.add_argument("--num-processes", type=int, default=1)

if __name__ == "__main__":
    """
    The original purpose is to test multi-process allocation followed by memory usage reading.
    Also test shm write performance during allocation/free/trim and shm read performance.
    """
    args = parser.parse_args()
    num_processes = args.num_processes

    processes = []
    context = torch.multiprocessing.get_context("spawn")
    print(f"spawn {num_processes} processes")
    for i in range(num_processes):
        pipe = context.Pipe()
        p = context.Process(target=run_kv_cached, args=(i, ))
        p.start()
        processes.append(p)

    read_kv_cached_memory_usage()
    for p in processes:
        p.join()

    print("shutdown kvcached")
