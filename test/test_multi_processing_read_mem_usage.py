import argparse
import time

import numpy as np
import torch

from kvcached import ops as kvcached_ops
from kvcached.ops import init_kvcached, shutdown_kvcached
from kvcached.slab_allocator import KVCacheManager

total_tokens = 10000
head_num = 32
head_dim = 128
dtype = torch.float16
device = "cuda:0"
layer_num = 32


def run_kv_cached(i):

    torch.set_default_device("cuda:0")
    print(f"Initializing kvcached in process {i}")
    init_kvcached()
    print(f"Initialized kvcached in process {i}")
    k_buffer, v_buffer = kvcached_ops.sgl_alloc_kv_cache(
        total_tokens,
        head_num,
        head_dim,
        dtype,
        device,
        layer_num,
    )

    cell_size = head_num * head_dim * dtype.itemsize

    kv_allocator = KVCacheManager(total_tokens,
                                  1,
                                  cell_size,
                                  num_layers=layer_num,
                                  ipc_name="ipc_gpu_id_x_model_id_y")

    allocated_indices = []

    for i in range(10):
        allocated_indices.extend(kv_allocator.alloc(500))
        time.sleep(1)
    kv_allocator.free(allocated_indices)
    time.sleep(5)

    kv_allocator.trim()
    time.sleep(5)

    print(
        f"average setting memory usage time {np.average(kv_allocator.page_allocator.write_shm_times) * 1e6}us"
    )
    shutdown_kvcached()
    return


def read_kv_cached_memory_usage():
    from kvcached.slab_allocator import MemoryUsageReader

    reader = MemoryUsageReader("ipc_gpu_id_x_model_id_y")

    times = []
    for i in range(20):
        t0 = time.perf_counter()
        mem_in_mb = reader.get_memory_usage_in_mb()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"Memory in use: {mem_in_mb}MB")
        time.sleep(1)
    print(f"average reading memory usage time {np.average(times) * 1e6}us")


parser = argparse.ArgumentParser()
parser.add_argument("--num-processes", type=int, default=1)

if __name__ == "__main__":
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
