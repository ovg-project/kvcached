import gc
import os

import torch

contiguous_layout = True
os.environ[
    "KVCACHED_CONTIGUOUS_LAYOUT"] = "true" if contiguous_layout else "false"

from kvcached.integration.sglang.interfaces import (  # noqa: E402
    alloc_kv_cache,
    init_kvcached,
    shutdown_kvcached,
)
from kvcached.kv_cache_manager import KVCacheManager  # noqa: E402

total_tokens = 2864
head_num = 32
head_dim = 128
dtype = torch.float16
device = "cuda:0"
layer_num = 32

print("Initializing kvcached")
init_kvcached()
print("Initialized kvcached")

k_buffer, v_buffer = alloc_kv_cache(
    kvcache_shape=(total_tokens, head_num, head_dim),
    dtype=dtype,
    device=device,
    num_layers=layer_num,
)

print("Initializing KV cache manager")
kv_cache_manager = KVCacheManager(
    num_blocks=total_tokens,
    block_size=1,
    cell_size=head_num * head_dim * dtype.itemsize,
    num_layers=layer_num,
    reserve_null_block=True,
)
print("Initialized KV cache manager")

cur_gpu_memory = torch.cuda.mem_get_info(device)[0]
print("-" * 100)
need_size = 500
print(
    f"Try to allocate {need_size} tokens. GPU memory before allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
indices = kv_cache_manager.alloc(need_size)

cur_gpu_memory = torch.cuda.mem_get_info(device)[0]
if indices is not None:
    print(
        f"Successfully allocated {need_size} tokens. GPU memory after allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
    )
else:
    print(f"Failed to allocate {need_size} tokens")
# input("Press Enter to continue")
print("-" * 100)
target_size = total_tokens - need_size
print(
    f"Try to resize to {target_size}. GPU memory before resize: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
kv_cache_manager.resize(target_size)
print(
    f"Available size after resize to {target_size}: {kv_cache_manager.available_size()}"
)

print("-" * 100)
need_size = 1000
print(
    f"Try to allocate {need_size} tokens. GPU memory before allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
indices_2 = kv_cache_manager.alloc(need_size)
if indices_2 is not None:
    print(
        f"Successfully allocated {need_size} tokens. GPU memory after allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
    )
else:
    print(f"Failed to allocate {need_size} tokens")
cur_gpu_memory = torch.cuda.mem_get_info(device)[0]

print("-" * 100)
free_size = 500
print(
    f"Freeing {free_size} tokens. GPU memory before free: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)

if indices is not None:
    kv_cache_manager.free(indices[:free_size])
    cur_gpu_memory = torch.cuda.mem_get_info(device)[0]
    print(
        f"GPU memory after free {free_size} tokens: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
    )

print("-" * 100)
need_size = 1000
print(f"Allocating {need_size} tokens")
indices_3 = kv_cache_manager.alloc(need_size)
if indices_3 is not None:
    print(
        f"Successfully allocated {len(indices_3)} tokens. GPU memory after allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
    )
else:
    print(f"Failed to allocate {need_size} tokens")
cur_gpu_memory = torch.cuda.mem_get_info(device)[0]

print("-" * 100)
resize_target_size = 1000
print(
    f"Try to resize to {resize_target_size}. GPU memory before resize: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
kv_cache_manager.resize(resize_target_size)
print(
    f"Available size after resize to {resize_target_size}: {kv_cache_manager.available_size()}"
)

print("-" * 100)
resize_target_size = 3000
print(
    f"Try to resize to {resize_target_size}. GPU memory before resize: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
kv_cache_manager.resize(resize_target_size)
print(
    f"Available size after resize to {resize_target_size}: {kv_cache_manager.available_size()}"
)

print("-" * 100)
need_size = 500
print(
    f"Try to allocate {need_size} tokens. GPU memory before allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
indices_4 = kv_cache_manager.alloc(need_size)
if indices_4 is not None:
    print(
        f"Successfully allocated {len(indices_4)} tokens. GPU memory after allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
    )
else:
    print(f"Failed to allocate {need_size} tokens")

print("-" * 100)
print(
    f"Try to trim. GPU memory before trim: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
kv_cache_manager.trim()
print(f"Available size after trim: {kv_cache_manager.available_size()}")

print("-" * 100)
need_size = 2000
print(
    f"Try to allocate {need_size} tokens. GPU memory before allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
indices_5 = kv_cache_manager.alloc(need_size)
if indices_5 is not None:
    print(
        f"Successfully allocated {len(indices_5)} tokens. GPU memory after allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
    )
else:
    print(f"Failed to allocate {need_size} tokens")

print("-" * 100)
resize_target_size = 1
print(
    f"Try to resize to {resize_target_size}. GPU memory before resize: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
kv_cache_manager.resize(resize_target_size)
print(
    f"Available size after resize to {resize_target_size}: {kv_cache_manager.available_size()}"
)

print("-" * 100)
for indices in [indices_5, indices_4, indices_3]:
    if indices is not None:
        kv_cache_manager.free(indices)

print(
    f"Available size after free: {kv_cache_manager.available_size()}, GPU memory after free: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB"
)

shutdown_kvcached()
print("Shutdown kvcached")
cur_gpu_memory = torch.cuda.mem_get_info(device)[0]
del k_buffer, v_buffer
del kv_cache_manager

gc.collect()
torch.cuda.empty_cache()

print(f"GPU memory after shutdown: {cur_gpu_memory / 1024 / 1024} MB. ")
# input("Press Enter to continue")
