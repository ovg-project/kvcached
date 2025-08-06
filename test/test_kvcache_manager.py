import gc

import torch

from kvcached.integration.sglang.interfaces import (alloc_kv_cache,
                                                    init_kvcached,
                                                    shutdown_kvcached)
from kvcached.kv_cache_manager import KVCacheManager

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
    contiguous_layout=True,
    reserve_null_block=True,
)
print("Initialized KV cache manager")

cur_gpu_memory = torch.cuda.mem_get_info(device)[0]
print("-" * 100)
need_size = 50000
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
target_size = total_tokens - 50000
print(
    f"Try to resize to {50000}. GPU memory before resize: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
kv_cache_manager.resize(50000)
print(
    f"Available size after resize to {50000}: {kv_cache_manager.available_size()}"
)

print("-" * 100)
print(
    f"Try to allocate {20000} tokens. GPU memory before allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
indices_2 = kv_cache_manager.alloc(20000)
if indices_2 is not None:
    print(
        f"Successfully allocated {20000} tokens. GPU memory after allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
    )
else:
    print(f"Failed to allocate {20000} tokens")
cur_gpu_memory = torch.cuda.mem_get_info(device)[0]

print("-" * 100)
print(
    f"Freeing {50000} tokens. GPU memory before free: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)

if indices is not None:
    kv_cache_manager.free(indices[:50000])
    cur_gpu_memory = torch.cuda.mem_get_info(device)[0]
    print(
        f"GPU memory after free {50000} tokens: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
    )

print("-" * 100)
print(f"Allocating {20000} tokens")
indices_3 = kv_cache_manager.alloc(20000)
if indices_3 is not None:
    print(
        f"Successfully allocated {len(indices_3)} tokens. GPU memory after allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
    )
else:
    print(f"Failed to allocate {20000} tokens")
cur_gpu_memory = torch.cuda.mem_get_info(device)[0]

print("-" * 100)
print(
    f"Try to resize to {20000}. GPU memory before resize: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
kv_cache_manager.resize(20000)
print(
    f"Available size after resize to {20000}: {kv_cache_manager.available_size()}"
)

print("-" * 100)
print(
    f"Try to resize to {100000}. GPU memory before resize: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
kv_cache_manager.resize(100000)
print(
    f"Available size after resize to {100000}: {kv_cache_manager.available_size()}"
)

print("-" * 100)
print(
    f"Try to allocate {50000} tokens. GPU memory before allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
indices_4 = kv_cache_manager.alloc(50000)
if indices_4 is not None:
    print(
        f"Successfully allocated {len(indices_4)} tokens. GPU memory after allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
    )
else:
    print(f"Failed to allocate {50000} tokens")

print("-" * 100)
print(
    f"Try to trim. GPU memory before trim: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
kv_cache_manager.trim()
print(f"Available size after trim: {kv_cache_manager.available_size()}")

print("-" * 100)
print(
    f"Try to allocate {50000} tokens. GPU memory before allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
indices_5 = kv_cache_manager.alloc(50000)
if indices_5 is not None:
    print(
        f"Successfully allocated {len(indices_5)} tokens. GPU memory after allocation: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
    )
else:
    print(f"Failed to allocate {50000} tokens")

print("-" * 100)
print(
    f"Try to resize to {0}. GPU memory before resize: {torch.cuda.mem_get_info(device)[0] / 1024 / 1024} MB. Available size: {kv_cache_manager.available_size()}"
)
kv_cache_manager.resize(0)
print(
    f"Available size after resize to {0}: {kv_cache_manager.available_size()}")

print("-" * 100)
kv_cache_manager.free(indices_4)
kv_cache_manager.free(indices_3)

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
input("Press Enter to continue")
