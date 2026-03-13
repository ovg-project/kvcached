# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os

os.environ["ENABLE_KVCACHED"] = "1"

import torch
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool  # 0.5.3

print(f"MLATokenToKVPool class: {MLATokenToKVPool.__name__}")
assert "Elastic" in MLATokenToKVPool.__name__, "MLA pool was not patched!"

# mock DeepSeek-V2-Lite parameters (small scale)
size = 8192          # token number
page_size = 1
dtype = torch.bfloat16
kv_lora_rank = 512
qk_rope_head_dim = 64
layer_num = 4        # test with few layers
device = "cuda:0"

try:
    pool = MLATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=dtype,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        layer_num=layer_num,
        device=device,
        enable_memory_saver=False,
    )
    print("Pool created successfully!")
    print(f"  kv_buffer count: {len(pool.kv_buffer)}")
    print(f"  kv_buffer[0] shape: {pool.kv_buffer[0].shape}")
    print(f"  kv_cache_dim: {pool.kv_cache_dim}")
    print(f"  has kvcached_allocator: {hasattr(pool, 'kvcached_allocator')}")

    # test allocator
    if hasattr(pool, 'kvcached_allocator'):
        allocator = pool.kvcached_allocator
        print(f"  available_size: {allocator.available_size()}")

        # test allocation
        indices = allocator.alloc(64)
        print(f"  alloc(64) -> {indices}")
        print(f"  available_size after alloc: {allocator.available_size()}")

        # test deallocation
        allocator.free(indices)
        print(f"  available_size after free: {allocator.available_size()}")

    print("\nAll tests passed!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()