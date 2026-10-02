# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

os.environ["ENABLE_KVCACHED"] = "1"

import torch
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

print(f"MLATokenToKVPool class: {MLATokenToKVPool.__name__}")
assert "Elastic" in MLATokenToKVPool.__name__, "MLA pool was not patched!"

# Small-pool regression: the logical capacity is below one 2 MiB kvcached
# physical page. The manager still needs one physical page to reserve block 0.
size = 1024          # token number
page_size = 64
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

    # Test the real page allocator and wait explicitly for null-block setup.
    if hasattr(pool, 'kvcached_allocator'):
        allocator = pool.kvcached_allocator
        assert allocator._post_init_done.wait(timeout=10.0)
        assert allocator.null_block == [0]
        assert allocator.num_blocks == 29
        assert allocator.page_allocator.get_num_total_pages() == 1

        logical_allocator = PagedTokenToKVPoolAllocator(
            size, page_size, dtype, device, pool, False
        )
        assert logical_allocator.available_size() == size
        print(f"  available_size: {logical_allocator.available_size()}")

        # Fill the logical pool, reject the extra backing-only capacity, and
        # verify that the mapped CUDA rows are writable through the inherited
        # MLA write path.
        indices = logical_allocator.alloc(size)
        assert indices is not None and len(indices) == size
        print(
            f"  alloc({size}) -> [{indices[0].item()}, ..., "
            f"{indices[-1].item()}]"
        )
        assert logical_allocator.available_size() == 0
        assert logical_allocator.alloc(page_size) is None
        print(
            "  available_size after alloc: "
            f"{logical_allocator.available_size()}"
        )

        loc = torch.tensor(
            [indices[0].item()], dtype=torch.int64, device=device
        )
        expected = torch.arange(
            pool.kv_cache_dim, dtype=torch.float32, device=device
        ).to(dtype).reshape(1, 1, pool.kv_cache_dim)
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=pool.start_layer), loc, expected, expected
        )
        torch.testing.assert_close(pool.kv_buffer[0][loc].cpu(), expected.cpu())

        # test deallocation
        logical_allocator.free(indices)
        assert logical_allocator.available_size() == size
        print(
            "  available_size after free: "
            f"{logical_allocator.available_size()}"
        )

    print("\nAll tests passed!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    raise
