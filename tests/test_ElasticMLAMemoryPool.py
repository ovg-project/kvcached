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

# Include pools whose logical byte count fits but whole-block packing does not.
size, page_size, backing_blocks, physical_pages, kv_lora_rank, physical_page_mb = {
    "small": (1024, 64, 29, 1, 512, 2),
    "page64": (8192, 64, 143, 5, 512, 2),
    "packing": (16320, 64, 285, 10, 512, 2),
    "page32": (8192, 32, 285, 5, 512, 2),
    "oversized": (256, 16, 17, 17, 131008, 4),
    "oversized_packing": (256, 16, 27, 11, 81856, 6),
    # This case is run with KVCACHED_PAGE_SIZE_MB=6.
    "oversized_explicit": (256, 16, 26, 17, 131008, 6),
}[os.environ.get("KVCACHED_TEST_MLA_CAPACITY", "small")]
dtype = torch.bfloat16
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
        assert allocator.page_size == physical_page_mb * 1024 * 1024
        assert allocator.num_blocks == backing_blocks
        assert allocator.page_allocator.get_num_total_pages() == physical_pages

        logical_allocator = PagedTokenToKVPoolAllocator(
            size, page_size, dtype, device, pool, False
        )
        assert logical_allocator.available_size() == size
        print(f"  available_size: {logical_allocator.available_size()}")

        if os.environ.get("KVCACHED_TEST_MLA_RESERVE") == "1":
            assert allocator.try_to_reserve(size // page_size)
            assert len(allocator.reserved_blocks) == size // page_size
            assert logical_allocator.available_size() == size

        # Fill the logical pool, reject the extra backing-only capacity, and
        # verify that the mapped CUDA rows are writable through the inherited
        # MLA write path.
        indices = logical_allocator.alloc(size)
        assert indices is not None and len(indices) == size
        assert indices.unique().numel() == size
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

        loc = indices
        expected = torch.arange(
            pool.kv_cache_dim, dtype=torch.float32, device=device
        ).to(dtype).reshape(1, 1, pool.kv_cache_dim).expand(size, 1, -1).contiguous()
        for layer in range(layer_num):
            pool.set_kv_buffer(
                SimpleNamespace(layer_id=pool.start_layer + layer), loc, expected, expected
            )
            torch.testing.assert_close(pool.kv_buffer[layer][loc].cpu(), expected.cpu())

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
