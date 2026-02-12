# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Demonstrates the zero_page aliasing problem when page_size > 1 and
KVCacheManager.alloc() is NOT called — i.e. when SGLang's default
PagedTokenToKVPoolAllocator is used instead of
ElasticPagedTokenToKVPoolAllocator.

All FTensor virtual pages are initially mapped to the same physical
zero_page (2 MB).  Without KVCacheManager.alloc() → PageAllocator.alloc_page()
→ map_to_kv_tensors(), they STAY mapped to zero_page, so writes to
tokens in different virtual pages silently overwrite each other once the
GPU L2 cache is evicted.

Run:
    python tests/test_paged_allocator_aliasing.py

Expected output (example):
    [PASS] test_without_alloc_data_corrupted
    [PASS] test_with_alloc_data_correct
"""

import sys
import time

import torch

from kvcached.integration.sglang.interfaces import (
    alloc_kv_cache,
    get_kv_cache_manager,
    init_kvcached,
    shutdown_kvcached,
)
from kvcached.utils import PAGE_SIZE
from kvcached.vmm_ops import kv_tensors_created

# ── Config ──────────────────────────────────────────────────────────
SGLANG_PAGE_SIZE = 16          # tokens per block (SGLang "page")
HEAD_NUM         = 8
HEAD_DIM         = 64
NUM_LAYERS       = 2
DTYPE            = torch.float16
DEVICE           = "cuda:0"
NUM_TOKENS       = 65536       # large enough to span many 2 MB pages
# ────────────────────────────────────────────────────────────────────

passed = 0
failed = 0


def _tokens_per_physical_page(k_buf):
    """How many tokens fit in one 2 MB physical page (accounting for stride)."""
    byte_stride = k_buf.stride()[0] * DTYPE.itemsize
    return PAGE_SIZE // byte_stride


# ── Test 1 ──────────────────────────────────────────────────────────
def test_without_alloc_data_corrupted(k_tensors, manager):
    """
    WITHOUT KVCacheManager.alloc(), writes to tokens in different 2 MB
    virtual pages alias via zero_page.

    We write enough pages to exceed the GPU L2 cache so that evicted
    cache lines actually land on zero_page and overwrite each other.
    """
    global passed, failed
    k_buf = k_tensors[0]                         # layer-0 K buffer
    tpp = _tokens_per_physical_page(k_buf)       # e.g. 512

    # Write to token 1 inside each of many virtual 2 MB pages.
    # This is what happens when SGLang's PagedTokenToKVPoolAllocator
    # hands out page IDs — it never calls KVCacheManager.alloc().
    num_pages = 60   # 60 × 2 MB = 120 MB > typical L2 (~50-80 MB)
    for i in range(num_pages):
        token = 1 + i * tpp
        k_buf[token] = torch.full(
            (HEAD_NUM, HEAD_DIM), float(i), dtype=DTYPE, device=DEVICE
        )
    torch.cuda.synchronize()

    # Read back every page's token 1.
    readbacks = []
    for i in range(num_pages):
        token = 1 + i * tpp
        readbacks.append(k_buf[token][0][0].item())
    torch.cuda.synchronize()

    expected = [float(i) for i in range(num_pages)]

    # With zero_page aliasing the readbacks will NOT match the expected
    # values; most of them will be overwritten by later writes.
    num_correct = sum(1 for a, b in zip(readbacks, expected) if a == b)

    if num_correct < num_pages:
        print(
            f"[PASS] test_without_alloc_data_corrupted\n"
            f"       Only {num_correct}/{num_pages} tokens retained their "
            f"value — zero_page aliasing confirmed.\n"
            f"       First 10 expected : {expected[:10]}\n"
            f"       First 10 actual   : {readbacks[:10]}"
        )
        passed += 1
    else:
        print(
            f"[FAIL] test_without_alloc_data_corrupted\n"
            f"       All {num_pages} tokens kept their value — aliasing "
            f"not observed (L2 may be larger than expected)."
        )
        failed += 1


# ── Test 2 ──────────────────────────────────────────────────────────
def test_with_alloc_data_correct(k_tensors, manager):
    """
    WITH KVCacheManager.alloc(), each block gets its own physical page
    via PageAllocator → map_to_kv_tensors().  No aliasing.
    """
    global passed, failed
    k_buf = k_tensors[0]

    # How many SGLang blocks fit in one kvcached physical page (2 MB)?
    blocks_per_phys = PAGE_SIZE // (SGLANG_PAGE_SIZE * HEAD_NUM * HEAD_DIM * DTYPE.itemsize)

    # Use fewer physical pages than test 1 so we fit within available blocks.
    avail = manager.available_size()
    num_phys_pages = min(30, avail // blocks_per_phys)
    assert num_phys_pages >= 4, (
        f"Not enough blocks for test: avail={avail}, "
        f"blocks_per_phys={blocks_per_phys}, num_phys_pages={num_phys_pages}"
    )
    num_blocks_needed = blocks_per_phys * num_phys_pages

    block_ids = manager.alloc(num_blocks_needed)
    assert block_ids is not None, (
        f"alloc({num_blocks_needed}) failed, available={manager.available_size()}"
    )
    torch.cuda.synchronize()

    # Write a unique value to one token per physical page.
    test_tokens = []
    for i in range(num_phys_pages):
        blk = block_ids[i * blocks_per_phys]
        token = blk * SGLANG_PAGE_SIZE
        test_tokens.append(token)
        k_buf[token] = torch.full(
            (HEAD_NUM, HEAD_DIM), float(i), dtype=DTYPE, device=DEVICE
        )
    torch.cuda.synchronize()

    # Read back — every value must survive.
    readbacks = []
    for token in test_tokens:
        readbacks.append(k_buf[token][0][0].item())
    torch.cuda.synchronize()

    expected = [float(i) for i in range(num_phys_pages)]
    num_correct = sum(1 for a, b in zip(readbacks, expected) if a == b)

    manager.free(block_ids)

    if num_correct == num_phys_pages:
        print(
            f"[PASS] test_with_alloc_data_correct\n"
            f"       All {num_phys_pages}/{num_phys_pages} tokens correct — "
            f"unique physical pages confirmed."
        )
        passed += 1
    else:
        print(
            f"[FAIL] test_with_alloc_data_correct\n"
            f"       {num_correct}/{num_phys_pages} tokens correct.\n"
            f"       First 10 expected : {expected[:10]}\n"
            f"       First 10 actual   : {readbacks[:10]}"
        )
        failed += 1


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Single init/shutdown — kvcached global state cannot be re-initialized.
    torch.cuda.set_device(0)
    init_kvcached(async_sched=False)

    k_tensors, v_tensors = alloc_kv_cache(
        kvcache_shape=(NUM_TOKENS, HEAD_NUM, HEAD_DIM),
        dtype=DTYPE,
        device=DEVICE,
        num_layers=NUM_LAYERS,
        page_size=SGLANG_PAGE_SIZE,
        attention_type="MHA",
        kv_layout="NHD",
    )

    t0 = time.time()
    while not kv_tensors_created():
        assert time.time() - t0 < 10, "KV tensors not created within 10 s"
        time.sleep(0.1)

    cell_size = HEAD_NUM * HEAD_DIM * DTYPE.itemsize
    num_blocks = NUM_TOKENS // SGLANG_PAGE_SIZE
    manager = get_kv_cache_manager(
        num_blocks=num_blocks + 1,
        block_size=SGLANG_PAGE_SIZE,
        cell_size=cell_size,
        num_layers=NUM_LAYERS,
        reserve_null_block=True,
    )
    manager._post_init_done.wait(timeout=10.0)
    assert manager._post_init_done.is_set(), "post-init timed out"

    print(f"Setup: {NUM_TOKENS} tokens, page_size={SGLANG_PAGE_SIZE}, "
          f"available blocks={manager.available_size()}\n")

    test_without_alloc_data_corrupted(k_tensors, manager)
    print()
    test_with_alloc_data_correct(k_tensors, manager)

    shutdown_kvcached()

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
