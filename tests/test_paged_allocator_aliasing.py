# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Physical-page identity when SGLang's default (non-elastic) allocator is used.

All FTensor virtual pages start mapped to one shared physical zero page, which
is the read-safety net for tokens no one has claimed yet. Without
``KVCacheManager.alloc()`` -> ``PageAllocator.alloc_page()`` ->
``map_to_kv_tensors()`` they *stay* mapped there, so writes to tokens in
different virtual pages alias once the accelerator's last-level cache evicts
them. With ``alloc()`` every block gets its own physical page and no aliasing
occurs.

Runs on whichever accelerator kvcached was built for: the device comes from
``get_current_device_str()`` and the barriers from ``get_device_module()``.
The aliasing demonstration additionally requires the zero-page safety net, which
``vmm_ops.has_zero_page_safety_net()`` reports -- the XPU backend cannot map one
shared physical page into many virtual pages, so there is nothing to alias
through there and that test is skipped rather than silently inverted.
"""
import time

import pytest

torch = pytest.importorskip("torch")

from kvcached.utils import (  # noqa: E402
    PAGE_SIZE,
    get_current_device_str,
    get_device_module,
)
from kvcached.vmm_ops import (  # noqa: E402
    has_zero_page_safety_net,
    kv_tensors_created,
)

SGLANG_PAGE_SIZE = 16  # tokens per block (SGLang "page")
HEAD_NUM = 8
HEAD_DIM = 64
NUM_LAYERS = 2
DTYPE = torch.float16
NUM_TOKENS = 65536  # large enough to span many 2 MB pages

if not get_device_module().is_available():
    pytest.skip("needs an accelerator", allow_module_level=True)


@pytest.fixture(scope="module")
def kv_pool():
    """Initialize kvcached once and yield (k_tensors, manager).

    Module-scoped and single-shot: kvcached keeps process-global state and
    cannot be re-initialized, so both tests share one pool.
    """
    from kvcached.integration.sglang.interfaces import (
        alloc_kv_cache,
        get_kv_cache_manager,
        init_kvcached,
        shutdown_kvcached,
    )

    device = get_current_device_str()
    device_module = get_device_module()
    device_module.set_device(device_module.current_device())
    init_kvcached(async_sched=False)
    try:
        k_tensors, _v_tensors = alloc_kv_cache(
            kvcache_shape=(NUM_TOKENS, HEAD_NUM, HEAD_DIM),
            dtype=DTYPE,
            device=device,
            num_layers=NUM_LAYERS,
            page_size=SGLANG_PAGE_SIZE,
            attention_type="MHA",
            kv_layout="NHD",
        )

        deadline = time.time() + 10
        while not kv_tensors_created():
            assert time.time() < deadline, "KV tensors not created within 10 s"
            time.sleep(0.1)

        manager = get_kv_cache_manager(
            num_blocks=NUM_TOKENS // SGLANG_PAGE_SIZE + 1,
            block_size=SGLANG_PAGE_SIZE,
            cell_size=HEAD_NUM * HEAD_DIM * DTYPE.itemsize,
            num_layers=NUM_LAYERS,
            reserve_null_block=True,
        )
        manager._post_init_done.wait(timeout=10.0)
        assert manager._post_init_done.is_set(), "post-init timed out"

        yield k_tensors, manager, device
    finally:
        shutdown_kvcached()


def _tokens_per_physical_page(k_buf):
    """How many tokens fit in one physical page, accounting for stride."""
    return PAGE_SIZE // (k_buf.stride()[0] * DTYPE.itemsize)


@pytest.mark.skipif(
    not has_zero_page_safety_net(),
    reason="backend maps no shared zero page, so there is nothing to alias through",
)
def test_without_alloc_data_corrupted(kv_pool):
    """Unallocated pages alias: writes to tokens in different virtual pages
    overwrite each other, because all of them still point at the zero page.

    Enough pages are written to exceed the last-level cache, so evicted lines
    actually reach the shared physical page.
    """
    k_tensors, _manager, device = kv_pool
    k_buf = k_tensors[0]
    tpp = _tokens_per_physical_page(k_buf)

    num_pages = 60  # 60 x 2 MB = 120 MB, past a typical ~50-80 MB LLC
    for i in range(num_pages):
        k_buf[1 + i * tpp] = torch.full(
            (HEAD_NUM, HEAD_DIM), float(i), dtype=DTYPE, device=device
        )
    get_device_module().synchronize()

    readbacks = [k_buf[1 + i * tpp][0][0].item() for i in range(num_pages)]
    get_device_module().synchronize()

    num_correct = sum(1 for i, got in enumerate(readbacks) if got == float(i))
    assert num_correct < num_pages, (
        f"all {num_pages} tokens kept their value, so no aliasing was observed. "
        f"Either the pages were backed after all, or the cache is larger than "
        f"the {num_pages * 2} MB written here."
    )


def test_with_alloc_data_correct(kv_pool):
    """Allocated blocks do not alias: KVCacheManager.alloc() backs each block
    with its own physical page, so every written token survives."""
    k_tensors, manager, device = kv_pool
    k_buf = k_tensors[0]

    blocks_per_phys = PAGE_SIZE // (
        SGLANG_PAGE_SIZE * HEAD_NUM * HEAD_DIM * DTYPE.itemsize
    )
    avail = manager.available_size()
    num_phys_pages = min(30, avail // blocks_per_phys)
    assert num_phys_pages >= 4, (
        f"not enough blocks for the test: avail={avail}, "
        f"blocks_per_phys={blocks_per_phys}, num_phys_pages={num_phys_pages}"
    )

    block_ids = manager.alloc(blocks_per_phys * num_phys_pages)
    assert block_ids is not None, (
        f"alloc({blocks_per_phys * num_phys_pages}) failed, "
        f"available={manager.available_size()}"
    )
    try:
        test_tokens = []
        for i in range(num_phys_pages):
            token = block_ids[i * blocks_per_phys] * SGLANG_PAGE_SIZE
            test_tokens.append(token)
            k_buf[token] = torch.full(
                (HEAD_NUM, HEAD_DIM), float(i), dtype=DTYPE, device=device
            )
        get_device_module().synchronize()

        readbacks = [k_buf[token][0][0].item() for token in test_tokens]
        get_device_module().synchronize()
    finally:
        manager.free(block_ids)

    expected = [float(i) for i in range(num_phys_pages)]
    assert readbacks == expected, (
        f"distinct physical pages expected, got aliasing.\n"
        f"  expected (first 10): {expected[:10]}\n"
        f"  actual   (first 10): {readbacks[:10]}"
    )
