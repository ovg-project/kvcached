# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the page create/map split (FTensor prepare()/commit()).

The split lets the background prealloc thread do only the physical allocation
(cuMemCreate) while the VA edit (mem_unmap zero page + cuMemMap real page) runs
on the main thread at the alloc_page handout. These tests lock in the observable
accounting invariants that the split must preserve, on the contiguous layout
with prealloc enabled (the config the split targets):

  1. alloc_page maps distinct physical pages -- no zero_page aliasing across
     blocks in different physical pages (prepare()+commit() correctness).
  2. free -> realloc round-trips data intact -- exercises handing back a page
     that was reserved while still mapped (commit() must be idempotent) as well
     as freshly prepared pages.
  3. trim() drops prealloc'd-but-uncommitted reserved pages without error and
     the pool stays usable afterwards -- exercises unmap() on a page that only
     exists in FTensor::prepared_, never committed to a real VA.

Needs the compiled extension and a CUDA/HIP device; skipped otherwise (like the
other real-extension GPU tests here). Contiguous layout is forced via env before
kvcached is imported, so run this module in its own process.

Run:
    KVCACHED_CONTIGUOUS_LAYOUT=true python tests/test_create_map_split.py
    # or: pytest tests/test_create_map_split.py
"""

import os
import time

# Must be set before importing kvcached (utils reads these at import time).
os.environ.setdefault("KVCACHED_CONTIGUOUS_LAYOUT", "true")
os.environ.setdefault("KVCACHED_PAGE_PREALLOC_ENABLED", "true")

import torch

try:
    import pytest
except ModuleNotFoundError:
    # Allow running as plain `python tests/test_create_map_split.py` in
    # environments without pytest (e.g. inside the serving image); the
    # decorators degrade to no-ops and the __main__ runner drives the checks.
    class _NoPytest:
        class mark:
            @staticmethod
            def skipif(*_a, **_k):
                return lambda f: f

        @staticmethod
        def fixture(*_a, **_k):
            return lambda f: f

    pytest = _NoPytest()  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="create/map split test needs a CUDA/HIP device and the compiled "
    "kvcached extension",
)

# ── Config ──────────────────────────────────────────────────────────
PAGE_TOKENS = 16          # tokens per block ("page" in SGLang terms)
HEAD_NUM = 8
HEAD_DIM = 64
NUM_LAYERS = 2
DTYPE = torch.float16
DEVICE = "cuda:0"
NUM_TOKENS = 65536        # spans many 2 MB physical pages
NUM_PHYS_PAGES = 12       # how many distinct physical pages to exercise
# ────────────────────────────────────────────────────────────────────


def _setup():
    """Init kvcached, create KV tensors, build the manager. Returns
    (k_tensors, manager); caller is responsible for shutdown_kvcached()."""
    from kvcached.integration.sglang.interfaces import (
        alloc_kv_cache,
        get_kv_cache_manager,
        init_kvcached,
    )
    from kvcached.vmm_ops import kv_tensors_created

    from kvcached.utils import CONTIGUOUS_LAYOUT
    assert CONTIGUOUS_LAYOUT, (
        "this test targets the contiguous layout; run with "
        "KVCACHED_CONTIGUOUS_LAYOUT=true in its own process")

    torch.cuda.set_device(0)
    init_kvcached(async_sched=False)

    k_tensors, _ = alloc_kv_cache(
        kvcache_shape=(NUM_TOKENS, HEAD_NUM, HEAD_DIM),
        dtype=DTYPE,
        device=DEVICE,
        num_layers=NUM_LAYERS,
        page_size=PAGE_TOKENS,
        attention_type="MHA",
        kv_layout="NHD",
    )

    t0 = time.time()
    while not kv_tensors_created():
        assert time.time() - t0 < 10, "KV tensors not created within 10 s"
        time.sleep(0.1)

    cell_size = HEAD_NUM * HEAD_DIM * DTYPE.itemsize
    manager = get_kv_cache_manager(
        num_blocks=NUM_TOKENS // PAGE_TOKENS + 1,
        block_size=PAGE_TOKENS,
        cell_size=cell_size,
        num_layers=NUM_LAYERS,
        reserve_null_block=True,
    )
    manager._post_init_done.wait(timeout=10.0)
    assert manager._post_init_done.is_set(), "post-init timed out"
    return k_tensors, manager


@pytest.fixture(scope="module")
def kv_setup():
    """One init/shutdown for the module (kvcached global state is a singleton)."""
    from kvcached.integration.sglang.interfaces import shutdown_kvcached
    k_tensors, manager = _setup()
    yield k_tensors, manager
    shutdown_kvcached()


def _blocks_per_phys(page_size_bytes):
    return page_size_bytes // (PAGE_TOKENS * HEAD_NUM * HEAD_DIM * DTYPE.itemsize)


def _write_read_distinct(k_buf, block_ids, blocks_per_phys, base):
    """Write base+i to the first token of each physical page, read it back."""
    tokens = [block_ids[i * blocks_per_phys] * PAGE_TOKENS
              for i in range(NUM_PHYS_PAGES)]
    for i, tok in enumerate(tokens):
        k_buf[tok] = torch.full((HEAD_NUM, HEAD_DIM), float(base + i),
                                dtype=DTYPE, device=DEVICE)
    torch.cuda.synchronize()
    got = [k_buf[tok][0][0].item() for tok in tokens]
    torch.cuda.synchronize()
    return got, [float(base + i) for i in range(NUM_PHYS_PAGES)]


# NOTE: defined first on purpose. Before any alloc, the reserved pool holds
# only prealloc'd-but-uncommitted pages, which is exactly the prepared-page
# drop path this exercises. Later tests reserve mapped pages too (free
# fast-path), which would dilute it.
def test_trim_drops_uncommitted_reserved_pages(kv_setup):
    """trim() must release prealloc'd-but-uncommitted reserved pages (present
    only in FTensor::prepared_, never committed to a real VA) without error,
    and the pool must stay usable afterwards."""
    k_tensors, manager = kv_setup
    from kvcached.utils import PAGE_SIZE
    pa = manager.page_allocator
    k_buf = k_tensors[0]
    bpp = _blocks_per_phys(PAGE_SIZE)

    # Wait for the initial prealloc fill, then stop the thread so the reserved
    # count can't change under the asserts (a prealloc already in flight could
    # otherwise re-insert just after trim()).
    t0 = time.time()
    while pa.get_num_reserved_pages() == 0:
        assert time.time() - t0 < 10, "prealloc never reserved any pages"
        time.sleep(0.05)
    pa.stop_prealloc_thread()

    assert pa.get_num_reserved_pages() > 0  # prepared-but-uncommitted pages
    pa.trim()                               # drops them with no VA edit
    assert pa.get_num_reserved_pages() == 0

    pa.start_prealloc_thread()  # restore for the remaining tests

    block_ids = manager.alloc(bpp * NUM_PHYS_PAGES)
    assert block_ids is not None
    try:
        got, expected = _write_read_distinct(k_buf, block_ids, bpp, base=200)
        assert got == expected, f"corruption after trim: {got} != {expected}"
    finally:
        manager.free(block_ids)


def test_alloc_maps_distinct_physical_pages(kv_setup):
    """prepare()+commit() must give each physical page its own mapping."""
    k_tensors, manager = kv_setup
    from kvcached.utils import PAGE_SIZE
    k_buf = k_tensors[0]
    bpp = _blocks_per_phys(PAGE_SIZE)
    assert manager.available_size() >= bpp * NUM_PHYS_PAGES

    block_ids = manager.alloc(bpp * NUM_PHYS_PAGES)
    assert block_ids is not None
    try:
        got, expected = _write_read_distinct(k_buf, block_ids, bpp, base=0)
        assert got == expected, f"aliasing / bad map: {got} != {expected}"
    finally:
        manager.free(block_ids)


def test_free_realloc_preserves_integrity(kv_setup):
    """Handing a page back out -- reserved-while-mapped (idempotent commit) or
    freshly prepared -- must still serve correct, non-aliased memory.

    Reliably hits the idempotent-commit branch only with the default
    min_reserved < max_reserved, which leaves room for free() to reserve pages
    that are still mapped; those then come back out via the fast path.
    """
    k_tensors, manager = kv_setup
    from kvcached.utils import PAGE_SIZE
    k_buf = k_tensors[0]
    bpp = _blocks_per_phys(PAGE_SIZE)

    first = manager.alloc(bpp * NUM_PHYS_PAGES)
    assert first is not None
    manager.free(first)

    second = manager.alloc(bpp * NUM_PHYS_PAGES)
    assert second is not None
    try:
        got, expected = _write_read_distinct(k_buf, second, bpp, base=100)
        assert got == expected, f"post-realloc corruption: {got} != {expected}"
    finally:
        manager.free(second)


if __name__ == "__main__":
    # Plain-python runner (no pytest needed): share one setup across the checks.
    import sys

    from kvcached.integration.sglang.interfaces import shutdown_kvcached

    setup = _setup()
    checks = [
        test_trim_drops_uncommitted_reserved_pages,
        test_alloc_maps_distinct_physical_pages,
        test_free_realloc_preserves_integrity,
    ]
    failed = 0
    for check in checks:
        try:
            check(setup)
            print(f"[PASS] {check.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {check.__name__}: {type(e).__name__}: {e}")
    shutdown_kvcached()
    print(f"\n{'=' * 50}\nResults: {len(checks) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
