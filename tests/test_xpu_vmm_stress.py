# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Repeated map/unmap churn on an Intel XPU, to catch handle leaks.

The XPU arm cannot store a sycl::physical_mem directly in GPUPage (it has no
default constructor), so it keeps one in a process-wide registry and hands out
integer tokens. A missed erase in mem_release() would leak device memory without
leaking virtual address space, which no functional test would notice. These tests
therefore watch free device memory across many grow/shrink cycles.

    KVCACHED_BACKEND=xpu python setup.py build_ext --inplace
    pytest tests/test_xpu_vmm_stress.py -v
"""

import pytest

torch = pytest.importorskip("torch")

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("requires an Intel XPU", allow_module_level=True)

from kvcached.utils import PAGE_SIZE, get_device_type  # noqa: E402

if get_device_type() != "xpu":
    pytest.skip(
        "kvcached is not built for the XPU backend", allow_module_level=True
    )

from kvcached.vmm_ops import (  # noqa: E402
    create_kv_tensors,
    init_kvcached,
    map_to_kv_tensors,
    shutdown_kvcached,
    unmap_from_kv_tensors,
)

DEVICE = "xpu:0"
NUM_LAYERS = 4
DTYPE = torch.float16
SIZE_PER_LAYER = 32 * PAGE_SIZE
K_PAGES = (SIZE_PER_LAYER // 2) // PAGE_SIZE
ELEMS_PER_PAGE = PAGE_SIZE // DTYPE.itemsize

CYCLES = 20
# One full cycle maps K_PAGES * 2 (K and V) * NUM_LAYERS pages. Allow a single
# cycle's worth of slack for allocator caching before calling it a leak.
LEAK_TOLERANCE_BYTES = K_PAGES * 2 * NUM_LAYERS * PAGE_SIZE


@pytest.fixture(scope="module")
def kv_tensors():
    torch.xpu.set_device(0)
    init_kvcached(DEVICE, PAGE_SIZE, False)
    tensors = create_kv_tensors(SIZE_PER_LAYER, DTYPE.itemsize, DEVICE, NUM_LAYERS, 2)
    try:
        yield tensors
    finally:
        shutdown_kvcached()


def _free_bytes():
    torch.xpu.synchronize()
    free, _total = torch.xpu.mem_get_info()
    return free


def test_full_map_unmap_cycles_do_not_leak_device_memory(kv_tensors):
    offsets = [p * PAGE_SIZE for p in range(K_PAGES)]

    # One warm-up cycle so any first-touch allocation is outside the baseline.
    assert map_to_kv_tensors(offsets)
    assert unmap_from_kv_tensors(offsets)
    baseline = _free_bytes()

    for cycle in range(CYCLES):
        assert map_to_kv_tensors(offsets), f"map failed on cycle {cycle}"
        assert unmap_from_kv_tensors(offsets), f"unmap failed on cycle {cycle}"

    leaked = baseline - _free_bytes()
    assert leaked <= LEAK_TOLERANCE_BYTES, (
        f"{leaked / 2**20:.1f} MiB not returned after {CYCLES} cycles; "
        f"physical pages are likely retained in the XPU handle registry"
    )


def test_data_stays_correct_across_cycles(kv_tensors):
    """Each cycle gets fresh physical pages; a stale mapping would surface as
    the previous cycle's value."""
    offsets = [p * PAGE_SIZE for p in range(K_PAGES)]
    tensor = kv_tensors[0]

    for cycle in range(1, 6):
        assert map_to_kv_tensors(offsets)
        try:
            for page in range(K_PAGES):
                start = page * ELEMS_PER_PAGE
                tensor[start:start + ELEMS_PER_PAGE].fill_(float(cycle * 100 + page))
            torch.xpu.synchronize()

            for page in range(K_PAGES):
                start = page * ELEMS_PER_PAGE
                chunk = tensor[start:start + ELEMS_PER_PAGE]
                expected = float(cycle * 100 + page)
                assert torch.all(chunk == expected), (
                    f"cycle {cycle} page {page}: expected {expected}"
                )
        finally:
            assert unmap_from_kv_tensors(offsets)


def test_partial_and_interleaved_map_unmap(kv_tensors):
    """Mapping and unmapping disjoint subsets in a non-LIFO order must work: the
    registry is keyed by token, not ordered like a stack."""
    evens = [p * PAGE_SIZE for p in range(0, K_PAGES, 2)]
    odds = [p * PAGE_SIZE for p in range(1, K_PAGES, 2)]

    assert map_to_kv_tensors(evens)
    assert map_to_kv_tensors(odds)
    # Release in the order they were acquired, not reversed.
    assert unmap_from_kv_tensors(evens)
    assert unmap_from_kv_tensors(odds)
    torch.xpu.synchronize()


def test_many_small_batches(kv_tensors):
    """Page-at-a-time churn, the pattern the prealloc thread produces under
    steady-state serving."""
    for page in range(K_PAGES):
        offsets = [page * PAGE_SIZE]
        assert map_to_kv_tensors(offsets)
        assert unmap_from_kv_tensors(offsets)
    torch.xpu.synchronize()
