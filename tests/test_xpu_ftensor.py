# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""FTensor semantics on an Intel XPU: physical page isolation.

The property the XPU arm must preserve, structural rather than
performance-dependent: once distinct offsets are mapped, each is backed by its
own physical page, so writes do not alias -- across pages, across layers, and
across the K and V halves of a layer.

What this file deliberately does *not* assert on XPU is the zero-page safety
net. On CUDA and HIP every virtual page of a fresh FTensor is backed by one
shared zero page, so a stray read of a never-allocated region returns data
instead of faulting. Level Zero cannot express that: physical_mem::map() accepts
a second virtual range for the same physical page without error and then kills
the context with UR_RESULT_ERROR_DEVICE_LOST on first access. FTensor therefore
leaves XPU reservations unbacked, and reading one is a genuine fault, so the
tests that cover the safety net skip themselves via
vmm_ops.has_zero_page_safety_net() rather than crashing the interpreter.

Note this file does not assert that *unmapped* pages alias each other. On CUDA
they do, but observing the aliasing requires evicting the GPU cache, which is
timing- and cache-size-dependent (see test_paged_allocator_aliasing.py, which
needs >100 MB of traffic to show it). Asserting the safe direction keeps this
test deterministic.

    KVCACHED_BACKEND=xpu python setup.py build_ext --inplace
    pytest tests/test_xpu_ftensor.py -v
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
    has_zero_page_safety_net,
    init_kvcached,
    map_to_kv_tensors,
    shutdown_kvcached,
    unmap_from_kv_tensors,
)

# Reading an unbacked virtual page faults the process rather than raising, so
# these must be skips decided up front, not try/except around the read.
requires_zero_page = pytest.mark.skipif(
    not has_zero_page_safety_net(),
    reason="backend has no shared zero page; reads of unmapped pages fault",
)

DEVICE = "xpu:0"
NUM_LAYERS = 2
DTYPE = torch.float16
SIZE_PER_LAYER = 8 * PAGE_SIZE
K_PAGES = (SIZE_PER_LAYER // 2) // PAGE_SIZE
ELEMS_PER_PAGE = PAGE_SIZE // DTYPE.itemsize


@pytest.fixture(scope="module")
def kv_tensors():
    torch.xpu.set_device(0)
    init_kvcached(DEVICE, PAGE_SIZE, False)
    tensors = create_kv_tensors(SIZE_PER_LAYER, DTYPE.itemsize, DEVICE, NUM_LAYERS, 2)
    try:
        yield tensors
    finally:
        shutdown_kvcached()


def _page_slice(tensor, page):
    start = page * ELEMS_PER_PAGE
    return tensor[start:start + ELEMS_PER_PAGE]


@requires_zero_page
def test_unmapped_pages_are_readable_via_the_zero_page(kv_tensors):
    """A read of a never-mapped page must not fault. init_with_zero_() maps the
    shared zero page across the whole reservation for exactly this reason.

    Only reachability is asserted: the zero page is shared by every unmapped
    virtual page, so its contents are whatever was written through any of them.
    """
    tensor = kv_tensors[0]
    for page in range(K_PAGES):
        # Copying to host forces the read to complete; a missing backing page
        # would fault here rather than return.
        host = _page_slice(tensor, page)[:8].cpu()
        assert host.numel() == 8
    torch.xpu.synchronize()


def test_mapped_pages_do_not_alias_each_other(kv_tensors):
    """The core physical-isolation property: distinct offsets get distinct
    physical pages, so a full-page write to one is invisible to the others."""
    offsets = [p * PAGE_SIZE for p in range(K_PAGES)]
    assert map_to_kv_tensors(offsets)
    try:
        tensor = kv_tensors[0]
        for page in range(K_PAGES):
            _page_slice(tensor, page).fill_(float(page + 1))
        torch.xpu.synchronize()

        for page in range(K_PAGES):
            chunk = _page_slice(tensor, page)
            assert torch.all(chunk == float(page + 1)), (
                f"page {page} was overwritten by another page's data"
            )
    finally:
        assert unmap_from_kv_tensors(offsets)


def test_layers_do_not_alias_each_other(kv_tensors):
    """Non-contiguous layout gives each layer its own reservation and its own
    physical pages; a write to layer 0 must not appear in layer 1."""
    offsets = [0]
    assert map_to_kv_tensors(offsets)
    try:
        for layer, tensor in enumerate(kv_tensors):
            _page_slice(tensor, 0).fill_(float(layer + 1))
        torch.xpu.synchronize()

        for layer, tensor in enumerate(kv_tensors):
            chunk = _page_slice(tensor, 0)
            assert torch.all(chunk == float(layer + 1)), (
                f"layer {layer} aliases another layer's physical page"
            )
    finally:
        assert unmap_from_kv_tensors(offsets)


def test_k_and_v_halves_do_not_alias(kv_tensors):
    """map_to_kv_tensors() maps offset and offset + v_base_offset as separate
    physical pages; K must not shadow V."""
    assert map_to_kv_tensors([0])
    try:
        tensor = kv_tensors[0]
        v_base_page = (SIZE_PER_LAYER // 2) // PAGE_SIZE

        _page_slice(tensor, 0).fill_(1.0)
        _page_slice(tensor, v_base_page).fill_(2.0)
        torch.xpu.synchronize()

        assert torch.all(_page_slice(tensor, 0) == 1.0), "V write leaked into K"
        assert torch.all(_page_slice(tensor, v_base_page) == 2.0), (
            "K write leaked into V"
        )
    finally:
        assert unmap_from_kv_tensors([0])


@requires_zero_page
def test_reads_after_unmap_do_not_fault(kv_tensors):
    """unmap must restore the zero-page backing rather than leaving a hole, or a
    late attention read would segfault the engine."""
    assert map_to_kv_tensors([0])
    tensor = kv_tensors[0]
    _page_slice(tensor, 0).fill_(7.0)
    torch.xpu.synchronize()

    assert unmap_from_kv_tensors([0])
    torch.xpu.synchronize()

    _ = _page_slice(tensor, 0)[0].item()  # must not fault
    torch.xpu.synchronize()
