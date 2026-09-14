# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""End-to-end VMM lifecycle on an Intel XPU, exercised through vmm_ops only.

Covers the seam implemented by the KVCACHED_USE_XPU arm of csrc/inc/gpu_vmm.hpp:
runtime init, granularity check, virtual reservation, physical page create/map,
unmap and release. Deliberately free of any serving-engine import so a failure
localizes to kvcached's own VMM layer.

Requires an Intel GPU and a kvcached built with KVCACHED_BACKEND=xpu:
    KVCACHED_BACKEND=xpu python setup.py build_ext --inplace
    pytest tests/test_xpu_vmm_lifecycle.py -v
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
    kv_tensors_created,
    map_to_kv_tensors,
    shutdown_kvcached,
    unmap_from_kv_tensors,
)

DEVICE = "xpu:0"
NUM_LAYERS = 2
NUM_KV_BUFFERS = 2
DTYPE = torch.float16
# Per-layer FTensor bytes. Must be a multiple of 2 * PAGE_SIZE because
# get_v_base_offset() splits the tensor into a K half and a V half.
SIZE_PER_LAYER = 8 * PAGE_SIZE
# Page-aligned offsets addressable within the K half.
K_PAGES = (SIZE_PER_LAYER // 2) // PAGE_SIZE


@pytest.fixture(scope="module")
def kv_tensors():
    """One init/shutdown for the module: kvcached keeps global allocator state."""
    torch.xpu.set_device(0)
    # contiguous_layout=False gives one FTensor per layer, the XPU default.
    init_kvcached(DEVICE, PAGE_SIZE, False)
    tensors = create_kv_tensors(
        SIZE_PER_LAYER, DTYPE.itemsize, DEVICE, NUM_LAYERS, NUM_KV_BUFFERS
    )
    try:
        yield tensors
    finally:
        shutdown_kvcached()


def test_init_rejects_a_device_from_another_backend():
    """An XPU build must refuse ``cuda:0`` instead of quietly using host memory.

    is_accelerator() is false for a foreign device family, which sends the
    reservation down the CPU mmap() path while from_blob() still labels the
    tensor ``cuda:0``. The engine would then launch kernels against host memory.
    Takes no fixture: the check runs before init_kvcached() touches any global
    state, so it cannot disturb an allocator this module already initialized.
    """
    with pytest.raises(RuntimeError, match="built for XPU"):
        init_kvcached("cuda:0", PAGE_SIZE, False)


def test_runtime_initializes_and_reports_tensors_created(kv_tensors):
    assert kv_tensors_created()


def test_tensors_land_on_the_xpu_with_the_requested_geometry(kv_tensors):
    assert len(kv_tensors) == NUM_LAYERS
    for layer, tensor in enumerate(kv_tensors):
        # An is_cuda()-based dispatch would have served these from host mmap,
        # which is exactly what device_utils.hpp::is_accelerator() prevents.
        assert tensor.device.type == "xpu", f"layer {layer} is not on the XPU"
        assert tensor.device.index == 0
        # create_kv_tensors() is given a cell width in bytes, not a dtype, so
        # the concrete scalar type is the allocator's choice; only the width is
        # part of the contract. Engines reinterpret the buffer themselves.
        assert tensor.element_size() == DTYPE.itemsize
        assert tensor.numel() * tensor.element_size() == SIZE_PER_LAYER


def test_virtual_reservation_is_page_aligned(kv_tensors):
    """address_reserve() must honor the alignment the caller asked for; the XPU
    arm verifies this itself because reserve_virtual_mem() takes no alignment
    argument."""
    for tensor in kv_tensors:
        assert tensor.data_ptr() % PAGE_SIZE == 0


def test_map_then_unmap_every_k_page(kv_tensors):
    offsets = [p * PAGE_SIZE for p in range(K_PAGES)]

    assert map_to_kv_tensors(offsets)
    # Each offset maps a K page and its mirrored V page, in every layer.
    torch.xpu.synchronize()

    assert unmap_from_kv_tensors(offsets)
    torch.xpu.synchronize()


def test_repeated_map_unmap_of_the_same_offset(kv_tensors):
    """Remapping an offset after unmap must succeed, with no handle leaked.

    The two offsets take different paths, because XPU has no shared zero page.
    Offset 0 trades its virtual page back and forth between a real page and a
    freshly created anchor page, since from_blob() needs page 0 backed;
    offset PAGE_SIZE just goes unbacked on unmap. Four cycles therefore churn
    both kinds of handle through PhysicalMemRegistry.

    Nothing here reads an unmapped page, and nothing added here should: an
    unbacked XPU read loses the device rather than raising. Reads of unmapped
    pages belong in test_xpu_ftensor.py behind its requires_zero_page marker."""
    offsets = [0, PAGE_SIZE]
    for _ in range(4):
        assert map_to_kv_tensors(offsets)
        assert unmap_from_kv_tensors(offsets)
    torch.xpu.synchronize()


def test_mapped_pages_are_writable_and_read_back(kv_tensors):
    offsets = [p * PAGE_SIZE for p in range(K_PAGES)]
    assert map_to_kv_tensors(offsets)
    try:
        elems_per_page = PAGE_SIZE // DTYPE.itemsize
        tensor = kv_tensors[0]
        for page, offset in enumerate(offsets):
            start = offset // DTYPE.itemsize
            tensor[start:start + elems_per_page] = float(page + 1)
        torch.xpu.synchronize()

        for page, offset in enumerate(offsets):
            start = offset // DTYPE.itemsize
            chunk = tensor[start:start + elems_per_page]
            assert torch.all(chunk == float(page + 1)), (
                f"page {page} did not read back its own value; physical pages "
                f"may be aliased"
            )
    finally:
        assert unmap_from_kv_tensors(offsets)
