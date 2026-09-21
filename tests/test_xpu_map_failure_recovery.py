# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""What the XPU arm does when the device runs out of memory mid-map.

Physical pages are committed at map time on Level Zero -- creating a
sycl::physical_mem never fails for lack of memory, mapping it does -- so
exhaustion surfaces from mem_map, inside a transaction (#418). Three things have
to hold afterwards: the pool the caller already has must stay usable, the range
whose map failed must be usable again once there is room, and freeing a page
must not itself need a page.

The last two do not come for free. A 2 MiB page is 32 chunks of the driver's
64 KiB granularity, and a map that fails part-way leaves the chunks it did map
behind; without the cleanup in gpu_vmm's mem_map, the next map of that range
fails with INVALID_ARGUMENT and the pool has lost those pages for good. And
releasing offset 0 has to put the anchor page back under virtual page 0, which
would be an allocation on the release path if the anchor were not parked.

This test fills the device, so it is slow (tens of seconds) and wants a card to
itself.

    KVCACHED_BACKEND=xpu python setup.py build_ext --inplace
    ZE_AFFINITY_MASK=0 pytest tests/test_xpu_map_failure_recovery.py -v
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
DTYPE = torch.float16
# Virtual room for more than any current card holds, so the reservation is never
# what runs out first. One KV buffer of one layer keeps the page accounting
# readable: each logical offset costs two physical pages, K and V.
VIRTUAL = 48 * 2**30
BATCH = 256
ELEMS_PER_PAGE = PAGE_SIZE // DTYPE.itemsize


class Exhausted:
    """What a filled-up device left behind.

    A plain class rather than a tuple because pytest reprs a failing test's
    arguments, and repr of a KV tensor reads every page of it from the host: any
    page this test could not map turns the assertion message into a segfault.
    The default object repr keeps a failure reportable.
    """

    def __init__(self, tensor, live, failed):
        self.tensor = tensor
        # Offsets that mapped, and the batch whose map raised and which the
        # allocator then rolled back (#418).
        self.live = live
        self.failed = failed


@pytest.fixture(scope="module")
def exhausted():
    """Map until the device says no, and hand the test what survived."""
    init_kvcached(DEVICE, PAGE_SIZE, False)
    try:
        tensors = create_kv_tensors(VIRTUAL, DTYPE.itemsize, DEVICE, 1, 2)
        live = []
        failed = None
        # Only the first half of the reservation is addressable as a K offset;
        # the V pages live at the matching offset in the upper half.
        for start in range(0, VIRTUAL // PAGE_SIZE // 2 - BATCH, BATCH):
            offsets = [(start + i) * PAGE_SIZE for i in range(BATCH)]
            try:
                map_to_kv_tensors(offsets)
            except RuntimeError:
                failed = offsets
                break
            live.extend(offsets)

        if failed is None:
            pytest.skip(
                f"{VIRTUAL // 2**30} GiB of pages mapped without exhausting the "
                "device; this test needs a card it can fill"
            )
        yield Exhausted(tensors[0], live, failed)

        for i in range(0, len(live), BATCH):
            unmap_from_kv_tensors(live[i:i + BATCH])
    finally:
        shutdown_kvcached()


def test_pages_mapped_before_the_failure_still_work(exhausted):
    """The transaction must not damage what the caller already holds."""
    tensor = exhausted.tensor
    start = exhausted.live[len(exhausted.live) // 2] // DTYPE.itemsize
    page = tensor[start:start + ELEMS_PER_PAGE]
    page.fill_(3)
    torch.xpu.synchronize()
    assert bool(torch.all(page == 3)), (
        "a page mapped before the failure is not writable"
    )


def test_the_range_that_ran_out_of_memory_is_reusable(exhausted):
    """The regression this file exists for: the failed range must not be lost.

    Freeing one batch's worth of pages and mapping the failed batch is the
    retry an engine performs after a KVCachePoolExhausted, so an
    INVALID_ARGUMENT here is a pool that shrinks every time it fills up.
    """
    live, failed = exhausted.live, exhausted.failed
    assert unmap_from_kv_tensors(live[-BATCH:])
    del live[-BATCH:]

    assert map_to_kv_tensors(failed), "the range that OOMed cannot be mapped"
    live.extend(failed)

    start = failed[0] // DTYPE.itemsize
    page = exhausted.tensor[start:start + ELEMS_PER_PAGE]
    page.fill_(5)
    torch.xpu.synchronize()
    assert bool(torch.all(page == 5)), "the remapped range is not writable"


def test_releasing_offset_zero_needs_no_spare_page(exhausted):
    """Offset 0 must be releasable on a device with nothing left to allocate.

    Virtual page 0 has to stay backed for from_blob()'s device lookup, so
    releasing the page mapped there puts the anchor page back under it. The
    anchor is allocated once in the FTensor constructor and parked so that this
    needs no allocation: otherwise freeing virtual page 0 would want one MORE
    physical page than the pool already holds, exactly when the caller is
    freeing memory to make room.

    The fixture stops at a failed batch, which leaves that batch's pages free,
    so this fills the rest: single offsets until one is refused, then plain
    torch allocations until the allocator cannot reserve a segment either.

    This guards the path rather than reproducing a failure. Allocating the
    anchor here instead of reusing the parked one passes on Level Zero too --
    the allocation still finds room on a full card. Only a backend that holds
    the physical allocation across the unmap would fail here, and those (CUDA,
    HIP) map a shared zero page and never reach the anchor at all.
    """
    live = exhausted.live
    assert 0 in live, "the fixture did not map offset 0"

    top_up = max(live) + PAGE_SIZE
    limit = VIRTUAL // 2  # only the lower half is addressable as a K offset
    refused = False
    while top_up < limit:
        try:
            map_to_kv_tensors([top_up])
        except RuntimeError:
            refused = True
            break
        live.append(top_up)
        top_up += PAGE_SIZE
    if not refused:
        pytest.skip("ran out of virtual space before the device ran out of memory")

    soak: list = []
    try:
        # Bounded so a device that never refuses fails the test rather than
        # allocating until the machine suffers.
        while len(soak) < 16384:
            soak.append(torch.empty(PAGE_SIZE, dtype=torch.uint8, device=DEVICE))
        pytest.fail("the device kept handing out memory; it is not full")
    except (RuntimeError, MemoryError):
        pass

    try:
        # The operation under test: no page is free, and this must still work.
        assert unmap_from_kv_tensors([0]), "releasing offset 0 was refused"
    finally:
        # Give the device back before anything else, so a failure above still
        # leaves the card usable for the rest of the session.
        del soak
        torch.xpu.empty_cache()
    live.remove(0)

    # Reading virtual page 0 must not fault: that is the anchor's whole job.
    # What comes back is the anchor's content, not the released page's, so only
    # the fact that the read completes means anything here.
    exhausted.tensor[:8].cpu()
    torch.xpu.synchronize()

    # Everything below launches kernels, and a launch needs device resources of
    # its own: on a card this full fill_() fails with OUT_OF_RESOURCES however
    # sound the mappings are. So give a batch back first -- which is what an
    # engine does next anyway -- and check the pool with room to run in.
    assert unmap_from_kv_tensors(live[-BATCH:])
    del live[-BATCH:]

    mid = live[len(live) // 2] // DTYPE.itemsize
    page = exhausted.tensor[mid:mid + ELEMS_PER_PAGE]
    page.fill_(7)
    torch.xpu.synchronize()
    assert bool(torch.all(page == 7)), (
        "a page elsewhere in the pool is not writable after offset 0 was freed"
    )

    # Recovery: offset 0 maps again, which also exercises evicting the anchor
    # from under virtual page 0.
    assert map_to_kv_tensors([0]), "offset 0 cannot be mapped again"
    live.insert(0, 0)
    page = exhausted.tensor[:ELEMS_PER_PAGE]
    page.fill_(9)
    torch.xpu.synchronize()
    assert bool(torch.all(page == 9)), "offset 0 is not writable after remapping"
