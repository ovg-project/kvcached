# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Why the XPU backend has no zero-page safety net.

On CUDA and HIP, FTensor backs every virtual page of a fresh reservation with
one shared physical zero page, so a stray read of a never-allocated KV region
returns zeros instead of faulting. The XPU arm cannot: mapping one physical page
into a second virtual range does not work on this driver, and it fails silently
rather than returning an error.

    zeVirtualMemMap of the second range   -> ZE_RESULT_SUCCESS
    read of the second range              -> zeros; it is NOT aliased to the first
    first write through the second range  -> ZE_RESULT_ERROR_DEVICE_LOST
    anything afterwards, incl. unmap      -> dead context, then SIGSEGV in the driver

Nor does avoiding the write help: unmapping the second range without ever writing
through it returns SUCCESS and loses the device anyway, after which the *first*
range cannot be read, written or unmapped either. Creating the second mapping is
itself the problem.

These tests pin that down against the Level Zero API directly -- no SYCL, no
Unified Runtime, no kvcached -- so a failure here localizes to the driver rather
than to anything this repo does. That matters because the L0 spec neither permits
nor forbids the sharing (see zeVirtualMemMap's documented details), so it is
implementation behavior that a future driver could change.

That is the point of these tests. They are a canary, not a bug report:
test_shared_physical_page_is_silently_not_aliased fails the day a driver starts
honoring the sharing, and the failure message says to flip
gpu_vmm::supports_shared_page_mapping() so XPU gets the safety net back. See
csrc/inc/gpu_vmm.hpp and FTensor::init_with_zero_().

Each case runs in a subprocess: a positive result leaves the L0 context unusable
and later crashes inside the driver, which would take the whole pytest run with
it.

    pytest tests/test_xpu_shared_page_limitation.py -v
"""

import ctypes.util
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("requires an Intel XPU", allow_module_level=True)

if ctypes.util.find_library("ze_loader") is None:
    pytest.skip("libze_loader not found", allow_module_level=True)

# Driver-level test: deliberately does not need kvcached to be built for XPU.

# Level Zero, driven through ctypes so this needs no compiler and no L0 headers.
# argv[1] selects the case; see the tests below for what each one asserts.
_WORKER = textwrap.dedent(
    """
    import ctypes
    import os
    import sys
    from ctypes import POINTER, byref, c_size_t, c_uint32, c_void_p

    ZE_RESULT_SUCCESS = 0
    ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0xD
    ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0xE
    ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 0x16
    ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC = 0x20
    ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE = 1
    ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS = 1
    ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0

    class ContextDesc(ctypes.Structure):
        _fields_ = [("stype", c_uint32), ("pNext", c_void_p), ("flags", c_uint32)]

    class PhysicalMemDesc(ctypes.Structure):
        _fields_ = [("stype", c_uint32), ("pNext", c_void_p),
                    ("flags", c_uint32), ("size", c_size_t)]

    class QueueDesc(ctypes.Structure):
        _fields_ = [("stype", c_uint32), ("pNext", c_void_p),
                    ("ordinal", c_uint32), ("index", c_uint32),
                    ("flags", c_uint32), ("mode", c_uint32),
                    ("priority", c_uint32)]

    class HostMemAllocDesc(ctypes.Structure):
        _fields_ = [("stype", c_uint32), ("pNext", c_void_p), ("flags", c_uint32)]

    ze = ctypes.CDLL("libze_loader.so.1")

    def ck(name, res):
        if res != ZE_RESULT_SUCCESS:
            raise SystemExit("%s failed: 0x%x" % (name, res))

    ck("zeInit", ze.zeInit(0))

    ndrv = c_uint32(1)
    drv = c_void_p()
    ck("zeDriverGet", ze.zeDriverGet(byref(ndrv), byref(drv)))

    ndev = c_uint32(0)
    ck("zeDeviceGet", ze.zeDeviceGet(drv, byref(ndev), None))
    if ndev.value == 0:
        raise SystemExit("no Level Zero devices")
    devs = (c_void_p * ndev.value)()
    ck("zeDeviceGet", ze.zeDeviceGet(drv, byref(ndev), devs))
    dev = c_void_p(devs[0])

    ctx = c_void_p()
    cdesc = ContextDesc(ZE_STRUCTURE_TYPE_CONTEXT_DESC, None, 0)
    ck("zeContextCreate", ze.zeContextCreate(drv, byref(cdesc), byref(ctx)))

    # One page, whatever the device calls a page.
    ps = c_size_t(0)
    ck("zeVirtualMemQueryPageSize",
       ze.zeVirtualMemQueryPageSize(ctx, dev, c_size_t(2 << 20), byref(ps)))
    sz = ps.value

    def make_phys():
        h = c_void_p()
        pdesc = PhysicalMemDesc(ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, None, 0, sz)
        ck("zePhysicalMemCreate",
           ze.zePhysicalMemCreate(ctx, dev, byref(pdesc), byref(h)))
        return h

    def reserve(n):
        p = c_void_p()
        ck("zeVirtualMemReserve",
           ze.zeVirtualMemReserve(ctx, None, c_size_t(n), byref(p)))
        return p

    def vmap(ptr, phys):
        return ze.zeVirtualMemMap(ctx, ptr, c_size_t(sz), phys, c_size_t(0),
                                  ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE)

    def set_rw(ptr):
        return ze.zeVirtualMemSetAccessAttribute(
            ctx, ptr, c_size_t(sz), ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE)

    case = sys.argv[1]
    shared = case != "control"

    phys_a = make_phys()
    phys_b = phys_a if shared else make_phys()

    # Two ranges from one reservation, which is exactly FTensor's zero-page
    # layout. Two independent reservations behave identically.
    base = reserve(2 * sz)
    a = c_void_p(base.value)
    b = c_void_p(base.value + sz)

    ck("zeVirtualMemMap(a)", vmap(a, phys_a))
    ck("zeVirtualMemSetAccessAttribute(a)", set_rw(a))

    # The call under test. Not ck()'d: whether it reports an error is the finding.
    map_b = vmap(b, phys_b)
    set_b = set_rw(b)

    cl = c_void_p()
    qdesc = QueueDesc(ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, None, 0, 0, 0,
                      ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                      ZE_COMMAND_QUEUE_PRIORITY_NORMAL)
    ck("zeCommandListCreateImmediate",
       ze.zeCommandListCreateImmediate(ctx, dev, byref(qdesc), byref(cl)))

    host = c_void_p()
    hdesc = HostMemAllocDesc(ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, None, 0)
    ck("zeMemAllocHost",
       ze.zeMemAllocHost(ctx, byref(hdesc), c_size_t(64), c_size_t(64),
                         byref(host)))
    view = (ctypes.c_ubyte * 64).from_address(host.value)

    def fill(ptr, byte):
        pat = ctypes.c_ubyte(byte)
        res = ze.zeCommandListAppendMemoryFill(
            cl, ptr, byref(pat), c_size_t(1), c_size_t(sz), None, 0, None)
        if res != ZE_RESULT_SUCCESS:
            return res
        return ze.zeCommandListHostSynchronize(cl, ctypes.c_uint64(-1))

    def peek(ptr):
        view[0] = 0
        res = ze.zeCommandListAppendMemoryCopy(
            cl, host, ptr, c_size_t(64), None, 0, None)
        if res != ZE_RESULT_SUCCESS:
            return res, None
        res = ze.zeCommandListHostSynchronize(cl, ctypes.c_uint64(-1))
        return res, view[0]

    ck("fill(a, 0xAB)", fill(a, 0xAB))
    res, seen_a = peek(a)
    ck("peek(a)", res)
    if seen_a != 0xAB:
        raise SystemExit("range A did not read back its own write: 0x%02x" % seen_a)

    # If the sharing worked, B would now read 0xAB.
    res, seen_b = peek(b)
    aliased = 1 if (res == ZE_RESULT_SUCCESS and seen_b == 0xAB) else 0

    # The write that kills the context when the page is shared.
    write_b = fill(b, 0xCD)

    print("map_b=0x%x" % map_b)
    print("set_b=0x%x" % set_b)
    print("aliased=%d" % aliased)
    print("write_b=0x%x" % write_b)
    sys.stdout.flush()

    if case == "teardown":
        # Deliberately unmap after the failed write; this is where the driver
        # crashes. Anything printed above has already been flushed.
        ze.zeVirtualMemUnmap(ctx, b, c_size_t(sz))
        ze.zeVirtualMemUnmap(ctx, a, c_size_t(sz))
        print("survived teardown")
        sys.stdout.flush()
        os._exit(0)

    if case == "control":
        # Only the control can be torn down safely, so only it does.
        ck("unmap(b)", ze.zeVirtualMemUnmap(ctx, b, c_size_t(sz)))
        ck("unmap(a)", ze.zeVirtualMemUnmap(ctx, a, c_size_t(sz)))
        ck("destroy(a)", ze.zePhysicalMemDestroy(ctx, phys_a))
        ck("destroy(b)", ze.zePhysicalMemDestroy(ctx, phys_b))
        ck("free", ze.zeVirtualMemFree(ctx, base, c_size_t(2 * sz)))
        ck("zeContextDestroy", ze.zeContextDestroy(ctx))
        print("clean")
        sys.stdout.flush()

    # The context is poisoned in the shared cases; skip atexit handlers rather
    # than let the driver crash on the way out and muddy the exit status.
    os._exit(0)
    """
)


def _run(case):
    return subprocess.run(
        [sys.executable, "-c", _WORKER, case],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _fields(result):
    """Parse the worker's key=value lines, or explain what it did instead."""
    assert result.returncode == 0, (
        f"worker died (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    out = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    for key in ("map_b", "set_b", "aliased", "write_b"):
        assert key in out, f"worker printed no {key}:\n{result.stdout}"
    return out


def test_control_two_physical_pages_works_end_to_end():
    """The control, and it has to come first: the same reservation, mapping,
    fill, read and teardown sequence, differing only in that each range gets its
    own physical page. If this fails, the test below proves nothing about
    sharing -- the sequence or the ctypes plumbing is simply wrong."""
    result = _run("control")
    fields = _fields(result)

    assert fields["map_b"] == "0x0"
    assert fields["set_b"] == "0x0"
    assert fields["write_b"] == "0x0", "writing an unshared second range failed"
    assert fields["aliased"] == "0", "distinct physical pages must not alias"
    assert "clean" in result.stdout, f"teardown was not clean:\n{result.stdout}"


def test_shared_physical_page_is_silently_not_aliased():
    """The finding, and the canary. One physical page mapped into two virtual
    ranges: the map is accepted, the ranges are not aliased, and the first write
    through the second range loses the device.

    A failure here is good news -- see the assertion messages."""
    fields = _fields(_run("shared"))

    assert fields["map_b"] == "0x0", (
        "zeVirtualMemMap now rejects a shared physical page outright "
        f"({fields['map_b']}). Still unsupported, but no longer silent -- "
        "update the comment in csrc/inc/gpu_vmm.hpp."
    )
    assert fields["aliased"] == "0", (
        "a shared physical page now aliases both virtual ranges. If "
        "write_b is 0x0 too, this driver supports the sharing: flip "
        "gpu_vmm::supports_shared_page_mapping() to true for XPU and the "
        "zero-page safety net comes back (FTensor::init_with_zero_)."
    )
    assert fields["write_b"] != "0x0", (
        "writing through the second virtual range no longer fails; the "
        "sharing may now be supported. See csrc/inc/gpu_vmm.hpp."
    )


def test_unmapping_after_the_lost_write_crashes_the_driver():
    """Why FTensor cannot just ignore the failure and unmap normally: once the
    write above has lost the device, teardown faults inside the driver. Note
    which call: unmapping the second range returns SUCCESS, and the segfault
    lands on unmapping the *first* range, the one that was working. This is what
    the destructor's per-page unmap in FTensor::~FTensor exists to stay clear
    of."""
    result = _run("teardown")

    assert result.returncode != 0, (
        "unmapping after the lost write no longer crashes; FTensor's teardown "
        f"path may be able to be simplified.\nstdout:\n{result.stdout}"
    )
    assert "survived teardown" not in result.stdout


def test_backend_reports_no_zero_page_safety_net():
    """Ties the driver behavior above to the decision kvcached makes about it."""
    vmm_ops = pytest.importorskip("kvcached.vmm_ops")
    from kvcached.utils import get_device_type

    if get_device_type() != "xpu":
        pytest.skip("kvcached is not built for the XPU backend")

    assert not vmm_ops.has_zero_page_safety_net(), (
        "the XPU backend claims a zero-page safety net, but the driver cannot "
        "map one physical page into several virtual ranges"
    )
