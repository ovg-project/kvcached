# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Device-index plumbing on a multi-XPU host.

The XPU arm resolves device indices in kvcached::xpu_runtime::resolve() and
passes the resulting sycl::device into physical_mem() and get_mem_granularity().
A hardcoded device 0 anywhere in that path would still pass every single-device
test while silently placing a TP rank's KV cache on the wrong GPU, so these tests
target a non-zero index specifically.

Each device is exercised in its own subprocess: kvcached binds one device per
process at init_kvcached() time, which is also how TP workers are laid out.

    KVCACHED_BACKEND=xpu python setup.py build_ext --inplace
    pytest tests/test_xpu_multi_device.py -v
"""

import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("requires an Intel XPU", allow_module_level=True)

from kvcached.utils import get_device_type  # noqa: E402

if get_device_type() != "xpu":
    pytest.skip(
        "kvcached is not built for the XPU backend", allow_module_level=True
    )

DEVICE_COUNT = torch.xpu.device_count()

# Body run in a subprocess, one device per process. Prints OK on success so the
# parent can distinguish a clean pass from a crash inside the native layer.
_WORKER = textwrap.dedent(
    """
    import sys
    import torch
    from kvcached.utils import PAGE_SIZE
    from kvcached.vmm_ops import (
        create_kv_tensors,
        init_kvcached,
        map_to_kv_tensors,
        shutdown_kvcached,
        unmap_from_kv_tensors,
    )

    dev_idx = int(sys.argv[1])
    device = f"xpu:{dev_idx}"
    num_layers, dtype = 2, torch.float16
    size = 8 * PAGE_SIZE

    # No torch.xpu.set_device() here: init_kvcached() both range-checks the
    # index and makes it current (gpu_vmm::set_device -> c10::xpu::set_device),
    # so binding it first would only preempt the check this module is here to
    # exercise. It is what makes the bare torch.xpu.synchronize() below wait on
    # dev_idx.
    init_kvcached(device, PAGE_SIZE, False)
    try:
        tensors = create_kv_tensors(size, dtype.itemsize, device, num_layers, 2)

        for t in tensors:
            assert t.device.type == "xpu", t.device
            assert t.device.index == dev_idx, (
                f"tensor landed on {t.device}, expected {device}"
            )

        offsets = [0, PAGE_SIZE]
        assert map_to_kv_tensors(offsets)
        elems = PAGE_SIZE // dtype.itemsize
        # create_kv_tensors() is given a cell width, not a dtype, so the scalar
        # type is the allocator's choice; the sentinel must be representable in
        # any type of that width, hence an integer rather than 3.5.
        sentinel = dev_idx + 1
        tensors[0][:elems].fill_(sentinel)
        torch.xpu.synchronize()
        assert torch.all(tensors[0][:elems] == sentinel), "readback failed"
        assert unmap_from_kv_tensors(offsets)
    finally:
        shutdown_kvcached()
    print("OK")
    """
)


def _run_on_device(dev_idx, env=None):
    return subprocess.run(
        [sys.executable, "-c", _WORKER, str(dev_idx)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


@pytest.mark.parametrize("dev_idx", range(DEVICE_COUNT))
def test_allocation_lands_on_the_requested_device(dev_idx):
    result = _run_on_device(dev_idx)
    assert result.returncode == 0, (
        f"xpu:{dev_idx} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


@pytest.mark.skipif(DEVICE_COUNT < 2, reason="requires at least 2 XPUs")
def test_non_default_device_is_not_silently_redirected_to_zero():
    """The specific failure this guards: a hardcoded device 0 in the XPU arm
    would make xpu:1 allocate on xpu:0 and still report success."""
    last = DEVICE_COUNT - 1
    result = _run_on_device(last)
    assert result.returncode == 0, (
        f"xpu:{last} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_out_of_range_device_fails_loudly():
    """resolve() range-checks against device_count(); an out-of-range index must
    raise rather than fall back to the current device.

    Deliberately unguarded: DEVICE_COUNT + 8 is out of range on any host, so a
    second device buys nothing and a `DEVICE_COUNT < 2` guard would hide this
    regression on the common single-GPU box. The module-level skip already
    guarantees at least one XPU, since torch.xpu.is_available() is
    device_count() > 0."""
    result = _run_on_device(DEVICE_COUNT + 8)
    assert result.returncode != 0, (
        "an out-of-range XPU index was accepted:\n" + result.stdout
    )
    # A non-zero exit alone would also be satisfied by an unrelated crash, or by
    # torch rejecting the index before kvcached ever saw it -- torch's own
    # message says "is out of range" too, so match resolve()'s distinctive
    # wording and keep this a test of the native range check.
    assert "XPU device(s) visible" in result.stderr, (
        "the worker failed, but not in resolve():\n" + result.stderr
    )
