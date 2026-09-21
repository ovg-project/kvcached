# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""What the XPU arm does when a failed map cannot be cleaned up.

A 2 MiB page is 32 chunks of the driver's 64 KiB granularity, and a map that
fails part-way leaves the chunks it did map behind, so mem_map() unmaps the range
itself before reporting the failure. That cleanup is what lets the caller treat
an out-of-memory map as a capacity miss and map the range again later --
``test_xpu_map_failure_recovery.py`` is that case.

If the cleanup fails too, nothing knows whether the range still holds part of the
page. Releasing the physical handle then hands the driver memory that may still
be mapped, and mapping it again writes over chunks that may still be there, so
the page has to stay owned and out of service instead: the state-consistency path
from #418. This drives that double failure through the real allocator with a
preloaded shim that fails one map and the cleanup unmap behind it.

The shim fails at the SYCL seam, not in the driver, so no mapping is really left
behind. What it tests is the decision the double failure forces, not the driver
state that would produce it.

    KVCACHED_BACKEND=xpu python setup.py build_ext --inplace
    ZE_AFFINITY_MASK=0 pytest tests/test_xpu_map_cleanup_failure.py -v
"""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("requires an Intel XPU", allow_module_level=True)

from kvcached.utils import get_device_type  # noqa: E402

if get_device_type() != "xpu":
    pytest.skip(
        "kvcached is not built for the XPU backend", allow_module_level=True
    )

PAGE_SIZE = 2 * 1024 * 1024
SHIM = Path(__file__).parent / "native" / "xpu_vmm_faults.cpp"


@pytest.fixture(scope="module")
def fault_library(tmp_path_factory):
    """The shim needs no SYCL headers: it names its targets by mangled symbol."""
    library = tmp_path_factory.mktemp("xpu-vmm-faults") / "faults.so"
    subprocess.run([
        "c++", "-std=c++17", "-shared", "-fPIC", "-O2", str(SHIM), "-ldl",
        "-o", str(library),
    ], check=True)
    return library


def test_map_cleanup_failure_keeps_the_page_out_of_service(fault_library):
    """The double failure must quarantine the page, not offer it again.

    Its own process: LD_PRELOAD has to be in place before the SYCL runtime
    loads, and the allocator is left DEGRADED on purpose, which no other test
    should inherit.
    """
    env = dict(os.environ,
               LD_PRELOAD=str(fault_library),
               KVCACHED_MIN_RESERVED_PAGES="2",
               KVCACHED_MAX_RESERVED_PAGES="0",
               KVCACHED_IPC_NAME=f"xpu-cleanup-{os.getpid()}",
               ENABLE_KVCACHED="false",
               KVCACHED_AUTOPATCH="0")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                            env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout, result.stdout + result.stderr


def _case():
    from kvcached import vmm_ops
    from kvcached.errors import MapQuarantinedError

    torch.xpu.set_device(0)
    vmm_ops.init_kvcached("xpu:0", PAGE_SIZE, False)
    tensors = vmm_ops.create_kv_tensors(8 * PAGE_SIZE, 2, "xpu:0", 2, 2, 0,
                                        False)
    allocator = vmm_ops.PageAllocator(2, 4 * PAGE_SIZE, PAGE_SIZE,
                                      contiguous_layout=False,
                                      enable_page_prealloc=False,
                                      ipc_name=os.environ["KVCACHED_IPC_NAME"])
    shim = ctypes.CDLL(None)  # preloaded, so its controls are already here
    shim.kvcached_xpu_fault_arm.argtypes = [ctypes.c_int] * 2
    for counter in ("kvcached_xpu_fault_map_hits",
                    "kvcached_xpu_fault_cleanup_hits"):
        getattr(shim, counter).restype = ctypes.c_int

    # Page 0 first, so the page that fails sits at a nonzero offset: virtual page
    # 0 has the anchor page under it and a release path of its own, which
    # test_xpu_map_failure_recovery.py covers.
    first = allocator.alloc_page()
    assert first.page_id == 0

    # Fail the next map, and the cleanup unmap that mem_map() runs behind it.
    shim.kvcached_xpu_fault_arm(1, 1)
    with pytest.raises(MapQuarantinedError) as raised:
        allocator.alloc_page()
    # Counted before disarming, which resets them.
    map_hits = shim.kvcached_xpu_fault_map_hits()
    cleanup_hits = shim.kvcached_xpu_fault_cleanup_hits()
    shim.kvcached_xpu_fault_arm(0, 0)
    assert (map_hits, cleanup_hits) == (1, 1), (map_hits, cleanup_hits)

    # Both errors survive, because neither explains the other: the map's says the
    # page could not be placed, the unmap's says that could not be undone.
    message = str(raised.value)
    assert "injected map failure" in message, message
    assert "injected cleanup unmap failure" in message, message

    state = allocator.get_transaction_state()
    assert state["quarantined_page_ids"] == [1], state
    assert state["state"] == "DEGRADED", state
    # Still owned, which is the whole point: with the range's state unknown, the
    # handles must not go back to the driver. Four physical pages per logical
    # page here -- two layers, K and V.
    assert state["retained_bytes_upper_bound"] == 4 * PAGE_SIZE, state

    # One page out of service, not the pool: the quarantined id is never offered
    # again and everything else still maps and is writable.
    assert allocator.get_num_free_pages() == 2
    pages = [allocator.alloc_page() for _ in range(2)]
    assert [page.page_id for page in pages] == [2, 3]
    for page in [first, *pages]:
        for tensor in tensors:
            data = tensor.view(-1)
            for base in (0, data.numel() // 2):
                start = base + page.page_id * PAGE_SIZE // data.element_size()
                data[start:start + 128].fill_(7)
                assert bool((data[start:start + 128] == 7).all())
    torch.xpu.synchronize()

    vmm_ops.shutdown_kvcached()
    print("PASS", flush=True)


if __name__ == "__main__":
    _case()
