# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Count real driver allocations and exercise single-buffer rollback/release."""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import pytest
import test_vmm_failure_policy

fault_library = test_vmm_failure_policy.fault_library


@pytest.mark.parametrize("buffers", [1, 2], ids=["mla", "mha-gqa"])
@pytest.mark.parametrize("case", ["footprint", "map-failure", "unmap-failure"])
def test_native_mapping_geometry(fault_library, buffers, case):
    env = dict(os.environ, LD_PRELOAD=str(fault_library),
               ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(buffers), case],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def _native_case(buffers, case):
    import torch

    from kvcached import vmm_ops as vmm

    device = int(os.environ.get("KVCACHED_TEST_DEVICE", "0"))
    torch.cuda.set_device(device)
    page = 2 * 1024 * 1024
    layers = 2
    vmm.init_kvcached(f"cuda:{device}", page, False)
    driver = ctypes.CDLL(None)
    driver.kvcached_fault_arm.argtypes = [ctypes.c_int] * 3
    for name in ("kvcached_fault_hits", "kvcached_create_count", "kvcached_release_count"):
        getattr(driver, name).restype = ctypes.c_int
    try:
        tensors = vmm.create_kv_tensors(
            8 * page, 1, f"cuda:{device}", layers, num_kv_buffers=buffers)
        # Reset after zero-page creation: count only the physical KV pages.
        driver.kvcached_fault_arm(0, 0, 0)
        assert vmm.map_to_kv_tensors_with_result([0]) == (True, [0])
        expected = layers * buffers
        assert driver.kvcached_create_count() == expected
        views = [tensor[start:start + 128] for tensor in tensors
                 for start in ([0] if buffers == 1 else [0, 4 * page])]
        for index, view in enumerate(views):
            view.fill_(index + 7)
        torch.cuda.synchronize()

        if case == "map-failure":
            # Fail after one new layer page succeeds; existing pages survive.
            driver.kvcached_fault_arm(2, 0, 0)
            with pytest.raises(RuntimeError):
                vmm.map_to_kv_tensors_with_result([0, page])
            assert driver.kvcached_fault_hits() == 1
            assert driver.kvcached_release_count() == 1
            driver.kvcached_fault_arm(0, 0, 0)
        elif case == "unmap-failure":
            # A later driver failure must restore earlier retained mappings.
            driver.kvcached_fault_arm(0, 2, 0)
            with pytest.raises(RuntimeError):
                vmm.prepare_unmap_from_kv_tensors([0], "failed-prepare")
            assert driver.kvcached_fault_hits() == 1
            assert driver.kvcached_release_count() == 0
            driver.kvcached_fault_arm(0, 0, 0)

        assert vmm.map_to_kv_tensors_with_result([0]) == (True, [])
        for index, view in enumerate(views):
            assert torch.all(view == index + 7).item()
        torch.cuda.synchronize()
        assert vmm.prepare_unmap_from_kv_tensors([0], "abort")
        assert vmm.abort_unmap_from_kv_tensors("abort")
        for index, view in enumerate(views):
            assert torch.all(view == index + 7).item()
        torch.cuda.synchronize()
        assert vmm.prepare_unmap_from_kv_tensors([0], "commit")
        assert vmm.commit_unmap_from_kv_tensors("commit")
        assert driver.kvcached_release_count() == expected

        # Direct release/reuse must also use the same geometry.
        driver.kvcached_fault_arm(0, 0, 0)
        offsets = [0, 4 * page] if buffers == 1 else [0, page]
        assert vmm.map_to_kv_tensors_with_result(offsets) == (True, offsets)
        assert driver.kvcached_create_count() == 2 * expected
        # MLA's upper half is usable storage, not a mirrored V region.
        for tensor in tensors:
            tensor[:128].fill_(23)
            tensor[offsets[1]:offsets[1] + 128].fill_(29)
            assert torch.all(tensor[:128] == 23).item()
            assert torch.all(tensor[offsets[1]:offsets[1] + 128] == 29).item()
        torch.cuda.synchronize()
        assert vmm.unmap_from_kv_tensors(offsets)
        assert driver.kvcached_release_count() == 2 * expected
        print(f"PASS buffers={buffers} case={case} pages_per_offset={expected}")
    finally:
        driver.kvcached_fault_arm(0, 0, 0)
        vmm.shutdown_kvcached()


if __name__ == "__main__":
    _native_case(int(sys.argv[1]), sys.argv[2])
