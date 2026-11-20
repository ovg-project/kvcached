# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Recycle bounded MLA pages with one physical grant left, using real CUDA."""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import pytest
import test_vmm_failure_policy

fault_library = test_vmm_failure_policy.fault_library


@pytest.mark.parametrize("fail_map", [False, True])
def test_bounded_pool_reuses_last_physical_grant(fault_library, fail_map):
    env = dict(
        os.environ, LD_PRELOAD=str(fault_library),
        ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0",
        KVCACHED_PAGE_PREALLOC_ENABLED="false",
        KVCACHED_MIN_RESERVED_PAGES="0", KVCACHED_MAX_RESERVED_PAGES="0",
        KVCACHED_CONTIGUOUS_LAYOUT="false",
        KVCACHED_IPC_NAME=f"bounded-mla-{os.getpid()}-{int(fail_map)}",
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(int(fail_map))],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "bounded MLA reuse complete" in result.stdout


def _exercise(fail_map):
    import torch

    from kvcached import vmm_ops
    from kvcached.kv_cache_manager import KVCacheManager

    page, block, layers = 2 * 1024 * 1024, 64 * 576 * 2, 2
    device = f"cuda:{int(os.environ.get('KVCACHED_TEST_DEVICE', '0'))}"
    torch.cuda.set_device(device)
    driver = ctypes.CDLL(None)
    driver.kvcached_fault_arm.argtypes = [ctypes.c_int] * 3
    driver.kvcached_fault_hits.restype = ctypes.c_int
    vmm_ops.init_kvcached(device, page, False)
    tensors = vmm_ops.create_kv_tensors(
        10 * page, 1, device, layers, num_kv_buffers=1,
        unified_pool=True, page_size=page)
    manager = KVCacheManager(
        285, 64, 576 * 2, layers, num_kv_buffers=1, page_size=page,
        logical_num_blocks=256, max_block_id=256, reserve_null_block=True)
    try:
        manager.wait_ready(timeout=15)
        assert vmm_ops.set_instance_budget_limit(9 * page * layers, 1) == "applied"
        blocks = manager.alloc(247)
        assert blocks is not None and len(blocks) == 247
        page3 = [b for b in blocks if b * block // page == 3]
        assert len(page3) == 27
        live = torch.tensor([b * block for b in blocks if b not in page3], device=device)
        for tensor in tensors:
            tensor[live] = 37
        torch.cuda.synchronize()

        for _ in range(3):
            manager.free(page3)
            manager.trim()
            assert manager.page_allocator.get_avail_physical_pages() == 1
            # Before the fix free order is [9, 3]: the empty page consumes
            # the only forecast grant, hiding all 27 reusable blocks.
            assert manager.available_size() == 27
            if fail_map:
                # Fail the second layer's allocation after the first succeeds.
                driver.kvcached_fault_arm(2, 0, 0)
                assert manager.alloc(27) is None
                assert driver.kvcached_fault_hits() == 1
                driver.kvcached_fault_arm(0, 0, 0)
                assert manager.available_size() == 27
            assert manager.alloc(27) == page3
            for tensor in tensors:
                assert torch.all(tensor[live] == 37).item()
                for b in page3:
                    tensor[b * block:b * block + 128].fill_(53)
                    assert torch.all(tensor[b * block:b * block + 128] == 53).item()
            torch.cuda.synchronize()
            assert manager.available_size() == 0
            assert manager.alloc(1) is None

        # A later growth request and clear must not restore the empty tail.
        assert manager.resize(10 * page)
        assert manager.page_allocator.get_num_total_pages() == 9
        manager.clear()
        assert manager.null_block == [0]
        blocks = manager.alloc(247)
        assert blocks is not None and len(blocks) == 247
        assert manager.available_size() == 0
    finally:
        driver.kvcached_fault_arm(0, 0, 0)
        assert manager.shutdown()
        vmm_ops.shutdown_kvcached()


if __name__ == "__main__":
    _exercise(bool(int(sys.argv[1])))
    print("bounded MLA reuse complete")
