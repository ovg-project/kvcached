# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Publish zero quotas through real shared memory and exercise real CUDA pages."""

import os
import subprocess
import sys
import time
import uuid

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a CUDA device", allow_module_level=True)

from kvcached import vmm_ops  # noqa: E402
from kvcached.cli.utils import update_kv_cache_limit  # noqa: E402
from kvcached.kv_cache_manager import KVCacheManager  # noqa: E402
from kvcached.runtime_control import RuntimeControl  # noqa: E402

PAGE_SIZE = 2 * 1024 * 1024
NUM_PAGES = 4
BLOCK_BYTES = PAGE_SIZE // 4


@pytest.mark.parametrize("keep_live_block", [False, True])
@pytest.mark.parametrize("reserve_null_block", [False, True])
@pytest.mark.parametrize("enable_prealloc", [False, True])
@pytest.mark.parametrize("api_takeover", [False, True])
def test_zero_quota_rejects_growth_then_recovers(
    keep_live_block, reserve_null_block, enable_prealloc, api_takeover,
):
    ipc_name = f"kvcached-zero-{os.getpid()}-{uuid.uuid4().hex}"
    env = dict(os.environ, KVCACHED_IPC_NAME=ipc_name)
    try:
        result = subprocess.run(
            [sys.executable, __file__, str(int(keep_live_block)),
             str(int(reserve_null_block)), str(int(enable_prealloc)),
             str(int(api_takeover))],
            env=env, capture_output=True, text=True, timeout=30,
        )
    finally:
        # Native shutdown closes the segment; the test owns its deletion.
        path = f"/dev/shm/{ipc_name}"
        if os.path.exists(path):
            os.unlink(path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "zero quota recovery complete" in result.stdout


def _exercise_zero_quota(
    monkeypatch, keep_live_block, reserve_null_block, enable_prealloc, api_takeover,
):
    import kvcached.kv_cache_manager as kcm

    monkeypatch.setattr(kcm, "CONTIGUOUS_LAYOUT", False)
    monkeypatch.setattr(kcm, "PAGE_PREALLOC_ENABLED", enable_prealloc)
    torch.cuda.set_device(0)
    vmm_ops.init_kvcached("cuda:0", PAGE_SIZE, False)
    tensors = vmm_ops.create_kv_tensors(
        NUM_PAGES * PAGE_SIZE, 1, "cuda:0", 1,
        num_kv_buffers=1, unified_pool=True, page_size=PAGE_SIZE,
    )
    manager = KVCacheManager(
        num_blocks=NUM_PAGES * 4, block_size=1, cell_size=BLOCK_BYTES,
        num_layers=1, num_kv_buffers=1, own_segment=True,
        reserve_null_block=reserve_null_block, page_size=PAGE_SIZE,
    )

    def publish(per_layer_bytes):
        result = update_kv_cache_limit(manager.ipc_name, per_layer_bytes)
        assert result["status"] == "accepted"
        runtime.poll(force=True)
        manager.apply_resize_target()
        return result

    try:
        manager.wait_ready(timeout=10)
        runtime = RuntimeControl(
            manager, worker_ipc=False, can_shrink=True, expected_participants=1)
        setattr(manager, "_runtime_control", runtime)
        if not enable_prealloc:
            time.sleep(0.2)
            assert manager.page_allocator.get_num_reserved_pages() == 0
        for cycle in range(3):
            blocks = manager.alloc(1)
            assert blocks is not None
            start = blocks[0] * BLOCK_BYTES
            view = tensors[0][start:start + 256]
            view.fill_(37)
            torch.cuda.synchronize()
            if not keep_live_block:
                manager.free(blocks)

            publish(0)
            # Shrink preserves in-use pages, whose free blocks remain reusable.
            # Asking for one more than those blocks would require a new page.
            mapped_free = manager.num_avail_blocks + len(manager.reserved_blocks)
            assert manager.alloc(mapped_free + 1) is None
            assert manager.available_size() == mapped_free
            reused = manager.alloc(mapped_free)
            assert reused is not None
            assert manager.alloc(1) is None
            if keep_live_block:
                assert torch.all(view == 37).item()
            manager.free(reused)
            if keep_live_block:
                manager.free(blocks)
            if reserve_null_block:
                assert not manager.in_shrink
                assert manager.page_allocator.get_num_total_pages() == 1
                state = manager.memory_limit_state()
                assert state["status"] == "pending"
                assert state["gpus"][0]["limit_bytes"] == 0
                assert state["gpus"][0]["charged_bytes"] == PAGE_SIZE
            else:
                assert manager.page_allocator.get_num_total_pages() == 0

            if api_takeover:
                state = manager.set_memory_limit(2 * PAGE_SIZE)
                assert state["status"] == "accepted"
                runtime.poll(force=True)
                manager.apply_resize_target()

            # Repeated clear must preserve the existing null receipt under a
            # zero cap; it cannot acquire a new background grant.
            manager.clear()
            assert manager.null_block == ([0] if reserve_null_block else None)
            assert not manager.in_shrink
            if api_takeover:
                assert manager.page_allocator.get_num_total_pages() == 2
                assert manager.alloc(0) == []
                assert manager.page_allocator.get_num_total_pages() == 2
            else:
                mapped_free = manager.num_avail_blocks
                assert manager.alloc(mapped_free + 1) is None
                time.sleep(0.2)
                assert manager.page_allocator.get_num_reserved_pages() == 0
                assert manager.page_allocator.get_num_total_pages() == int(reserve_null_block)
                state = manager.memory_limit_state()
                assert state["status"] == ("pending" if reserve_null_block else "converged")

            publish(NUM_PAGES * PAGE_SIZE)
            recovered = manager.alloc(1)
            assert recovered is not None
            start = recovered[0] * BLOCK_BYTES
            recovered_view = tensors[0][start:start + 256]
            recovered_view.fill_(53)
            assert torch.all(recovered_view == 53).item()
            manager.free(recovered)
            assert not manager.in_shrink
    finally:
        # Leave the pool at its startup capacity before teardown.
        publish(NUM_PAGES * PAGE_SIZE)
        manager.alloc(0)
        manager.clear()
        assert manager.shutdown()
        vmm_ops.shutdown_kvcached()


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as patch:
        _exercise_zero_quota(patch, *(bool(int(arg)) for arg in sys.argv[1:]))
    print("zero quota recovery complete")
