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
from kvcached.cli.utils import (  # noqa: E402
    get_kv_cache_limit,
    update_kv_cache_limit,
)
from kvcached.kv_cache_manager import KVCacheManager  # noqa: E402

PAGE_SIZE = 2 * 1024 * 1024
NUM_PAGES = 4
NUM_LAYERS = 2
NUM_KV_BUFFERS = 2
PAGE_BUNDLE = PAGE_SIZE * NUM_LAYERS * NUM_KV_BUFFERS


@pytest.mark.parametrize("keep_live_block", [False, True])
@pytest.mark.parametrize("reserve_null_block", [False, True])
@pytest.mark.parametrize("enable_prealloc", [False, True])
@pytest.mark.parametrize("api_takeover", [False, True])
@pytest.mark.parametrize("block_bytes", [PAGE_SIZE // 4, PAGE_SIZE // 3])
def test_zero_quota_rejects_growth_then_recovers(
    keep_live_block, reserve_null_block, enable_prealloc, api_takeover, block_bytes,
):
    ipc_name = f"kvcached-zero-{os.getpid()}-{uuid.uuid4().hex}"
    env = dict(os.environ, KVCACHED_IPC_NAME=ipc_name)
    try:
        result = subprocess.run(
            [sys.executable, __file__, str(int(keep_live_block)),
             str(int(reserve_null_block)), str(int(enable_prealloc)),
             str(int(api_takeover)), str(block_bytes)],
            env=env, capture_output=True, text=True, timeout=45,
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
    *, block_bytes,
):
    import kvcached.kv_cache_manager as kcm

    monkeypatch.setattr(kcm, "CONTIGUOUS_LAYOUT", False)
    monkeypatch.setattr(kcm, "PAGE_PREALLOC_ENABLED", enable_prealloc)
    torch.cuda.set_device(0)
    vmm_ops.init_kvcached("cuda:0", PAGE_SIZE, False)
    tensors = vmm_ops.create_kv_tensors(
        NUM_PAGES * PAGE_SIZE * NUM_KV_BUFFERS, 1, "cuda:0", NUM_LAYERS,
        num_kv_buffers=NUM_KV_BUFFERS, page_size=PAGE_SIZE,
    )
    manager = KVCacheManager(
        num_blocks=NUM_PAGES * PAGE_SIZE // block_bytes,
        block_size=1, cell_size=block_bytes,
        num_layers=NUM_LAYERS, num_kv_buffers=NUM_KV_BUFFERS, own_segment=True,
        reserve_null_block=reserve_null_block, page_size=PAGE_SIZE,
    )

    def publish(per_layer_bytes):
        total_bytes = per_layer_bytes * NUM_LAYERS * NUM_KV_BUFFERS
        assert update_kv_cache_limit(manager.ipc_name, total_bytes) is not None
        deadline = time.monotonic() + 5
        while manager.page_allocator.get_resize_target() != per_layer_bytes:
            assert time.monotonic() < deadline, "resize watcher did not publish quota"
            time.sleep(0.01)

    def assert_minimum_mapping():
        # Observe several watcher/worker cycles without foreground resize calls:
        # a foreground-only allocation assertion missed the prealloc rebound.
        deadline = time.monotonic() + 0.35
        expected_pages = int(reserve_null_block)
        while time.monotonic() < deadline:
            pages = manager.page_allocator.get_page_state()
            assert pages == dict(total_pages=expected_pages, free_pages=0,
                                 inuse_pages=expected_pages, reserved_pages=0)
            assert vmm_ops.physical_budget_state()["instance_charged_bytes"] == (
                expected_pages * PAGE_BUNDLE)
            assert manager.page_allocator.get_resize_target() == 0
            control = get_kv_cache_limit(manager.ipc_name)
            assert control is not None and control.total_size == 0
            time.sleep(0.01)
        assert not manager.in_shrink

    try:
        manager.wait_ready(timeout=10)
        if enable_prealloc:
            deadline = time.monotonic() + 5
            while manager.page_allocator.get_num_reserved_pages() == 0:
                assert time.monotonic() < deadline, "preallocator never mapped pages"
                time.sleep(0.01)
        else:
            time.sleep(0.2)
            assert manager.page_allocator.get_num_reserved_pages() == 0
        for cycle in range(3):
            # With a null block this crosses into a second live page, so a
            # deferred shrink must really complete when that page is released.
            blocks = manager.alloc(PAGE_SIZE // block_bytes)
            assert blocks is not None
            start = blocks[0] * block_bytes
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
            assert_minimum_mapping()

            # The revisioned API must report the null mapping as overage even
            # though the reachable physical shrink has finished.
            state = manager.set_memory_limit(0, revision=2 * cycle)
            assert state["status"] == ("deferred" if reserve_null_block else "applied")
            assert state["limit_bytes"] == state["effective_limit_bytes"] == 0
            assert state["overage_bytes"] == int(reserve_null_block) * PAGE_BUNDLE
            # A new CLI publication takes back control without rounding its
            # positive subpage value up in shared memory or watcher state.
            publish(PAGE_SIZE // 2)
            manager.apply_resize_target()
            assert manager.page_allocator.get_num_total_pages() == int(reserve_null_block)
            publish(0)
            manager.apply_resize_target()

            if api_takeover:
                state = manager.set_memory_limit(2 * PAGE_BUNDLE, revision=2 * cycle + 1)
                assert state["status"] == "applied"

            # clear() must finish while the watcher still retains zero, including
            # rebuilding the null block while the quota watcher is stopped.
            manager.clear()
            assert manager.null_block == ([0] if reserve_null_block else None)
            assert not manager.in_shrink
            if api_takeover:
                assert manager.page_allocator.get_resize_target() == 0
                assert manager.page_allocator.get_num_total_pages() == 2
                assert manager.alloc(0) == []
                assert manager.page_allocator.get_num_total_pages() == 2
            else:
                assert_minimum_mapping()
                mapped_free = manager.num_avail_blocks
                assert manager.alloc(mapped_free + 1) is None
                # Repeated clear at the retained zero must also leave no
                # free IDs for the restarted worker to map.
                manager.clear()
                assert_minimum_mapping()

            publish(NUM_PAGES * PAGE_SIZE)
            recovered = manager.alloc(PAGE_SIZE // block_bytes + 1)
            assert recovered is not None
            assert manager.page_allocator.get_num_inuse_pages() >= 2
            start = recovered[0] * block_bytes
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
        _exercise_zero_quota(patch, *(bool(int(arg)) for arg in sys.argv[1:5]),
                             block_bytes=int(sys.argv[5]))
    print("zero quota recovery complete")
