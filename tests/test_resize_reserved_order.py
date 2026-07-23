# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for KVCacheManager.resize(): when the page allocator cannot
shrink immediately (too many blocks in use), resize() must free any
outstanding reserved blocks before it enters the lazy in_shrink wait, not
crash on an assertion that only the free it hasn't reached yet could satisfy.

This test stubs out the compiled kvcached.vmm_ops C++ extension (which
requires a CUDA/HIP build and a physical GPU to import and run) so the pure
control-flow logic in KVCacheManager.resize()/free_reserved() can be
exercised deterministically on CPU-only hardware. The lower-level `free()`
call that free_reserved() makes is stubbed too: this test is about the
*ordering* between the reserved-blocks assertion and the free, not about
free()'s own page-bookkeeping, which is exercised by the existing
GPU-dependent test_kvcache_manager.py.
"""

import sys
import threading
import types

import pytest


def _install_fake_vmm_ops():
    """Register a minimal fake kvcached.vmm_ops so kv_cache_manager.py's
    hard import succeeds without the compiled CUDA/HIP extension."""

    class FakeInternalPage:
        pass

    class FakePageAllocator:

        def __init__(self, *args, **kwargs):
            pass

        def set_should_use_worker_ipc_callback(self, cb):
            pass

        def set_broadcast_map_callback(self, cb):
            pass

        def set_broadcast_unmap_callback(self, cb):
            pass

        def start_prealloc_thread(self):
            pass

        def resize(self, new_mem_size):
            return False

        def trim(self):
            pass

    fake_module = types.ModuleType("kvcached.vmm_ops")
    fake_module.PageAllocator = FakePageAllocator  # type: ignore[attr-defined]
    fake_module.InternalPage = FakeInternalPage  # type: ignore[attr-defined]
    fake_module.kv_tensors_created = lambda *a, **kw: True  # type: ignore[attr-defined]
    fake_module.map_to_kv_tensors = lambda *a, **kw: None  # type: ignore[attr-defined]
    fake_module.unmap_from_kv_tensors = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["kvcached.vmm_ops"] = fake_module


_install_fake_vmm_ops()

from kvcached.kv_cache_manager import KVCacheManager  # noqa: E402
from kvcached.locks import NoOpLock  # noqa: E402


def _make_manager(resize_return=False):
    """Build a KVCacheManager instance without running __init__ (which
    spawns a background thread and talks to the real page allocator);
    set exactly the attributes resize()/free_reserved() read or write."""
    manager = object.__new__(KVCacheManager)
    manager._lock = NoOpLock()
    manager.block_mem_size = 1024
    manager.reserved_blocks = [1, 2, 3]
    manager.in_shrink = False
    manager.target_num_blocks = None
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()

    class StubPageAllocator:

        def resize(self, new_mem_size):
            return resize_return

    manager.page_allocator = StubPageAllocator()

    freed = []
    manager.free = lambda indices: freed.append(list(indices))
    return manager, freed


def test_resize_frees_reserved_blocks_before_entering_shrink_wait():
    """When the allocator can't shrink immediately, resize() must free the
    outstanding reserved blocks and enter in_shrink, not crash."""
    manager, freed = _make_manager(resize_return=False)

    result = manager.resize(4096)

    assert result is False
    assert manager.in_shrink is True
    assert manager.target_num_blocks == 4096 // manager.block_mem_size
    assert manager.reserved_blocks == []
    assert freed == [[1, 2, 3]]


def test_resize_with_no_reserved_blocks_still_works():
    """Sanity check: the common case (nothing reserved) is unaffected."""
    manager, _freed = _make_manager(resize_return=False)
    manager.reserved_blocks = []

    result = manager.resize(4096)

    assert result is False
    assert manager.in_shrink is True
    assert manager.reserved_blocks == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
