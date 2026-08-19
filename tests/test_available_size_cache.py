# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""CPU-only test for the get_avail_physical_pages() TTL cache in
KVCacheManager.available_size().

GPU-free: ``kvcached.vmm_ops`` is stubbed with a pure-Python fake when the
compiled extension is unavailable, mirroring tests/test_alloc_rollback.py.
available_size() otherwise fires a cudaMemGetInfo driver call
(csrc/page_allocator.cpp:481) on every invocation, and it runs per allocation
(kvcached/integration/vllm/patches.py:792) and per scheduler step (:927). These
tests assert that N back-to-back calls within one 100 ms TTL window invoke the
(expensive) page_allocator.get_avail_physical_pages exactly once instead of N
times, and that the cache re-fetches after the window elapses or resize()
invalidates it.
"""

import sys
import threading
import time
import types

BLOCKS_PER_PAGE = 4


class _FakeInternalPage:
    """Minimal stand-in for kvcached_cpp.InternalPage (no GPU needed)."""

    @staticmethod
    def get_num_blocks(page_size: int, block_mem_size: int) -> int:
        return page_size // block_mem_size


class CountingPageAllocator:
    """Pure-Python stand-in for the C++ PageAllocator that counts
    get_avail_physical_pages() calls -- the call available_size() caches."""

    def __init__(self) -> None:
        self.get_avail_call_count = 0

    def get_avail_physical_pages(self) -> int:
        self.get_avail_call_count += 1
        return 100

    def get_num_free_pages(self) -> int:
        return 100

    def get_num_reserved_pages(self) -> int:
        return 0

    def get_resize_target(self) -> int:
        return 0

    def resize(self, new_mem_size: int) -> bool:
        # Mirror a successful C++ resize so KVCacheManager.resize() invalidates
        # and returns True without entering the lazy-shrink path.
        return True


def _install_vmm_ops_stub() -> None:
    stub = types.ModuleType("kvcached.vmm_ops")
    stub.PageAllocator = CountingPageAllocator  # type: ignore[attr-defined]
    stub.InternalPage = _FakeInternalPage  # type: ignore[attr-defined]
    stub.kv_tensors_created = lambda group_id=0: True  # type: ignore[attr-defined]
    stub.map_to_kv_tensors = lambda *a, **k: None  # type: ignore[attr-defined]
    stub.unmap_from_kv_tensors = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["kvcached.vmm_ops"] = stub


try:
    import kvcached.vmm_ops  # noqa: F401
except ImportError:
    _install_vmm_ops_stub()

import kvcached.kv_cache_manager as _kvc_mod  # noqa: E402
from kvcached.kv_cache_manager import KVCacheManager  # noqa: E402
from kvcached.locks import NoOpLock  # noqa: E402


def make_manager() -> KVCacheManager:
    """Build a KVCacheManager around fakes without running __init__ (which
    needs the C++ extension, KV tensors, and background threads)."""
    manager = object.__new__(KVCacheManager)
    manager.page_size = BLOCKS_PER_PAGE
    manager.block_mem_size = 1
    manager.page_allocator = CountingPageAllocator()
    manager.num_avail_blocks = 0
    manager.avail_pages = {}
    manager.full_pages = {}
    manager.reserved_blocks = []
    manager.null_block = None
    manager.in_shrink = False
    manager.target_num_blocks = None
    manager._lock = NoOpLock()
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()
    return manager


def test_available_size_caches_get_avail_physical_pages():
    """Within one TTL window, N available_size() calls hit the driver-backed
    get_avail_physical_pages exactly once (green on branch); on master the
    same calls invoke it N times (red)."""
    manager = make_manager()
    n = 5
    for _ in range(n):
        manager.available_size()
    assert manager.page_allocator.get_avail_call_count == 1


def test_available_size_refetches_after_ttl_window(monkeypatch):
    """Once the TTL window elapses, available_size() re-reads the driver call
    and serves the fresh value from the cache until the window elapses again."""
    # Fallback to 0.1 so this test also runs (red) on master, where the
    # constant does not exist; on the branch it tracks the real value.
    ttl = getattr(_kvc_mod, "_AVAIL_PHYSICAL_PAGES_TTL_S", 0.1)
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    manager = make_manager()
    manager.available_size()
    assert manager.page_allocator.get_avail_call_count == 1

    clock["t"] = ttl  # exactly one window -> stale
    manager.available_size()
    assert manager.page_allocator.get_avail_call_count == 2

    manager.available_size()  # same instant -> still fresh
    assert manager.page_allocator.get_avail_call_count == 2

    clock["t"] = ttl * 3  # well past -> stale again
    manager.available_size()
    assert manager.page_allocator.get_avail_call_count == 3


def test_available_size_refetches_after_resize():
    """resize() invalidates the cache so the next available_size() re-reads
    the driver instead of serving pre-resize physical-free data."""
    manager = make_manager()
    manager.available_size()
    assert manager.page_allocator.get_avail_call_count == 1

    manager.resize(1024)
    manager.available_size()
    assert manager.page_allocator.get_avail_call_count == 2
