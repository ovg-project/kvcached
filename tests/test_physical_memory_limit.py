# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
    sys.modules.setdefault("torch", types.ModuleType("torch"))


def _load_manager_class():
    fake_module = types.ModuleType("kvcached.vmm_ops")
    setattr(fake_module, "PageAllocator", object)
    setattr(
        fake_module,
        "InternalPage",
        types.SimpleNamespace(
            get_num_blocks=lambda page_size, block_mem_size: (
                page_size // block_mem_size
            )
        ),
    )
    setattr(fake_module, "kv_tensors_created", lambda *args, **kwargs: True)
    setattr(fake_module, "map_to_kv_tensors", lambda *args, **kwargs: True)
    setattr(fake_module, "unmap_from_kv_tensors", lambda *args, **kwargs: True)
    previous = sys.modules.get("kvcached.vmm_ops")
    sys.modules["kvcached.vmm_ops"] = fake_module
    sys.modules.pop("kvcached.kv_cache_manager", None)
    try:
        from kvcached.kv_cache_manager import KVCacheManager

        return KVCacheManager
    finally:
        if previous is None:
            sys.modules.pop("kvcached.vmm_ops", None)
        else:
            sys.modules["kvcached.vmm_ops"] = previous


KVCacheManager = _load_manager_class()
ROOT = Path(__file__).resolve().parents[1]


class FakeAllocator:
    def __init__(self, *, total_pages: int = 16, mapped_pages: int = 0):
        self.total_pages = total_pages
        self.mapped_pages = mapped_pages
        self.limit = -1
        self.trim_calls = 0
        self.physical_free_queries = 0

    def get_num_total_pages(self):
        return self.total_pages

    def set_physical_page_limit(self, max_pages):
        self.limit = max_pages

    def get_physical_page_limit(self):
        return self.limit

    def get_num_mapped_pages(self):
        return self.mapped_pages

    def get_num_physical_limit_remaining_pages(self):
        if self.limit < 0:
            return self.total_pages
        return max(0, self.limit - self.mapped_pages)

    def get_num_free_pages(self):
        return self.total_pages

    def get_avail_physical_pages(self):
        self.physical_free_queries += 1
        return self.total_pages

    def get_num_reserved_pages(self):
        return 0

    def trim(self):
        self.trim_calls += 1


def _manager(*, mapped_pages: int = 0):
    manager = object.__new__(KVCacheManager)
    manager.page_size = 2 * 1024 * 1024
    manager.num_layers = 2
    manager.num_kv_buffers = 2
    manager.mem_size = manager.page_size * 8
    manager.group_id = 0
    manager._physical_memory_limit_bytes = None
    manager._physical_memory_limit_revision = -1
    manager.page_allocator = FakeAllocator(mapped_pages=mapped_pages)
    manager._lock = types.SimpleNamespace(
        __enter__=lambda self: self,
        __exit__=lambda self, *args: None,
    )
    manager.num_avail_blocks = 0
    manager.reserved_blocks = []
    manager.in_shrink = False
    manager.skip_physical_free_check = False
    manager.block_mem_size = manager.page_size
    return manager


def test_limit_is_page_bundle_aligned_and_versioned():
    manager = _manager(mapped_pages=1)
    page_bundle = manager._physical_page_bundle_bytes()

    state = KVCacheManager.set_physical_memory_limit.__wrapped__(
        manager, page_bundle * 3 + 123, revision=7
    )

    assert manager.page_allocator.limit == 3
    assert state["status"] == "applied"
    assert state["limit_bytes"] == page_bundle * 3 + 123
    assert state["effective_limit_bytes"] == page_bundle * 3
    assert state["revision"] == 7


def test_lower_limit_defers_without_revoking_active_pages():
    manager = _manager(mapped_pages=4)
    page_bundle = manager._physical_page_bundle_bytes()

    state = KVCacheManager.set_physical_memory_limit.__wrapped__(
        manager, page_bundle * 2, revision=1
    )

    assert state["status"] == "deferred"
    assert state["mapped_bytes"] == page_bundle * 4
    assert state["overage_bytes"] == page_bundle * 2
    assert manager.page_allocator.trim_calls == 1


def test_stale_limit_update_is_ignored():
    manager = _manager()
    page_bundle = manager._physical_page_bundle_bytes()
    KVCacheManager.set_physical_memory_limit.__wrapped__(manager, page_bundle * 4, revision=9)

    state = KVCacheManager.set_physical_memory_limit.__wrapped__(
        manager, page_bundle * 2, revision=8
    )

    assert state["status"] == "stale"
    assert manager.page_allocator.limit == 4
    assert state["revision"] == 9


def test_current_revision_can_be_redistributed_across_local_pools():
    manager = _manager()
    page_bundle = manager._physical_page_bundle_bytes()
    KVCacheManager.set_physical_memory_limit.__wrapped__(manager, page_bundle * 4, revision=9)

    state = KVCacheManager.set_physical_memory_limit.__wrapped__(
        manager, page_bundle * 2, revision=9
    )

    assert state["status"] == "applied"
    assert manager.page_allocator.limit == 2


def test_available_size_uses_limit_without_querying_physical_memory():
    manager = _manager(mapped_pages=2)
    manager.page_allocator.limit = 5

    available = KVCacheManager.available_size.__wrapped__(manager)

    assert available == 3
    assert manager.page_allocator.physical_free_queries == 0


def test_zero_limit_is_active_without_querying_physical_memory():
    manager = _manager()
    manager.page_allocator.limit = 0

    available = KVCacheManager.available_size.__wrapped__(manager)

    assert available == 0
    assert manager.page_allocator.physical_free_queries == 0


def test_available_size_checks_physical_memory_without_provider_limit():
    manager = _manager()

    available = KVCacheManager.available_size.__wrapped__(manager)

    assert available == manager.page_allocator.total_pages
    assert manager.page_allocator.physical_free_queries == 1


def test_instance_limit_is_split_without_exceeding_total():
    from kvcached import control
    from kvcached.pool_registry import (
        clear_registered_kv_cache_pools,
        register_kv_cache_pool,
    )

    clear_registered_kv_cache_pools()

    class Pool:
        def __init__(self, group_id, name, capacity):
            self.group_id = group_id
            self.pool_name = name
            self.mem_size = capacity
            self.num_layers = 1
            self.num_kv_buffers = 1
            self.received: tuple[int, int] | None = None

        def set_physical_memory_limit(self, limit_bytes, *, revision):
            self.received = (limit_bytes, revision)
            return {
                "status": "applied",
                "mapped_bytes": 0,
                "effective_limit_bytes": limit_bytes,
            }

    pools = [Pool(0, "large", 3), Pool(1, "small", 1)]
    register_kv_cache_pool(pools[0], integration="vllm")
    register_kv_cache_pool(pools[1], integration="vllm")

    result = control.set_instance_physical_memory_limit(
        101,
        revision=4,
    )

    assert pools[0].received == (75, 4)
    assert pools[1].received == (26, 4)
    received = [pool.received for pool in pools]
    assert all(item is not None for item in received)
    assert sum(item[0] for item in received if item is not None) == 101
    assert result["status"] == "applied"
    clear_registered_kv_cache_pools()


def test_native_limit_accounts_for_inflight_foreground_allocations():
    source = (ROOT / "csrc" / "page_allocator.cpp").read_text(encoding="utf-8")
    helper = source.split(
        "int64_t PageAllocator::get_num_mapped_or_pending_pages_unlocked() const {",
        1,
    )[1].split("}", 1)[0]

    assert "pending_foreground_pages_" in helper
    assert "pending_prealloc_pages_" in helper
