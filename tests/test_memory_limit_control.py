# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import threading
import types

if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
    sys.modules.setdefault("torch", types.ModuleType("torch"))


def _load_manager_class():
    fake_module = types.ModuleType("kvcached.vmm_ops")
    setattr(fake_module, "PageAllocator", object)
    setattr(
        fake_module,
        "InternalPage",
        types.SimpleNamespace(
            get_num_blocks=lambda page_size, block_mem_size: page_size // block_mem_size
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


class FakeAllocator:
    def __init__(
        self,
        *,
        page_size: int,
        total_pages: int = 16,
        inuse_pages: int = 0,
        reserved_pages: int = 0,
    ):
        self.page_size = page_size
        self.total_pages = total_pages
        self.inuse_pages = inuse_pages
        self.reserved_pages = reserved_pages
        self.resize_calls: list[int] = []

    def resize(self, new_mem_size):
        target_pages = new_mem_size // self.page_size
        self.resize_calls.append(target_pages)
        if target_pages < self.inuse_pages:
            return False
        self.total_pages = target_pages
        self.reserved_pages = min(
            self.reserved_pages,
            max(0, target_pages - self.inuse_pages),
        )
        return True

    def get_page_state(self):
        free_pages = max(0, self.total_pages - self.inuse_pages)
        return {
            "total_pages": self.total_pages,
            "free_pages": free_pages,
            "inuse_pages": self.inuse_pages,
            "reserved_pages": self.reserved_pages,
        }


def _manager(*, inuse_pages: int = 0, reserved_pages: int = 0):
    manager = object.__new__(KVCacheManager)
    manager.page_size = 2 * 1024 * 1024
    manager.num_layers = 2
    manager.num_kv_buffers = 2
    manager.mem_size = manager.page_size * 16
    manager.num_blocks = 16
    manager.group_id = 0
    manager._pool_name = "unified"
    manager._memory_limit_bytes = None
    manager._memory_limit_effective_bytes = None
    manager._memory_limit_revision = -1
    manager.page_allocator = FakeAllocator(
        page_size=manager.page_size,
        total_pages=16,
        inuse_pages=inuse_pages,
        reserved_pages=reserved_pages,
    )
    manager._lock = threading.RLock()
    manager._wait_post_init = lambda: None
    manager.num_avail_blocks = 0
    manager.reserved_blocks = []
    manager.in_shrink = False
    manager.target_num_blocks = None
    manager.block_mem_size = manager.page_size
    return manager


def test_idle_limit_applies_immediately_through_resize():
    manager = _manager(inuse_pages=1, reserved_pages=2)
    page_bundle = manager._memory_limit_page_bundle_bytes()

    state = manager.set_memory_limit(page_bundle * 3 + 123, revision=7)

    assert manager.page_allocator.resize_calls == [3]
    assert manager.page_allocator.total_pages == 3
    assert manager.mem_size == manager.page_size * 16
    assert state["status"] == "applied"
    assert state["limit_bytes"] == page_bundle * 3 + 123
    assert state["effective_limit_bytes"] == page_bundle * 3
    assert state["current_capacity_bytes"] == page_bundle * 3
    assert state["revision"] == 7


def test_active_limit_defers_without_revoking_and_converges_on_release():
    manager = _manager(inuse_pages=4)
    page_bundle = manager._memory_limit_page_bundle_bytes()

    deferred = manager.set_memory_limit(page_bundle * 2, revision=1)

    assert deferred["status"] == "deferred"
    assert deferred["mapped_bytes"] == page_bundle * 4
    assert deferred["overage_bytes"] == page_bundle * 2
    assert deferred["reason"] == "inuse_capacity_above_limit"
    assert manager.in_shrink
    assert manager.page_allocator.total_pages == 16

    manager.page_allocator.inuse_pages = 2
    assert manager.resize(manager.page_size * 2)
    applied = manager.set_memory_limit(page_bundle * 2, revision=1)

    assert applied["status"] == "applied"
    assert applied["current_capacity_bytes"] == page_bundle * 2
    assert not manager.in_shrink


def test_limit_can_regrow_within_the_original_reservation():
    manager = _manager()
    page_bundle = manager._memory_limit_page_bundle_bytes()

    manager.set_memory_limit(page_bundle * 3, revision=1)
    state = manager.set_memory_limit(page_bundle * 6, revision=2)

    assert manager.page_allocator.resize_calls == [3, 6]
    assert manager.page_allocator.total_pages == 6
    assert manager.mem_size == manager.page_size * 16
    assert state["status"] == "applied"
    assert state["effective_limit_bytes"] == page_bundle * 6


def test_limit_is_clamped_to_the_original_reservation():
    manager = _manager()
    page_bundle = manager._memory_limit_page_bundle_bytes()

    state = manager.set_memory_limit(page_bundle * 32, revision=1)

    assert manager.page_allocator.resize_calls == [16]
    assert state["effective_limit_bytes"] == page_bundle * 16


def test_zero_limit_drains_an_idle_pool():
    manager = _manager()

    state = manager.set_memory_limit(0, revision=1)

    assert manager.page_allocator.resize_calls == [0]
    assert manager.page_allocator.total_pages == 0
    assert state["status"] == "applied"
    assert state["effective_limit_bytes"] == 0


def test_stale_update_is_ignored_and_revision_reuse_must_match():
    manager = _manager()
    page_bundle = manager._memory_limit_page_bundle_bytes()
    first = manager.set_memory_limit(page_bundle * 4, revision=9)
    repeated = manager.set_memory_limit(page_bundle * 4, revision=9)
    conflict = manager.set_memory_limit(page_bundle * 3, revision=9)
    stale = manager.set_memory_limit(page_bundle * 2, revision=8)

    assert repeated == first
    assert conflict["status"] == "conflict"
    assert conflict["reason"] == "revision_reused_with_different_limit"
    assert stale["status"] == "stale"
    assert stale["revision"] == 9
    assert manager.page_allocator.resize_calls == [4]


def test_instance_limit_is_split_by_pool_capacity():
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
            self.received = None

        def set_memory_limit(self, limit_bytes, *, revision):
            self.received = (limit_bytes, revision)
            return {
                "status": "applied",
                "mapped_bytes": 0,
                "effective_limit_bytes": limit_bytes,
            }

    pools = [Pool(0, "large", 3), Pool(1, "small", 1)]
    register_kv_cache_pool(pools[0], integration="vllm")
    register_kv_cache_pool(pools[1], integration="vllm")
    try:
        result = control.set_instance_memory_limit(101, revision=4)

        assert pools[0].received == (3, 4)
        assert pools[1].received == (1, 4)
        assert result["status"] == "applied"
        assert result["effective_limit_bytes"] == 4
    finally:
        clear_registered_kv_cache_pools()


def test_instance_limit_reports_unavailable_without_live_pools():
    from kvcached import control
    from kvcached.pool_registry import clear_registered_kv_cache_pools

    clear_registered_kv_cache_pools()
    result = control.set_instance_memory_limit(1024, revision=1)

    assert result["status"] == "unavailable"
    assert result["reason"] == "no_registered_kv_cache_pool"
    assert result["pools"] == []
