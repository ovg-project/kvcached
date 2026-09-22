# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import types

import pytest
import torch

from kvcached.integration.sglang.patches import ElasticMambaPoolPatch


class MambaSlotAllocator:
    """Stand-in used to make the SGLang 0.5.13 split detectable."""


class FakeManager:
    def __init__(self, size):
        self.size = size
        self.free_ids = list(range(1, size + 1))
        self.alloc_calls = []
        self.free_calls = []
        self.clear_calls = 0

    def available_size(self):
        return len(self.free_ids)

    def alloc(self, count):
        self.alloc_calls.append(count)
        if count > len(self.free_ids):
            return None
        result, self.free_ids = self.free_ids[:count], self.free_ids[count:]
        return result

    def free(self, ids):
        self.free_calls.append(ids)
        self.free_ids.extend(ids)

    def clear(self):
        self.clear_calls += 1
        self.free_ids = list(range(1, self.size + 1))


class FakePool:
    def __init__(self, size):
        self.kvcached_allocator = FakeManager(size)
        self.alloc_calls = []
        self.free_calls = []
        self.clear_calls = 0

    def available_size(self):
        return self.kvcached_allocator.available_size()

    def alloc(self, count):
        self.alloc_calls.append(count)
        result = self.kvcached_allocator.alloc(count)
        if result is None:
            return None
        return torch.tensor(result, dtype=torch.int64)

    def free(self, slots):
        self.free_calls.append(slots.tolist())
        self.kvcached_allocator.free(slots.tolist())

    def clear(self):
        self.clear_calls += 1
        self.kvcached_allocator.clear()


@pytest.fixture
def allocator_cls(monkeypatch):
    class FakeMambaPool:
        State = object

    class FakeHybridReqToTokenPool:
        def _init_mamba_pool(self):
            # The production patch detects the allocator split through this
            # function's globals, matching SGLang 0.5.13 and newer.
            self.mamba_allocator = MambaSlotAllocator()

    memory_pool = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    setattr(memory_pool, "MambaPool", FakeMambaPool)
    setattr(memory_pool, "HybridReqToTokenPool", FakeHybridReqToTokenPool)

    patch = ElasticMambaPoolPatch()
    monkeypatch.setattr(
        patch.version_manager, "detect_version", lambda library: "0.5.13"
    )
    assert patch.apply(memory_pool)
    return memory_pool.ElasticMambaSlotAllocator


def assert_free_slot_ownership(allocator, pool):
    pool_free_ids = pool.kvcached_allocator.free_ids
    assert len(pool_free_ids) == len(set(pool_free_ids))
    assert allocator.free_slots.tolist() == sorted(pool_free_ids)
    assert allocator.available_size() == len(pool_free_ids)


def test_alloc_and_free_delegate_to_kvcached(allocator_cls):
    pool = FakePool(4)
    manager = pool.kvcached_allocator
    allocator = allocator_cls(4, "cpu", pool)

    slots = allocator.alloc(2)

    assert slots.tolist() == [1, 2]
    assert manager.alloc_calls == [2]
    assert pool.alloc_calls == [2]
    assert allocator.available_size() == 2

    allocator.free(slots)
    assert manager.free_calls == [[1, 2]]
    assert pool.free_calls == [[1, 2]]
    assert allocator.available_size() == 4


def test_group_allocation_returns_unused_slots(allocator_cls):
    pool = FakePool(4)
    manager = pool.kvcached_allocator
    allocator = allocator_cls(4, "cpu", pool)

    allocator.alloc_group_begin(3)
    assert allocator.alloc(1).tolist() == [1]
    allocator.alloc_group_end()

    assert manager.alloc_calls == [3]
    assert manager.free_calls == [[2, 3]]
    assert allocator.available_size() == 3


def test_exhaustion_and_reuse_preserve_slot_identity(allocator_cls):
    pool = FakePool(4)
    allocator = allocator_cls(4, "cpu", pool)

    first = allocator.alloc(4)
    assert first.tolist() == [1, 2, 3, 4]
    assert len(set(first.tolist())) == 4
    assert allocator.available_size() == 0
    assert allocator.alloc(1) is None

    allocator.free(first[1:3])
    reused = allocator.alloc(2)
    assert reused.tolist() == [2, 3]
    assert allocator.available_size() == 0


def test_free_slots_debug_view_matches_manager_ownership(allocator_cls):
    pool = FakePool(4)
    allocator = allocator_cls(4, "cpu", pool)

    slots = allocator.alloc(2)
    assert allocator.free_slots.tolist() == [3, 4]

    allocator.free(slots[:1])
    assert allocator.free_slots.tolist() == [1, 3, 4]


def test_failed_group_allocation_preserves_ownership_and_recovers(allocator_cls):
    pool = FakePool(2)
    allocator = allocator_cls(2, "cpu", pool)
    held = allocator.alloc(2)

    allocator.alloc_group_begin(1)

    assert allocator._alloc_iter is None
    assert pool.kvcached_allocator.alloc_calls == [2, 1]
    assert pool.kvcached_allocator.free_ids == []
    assert_free_slot_ownership(allocator, pool)

    allocator.free(held[:1])
    recovered = allocator.alloc(1)
    allocator.alloc_group_end()

    assert recovered.tolist() == [1]
    assert pool.kvcached_allocator.alloc_calls == [2, 1, 1]
    assert allocator._alloc_iter is None
    assert_free_slot_ownership(allocator, pool)

    allocator.free(torch.cat((recovered, held[1:])))
    assert_free_slot_ownership(allocator, pool)


def test_clear_discards_partial_group_before_fresh_allocation(allocator_cls):
    pool = FakePool(4)
    allocator = allocator_cls(4, "cpu", pool)

    allocator.alloc_group_begin(3)
    assert allocator.alloc(1).tolist() == [1]
    assert allocator.free_slots.tolist() == [4]

    allocator.clear()

    assert pool.clear_calls == 1
    assert allocator._alloc_iter is None
    assert_free_slot_ownership(allocator, pool)

    fresh = torch.cat([allocator.alloc(1) for _ in range(4)])
    assert fresh.tolist() == [1, 2, 3, 4]
    assert len(set(fresh.tolist())) == 4
    assert_free_slot_ownership(allocator, pool)

    allocator.free(fresh)
    assert_free_slot_ownership(allocator, pool)
