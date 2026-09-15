# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import pytest

from kvcached.integration.sglang.patches import ElasticMambaPoolPatch

torch = pytest.importorskip("torch")
memory_pool = pytest.importorskip("sglang.srt.mem_cache.memory_pool")


class FakeManager:
    def __init__(self, size):
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
        size = len(self.free_ids) + sum(self.alloc_calls) - sum(map(len, self.free_calls))
        self.free_ids = list(range(1, size + 1))


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
def allocator_cls():
    patch = ElasticMambaPoolPatch()
    assert patch.apply(memory_pool)
    return memory_pool.ElasticMambaSlotAllocator


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
