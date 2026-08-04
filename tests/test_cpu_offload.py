# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pytest


MODULE_PATH = Path(__file__).parents[1] / "kvcached" / "cpu_offload.py"
SPEC = importlib.util.spec_from_file_location("kvcached_cpu_offload", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
offload = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = offload
SPEC.loader.exec_module(offload)


class FakeBackend:
    def __init__(self, gpu_pages: Dict[int, Sequence[bytes]]):
        self.gpu_pages = {page_id: list(payloads) for page_id, payloads in gpu_pages.items()}
        self.calls: List[str] = []
        self.fail_release = False
        self.fail_write = False

    def read_gpu_page(self, page_id, geometry):
        self.calls.append(f"read:{page_id}")
        return self.gpu_pages[page_id]

    def release_gpu_page(self, page_id):
        self.calls.append(f"release:{page_id}")
        if self.fail_release:
            raise RuntimeError("simulated unmap failure")
        del self.gpu_pages[page_id]

    def allocate_gpu_page(self, page_id):
        self.calls.append(f"allocate:{page_id}")
        self.gpu_pages[page_id] = []

    def write_gpu_page(self, page_id, payloads, geometry):
        self.calls.append(f"write:{page_id}")
        if self.fail_write:
            raise RuntimeError("simulated copy failure")
        self.gpu_pages[page_id] = list(payloads)

    def commit_gpu_page(self, page_id):
        self.calls.append(f"commit:{page_id}")

    def rollback_gpu_page(self, page_id):
        self.calls.append(f"rollback:{page_id}")
        self.gpu_pages.pop(page_id, None)


def payloads(byte_value: int, count: int = 2, page_size: int = 4):
    return [bytes([byte_value]) * page_size for _ in range(count)]


class FakeDevice:
    type = "cpu"


class FakePinnedTensor:
    device = FakeDevice()

    def __init__(self, size_bytes: int, *, pinned: bool = True):
        self.size_bytes = size_bytes
        self.pinned = pinned

    def is_pinned(self):
        return self.pinned

    def is_contiguous(self):
        return True

    def numel(self):
        return self.size_bytes

    def element_size(self):
        return 1


def test_store_validates_logical_page_geometry():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    store = offload.CPUOffloadStore(geometry, max_bytes=16)

    with pytest.raises(ValueError, match="expected 2"):
        store.put(0, payloads(1, count=1))
    with pytest.raises(ValueError, match="exactly 4 bytes"):
        store.put(0, [b"1234", b"12"])


def test_store_rejects_capacity_overflow_without_dropping_existing_pages():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    store = offload.CPUOffloadStore(geometry, max_bytes=16)

    assert store.put(1, payloads(1)).stored
    assert store.put(2, payloads(2)).stored
    assert store.get(1) is not None

    result = store.put(3, payloads(3))

    assert not result.stored
    assert store.page_ids() == (2, 1)
    assert store.used_bytes == 16


def test_store_owns_an_immutable_copy_of_payloads():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    store = offload.CPUOffloadStore(geometry, max_bytes=8)
    first = bytearray(b"aaaa")
    second = bytearray(b"bbbb")

    assert store.put(1, [first, second]).stored
    first[:] = b"xxxx"
    second[:] = b"yyyy"

    assert store.get(1).payloads == (b"aaaa", b"bbbb")


def test_replacing_page_does_not_drift_capacity_accounting():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    store = offload.CPUOffloadStore(geometry, max_bytes=16)
    store.put(1, payloads(1))
    store.put(2, payloads(2))

    result = store.put(1, payloads(3))

    assert result.stored
    assert store.used_bytes == 16
    assert store.page_ids() == (2, 1)
    assert store.get(1).payloads == tuple(payloads(3))


def test_offload_commits_cpu_copy_before_releasing_gpu_page():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    store = offload.CPUOffloadStore(geometry, max_bytes=8)
    backend = FakeBackend({7: payloads(7)})
    manager = offload.CPUOffloadManager(store, backend)

    result = manager.offload(7)

    assert result.stored
    assert backend.calls == ["read:7", "release:7"]
    assert 7 not in backend.gpu_pages
    assert store.get(7).payloads == tuple(payloads(7))


def test_capacity_rejection_never_releases_gpu_page():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    store = offload.CPUOffloadStore(geometry, max_bytes=7)
    backend = FakeBackend({4: payloads(4)})
    manager = offload.CPUOffloadManager(store, backend)

    result = manager.offload(4)

    assert not result.stored
    assert backend.calls == ["read:4"]
    assert 4 in backend.gpu_pages
    assert manager.stats() == {
        "offloaded_pages": 0,
        "used_bytes": 0,
        "capacity_bytes": 7,
    }


def test_restore_keeps_cpu_copy_until_gpu_copy_finishes():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    store = offload.CPUOffloadStore(geometry, max_bytes=8)
    store.put(5, payloads(5))
    backend = FakeBackend({})
    manager = offload.CPUOffloadManager(store, backend)

    assert manager.restore(5)
    assert backend.calls == ["allocate:5", "write:5", "commit:5"]
    assert backend.gpu_pages[5] == payloads(5)
    assert 5 not in store


def test_offload_keeps_cpu_copy_when_gpu_release_fails():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    store = offload.CPUOffloadStore(geometry, max_bytes=8)
    backend = FakeBackend({8: payloads(8)})
    backend.fail_release = True
    manager = offload.CPUOffloadManager(store, backend)

    with pytest.raises(offload.OffloadError, match="retained for recovery") as exc_info:
        manager.offload(8)

    assert exc_info.value.page_id == 8
    assert exc_info.value.operation == "release"
    assert backend.calls == ["read:8", "release:8"]
    assert 8 in backend.gpu_pages
    assert 8 in store


def test_full_store_rejection_does_not_attempt_gpu_release():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    store = offload.CPUOffloadStore(geometry, max_bytes=8)
    store.put(1, payloads(1))
    backend = FakeBackend({2: payloads(2)})
    manager = offload.CPUOffloadManager(store, backend)

    result = manager.offload(2)

    assert not result.stored
    assert backend.calls == ["read:2"]
    assert store.page_ids() == (1,)
    assert 2 in backend.gpu_pages


def test_restore_rolls_back_gpu_allocation_after_copy_failure():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    store = offload.CPUOffloadStore(geometry, max_bytes=8)
    store.put(9, payloads(9))
    backend = FakeBackend({})
    backend.fail_write = True
    manager = offload.CPUOffloadManager(store, backend)

    with pytest.raises(offload.OffloadError, match="restore page 9") as exc_info:
        manager.restore(9)

    assert exc_info.value.page_id == 9
    assert exc_info.value.operation == "restore"
    assert backend.calls == ["allocate:9", "write:9", "rollback:9"]
    assert 9 in store
    assert 9 not in backend.gpu_pages


def test_planner_never_selects_a_page_with_active_blocks():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    planner = offload.PageOffloadPlanner(geometry)
    candidates = [
        offload.OffloadCandidate(1, 1, 1, 2, 0, 10.0),
        offload.OffloadCandidate(2, 0, 2, 2, 2, 10.0),
        offload.OffloadCandidate(3, 0, 2, 2, 1, 5.0),
    ]

    plan = planner.plan(candidates, pages_needed=2, cpu_available_bytes=16)

    assert plan.selected_page_ids == (3, 2)
    assert plan.skipped_active_pages == 1
    assert plan.bytes_to_offload == 16


def test_planner_reports_cpu_capacity_limit():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    planner = offload.PageOffloadPlanner(geometry)
    candidates = [
        offload.OffloadCandidate(page_id, 0, 1, 1, page_id, 1.0)
        for page_id in range(3)
    ]

    plan = planner.plan(candidates, pages_needed=3, cpu_available_bytes=8)

    assert plan.selected_page_ids == (0,)
    assert plan.limited_by_cpu_capacity


def test_planner_rejects_duplicate_page_ids():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    planner = offload.PageOffloadPlanner(geometry)
    candidates = [
        offload.OffloadCandidate(1, 0, 1, 1, 1, 1.0),
        offload.OffloadCandidate(1, 0, 1, 1, 2, 2.0),
    ]

    with pytest.raises(ValueError, match="duplicate page ids"):
        planner.plan(candidates, pages_needed=1, cpu_available_bytes=8)


def test_restore_break_even_uses_complete_logical_page_size():
    geometry = offload.PageGeometry(
        page_size=2_000_000,
        num_layers=10,
        num_kv_buffers=2,
    )
    planner = offload.PageOffloadPlanner(geometry)

    assert planner.estimated_transfer_ms(10.0) == pytest.approx(4.0)
    assert planner.restore_beats_recompute(estimated_recompute_ms=5.0, bandwidth_gbps=10.0)
    assert not planner.restore_beats_recompute(
        estimated_recompute_ms=3.0,
        bandwidth_gbps=10.0,
    )


def test_pinned_store_preserves_existing_pages_when_capacity_is_full():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    store = offload.PinnedMemoryOffloadStore(geometry, max_bytes=16)
    first = (FakePinnedTensor(4), FakePinnedTensor(4))
    second = (FakePinnedTensor(4), FakePinnedTensor(4))
    third = (FakePinnedTensor(4), FakePinnedTensor(4))

    assert store.put(1, first).stored
    assert store.put(2, second).stored
    assert store.get(1).payloads == first
    result = store.put(3, third)

    assert not result.stored
    assert store.page_ids() == (2, 1)
    assert store.used_bytes == 16


def test_pinned_store_rejects_pageable_or_wrong_sized_payloads():
    geometry = offload.PageGeometry(page_size=4, num_layers=1, num_kv_buffers=2)
    store = offload.PinnedMemoryOffloadStore(geometry, max_bytes=8)

    with pytest.raises(ValueError, match="pinned memory"):
        store.put(1, [FakePinnedTensor(4, pinned=False), FakePinnedTensor(4)])
    with pytest.raises(ValueError, match="exactly 4 bytes"):
        store.put(1, [FakePinnedTensor(3), FakePinnedTensor(4)])


def test_non_contiguous_page_layout_splits_k_and_v_per_layer():
    geometry = offload.PageGeometry(page_size=4, num_layers=2, num_kv_buffers=2)
    layout = offload.PageTensorLayout(
        geometry,
        raw_tensor_nbytes=[32, 32],
        contiguous_layout=False,
    )

    assert layout.spans(1) == (
        offload.TensorSpan(0, 4, 4),
        offload.TensorSpan(0, 20, 4),
        offload.TensorSpan(1, 4, 4),
        offload.TensorSpan(1, 20, 4),
    )


def test_contiguous_page_layout_splits_one_compound_page():
    geometry = offload.PageGeometry(page_size=4, num_layers=2, num_kv_buffers=2)
    layout = offload.PageTensorLayout(
        geometry,
        raw_tensor_nbytes=[64],
        contiguous_layout=True,
    )

    assert layout.spans(2) == (
        offload.TensorSpan(0, 32, 4),
        offload.TensorSpan(0, 36, 4),
        offload.TensorSpan(0, 40, 4),
        offload.TensorSpan(0, 44, 4),
    )
    with pytest.raises(IndexError, match="exceeds"):
        layout.spans(4)
