# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""CPU-only ownership checks; native engine cases run separately."""

import importlib.util
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "_allocation_attempt_under_test",
    Path(__file__).parents[1] / "kvcached/integration/vllm/allocation_attempt.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
AllocationAttempt = _module.AllocationAttempt


def block(number, refs=1):
    return SimpleNamespace(block_id=number, ref_cnt=refs, is_null=number == 0)


def group():
    released = []

    def free(blocks):
        for item in blocks:
            assert item.ref_cnt > 0
            item.ref_cnt -= 1
            released.append(item.block_id)

    return SimpleNamespace(req_to_blocks=defaultdict(list), num_cached_block={},
                           _partial_hit_reqs={}, new_block_ids=[], _pending_cow_copies=[],
                           block_pool=SimpleNamespace(free_blocks=free), released=released)


def attempt(manager, allocate):
    coordinator = SimpleNamespace(single_type_managers=[manager],
                                  get_num_blocks_to_allocate=lambda: 1,
                                  allocate_new_computed_blocks=lambda *_: None,
                                  allocate_new_blocks=allocate)
    journal = AllocationAttempt(coordinator)
    return journal, coordinator


def test_capture_follows_skipped_release_and_retains_other_request_copies():
    manager = group()
    old, tail, new = block(1), block(2), block(3)
    manager.req_to_blocks["req"] = [old, tail]
    manager._pending_cow_copies = [(block(8), block(9))]

    def allocate(_):
        manager.req_to_blocks["req"].append(new)
        manager.new_block_ids.append(new.block_id)

    journal, coordinator = attempt(manager, allocate)
    journal.begin("req")
    coordinator.get_num_blocks_to_allocate()
    manager.block_pool.free_blocks([old])
    manager.req_to_blocks["req"][0] = block(0, 0)
    coordinator.allocate_new_blocks("req")
    journal.rollback()
    journal.end()
    assert [b.block_id for b in manager.req_to_blocks["req"]] == [0, 2]
    assert manager.released == [1, 3]
    assert tail.ref_cnt == 1 and old.ref_cnt == new.ref_cnt == 0
    assert len(manager._pending_cow_copies) == 1
    assert manager.new_block_ids == []


def test_partial_copy_restores_source_and_releases_both_destination_references():
    manager = group()
    source, destination = block(1), block(2, 2)
    manager.req_to_blocks["req"] = [source]
    manager._partial_hit_reqs["req"] = (0, source)

    def allocate(_):
        manager._partial_hit_reqs.pop("req")
        manager.req_to_blocks["req"][0] = destination
        manager._pending_cow_copies.append((source, destination))

    journal, coordinator = attempt(manager, allocate)
    journal.begin("req")
    coordinator.get_num_blocks_to_allocate()
    coordinator.allocate_new_blocks("req")
    journal.rollback()
    journal.end()
    assert manager.req_to_blocks["req"] == [source]
    assert manager._partial_hit_reqs["req"] == (0, source)
    assert source.ref_cnt == 1 and destination.ref_cnt == 0
    assert manager.released == [2, 2]
    assert manager._pending_cow_copies == []


def test_existing_attention_table_is_not_copied_on_success_path():
    class NoIteration(list):
        def __iter__(self):
            raise AssertionError("Successful attention allocation must not copy the table")

    manager = group()
    manager.req_to_blocks["req"] = NoIteration([block(1)] * 10000)
    journal, coordinator = attempt(manager, lambda _: 42)
    journal.begin("req")
    coordinator.get_num_blocks_to_allocate()
    assert coordinator.allocate_new_blocks("req") == 42
    assert journal.groups[0].blocks is None
    journal.end()
    assert journal.groups == [] and journal.request_id is None
    assert manager.released == []


def test_admission_miss_before_allocation_does_not_capture_or_release():
    manager = group()
    journal, _ = attempt(manager, lambda _: None)
    journal.begin("req")
    journal.rollback()
    journal.end()
    assert manager.req_to_blocks == {} and manager.released == []


def test_nested_allocation_is_not_silently_accepted():
    journal, _ = attempt(group(), lambda _: None)
    journal.begin("first")
    with pytest.raises(RuntimeError, match="Nested"):
        journal.begin("second")
    assert journal.request_id == "first"
    journal.end()


def test_allocation_outside_scheduler_attempt_keeps_native_behavior():
    journal, coordinator = attempt(group(), lambda _: 42)
    assert coordinator.allocate_new_blocks("req") == 42
    assert journal.groups == []


def test_zero_native_demand_does_not_snapshot_any_group():
    coordinator = SimpleNamespace(single_type_managers=[object()],
                                  get_num_blocks_to_allocate=lambda: 0,
                                  allocate_new_computed_blocks=lambda *_: None,
                                  allocate_new_blocks=lambda *_: [])
    journal = AllocationAttempt(coordinator)
    journal.begin("req")
    assert coordinator.get_num_blocks_to_allocate() == 0
    assert coordinator.allocate_new_blocks("req") == []
    assert journal.groups == []
    journal.end()
