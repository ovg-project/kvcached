# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Undo unpublished logical allocations after a recoverable physical miss."""

from collections import Counter
from functools import wraps
from typing import Any

_MISSING = object()
_REQUEST_MAPS = (
    "num_cached_block", "_partial_hit_reqs", "last_state_block_idx",
    "_producer_partial_tail_reqs", "_num_checkpoint_blocks",
)
_APPEND_LISTS = (
    "new_block_ids", "_pending_cow_copies", "_pending_partial_tail_offloads",
    "_pending_boundary_state_offloads",
)


class _GroupState:
    def __init__(self, manager: Any, request_id: str):
        self.manager = manager
        self.request_id = request_id
        self.present = request_id in manager.req_to_blocks
        blocks = manager.req_to_blocks.get(request_id, ())
        self.length = len(blocks)
        # Aligned Mamba can relocate existing speculative slots, not just append.
        self.blocks = list(blocks) if hasattr(manager, "last_state_block_idx") else None
        self.maps = {
            name: getattr(manager, name).get(request_id, _MISSING)
            for name in _REQUEST_MAPS if hasattr(manager, name)
        }
        self.lengths = {
            name: len(getattr(manager, name))
            for name in _APPEND_LISTS if hasattr(manager, name)
        }
        self.allocated = request_id in getattr(manager, "_allocated_block_reqs", ())
        partial = self.maps.get("_partial_hit_reqs", _MISSING)
        self.producer_source = partial[1] if self.allocated and partial is not _MISSING else None
        self.cached_this_step = (
            self.producer_source.block_hash in manager.cached_blocks_this_step
            if self.producer_source is not None else False
        )

    def restore(self) -> None:
        manager, request_id = self.manager, self.request_id
        current = manager.req_to_blocks.get(request_id, [])
        copies = getattr(manager, "_pending_cow_copies", [])[self.lengths.get(
            "_pending_cow_copies", 0):]
        desired = list(self.blocks) if self.blocks is not None else current[:self.length]
        partial = self.maps.get("_partial_hit_reqs", _MISSING)
        if partial is not _MISSING and self.blocks is None:
            index, source = partial
            desired[index] = source

        # Each pending copy owns one reference to each endpoint, independently
        # of the request table. Only this attempt's appended copies are removed.
        held = list(current) + [block for pair in copies for block in pair]
        owned = Counter(block.block_id for block in held if not block.is_null)
        wanted = Counter(block.block_id for block in desired if not block.is_null)
        if wanted - owned:
            raise RuntimeError("Cannot restore KV allocation: a retained block is missing")
        by_id = {block.block_id: block for block in held}
        released = [by_id[bid] for bid, count in (owned - wanted).items()
                    for _ in range(count)]

        for source, destination in copies:
            if source is self.producer_source:
                moved_hash = destination.block_hash
                manager.block_pool.move_block_hashes(destination, source)
                if not self.cached_this_step:
                    manager.cached_blocks_this_step.discard(moved_hash)

        if self.present:
            manager.req_to_blocks[request_id] = desired
        else:
            manager.req_to_blocks.pop(request_id, None)
        for name, value in self.maps.items():
            mapping = getattr(manager, name)
            if value is _MISSING:
                mapping.pop(request_id, None)
            else:
                mapping[request_id] = value
        if hasattr(manager, "_allocated_block_reqs"):
            if self.allocated:
                manager._allocated_block_reqs.add(request_id)
            else:
                manager._allocated_block_reqs.discard(request_id)
        for name, length in self.lengths.items():
            del getattr(manager, name)[length:]
        # Use the pool's reference-aware release, never a physical unmap here.
        manager.block_pool.free_blocks(reversed(released))


class AllocationAttempt:
    """Snapshot at the native allocation boundary, after skipped-block release.

    The scheduler owns the coordinator on one thread. Its native admission and
    skipped-block logic remains intact; only unpublished per-request mutations
    are rolled back. Running requests and earlier scheduled copies stay owned.
    """

    def __init__(self, coordinator: Any):
        self.coordinator = coordinator
        self.request_id: Any = None
        self.groups: list[_GroupState] = []
        self.needs_allocation = False
        original_demand = coordinator.get_num_blocks_to_allocate

        @wraps(original_demand)
        def demand(*args: Any, **kwargs: Any) -> Any:
            count = original_demand(*args, **kwargs)
            if self.request_id is not None:
                self.needs_allocation = count > 0
            return count

        coordinator.get_num_blocks_to_allocate = demand
        for name in ("allocate_new_computed_blocks", "allocate_new_blocks"):
            original = getattr(coordinator, name)
            setattr(coordinator, name, self._wrap(original))

    def _wrap(self, original: Any) -> Any:
        @wraps(original)
        def before_allocate(*args: Any, **kwargs: Any) -> Any:
            if self.request_id is not None and self.needs_allocation and not self.groups:
                self.groups = [_GroupState(manager, self.request_id)
                               for manager in self.coordinator.single_type_managers]
            return original(*args, **kwargs)
        return before_allocate

    def begin(self, request_id: str) -> None:
        if self.request_id is not None:
            raise RuntimeError("Nested KV allocation attempt")
        self.request_id = request_id
        self.needs_allocation = False

    def rollback(self) -> None:
        for group in reversed(self.groups):
            group.restore()

    def end(self) -> None:
        self.groups = []
        self.request_id = None
