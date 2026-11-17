# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""One byte allowance for coupled, single-process token pools."""

from __future__ import annotations

import threading
from typing import Any


class PriorityReclaimGroup:
    def __init__(self, members: tuple[Any, ...]):
        identities = []
        slots = []
        for member in members:
            control = member._runtime_control
            if control.worker_ipc or len(control.sources) != 1:
                raise ValueError("coupled reclaim requires local single-rank pools")
            source = control.sources[0]
            identities.append(tuple(source[key] for key in ("gpu", "instance", "generation")))
            slots.append((source["pool_slot"], source["pool_owner"]))
        if len(members) != 2 or len(set(identities)) != 1 or len(set(slots)) != len(members):
            raise ValueError("coupled reclaim requires distinct pools of one instance")
        self.members = members
        self.release_members = tuple(sorted(members, key=lambda member: (
            member._runtime_control.sources[0]["pool_slot"])))
        # Every synchronized member operation takes this before its own lock.
        # Native SWA eviction can call both members recursively, in either order.
        self.lock = threading.RLock()
        self.active = False
        self.failed = False
        self.limit = 0
        self.asks = [0, 0, 0]
        self.posted: tuple[tuple[int, ...], ...] | None = None
        for member in members:
            member._priority_reclaim_group = self

    def publish(self, *, force: bool = False) -> None:
        from kvcached.vmm_ops import _publish_pool_reclaimable

        with self.lock:
            levels = []
            for member in self.members:
                source = member._runtime_control.sources[0]
                bundle = member._memory_limit_page_bundle_bytes()
                levels.append((source["pool_slot"], source["pool_owner"],
                               *(count * bundle for count in member._reclaimable_levels())))
            # Pool ownership matters even when the instance totals do not change.
            snapshot = tuple(levels)
            if force or snapshot != self.posted:
                source = self.members[0]._runtime_control.sources[0]
                if not _publish_pool_reclaimable(source["gpu"], source["instance"],
                                                 source["generation"], snapshot):
                    raise RuntimeError("coupled reclaim participants changed during publication")
                self.posted = snapshot

    def _consume(self, size: int) -> None:
        self.limit -= size
        for i, ask in enumerate(self.asks):
            taken = min(ask, size)
            self.asks[i] -= taken
            size -= taken

    def claim(self, member: Any, page_id: int) -> bool:
        claimed: set[int] = getattr(member, "_priority_retired_ids", set())
        if page_id in claimed:
            return True
        size = member._memory_limit_page_bundle_bytes()
        if self.failed or not self.active or not any(self.asks) or size > self.limit:
            return False
        self._consume(size)
        member._priority_retired_ids = claimed | {page_id}
        retained: set[int] = getattr(member, "_priority_retained_ids", set())
        retained.discard(page_id)
        return True

    def pending(self) -> bool:
        return (self.active and not self.failed and self.asks[2] > 0
                and any(m._memory_limit_page_bundle_bytes() <= self.limit for m in self.members))

    def finish(self, *, force_publish: bool = False) -> None:
        with self.lock:
            for member in self.members:
                member._refund_priority_reserve()
            self.publish(force=force_publish)

    def cancel(self) -> None:
        # Existing page promises remain owned by their managers until refund.
        self.active = False
        self.asks = [0, 0, 0]
        self.limit = 0

    def reclaim(self) -> int:
        from kvcached.vmm_ops import kv_reclaim

        with self.lock:
            if self.failed:
                raise RuntimeError("coupled reclaim has an unconfirmed native release")
            if any(getattr(m, "in_shrink", False) for m in self.members):
                self.cancel()
                return 0
            for member in self.members:
                member._runtime_control.poll(force=True)
            self.finish(force_publish=True)
            raw = kv_reclaim(self.members[0].group_id)
            self.limit = int(raw["limit_bytes"])
            self.asks = [int(raw[f"{tier}_bytes"]) for tier in ("empty", "cache", "running")]
            self.active = any(self.asks)
            if not self.active:
                return 0
            # Previously retired pages may already satisfy this ask. Protect
            # surplus too: their eventual native free can overflow the reserve.
            for member in self.release_members:
                retained: set[int] = getattr(member, "_priority_retained_ids", set())
                for page_id in member._retiring_page_ids():
                    if not self.claim(member, page_id):
                        retained.add(page_id)
                member._priority_retained_ids = retained
            for member in self.release_members:
                bundle = member._memory_limit_page_bundle_bytes()
                count = min(-(-self.asks[0] // bundle), self.limit // bundle,
                            member.page_allocator.get_num_reserved_pages() + len(member._priority_empty_ids()))
                if not count:
                    continue
                before_limit, before_asks = self.limit, self.asks[:]
                debt = getattr(member, "_priority_reserve_debt", 0)
                self._consume(count * bundle)
                try:
                    released = member.release_empty_pages(count)
                except Exception:
                    # Native free/trim may have partly completed. Do not reuse
                    # its allowance or repeat an ambiguous free.
                    self.failed = True
                    raise
                self.limit, self.asks = before_limit, before_asks
                promised = max(0, getattr(member, "_priority_reserve_debt", 0) - debt)
                self._consume((released + promised) * bundle)
            for member in self.release_members:
                bundle = member._memory_limit_page_bundle_bytes()
                for _, evict in member._live_cache_owners():
                    pages = min(-(-self.asks[1] // bundle), self.limit // bundle)
                    if not pages:
                        break
                    # A native victim can free both pools. Only concrete page
                    # claims consume the ask; evicted token counts do not.
                    evict(pages)
            self.finish()
            bundle = self.members[0]._memory_limit_page_bundle_bytes()
            return -(-self.asks[2] // bundle) if self.pending() else 0
