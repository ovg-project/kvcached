# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Incremental page ordering for the elastic prefix cache."""

from __future__ import annotations

import heapq
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any


class PageEvictionIndex:
    """Keep whole-page candidates ordered by cost, then most recent use.

    Only pages changed by block allocation, release or reuse need an occupancy
    query. The manager remains the authority on physical occupancy, including
    the null block and blocks retained for pending copies.
    """

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self.pages: dict[int, OrderedDict[int, int]] = {}
        self.dirty: set[int] = set()
        self.candidates: dict[int, tuple[int, int, int, int]] = {}
        self.heap: list[tuple[int, int, int, int]] = []
        self.clock = 0
        watch_releases = getattr(manager, "_register_page_release_callback", None)
        if watch_releases is not None:
            watch_releases(self._pages_released)

    def page_id(self, block_id: int) -> int:
        # Same byte-address calculation as PageAllocator::get_page_id, also
        # when the block size does not divide the physical page size.
        return block_id * self.manager.block_mem_size // self.manager.page_size

    def add(self, block_id: int) -> None:
        page = self.page_id(block_id)
        self.clock += 1
        self.pages.setdefault(page, OrderedDict())[block_id] = self.clock
        self.dirty.add(page)

    def remove(self, block_id: int) -> None:
        page = self.page_id(block_id)
        blocks = self.pages.get(page)
        if blocks is not None and block_id in blocks:
            del blocks[block_id]
            if not blocks:
                del self.pages[page]
            self.dirty.add(page)

    def changed(self, block_ids: Iterable[int]) -> None:
        self.dirty.update(self.page_id(bid) for bid in block_ids)

    def _pages_released(self, page_ids: Iterable[int]) -> None:
        self.dirty.update(page for page in page_ids if page in self.pages)

    def clear(self) -> None:
        self.pages.clear()
        self.dirty.clear()
        self.candidates.clear()
        self.heap.clear()

    def victims(self, budget: int) -> list[int]:
        dirty, self.dirty = self.dirty, set()
        occupancy = self.manager.get_page_occupancy(
            [page for page in dirty if page in self.pages]) if dirty else {}
        for page in dirty:
            self.candidates.pop(page, None)
            blocks = self.pages.get(page)
            if blocks and len(blocks) >= occupancy.get(page, 0):
                self.clock += 1
                entry = (len(blocks), next(reversed(blocks.values())), page, self.clock)
                self.candidates[page] = entry
                heapq.heappush(self.heap, entry)

        # Lazy deletion keeps updates cheap. Bound stale entries even when
        # touches repeatedly change a page that is never selected for eviction.
        if len(self.heap) > max(64, 2 * len(self.candidates)):
            self.heap = list(self.candidates.values())
            heapq.heapify(self.heap)

        victims: list[int] = []
        selected = []
        while self.heap:
            entry = self.heap[0]
            cost, _, page, _ = entry
            if self.candidates.get(page) != entry:
                heapq.heappop(self.heap)
                continue
            if cost > budget - len(victims):
                break
            heapq.heappop(self.heap)
            # Recheck selected pages before eviction: allocator reservations
            # can also change occupancy outside the block pool's methods.
            if self.manager.get_page_occupancy([page]).get(page, 0) > cost:
                self.candidates.pop(page)
                continue
            victims.extend(self.pages[page])
            selected.append(entry)
        for entry in selected:
            heapq.heappush(self.heap, entry)
        return victims
