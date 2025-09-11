# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import threading
from collections import deque
from typing import List, Optional, Tuple, cast

import torch

from kvcached.locks import ConditionLike, LockLike, NoOpCondition, NoOpLock
from kvcached.mem_info_tracker import MemInfoTracker
from kvcached.tp_ipc_util import broadcast_map_to_kv_tensors, broadcast_unmap_from_kv_tensors
from kvcached.utils import (
    CONTIGUOUS_LAYOUT,
    GPU_UTILIZATION,
    MAX_RESERVED_PAGES,
    MIN_RESERVED_PAGES,
    PAGE_PREALLOC_ENABLED,
    SANITY_CHECK,
    get_kvcached_logger,
)
from kvcached.vmm_ops import map_to_kv_tensors, unmap_from_kv_tensors

logger = get_kvcached_logger()

PREALLOC_THREAD_TIMEOUT: float = 2.0  # seconds


class Page:

    def __init__(self, page_id: int, page_size: int):
        self.page_id = page_id
        self.page_size = page_size

        self.start_block: Optional[int] = None
        self.end_block: Optional[int] = None
        self.num_kv_blocks: Optional[int] = None
        self.free_list: List[int] = []

    def _require_init(self) -> None:
        """Raise AssertionError if the page has not been initialised.
        """
        assert self.start_block is not None, "Page not initialised"
        assert self.end_block is not None, "Page not initialised"
        assert self.num_kv_blocks is not None, "Page not initialised"

    def init(self, block_mem_size: int) -> None:
        self.start_block, self.end_block = self.get_block_range(
            self.page_id, self.page_size, block_mem_size)

        self.num_kv_blocks = self.end_block - self.start_block
        self.free_list = list(range(self.start_block, self.end_block))

    def alloc(self, num_blocks: int = 1) -> List[int]:
        self._require_init()
        if self.full():
            raise ValueError(f"Page {self.page_id} is already full")
        block_ids = self.free_list[:num_blocks]
        self.free_list = self.free_list[num_blocks:]
        return block_ids

    def free(self, block_id: int) -> None:
        self._require_init()
        if SANITY_CHECK:
            self._sanity_check(block_id)
        self.free_list.append(block_id)

    def free_batch(self, block_ids: List[int]) -> None:
        self._require_init()
        if SANITY_CHECK:
            for block_id in block_ids:
                self._sanity_check(block_id)
        self.free_list.extend(block_ids)

    def empty(self) -> bool:
        self._require_init()
        return len(self.free_list) == self.num_kv_blocks

    def full(self) -> bool:
        self._require_init()
        return not self.free_list

    def num_free_blocks(self) -> int:
        self._require_init()
        return len(self.free_list)

    def get_free_blocks(self) -> List[int]:
        self._require_init()
        return self.free_list

    def _has_block(self, block_id: int) -> bool:
        self._require_init()
        return block_id >= cast(int, self.start_block) and block_id < cast(
            int, self.end_block)

    def _sanity_check(self, block_id: int) -> None:
        self._require_init()
        if not self._has_block(block_id):
            raise ValueError(
                f"Page {self.page_id} does not have block {block_id}")
        if block_id in self.free_list:
            raise ValueError(f"Block {block_id} is already free")

    @staticmethod
    def get_block_range(page_id: int, page_size: int,
                        block_mem_size: int) -> Tuple[int, int]:
        """
        Get the block range of a page.
        The page contains [start_block, end_block), which handles the case where
        page_size is not divisible by block_mem_size.
        For example, if page_size = 16 and block_mem_size = 6, the page 0
        contains [0, 2) blocks, and the page 1 contains [3, 5) blocks.
        Pages:  |      0-16       |        16-32        |
                | 0-6 | 6-12 | 12-18 | 18-24 | 24-30 | 30-32 |
        Blocks: |  0  |  1   |2<skip>|   3   |   4   |5<skip>|
        """
        start_block = (page_id * page_size + block_mem_size -
                       1) // block_mem_size
        end_block = ((page_id + 1) * page_size) // block_mem_size
        return start_block, end_block

    @staticmethod
    def get_num_blocks(page_size: int, block_mem_size: int) -> int:
        """
        Calculate the number of blocks that can fit in a page.
        This calculation is accurate even when page_size is not divisible by
        block_mem_size.
        """
        return page_size // block_mem_size


class PageAllocator:

    def __init__(self,
                 num_layers: int,
                 mem_size_per_layer: int,
                 page_size: int,
                 tp_size: int = 1,
                 async_sched: bool = False,
                 contiguous_layout: bool = CONTIGUOUS_LAYOUT,
                 enable_page_prealloc: bool = PAGE_PREALLOC_ENABLED):
        """
        Args:
            num_layers: Number of layers (for physical memory calculation).
            mem_size_per_layer: Memory size per layer per K/V tensor in bytes.
            page_size: Page size in bytes.
            tp_size: Tensor parallel size.
            async_sched: Whether asynchronous scheduling is enabled.
            contiguous_layout: Whether to use contiguous layout.
            enable_page_prealloc: Whether to enable page preallocation.
        """
        logger.info(
            f"Init kvcached KV cache allocator: "
            f"num_layers={num_layers}, "
            f"mem_size_per_layer={mem_size_per_layer//(1024*1024)}MB, "
            f"total_mem_size={2 * num_layers * mem_size_per_layer//(1024*1024)}MB, "
            f"page_size={page_size//(1024*1024)}MB, "
            f"tp_size={tp_size}, "
            f"async_sched={async_sched}, "
            f"contiguous_layout={contiguous_layout}, "
            f"enable_prealloc={enable_page_prealloc}")
        # WARNING (YIFAN): kvcached_ops.init_kvcached must have been called
        # before this.

        self.num_layers = num_layers
        self.mem_size_per_layer = mem_size_per_layer
        self.page_size = page_size
        self.tp_size = tp_size
        self.async_sched = async_sched
        self.contiguous_layout = contiguous_layout
        # TODO: make this compatible with engine's memory limit after getting
        # better configuration management.
        self.gpu_utilization = GPU_UTILIZATION
        self.num_free_pages = mem_size_per_layer // page_size
        self.num_total_pages = mem_size_per_layer // page_size

        self.free_page_list: deque[int] = deque(range(self.num_free_pages))

        self.min_reserved_pages: int = MIN_RESERVED_PAGES
        self.max_reserved_pages: int = MAX_RESERVED_PAGES
        self.reserved_page_list: deque[int] = deque()  # Fast path allocation

        self.reclaimed_page_list: deque[int] = deque()  # Reclaimed page ids

        # Initialize memory info tracker
        self.mem_info_tracker = MemInfoTracker(self.mem_size_per_layer *
                                               num_layers * 2)

        # Preallocation thread management
        self.enable_page_prealloc: bool = enable_page_prealloc

        self._lock: LockLike
        self._cond: ConditionLike

        if self.enable_page_prealloc:
            self._lock = threading.RLock()
            self._cond = threading.Condition(self._lock)
        else:  # No preallocation lock and condition are needed.
            self._lock = NoOpLock()
            self._cond = NoOpCondition(self._lock)
        self.prealloc_running: bool = False
        self.prealloc_needed: bool = False
        self.prealloc_thd: Optional[threading.Thread] = None

    def __del__(self):
        try:
            if self.enable_page_prealloc and self.prealloc_thd is not None:
                self._stop_prealloc_thread(timeout=PREALLOC_THREAD_TIMEOUT)
        except Exception:
            # Silently ignore exceptions during cleanup
            pass

    def start_prealloc_thread(self):
        # NOTE: called by KVCacheManager after reserving the null block
        if self.enable_page_prealloc:
            self._lock = threading.RLock()
            self._cond = threading.Condition(self._lock)
            self._start_prealloc_thread()

    def alloc_page(self) -> Page:
        with self._lock:
            if self.num_free_pages <= 0:
                raise ValueError("No free pages left")
            self.num_free_pages -= 1

            # Fast path: allocate pages with reserved physical memory mapping.
            if self.reserved_page_list:
                page_id = self.reserved_page_list.popleft()

                # Trigger preallocation to refill reserved pool if getting low
                if len(self.reserved_page_list) < self.min_reserved_pages:
                    self.prealloc_needed = True
                    self._cond.notify()

                # Update memory usage after fast path allocation
                self._update_memory_usage()
                return Page(page_id, self.page_size)

            # Slow path: allocate pages with new physical memory mapping.
            page_id = self.free_page_list.popleft()

        try:
            self._map_pages([page_id])
        except Exception as e:
            # If mapping fails, return page to free list and restore count
            with self._lock:
                self.free_page_list.appendleft(page_id)
                self.num_free_pages += 1
            raise RuntimeError(f"Failed to map page {page_id}: {e}") from e

        if self.enable_page_prealloc:
            # Trigger preallocation to refill the pool
            self._trigger_preallocation()

        # Update memory usage after mapping pages
        self._update_memory_usage()
        return Page(page_id, self.page_size)

    def free_page(self, page_id: int) -> None:
        with self._lock:
            if SANITY_CHECK and (page_id in self.free_page_list
                                 or page_id in self.reserved_page_list):
                raise ValueError(f"Page {page_id} is already free or reserved")

            self.num_free_pages += 1
            if len(self.reserved_page_list) < self.max_reserved_pages:
                # Fast path: reserve page with its physical memory mapping.
                self.reserved_page_list.append(page_id)
                # Update memory usage after fast path free/reserve
                self._update_memory_usage()
                return

        # Slow path: free page and its physical memory mapping.
        self._unmap_pages([page_id])
        with self._lock:
            self.free_page_list.append(page_id)
            # Update memory usage after unmapping pages
            self._update_memory_usage()

    def free_pages(self, page_ids: List[int]) -> None:
        with self._lock:
            if SANITY_CHECK:
                for page_id in page_ids:
                    if (page_id in self.free_page_list
                            or page_id in self.reserved_page_list):
                        raise ValueError(
                            f"Page {page_id} is already free or reserved")

            self.num_free_pages += len(page_ids)
            num_to_reserve = self.max_reserved_pages - len(
                self.reserved_page_list)
            if num_to_reserve > 0:
                # Fast path: reserve pages with their physical memory mapping.
                self.reserved_page_list.extend(page_ids[:num_to_reserve])
                page_ids = page_ids[num_to_reserve:]

        if len(page_ids) == 0:
            # Update memory usage after fast path free/reserve
            self._update_memory_usage()
            return

        # Slow path: free page_ids and their physical memory mapping.
        self._unmap_pages(page_ids)
        with self._lock:
            self.free_page_list.extend(page_ids)
            # Update memory usage after unmapping pages
            self._update_memory_usage()

    def resize(self, new_mem_size: int) -> bool:
        new_num_pages = new_mem_size // self.page_size
        with self._lock:
            if new_num_pages < self.get_num_inuse_pages():
                return False
            if new_num_pages == self.num_total_pages:
                return True
            elif new_num_pages > self.num_total_pages:
                num_to_expand = new_num_pages - self.num_total_pages

                # Reuse previously reclaimed pages first.
                num_to_reuse = min(len(self.reclaimed_page_list),
                                   num_to_expand)
                if num_to_reuse > 0:
                    for _ in range(num_to_reuse):
                        self.free_page_list.append(
                            self.reclaimed_page_list.popleft())
                    num_to_expand -= num_to_reuse
                    self.num_free_pages += num_to_reuse

                # Allocate new pages if needed.
                if num_to_expand > 0:
                    new_page_ids = list(
                        range(self.num_total_pages,
                              self.num_total_pages + num_to_expand))
                    self.free_page_list.extend(new_page_ids)
                    self.num_free_pages += num_to_expand
                self.num_total_pages = new_num_pages
                self._update_memory_usage()
            else:  # new_num_pages < self.num_total_pages and new_num_pages >= num_inuse_pages
                num_to_reclaim = self.num_total_pages - new_num_pages

                if len(self.free_page_list) < num_to_reclaim:
                    # Need to trim reserved pages first
                    reserved_count = len(self.reserved_page_list)
                    if reserved_count > 0:
                        # Move reserved pages back to free list
                        pages_to_unmap = list(self.reserved_page_list)
                        self.reserved_page_list.clear()
                        # Unmap outside the lock to avoid holding it during I/O
                        try:
                            self._lock.release()
                            self._unmap_pages(pages_to_unmap)
                        finally:
                            self._lock.acquire()
                        self.free_page_list.extend(pages_to_unmap)
                        # Update memory usage after unmapping pages
                        self._update_memory_usage()

                if len(self.free_page_list) < num_to_reclaim:
                    # Still not enough free pages
                    return False

                for _ in range(num_to_reclaim):
                    self.reclaimed_page_list.append(self.free_page_list.pop())
                self.num_free_pages -= num_to_reclaim
                self.num_total_pages = new_num_pages
        return True

    def trim(self) -> None:
        with self._lock:
            pages_to_unmap = list(self.reserved_page_list)  # copy
            self.reserved_page_list.clear()

            if not pages_to_unmap:
                # Update memory usage after trimming
                self._update_memory_usage()
                return

            try:
                self._lock.release()
                self._unmap_pages(pages_to_unmap)
            finally:
                self._lock.acquire()

            self.free_page_list.extend(pages_to_unmap)
            # Update memory usage after unmapping pages
            self._update_memory_usage()

    def get_num_free_pages(self) -> int:
        return self.num_free_pages

    def get_num_inuse_pages(self) -> int:
        return self.num_total_pages - self.num_free_pages

    def get_num_total_pages(self) -> int:
        return self.num_total_pages

    def get_num_reserved_pages(self) -> int:
        with self._lock:
            return len(self.reserved_page_list)

    def get_avail_physical_pages(self) -> int:
        avail_phy_mem_size, total_phy_mem_size = torch.cuda.mem_get_info()
        headroom = total_phy_mem_size * (1 - self.gpu_utilization)
        avail_phy_mem_size = max(avail_phy_mem_size - headroom, 0)

        # Calculate available pages considering layers and K/V split
        avail_phy_pages = avail_phy_mem_size // self.page_size
        # Each layer needs to reserve K and V tensors so we divide by 2.
        avail_pages_per_layer = avail_phy_pages // self.num_layers // 2
        return avail_pages_per_layer

    def get_page_id(self, block_id: int, block_mem_size: int) -> int:
        return block_id * block_mem_size // self.page_size

    # Private methods
    def _prealloc_worker(self):
        """Worker thread that preallocates and maps physical pages."""
        while self.prealloc_running:
            with self._lock:
                # Wait until preallocation is needed or thread is stopped
                while not self.prealloc_needed and self.prealloc_running:
                    self._cond.wait()

                if not self.prealloc_running:
                    break

                self.prealloc_needed = False
                current_reserved = len(self.reserved_page_list)
                to_reserve = max(0, self.min_reserved_pages - current_reserved)
                # Only try to reserve up to the available free pages
                to_reserve = min(to_reserve, len(self.free_page_list),
                                 self.get_avail_physical_pages())
                if to_reserve <= 0:
                    continue

                pages_to_reserve = []

                # Get pages from free list
                for _ in range(to_reserve):
                    if self.free_page_list:
                        pages_to_reserve.append(self.free_page_list.popleft())
                    else:
                        break

            if pages_to_reserve:
                try:
                    self._map_pages(pages_to_reserve)
                    with self._lock:
                        self.reserved_page_list.extend(pages_to_reserve)
                        # Update memory usage after mapping pages
                        self._update_memory_usage()
                    logger.debug(
                        f"Preallocated {len(pages_to_reserve)} pages, "
                        f"reserved={len(self.reserved_page_list)}")
                except Exception as e:
                    # If mapping fails, return pages to free list
                    with self._lock:
                        self.free_page_list.extendleft(pages_to_reserve)
                    logger.error(
                        f"Failed to preallocate {len(pages_to_reserve)} pages: "
                        f"{e}")

    def _start_prealloc_thread(self):
        if self.prealloc_thd is None:
            self.prealloc_running = True
            self.prealloc_thd = threading.Thread(target=self._prealloc_worker,
                                                 daemon=True)
            self.prealloc_thd.start()

            # Initial preallocation trigger
            self._trigger_preallocation()

    def _stop_prealloc_thread(self, timeout: Optional[float] = None):
        if self.prealloc_thd is not None:
            with self._lock:
                self.prealloc_running = False
                self._cond.notify_all()
            self.prealloc_thd.join(timeout)
            if self.prealloc_thd.is_alive():
                logger.warning(
                    "Preallocation thread did not stop within timeout")
            self.prealloc_thd = None
            logger.debug("Stopped page preallocation thread")

    def _trigger_preallocation(self):
        """Trigger the preallocation thread to fill up reserved blocks"""
        with self._lock:
            self.prealloc_needed = True
            self._cond.notify()

    def _map_pages(self, page_ids: list[int]) -> None:
        if self.contiguous_layout:
            offsets = [
                pid * self.page_size * self.num_layers * 2 for pid in page_ids
            ]
        else:
            offsets = [pid * self.page_size for pid in page_ids]
        if self.tp_size > 1:  # map pages across all tensor parallel workers.
            broadcast_map_to_kv_tensors(self.tp_size, offsets)
        else:
            map_to_kv_tensors(offsets)

    def _unmap_pages(self, page_ids: list[int]) -> None:
        if self.contiguous_layout:
            offsets = [
                pid * self.page_size * self.num_layers * 2 for pid in page_ids
            ]
        else:
            offsets = [pid * self.page_size for pid in page_ids]
        if self.tp_size > 1:  # unmap pages across all tensor parallel workers.
            broadcast_unmap_from_kv_tensors(self.tp_size, offsets)
        else:
            if self.async_sched:
                torch.cuda.synchronize()
            unmap_from_kv_tensors(offsets)

    def _update_memory_usage(self):
        """Update memory usage information in shared memory."""
        # Memory actively used by allocations (excludes preallocated pages).
        used_phy_mem_size = (self.get_num_inuse_pages() * self.num_layers *
                             self.page_size * 2)
        # Memory held by preallocated pages that are not yet actively used.
        prealloc_phy_mem_size = (self.get_num_reserved_pages() *
                                 self.num_layers * self.page_size * 2)

        self.mem_info_tracker.update_memory_usage(
            used_size=used_phy_mem_size, prealloc_size=prealloc_phy_mem_size)
