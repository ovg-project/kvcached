import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Optional, Tuple

import torch

from kvcached.locks import NoOpCondition, NoOpLock
from kvcached.tp_ipc_util import (broadcast_map_to_kv_tensors_to_workers,
                                  broadcast_unmap_from_kv_tensors_to_workers)
from kvcached.utils import (MAX_RESERVED_PAGES, MIN_RESERVED_PAGES,
                            PAGE_PREALLOC_ENABLED, SANITY_CHECK,
                            get_kvcached_logger)
from kvcached.vmm_ops import map_to_kv_tensors, unmap_from_kv_tensors

logger = get_kvcached_logger()

PREALLOC_THREAD_TIMEOUT: float = 2.0  # seconds


class Page:

    def __init__(self, page_id: int, page_size: int):
        self.page_id = page_id
        self.page_size = page_size

        self.start_block = None
        self.end_block = None
        self.num_kv_blocks = None
        self.free_list = None

    def init(self, block_mem_size: int) -> None:
        self.start_block, self.end_block = self.get_block_range(
            self.page_id, self.page_size, block_mem_size)

        self.num_kv_blocks = self.end_block - self.start_block
        self.free_list = list(range(self.start_block, self.end_block))

    def alloc(self) -> int:
        if self.full():
            raise ValueError(f"Page {self.page_id} is already full")
        block_id = self.free_list.pop()
        return block_id

    def alloc_all_remaining(self) -> List[int]:
        if self.full():
            raise ValueError(f"Page {self.page_id} is already full")
        block_ids = self.free_list
        self.free_list = []
        return block_ids

    def free(self, block_id: int) -> None:
        if SANITY_CHECK:
            self._sanity_check(block_id)
        self.free_list.append(block_id)

    def free_batch(self, block_ids: List[int]) -> None:
        if SANITY_CHECK:
            for block_id in block_ids:
                self._sanity_check(block_id)
        self.free_list.extend(block_ids)

    def empty(self) -> bool:
        return len(self.free_list) == self.num_kv_blocks

    def full(self) -> bool:
        return not self.free_list

    def num_free_blocks(self) -> int:
        return len(self.free_list)

    def get_free_blocks(self) -> List[int]:
        return self.free_list

    def _has_block(self, block_id: int) -> bool:
        return block_id >= self.start_block and block_id < self.end_block

    def _sanity_check(self, block_id: int) -> None:
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


class PageAllocatorBase(ABC):

    @abstractmethod
    def __init__(self, total_mem_size: int, page_size: int):
        pass

    @abstractmethod
    def alloc_page(self) -> Page:
        pass

    @abstractmethod
    def free_page(self, page_id: int) -> None:
        pass

    @abstractmethod
    def free_pages(self, page_ids: List[int]) -> None:
        pass

    @abstractmethod
    def get_num_free_pages(self) -> int:
        pass

    @abstractmethod
    def get_num_total_pages(self) -> int:
        pass


class PageAllocator(PageAllocatorBase):

    def __init__(self,
                 total_mem_size: int,
                 page_size: int,
                 tp_size: int = 1,
                 async_sched: bool = False,
                 enable_page_prealloc: bool = PAGE_PREALLOC_ENABLED):
        """
        Args:
            total_mem_size: Total memory size in bytes.
            page_size: Page size in bytes.
            async_sched: Whether asynchronous scheduling is enabled.
            enable_page_prealloc: Whether to enable page preallocation.
        """
        logger.info(f"Init KVCached PageAllocator: "
                    f"total_mem_size={total_mem_size//(1024*1024)}MB, "
                    f"page_size={page_size//(1024*1024)}MB, "
                    f"tp_size={tp_size}, "
                    f"async_sched={async_sched}, "
                    f"enable_prealloc={enable_page_prealloc}")
        # WARNING (YIFAN): kvcached_ops.init_kvcached must have been called
        # before this.

        self.total_mem_size = total_mem_size
        self.page_size = page_size
        self.tp_size = tp_size
        self.async_sched = async_sched
        self.num_free_pages = total_mem_size // page_size
        self.num_total_pages = total_mem_size // page_size

        self.free_page_list: deque[int] = deque(range(self.num_free_pages))

        self.min_reserved_pages: int = MIN_RESERVED_PAGES
        self.max_reserved_pages: int = MAX_RESERVED_PAGES
        self.reserved_page_list: deque[int] = deque()  # Fast path allocation

        self.reclaimed_page_list: deque[int] = deque()  # Reclaimed page ids

        # Preallocation thread management
        self.enable_page_prealloc: bool = enable_page_prealloc
        if self.enable_page_prealloc:
            self.prealloc_lock = threading.RLock()
            self.prealloc_cond = threading.Condition(self.prealloc_lock)
        else:  # No preallocation lock and condition are needed.
            self.prealloc_lock = NoOpLock()
            self.prealloc_cond = NoOpCondition(self.prealloc_lock)
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
            self.prealloc_lock = threading.RLock()
            self.prealloc_cond = threading.Condition(self.prealloc_lock)
            self._start_prealloc_thread()

    def alloc_page(self) -> Page:
        with self.prealloc_lock:
            if self.num_free_pages <= 0:
                raise ValueError("No free pages left")
            self.num_free_pages -= 1

            # Fast path: allocate pages with reserved physical memory mapping.
            if self.reserved_page_list:
                page_id = self.reserved_page_list.popleft()

                # Trigger preallocation to refill reserved pool if getting low
                if len(self.reserved_page_list) < self.min_reserved_pages:
                    self.prealloc_needed = True
                    self.prealloc_cond.notify()

                return Page(page_id, self.page_size)

            # Slow path: allocate pages with new physical memory mapping.
            page_id = self.free_page_list.popleft()

        try:
            self._map_pages([page_id])
        except Exception as e:
            # If mapping fails, return page to free list and restore count
            with self.prealloc_lock:
                self.free_page_list.appendleft(page_id)
                self.num_free_pages += 1
            raise RuntimeError(f"Failed to map page {page_id}: {e}") from e

        if self.enable_page_prealloc:
            # Trigger preallocation to refill the pool
            self._trigger_preallocation()

        return Page(page_id, self.page_size)

    def free_page(self, page_id: int) -> None:
        if SANITY_CHECK:
            with self.prealloc_lock:
                if (page_id in self.free_page_list
                        or page_id in self.reserved_page_list):
                    raise ValueError(f"Page {page_id} is already free")

        self.num_free_pages += 1

        with self.prealloc_lock:
            if len(self.reserved_page_list) < self.max_reserved_pages:
                # Fast path: reserve page with its physical memory mapping.
                self.reserved_page_list.append(page_id)
                return

        # Slow path: free page and its physical memory mapping.
        self._unmap_pages([page_id])
        with self.prealloc_lock:
            self.free_page_list.append(page_id)

    def free_pages(self, page_ids: List[int]) -> None:
        self.num_free_pages += len(page_ids)
        with self.prealloc_lock:
            num_to_reserve = self.max_reserved_pages - len(
                self.reserved_page_list)
            if num_to_reserve > 0:
                # Fast path: reserve pages with their physical memory mapping.
                self.reserved_page_list.extend(page_ids[:num_to_reserve])
                page_ids = page_ids[num_to_reserve:]

        if len(page_ids) == 0:
            return

        # Slow path: free page_ids and their physical memory mapping.
        self._unmap_pages(page_ids)
        with self.prealloc_lock:
            self.free_page_list.extend(page_ids)

    def resize(self, new_mem_size: int) -> bool:
        new_num_pages = new_mem_size // self.page_size

        with self.prealloc_lock:
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
                        self.prealloc_lock.release()
                        self._unmap_pages(pages_to_unmap)
                        self.prealloc_lock.acquire()
                        self.free_page_list.extend(pages_to_unmap)

                if len(self.free_page_list) < num_to_reclaim:
                    # Still not enough free pages
                    return False

                for _ in range(num_to_reclaim):
                    self.reclaimed_page_list.append(self.free_page_list.pop())
                self.num_free_pages -= num_to_reclaim
                self.num_total_pages = new_num_pages
        return True

    def trim(self) -> None:
        with self.prealloc_lock:
            pages_to_unmap = list(self.reserved_page_list)  # copy
            self.reserved_page_list.clear()

            if not pages_to_unmap:
                return

            self.prealloc_lock.release()
            self._unmap_pages(pages_to_unmap)
            self.prealloc_lock.acquire()

            self.free_page_list.extend(pages_to_unmap)

    def get_num_free_pages(self) -> int:
        return self.num_free_pages

    def get_num_inuse_pages(self) -> int:
        return self.num_total_pages - self.num_free_pages

    def get_num_total_pages(self) -> int:
        return self.num_total_pages

    def get_num_reserved_pages(self) -> int:
        with self.prealloc_lock:
            return len(self.reserved_page_list)

    def get_page_id(self, block_id: int, block_mem_size: int) -> int:
        return block_id * block_mem_size // self.page_size

    def get_num_free_blocks(self, block_mem_size: int) -> int:
        return self.get_num_free_pages() * Page.get_num_blocks(
            self.page_size, block_mem_size)

    def get_num_inuse_blocks(self, block_mem_size: int) -> int:
        return self.get_num_inuse_pages() * Page.get_num_blocks(
            self.page_size, block_mem_size)

    def get_num_total_blocks(self, block_mem_size: int) -> int:
        return self.get_num_total_pages() * Page.get_num_blocks(
            self.page_size, block_mem_size)

    # Private methods
    def _prealloc_worker(self):
        """Worker thread that preallocates and maps physical pages."""
        while self.prealloc_running:
            with self.prealloc_lock:
                # Wait until preallocation is needed or thread is stopped
                while not self.prealloc_needed and self.prealloc_running:
                    self.prealloc_cond.wait()

                if not self.prealloc_running:
                    break

                self.prealloc_needed = False
                current_reserved = len(self.reserved_page_list)
                to_reserve = max(0, self.min_reserved_pages - current_reserved)
                # Only try to reserve up to the available free pages
                to_reserve = min(to_reserve, len(self.free_page_list))
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
                    with self.prealloc_lock:
                        self.reserved_page_list.extend(pages_to_reserve)
                    logger.debug(
                        f"Preallocated {len(pages_to_reserve)} pages, reserved={len(self.reserved_page_list)}"
                    )
                except Exception as e:
                    # If mapping fails, return pages to free list
                    with self.prealloc_lock:
                        self.free_page_list.extendleft(pages_to_reserve)
                    logger.error(
                        f"Failed to preallocate {len(pages_to_reserve)} pages: {e}"
                    )

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
            with self.prealloc_lock:
                self.prealloc_running = False
                self.prealloc_cond.notify_all()
            self.prealloc_thd.join(timeout)
            if self.prealloc_thd.is_alive():
                logger.warning(
                    "Preallocation thread did not stop within timeout")
            self.prealloc_thd = None
            logger.debug("Stopped page preallocation thread")

    def _trigger_preallocation(self):
        """Trigger the preallocation thread to fill up reserved blocks"""
        with self.prealloc_lock:
            self.prealloc_needed = True
            self.prealloc_cond.notify()

    def _map_pages(self, page_ids: list[int]) -> None:
        offsets = [pid * self.page_size for pid in page_ids]
        if self.tp_size > 1:  # map pages across all tensor parallel workers.
            broadcast_map_to_kv_tensors_to_workers(self.tp_size, offsets)
        else:
            map_to_kv_tensors(offsets)

    def _unmap_pages(self, page_ids: list[int]) -> None:
        offsets = [pid * self.page_size for pid in page_ids]
        if self.tp_size > 1:  # unmap pages across all tensor parallel workers.
            broadcast_unmap_from_kv_tensors_to_workers(self.tp_size, offsets)
        else:
            if self.async_sched:
                torch.cuda.synchronize()
            unmap_from_kv_tensors(offsets)
