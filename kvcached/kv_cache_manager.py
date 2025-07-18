import atexit
import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from functools import wraps
from typing import Dict, List, Optional

import posix_ipc
import torch

from controller.utils import (MemInfoStruct, RwLockedShm, get_ipc_name,
                              get_ipc_path, init_kv_cache_limit)
from kvcached.tp_ipc_util import (broadcast_kv_tensors_created_to_workers,
                                  broadcast_map_to_kv_tensors_to_workers,
                                  broadcast_unmap_from_kv_tensors_to_workers)
from kvcached.utils import DEFAULT_IPC_NAME, PAGE_SIZE, get_kvcached_logger
from kvcached.vmm_ops import (kv_tensors_created, map_to_kv_tensors,
                              unmap_from_kv_tensors)

logger = get_kvcached_logger()

SANITY_CHECK = False
GPU_UTILIZATION = 0.95
PAGE_PREALLOC_ENABLED = True


class NoOpLock:
    """A no-op lock that implements the same interface as threading.RLock"""

    def acquire(self, blocking=True, timeout=-1):
        return True

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def locked(self):
        return False


class NoOpCondition:
    """A no-op condition that implements the same interface as threading.Condition"""

    def __init__(self, lock: threading.RLock):
        self.lock = lock

    def wait(self, timeout=None):
        return True

    def wait_for(self, predicate, timeout=None):
        return predicate()

    def notify(self, n=1):
        pass

    def notify_all(self):
        pass

    def acquire(self, *args):
        return self.lock.acquire(*args)

    def release(self):
        return self.lock.release()

    def __enter__(self):
        return self.lock.__enter__()

    def __exit__(self, *args):
        return self.lock.__exit__(*args)


def synchronized(method):

    @wraps(method)
    def synchronized_method(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return synchronized_method


class Page:

    def __init__(self, page_id: int, page_size: int):
        self.page_id = page_id
        self.page_size = page_size

        self.num_kv_blocks = None
        self.free_list = None

    def init(self, block_mem_size: int) -> None:
        assert not self.initialized()

        assert self.page_size % block_mem_size == 0
        self.num_kv_blocks = self.page_size // block_mem_size

        stt_idx = self.page_id * self.num_kv_blocks
        self.free_list = [stt_idx + i for i in range(self.num_kv_blocks)]

    def destroy(self) -> None:
        assert self.initialized() and len(self.free_list) == self.num_kv_blocks
        self.num_kv_blocks = None
        self.free_list = None

    def initialized(self) -> bool:
        return self.num_kv_blocks is not None and self.free_list is not None

    def alloc(self) -> int:
        if self.full():
            raise ValueError(f"Page {self.page_id} is already full")
        block_id = self.free_list.pop()
        return block_id

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

    def get_used_blocks(self) -> List[int]:
        all_blk_ids = [
            block_id for block_id in range(
                self.page_id * self.num_kv_blocks,
                (self.page_id + 1) * self.num_kv_blocks,
            )
        ]
        return list(set(all_blk_ids) - set(self.free_list))

    def _has_block(self, block_id: int) -> bool:
        stt_idx = self.page_id * self.num_kv_blocks
        end_idx = stt_idx + self.num_kv_blocks
        return block_id >= stt_idx and block_id < end_idx

    def _sanity_check(self, block_id: int) -> None:
        if not self._has_block(block_id):
            raise ValueError(
                f"Page {self.page_id} does not have block {block_id}")
        if block_id in self.free_list:
            raise ValueError(f"Block {block_id} is already free")


class PageAllocatorBase(ABC):

    @abstractmethod
    def __init__(self, total_mem_size: int, page_size: int):
        pass

    @abstractmethod
    def alloc_page(self) -> int:
        pass

    @abstractmethod
    def free_page(self, page: int) -> None:
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

        self.min_reserved_pages: int = 5
        self.max_reserved_pages: int = 10
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
        if self.enable_page_prealloc:  # Stop preallocation thread
            self._stop_prealloc_thread()

    def start_prealloc_thread(self):
        # NOTE: called by KVCacheManager after reserving the null block
        if self.enable_page_prealloc:
            self.prealloc_lock = threading.RLock()
            self.prealloc_cond = threading.Condition(self.prealloc_lock)
            self._start_prealloc_thread()

    def alloc_page(self) -> Page:
        if self.num_free_pages <= 0:
            raise ValueError("No free pages left")
        self.num_free_pages -= 1

        # Fast path: allocate pages with reserved physical memory mapping.
        with self.prealloc_lock:
            if self.reserved_page_list:
                page_id = self.reserved_page_list.popleft()

                # Trigger preallocation to refill reserved pool if getting low
                if len(self.reserved_page_list) < self.min_reserved_pages:
                    self.prealloc_needed = True
                    self.prealloc_cond.notify()

                return Page(page_id, self.page_size)

        # Slow path: allocate pages with new physical memory mapping.
        with self.prealloc_lock:
            page_id = self.free_page_list.popleft()
        page = Page(page_id, self.page_size)
        self._map_pages([page_id])

        if self.enable_page_prealloc:
            # Trigger preallocation to refill the pool
            self._trigger_preallocation()

        return page

    def free_page(self, page: Page) -> None:
        page_id = page.page_id
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

        if new_num_pages < self.get_num_inuse_pages():
            return False
        if new_num_pages == self.num_total_pages:
            return True
        elif new_num_pages > self.num_total_pages:
            num_to_expand = new_num_pages - self.num_total_pages

            # Reuse previously reclaimed pages first.
            num_to_reuse = min(len(self.reclaimed_page_list), num_to_expand)
            with self.prealloc_lock:
                self.free_page_list.extend(
                    self.reclaimed_page_list[:num_to_reuse])
            self.reclaimed_page_list = self.reclaimed_page_list[num_to_reuse:]
            num_to_expand -= num_to_reuse
            self.num_free_pages += num_to_reuse

            # Allocate new pages if needed.
            if num_to_expand > 0:
                new_page_ids = range(self.num_total_pages,
                                     self.num_total_pages + num_to_expand)
                with self.prealloc_lock:
                    self.free_page_list.extend(new_page_ids)
                self.num_free_pages += num_to_expand
            self.num_total_pages = new_num_pages
        else:  # new_num_pages < self.num_total_pages and new_num_pages >= num_inuse_pages
            num_to_reclaim = self.num_total_pages - new_num_pages
            with self.prealloc_lock:
                if len(self.free_page_list) < num_to_reclaim:
                    self.trim()
                assert len(self.free_page_list) >= num_to_reclaim
                for _ in range(num_to_reclaim):
                    self.reclaimed_page_list.append(self.free_page_list.pop())
            self.num_free_pages -= num_to_reclaim
            self.num_total_pages = new_num_pages
        return True

    def trim(self) -> None:
        with self.prealloc_lock:
            pages_to_unmap = self.reserved_page_list[:]  # copy
            self.reserved_page_list.clear()

        if not pages_to_unmap:
            return

        self._unmap_pages(pages_to_unmap)

        with self.prealloc_lock:
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
        return self.get_num_free_pages() * self._num_blocks_per_page(
            block_mem_size)

    def get_num_inuse_blocks(self, block_mem_size: int) -> int:
        return self.get_num_inuse_pages() * self._num_blocks_per_page(
            block_mem_size)

    def get_num_total_blocks(self, block_mem_size: int) -> int:
        return self.get_num_total_pages() * self._num_blocks_per_page(
            block_mem_size)

    # Private methods
    def _num_blocks_per_page(self, block_mem_size: int):
        assert self.page_size % block_mem_size == 0
        return self.page_size // block_mem_size

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

    def _stop_prealloc_thread(self):
        if self.prealloc_thd is not None:
            with self.prealloc_lock:
                self.prealloc_running = False
                self.prealloc_cond.notify_all()
            self.prealloc_thd.join()
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


class KVCacheManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        cell_size: int,
        num_layers: int,
        tp_size: int = 1,
        async_sched: bool = False,
        reserve_null_block: bool = False,
    ):
        """
        Args:
            num_blocks: Number of blocks.
            block_size: Size of each block in bytes.
            cell_size: Size of each cell in bytes.
            num_layers: Number of layers.
            async_sched: Whether asynchronous scheduling is enabled.
        """
        self.num_blocks = num_blocks
        self.block_mem_size = block_size * cell_size
        self.num_layers = num_layers
        self.reserve_null_block = reserve_null_block

        # NOTE: this is the memory size of the K or V tensor in one layer
        self.mem_size = self.num_blocks * self.block_mem_size
        self.tp_size = tp_size
        self.page_allocator = PageAllocator(self.mem_size, PAGE_SIZE,
                                            self.tp_size, async_sched)

        self.num_avail_blocks = 0  # Only count free blocks in avail_pages
        self.avail_pages: Dict[int, Page] = {}
        self.full_pages: Dict[int, Page] = {}

        self.reserved_blocks: List[int] = []

        self.in_shrink: bool = False
        self.target_num_blocks: Optional[int] = None
        # NOTE: we use a no-op lock for sync scheduling to avoid overhead
        self._lock = threading.RLock() if async_sched else NoOpLock()

        self.null_block: Optional[list[int]] = None

        # Event used to signal that _post_init() has finished.
        self._post_init_done = threading.Event()
        # Launch _post_init in the background; it will block until KV tensors
        # exist, then complete the remaining setup (reserve null block, start
        # pre-alloc thread) and finally set the event.
        threading.Thread(target=self._post_init, daemon=True).start()

        self.ipc_name = get_ipc_name(DEFAULT_IPC_NAME)
        init_kv_cache_limit(self.ipc_name, self.mem_size * num_layers * 2)
        self._register_cleanup()

    def _post_init(self):
        if self.null_block is not None:
            return

        def _check_kv_tensors_created():
            if self.tp_size > 1:
                return broadcast_kv_tensors_created_to_workers(self.tp_size)
            else:
                return kv_tensors_created()

        # Busy-wait until the KV tensors become available.
        while not _check_kv_tensors_created():
            time.sleep(0.001)  # 1ms

        # Reserve the first block as null block for padding tokens
        if self.reserve_null_block:
            # Skip the wait to avoid dead-lock with the event.
            self.null_block = self.alloc(1, _skip_wait=True)
            assert self.null_block == [0], "Failed to reserve null block"
        self.page_allocator.start_prealloc_thread()

        self._post_init_done.set()

    def _wait_post_init(self):
        if not self._post_init_done.is_set():
            self._post_init_done.wait()

    @synchronized
    def alloc(self,
              need_size: int,
              _skip_wait: bool = False) -> Optional[List[int]]:
        if not _skip_wait:
            # Normal callers must wait until background initialisation is
            # finished and then perform the usual capacity check.
            self._wait_post_init()

        with RwLockedShm(self.ipc_name, MemInfoStruct.SHM_SIZE,
                         RwLockedShm.RLOCK) as mm:
            mem_info = MemInfoStruct.from_buffer(mm)
            new_mem_size = mem_info.total_size // self.num_layers // 2
            if new_mem_size != self.mem_size:
                self.resize(new_mem_size)

        if self.available_size() < need_size:
            logger.warning(f"available_size()={self.available_size()} < "
                           f"need_size={need_size}")
            return None

        ret_index = []
        page: Page = None

        remaining_need = need_size

        if self.reserved_blocks:
            # assert (
            #     len(self.reserved_blocks) == need_size
            # ), "Currently, we must have all blocks reserved before allocation to avoid OOM."
            # # NOTE: we can support len(self.reserved_blocks) != need_size cases,
            # # but we want to check reservation size == allocation size for now
            # # to ensure correctness.
            if len(self.reserved_blocks) >= remaining_need:
                ret_index = self.reserved_blocks[:remaining_need]
                self.reserved_blocks = self.reserved_blocks[remaining_need:]
                remaining_need = 0
            else:
                ret_index = self.reserved_blocks
                remaining_need -= len(self.reserved_blocks)
                self.reserved_blocks = []

        while remaining_need > 0:
            if not self.avail_pages:
                page = self.page_allocator.alloc_page()
                page.init(self.block_mem_size)
                self.num_avail_blocks += page.num_free_blocks()
            else:
                _, page = self.avail_pages.popitem()
            assert page is not None
            if page.num_free_blocks() > remaining_need:
                self.num_avail_blocks -= remaining_need
                alloced_index = page.free_list[:remaining_need]
                page.free_list = page.free_list[remaining_need:]
                ret_index.extend(alloced_index)
                remaining_need = 0
                self.avail_pages[page.page_id] = page
            else:
                self.num_avail_blocks -= page.num_free_blocks()
                ret_index.extend(page.free_list)
                remaining_need -= len(page.free_list)
                page.free_list = []
                self.full_pages[page.page_id] = page
        assert remaining_need == 0, "Insufficient memory for allocation."

        used_size = (self.page_allocator.get_num_inuse_pages() *
                     self.num_layers * PAGE_SIZE * 2)
        with RwLockedShm(self.ipc_name, MemInfoStruct.SHM_SIZE,
                         RwLockedShm.WLOCK) as mm:
            mem_info = MemInfoStruct.from_buffer(mm)
            mem_info.used_size = used_size
            mem_info.write_to_buffer(mm)

        return ret_index

    @synchronized
    def free(self, indices: List[int]):
        self._wait_post_init()
        # assert (
        #     len(self.reserved_blocks) == 0
        # ), "Reserved blocks must be used or freed before freeing other blocks."
        # # NOTE: we can support freeing reserved blocks, but we want to enforce
        # # this check for now to ensure correctness.
        unique_indices = set(indices)
        if self.reserved_blocks:
            self.reserved_blocks = [
                idx for idx in self.reserved_blocks
                if idx not in unique_indices
            ]

        idx_dict = defaultdict(list)
        for idx in unique_indices:
            page_id = self.page_allocator.get_page_id(idx, self.block_mem_size)
            idx_dict[page_id].append(idx)

        pages_to_free: List[int] = []
        for page_id, idxs in idx_dict.items():
            if (SANITY_CHECK and page_id not in self.full_pages
                    and page_id not in self.avail_pages):
                logger.warning(
                    f"Page {page_id} is not in avail_pages or full_pages, it is possible that the page is already freed."
                )
                continue
            if page_id in self.full_pages:
                page = self.full_pages.pop(page_id)
            else:
                page = self.avail_pages.pop(page_id)

            self.num_avail_blocks += len(idxs)
            page.free_batch(idxs)
            if page.empty():
                pages_to_free.append(page.page_id)
                self.num_avail_blocks -= page.num_free_blocks()
            else:
                self.avail_pages[page_id] = page

        if len(pages_to_free) > 0:
            self.page_allocator.free_pages(pages_to_free)

        if (self.in_shrink and self.page_allocator.get_num_inuse_blocks(
                self.block_mem_size) <= self.target_num_blocks):
            self.page_allocator.resize(self.target_num_blocks *
                                       self.block_mem_size)
            self.in_shrink = False
            self.target_num_blocks = None

        used_size = (self.page_allocator.get_num_inuse_pages() *
                     self.num_layers * PAGE_SIZE * 2)
        with RwLockedShm(self.ipc_name, MemInfoStruct.SHM_SIZE,
                         RwLockedShm.WLOCK) as mm:
            mem_info = MemInfoStruct.from_buffer(mm)
            mem_info.used_size = used_size
            mem_info.write_to_buffer(mm)

    @synchronized
    def try_to_reserve(self, need_size: int) -> bool:
        self._wait_post_init()
        if self.available_size() < need_size:
            return False
        # assert (
        #     len(self.reserved_blocks) == 0
        # ), "Reserved blocks must be used or freed before reserving more."
        reserved = self.alloc(need_size)
        assert reserved is not None, "Failed to reserve blocks."
        self.reserved_blocks.extend(reserved)
        return True

    @synchronized
    def free_reserved(self):
        if self.reserved_blocks:
            self.free(self.reserved_blocks)
            self.reserved_blocks = []

    @synchronized
    def resize(self, new_mem_size: int):
        """
        Reset the limit of the K or V tensor in one layer.
        new_mem_size: the memory size of the K or V tensor in one layer
        """
        self._wait_post_init()
        assert new_mem_size > 0, "new_mem_size must be positive"
        if self.page_allocator.resize(new_mem_size):
            if self.in_shrink:
                self.in_shrink = False
                self.target_num_blocks = None
            return True  # Successfully resized.
        # Failed to resize due to too many in-use blocks.
        assert (len(self.reserved_blocks) == 0
                ), "Reserved blocks must be freed before resizing."
        # NOTE: we can support resizing with reserved blocks, but we want to enforce
        # this check for now to ensure correctness.
        self.in_shrink = True
        self.target_num_blocks = new_mem_size // self.block_mem_size
        self.free_reserved()
        return False

    @synchronized
    def trim(self):
        self._wait_post_init()
        self.page_allocator.trim()

    @synchronized
    def available_size(self) -> int:
        avail_size = self.num_avail_blocks + len(self.reserved_blocks)
        if self.in_shrink:
            free_size = 0
        else:
            virtual_free_size = self.page_allocator.get_num_free_blocks(
                self.block_mem_size)
            physical_free_size = self._physical_free_size()
            free_size = min(virtual_free_size, physical_free_size)
        # logger.info(f"YIFAN: avail_size: {avail_size}, free_size: {free_size}, virtual_free_size: {virtual_free_size}, physical_free_size: {physical_free_size}")
        return avail_size + free_size

    @synchronized
    def get_mapped_memory_size(self, unit='bytes') -> float:
        """Get memory usage in specified unit (bytes, kb, mb, gb)."""
        memory_bytes = (self.page_allocator.get_num_inuse_pages() +
                        self.page_allocator.get_num_reserved_pages()
                        ) * self.num_layers * PAGE_SIZE * 2  # K and V tensors

        if unit == 'bytes':
            return memory_bytes
        elif unit == 'kb':
            return memory_bytes / 1024
        elif unit == 'mb':
            return memory_bytes / (1024**2)
        elif unit == 'gb':
            return memory_bytes / (1024**3)
        else:
            raise ValueError(f"Unknown unit: {unit}")

    def _physical_free_size(self) -> int:
        avail_phy_mem_size, total_phy_mem_size = torch.cuda.mem_get_info()
        headroom = total_phy_mem_size * (1 - GPU_UTILIZATION)
        avail_phy_mem_size = max(avail_phy_mem_size - headroom, 0)

        avail_phy_pages = avail_phy_mem_size // PAGE_SIZE
        # Each layer needs to reserve K and V tensors.
        avail_phy_blocks = (avail_phy_pages // self.num_layers //
                            2) * (PAGE_SIZE // self.block_mem_size)
        return avail_phy_blocks

    def clear(self):
        raise NotImplementedError

    def _cleanup_shm(self, *args):
        """Remove the POSIX shared-memory segment and its backing file."""
        try:
            # Unlink the POSIX shared memory object (no-op if already removed)
            posix_ipc.unlink_shared_memory(self.ipc_name)
        except Exception:
            pass

        # Also attempt to remove the backing file in /dev/shm (created by RwLockedShm)
        try:
            os.unlink(get_ipc_path(self.ipc_name))
        except FileNotFoundError:
            pass

        # If invoked as a signal handler, re-raise the default behaviour so the
        # process exits with the expected status code.
        if args and isinstance(args[0], int):
            signum = args[0]
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    def _register_cleanup(self):
        """Register atexit and signal handlers for shared-memory cleanup."""
        # Run on normal interpreter shutdown
        atexit.register(self._cleanup_shm)

        # Handle common termination signals (e.g., Ctrl-C or docker stop)
        for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP,
                     signal.SIGQUIT):
            try:
                signal.signal(_sig, self._cleanup_shm)
            except Exception:
                pass

    def __del__(self):
        try:
            self._cleanup_shm()
        except Exception:
            pass
