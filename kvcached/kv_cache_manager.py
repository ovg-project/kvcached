# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
kvcached Memory Manager

This module implements a hierarchical memory management system for KV cache:
- Pages: Large memory chunks (e.g., 2MB) that are mapped/unmapped to physical memory
- Blocks: Smaller units within pages that are allocated to store KV cache data
"""

import functools
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

from kvcached.locks import NoOpLock
from kvcached.page_allocator import Page, PageAllocator
from kvcached.tp_ipc_util import broadcast_kv_tensors_created
from kvcached.utils import PAGE_SIZE, SANITY_CHECK, get_kvcached_logger
from kvcached.vmm_ops import kv_tensors_created

logger = get_kvcached_logger()

KV_TENSOR_WAIT_TIMEOUT: float = 10.0  # seconds


def synchronized(method):
    """
    A helper decorator to synchronize access to a method.
    """

    @functools.wraps(method)
    def synchronized_method(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return synchronized_method


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
            tp_size: Number of tensor parallel processes.
            async_sched: Whether asynchronous scheduling is enabled.
            reserve_null_block: Whether to reserve the first block as null block
                for padding tokens. This is required by SGLang which assumes the
                first block is always reserved as padded tokens.
        """
        self.num_blocks = num_blocks
        self.block_mem_size = block_size * cell_size
        self.num_layers = num_layers
        self.reserve_null_block = reserve_null_block

        # The physical page size used by kvcached page allocator.
        self.page_size = PAGE_SIZE
        # NOTE: this is the memory size of the K or V tensor in one layer
        self.mem_size = self.num_blocks * self.block_mem_size
        self.tp_size = tp_size
        self.page_allocator = PageAllocator(
            self.num_layers,
            self.mem_size,
            self.page_size,
            self.tp_size,
            async_sched=async_sched,
        )

        self.num_avail_blocks = 0  # Only count free blocks in avail_pages
        self.avail_pages: Dict[int, Page] = {}
        self.full_pages: Dict[int, Page] = {}

        self.reserved_blocks: List[int] = []
        self.null_block: Optional[list[int]] = None

        self.in_shrink: bool = False
        self.target_num_blocks: Optional[int] = None
        # NOTE: we use a no-op lock for sync scheduling to avoid overhead
        self._lock = threading.RLock() if async_sched else NoOpLock()

        # Event used to signal that _post_init() has finished.
        self._post_init_done = threading.Event()
        # Launch _post_init in the background; it will block until KV tensors
        # exist, then complete the remaining setup (reserve null block, start
        # pre-alloc thread) and finally set the event.
        threading.Thread(target=self._post_init, daemon=True).start()

    def _post_init(self):
        if self.null_block is not None:
            return

        def _check_kv_tensors_created():
            if self.tp_size > 1:
                return broadcast_kv_tensors_created(self.tp_size)
            else:
                return kv_tensors_created()

        try:
            total_wait = 0.0
            while not _check_kv_tensors_created():
                if total_wait >= KV_TENSOR_WAIT_TIMEOUT:
                    raise TimeoutError("KV tensors not created after "
                                       f"{KV_TENSOR_WAIT_TIMEOUT} seconds")

                time.sleep(0.001)  # 1ms
                total_wait += 0.001

            # Reserve the first block as null block for padding tokens
            if self.reserve_null_block:
                # Skip the wait to avoid dead-lock with the event.
                self.null_block = self._alloc(1, _skip_wait=True)
                if self.null_block != [0]:
                    logger.error(
                        f"Failed to reserve null block, got {self.null_block}")
                    raise RuntimeError(
                        "Failed to reserve null block at index 0")

            self.page_allocator.start_prealloc_thread()
        except Exception as e:
            logger.error(
                f"Error during KVCacheManager post-initialization: {e}")
            # Set the event even on error to unblock waiting threads
            raise
        finally:
            self._post_init_done.set()

    def _wait_post_init(self):
        if not self._post_init_done.is_set():
            self._post_init_done.wait()

    def alloc(self, need_size: int) -> Optional[List[int]]:
        return self._alloc(need_size)

    @synchronized
    def _alloc(self,
               need_size: int,
               _skip_wait: bool = False) -> Optional[List[int]]:
        if not _skip_wait:
            # Normal callers must wait until background initialisation is
            # finished and then perform the usual capacity check.
            self._wait_post_init()

        new_mem_size = self.page_allocator.mem_info_tracker.check_and_get_resize_target(
            self.mem_size, self.num_layers)
        if new_mem_size is not None:
            self.resize(new_mem_size)

        if self.available_size() < need_size:
            logger.warning(f"available_size()={self.available_size()} < "
                           f"need_size={need_size}")
            return None

        ret_index = []
        page: Optional[Page] = None

        remaining_need = need_size

        if self.reserved_blocks:  # Try to allocate from reserved blocks first
            num_from_reserved = min(len(self.reserved_blocks), remaining_need)
            # ret_index is empty before so we directly assign it
            ret_index = self.reserved_blocks[:num_from_reserved]
            self.reserved_blocks = self.reserved_blocks[num_from_reserved:]
            remaining_need -= num_from_reserved

        while remaining_need > 0:  # Allocate the remaining blocks from pages
            if not self.avail_pages:
                page = self.page_allocator.alloc_page()
                page.init(self.block_mem_size)
                self.num_avail_blocks += page.num_free_blocks()
            else:
                _, page = self.avail_pages.popitem()
            num_from_page = min(page.num_free_blocks(), remaining_need)
            alloced_index = page.alloc(num_from_page)
            ret_index.extend(alloced_index)
            if page.full():
                self.full_pages[page.page_id] = page
            else:
                self.avail_pages[page.page_id] = page

            self.num_avail_blocks -= num_from_page
            remaining_need -= num_from_page

        return ret_index

    @synchronized
    def free(self, indices: List[int]):
        self._wait_post_init()

        if len(indices) == 0:
            return  # Nothing to free

        if SANITY_CHECK:
            for idx in indices:
                if idx in self.reserved_blocks:
                    raise ValueError(f"Freed index {idx} is in "
                                     " reserved_blocks, which is not allowed.")

        # Group indices by page_id
        idx_dict = defaultdict(list)
        for idx in indices:
            page_id = self.page_allocator.get_page_id(idx, self.block_mem_size)
            idx_dict[page_id].append(idx)

        pages_to_free: List[int] = []
        for page_id, idxs in idx_dict.items():
            # Find the page - it must be in either full_pages or avail_pages
            page = None
            if page_id in self.full_pages:
                page = self.full_pages.pop(page_id)
            elif page_id in self.avail_pages:
                page = self.avail_pages.pop(page_id)
            else:
                if SANITY_CHECK:
                    # This is a serious error - the page should exist
                    raise ValueError(
                        f"Page {page_id} not found in avail_pages or full_pages. "
                        f"This indicates a serious state inconsistency.")
                else:
                    logger.error(
                        f"Page {page_id} not found in avail_pages or full_pages. "
                        f"Skipping to avoid crash, but this indicates a serious bug."
                    )
                    continue

            self.num_avail_blocks += len(idxs)
            page.free_batch(idxs)

            if page.empty():
                pages_to_free.append(page.page_id)
                self.num_avail_blocks -= page.num_free_blocks()
            else:
                self.avail_pages[page_id] = page

        if pages_to_free:
            self.page_allocator.free_pages(pages_to_free)

        if self.in_shrink:
            assert self.target_num_blocks is not None
            if self._get_num_alloced_blocks() <= self.target_num_blocks:
                self.page_allocator.resize(self.target_num_blocks *
                                           self.block_mem_size)
                self.in_shrink = False
                self.target_num_blocks = None

    @synchronized
    def try_to_reserve(self, need_size: int) -> bool:
        self._wait_post_init()
        if self.available_size() < need_size:
            return False
        reserved = self.alloc(need_size)
        if reserved is None:
            logger.warning("Failed to reserve blocks.")
            return False
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
        # NOTE: we can support resizing with reserved blocks, but we want to
        # enforce this check for now to ensure correctness.
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
        avail_blocks = self.num_avail_blocks + len(self.reserved_blocks)
        if self.in_shrink:
            blocks_from_free_pages = 0
        else:
            virtual_free_pages = self.page_allocator.get_num_free_pages()
            physical_free_pages = self.page_allocator.get_avail_physical_pages(
            ) + self.page_allocator.get_num_reserved_pages()
            free_pages = min(virtual_free_pages, physical_free_pages)
            blocks_from_free_pages = free_pages * Page.get_num_blocks(
                self.page_size, self.block_mem_size)
        return avail_blocks + blocks_from_free_pages

    @synchronized
    def get_mapped_memory_size(self, unit='bytes') -> float:
        """Get memory usage in specified unit (bytes, kb, mb, gb)."""
        memory_bytes = (self.page_allocator.get_num_inuse_pages() *
                        self.num_layers * self.page_size * 2)

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

    def clear(self):
        raise NotImplementedError("kvcached does not support clear() for now")

    # Private methods
    @synchronized
    def _get_num_alloced_blocks(self) -> int:
        # Blocks from fully allocated pages
        blocks_from_full_pages = len(self.full_pages) * Page.get_num_blocks(
            self.page_size, self.block_mem_size)
        # Blocks from partially allocated pages. num_avail_blocks is the number
        # of free blocks in the partially allocated pages so the number of
        # allocated blocks is the total number of blocks in the partially
        # allocated pages minus the number of free blocks.
        blocks_from_avail_pages = len(self.avail_pages) * Page.get_num_blocks(
            self.page_size, self.block_mem_size) - self.num_avail_blocks
        # Blocks from reserved blocks
        blocks_from_reserved_blocks = len(self.reserved_blocks)
        return (blocks_from_full_pages + blocks_from_avail_pages +
                blocks_from_reserved_blocks)
