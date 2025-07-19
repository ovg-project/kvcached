"""
kvcached Memory Manager

This module implements a hierarchical memory management system for KV cache:
- Pages: Large memory chunks (e.g., 2MB) that are mapped/unmapped to physical memory
- Blocks: Smaller units within pages that are allocated to store KV cache data
"""

import atexit
import functools
import os
import signal
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

import posix_ipc
import torch

from kvcached.cli.utils import (MemInfoStruct, RwLockedShm, get_ipc_name,
                                get_ipc_path, init_kv_cache_limit)
from kvcached.locks import NoOpLock
from kvcached.page_allocator import Page, PageAllocator
from kvcached.tp_ipc_util import broadcast_kv_tensors_created_to_workers
from kvcached.utils import (DEFAULT_IPC_NAME, GPU_UTILIZATION, PAGE_SIZE,
                            SANITY_CHECK, get_kvcached_logger)
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
            if page.num_free_blocks() > remaining_need:
                self.num_avail_blocks -= remaining_need
                alloced_index = page.free_list[:remaining_need]
                page.free_list = page.free_list[remaining_need:]
                ret_index.extend(alloced_index)
                remaining_need = 0
                self.avail_pages[page.page_id] = page
            else:
                self.num_avail_blocks -= page.num_free_blocks()
                additional_blocks = page.alloc_all_remaining()
                ret_index.extend(additional_blocks)
                remaining_need -= len(additional_blocks)
                self.full_pages[page.page_id] = page

        with RwLockedShm(self.ipc_name, MemInfoStruct.SHM_SIZE,
                         RwLockedShm.WLOCK) as mm:
            mem_info = MemInfoStruct.from_buffer(mm)
            mem_info.used_size = self._get_used_size()
            mem_info.prealloc_size = self._get_prealloc_size()
            mem_info.write_to_buffer(mm)

        return ret_index

    @synchronized
    def free(self, indices: List[int]):
        self._wait_post_init()

        if len(indices) == 0:
            return  # Nothing to free

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

        if (self.in_shrink and self.page_allocator.get_num_inuse_blocks(
                self.block_mem_size) <= self.target_num_blocks):
            self.page_allocator.resize(self.target_num_blocks *
                                       self.block_mem_size)
            self.in_shrink = False
            self.target_num_blocks = None

        with RwLockedShm(self.ipc_name, MemInfoStruct.SHM_SIZE,
                         RwLockedShm.WLOCK) as mm:
            mem_info = MemInfoStruct.from_buffer(mm)
            mem_info.used_size = self._get_used_size()
            mem_info.prealloc_size = self._get_prealloc_size()
            mem_info.write_to_buffer(mm)

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
        return avail_size + free_size

    @synchronized
    def _get_used_size(self) -> int:
        # Memory actively used by allocations (excludes preallocated pages)
        return (self.page_allocator.get_num_inuse_pages() * self.num_layers *
                PAGE_SIZE * 2)

    def _get_prealloc_size(self) -> int:
        # Memory held by preallocated pages that are not yet actively used
        return (self.page_allocator.get_num_reserved_pages() *
                self.num_layers * PAGE_SIZE * 2)

    @synchronized
    def get_mapped_memory_size(self, unit='bytes') -> float:
        """Get memory usage in specified unit (bytes, kb, mb, gb)."""
        memory_bytes = self._get_used_size()

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
        # Use the page allocator's method to get accurate block count per page
        blocks_per_page = Page.get_num_blocks(PAGE_SIZE, self.block_mem_size)
        avail_phy_blocks = (avail_phy_pages // self.num_layers //
                            2) * blocks_per_page
        return avail_phy_blocks

    def clear(self):
        raise NotImplementedError("kvcached does not support clear() for now")

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
