# Add logging support
import logging
import os
import threading
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Dict, List, Optional, Union

import torch

try:
    from kvcached.vmm_ops import map_to_kv_tensors, unmap_from_kv_tensors
except ImportError:
    # If direct import fails, try importing from vmm_ops module
    # This handles the case where kvcached is used as a standalone module
    try:
        from vmm_ops import map_to_kv_tensors, unmap_from_kv_tensors
    except ImportError:
        # Final fallback: try to add local csrc path
        import sys

        SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
        KVCACHED_PATH = os.path.realpath(f"{SCRIPT_PATH}/../csrc")
        if os.path.exists(KVCACHED_PATH):
            sys.path.append(KVCACHED_PATH)

        # Try to import from kvcached.vmm_ops or vmm_ops
        try:
            from kvcached.vmm_ops import (map_to_kv_tensors,
                                          unmap_from_kv_tensors)
        except ImportError:
            from vmm_ops import map_to_kv_tensors, unmap_from_kv_tensors

# >>> ADD NVTX utilities
import contextlib
import time
from multiprocessing import shared_memory

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


def _is_kvcached_profile_enabled():
    """Check if KVCACHED_PROFILE environment variable is enabled"""
    return os.getenv("KVCACHED_PROFILE",
                     "false").lower() in ("true", "1", "yes")


try:
    from torch.cuda import nvtx as _nvtx
    print("✅ [NVTX-INIT] torch.cuda.nvtx imported successfully")
except Exception as e:
    print(f"❌ [NVTX-INIT] Failed to import torch.cuda.nvtx: {e}")
    _nvtx = None

# >>> ADD CUDA synchronization control
ENABLE_CUDA_SYNC = os.getenv("KVCACHED_ENABLE_CUDA_SYNC",
                             "false").lower() in ("true", "1", "yes")
if ENABLE_CUDA_SYNC:
    print("✅ [CUDA-SYNC] CUDA synchronization enabled for NVTX profiling")
else:
    print("ℹ️  [CUDA-SYNC] CUDA synchronization disabled")


def cuda_sync_if_enabled():
    """Conditionally perform CUDA sync based on environment variable"""
    if ENABLE_CUDA_SYNC and torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def nvtx_range(name: str):
    if _nvtx is None:
        yield
        return

    if ENABLE_CUDA_SYNC and torch.cuda.is_available():
        torch.cuda.synchronize()
    _nvtx.range_push(name)
    try:
        yield
    finally:
        _nvtx.range_pop()
        if ENABLE_CUDA_SYNC and torch.cuda.is_available():
            torch.cuda.synchronize()


SANITY_CHECK = False
PAGE_SIZE = 2 * 1024 * 1024  # 2MB
GPU_UTILIZATION = 0.95
PAGE_PREALLOC_ENABLED = True


class Timer:

    def __init__(self, timings: List[float]):
        self.timings = timings

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        duration = time.time() - self.start
        self.timings.append(duration)
        # Prevent unlimited growth by keeping only recent entries
        if len(self.timings) > 1024:
            self.timings.pop(0)


class MemoryUsageReader():

    def __init__(self, ipc_name: str, create_timeout=60):
        self.ipc_name = ipc_name
        self.create_timeout = create_timeout
        self.shared_memory = None
        self.memory_usage_array = None

        # Try to access the shared memory segment
        try:
            self.shared_memory = shared_memory.SharedMemory(name=ipc_name,
                                                            create=False)
            self.memory_usage_array = np.ndarray((1, ),
                                                 dtype=np.int64,
                                                 buffer=self.shared_memory.buf)
        except FileNotFoundError:
            # If shared memory doesn't exist, create it
            self.shared_memory = shared_memory.SharedMemory(
                name=ipc_name, create=True, size=np.int64().itemsize)
            self.memory_usage_array = np.ndarray((1, ),
                                                 dtype=np.int64,
                                                 buffer=self.shared_memory.buf)
            self.memory_usage_array[0] = 0

    def get_memory_usage_in_mb(self):
        return self.memory_usage_array[0] / 1024 / 1024

    def get_memory_usage_in_gb(self):
        return self.memory_usage_array[0] / 1024 / 1024 / 1024

    def get_memory_usage_in_bytes(self):
        return self.memory_usage_array[0]

    def __del__(self):
        if self.shared_memory:
            self.shared_memory.close()


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
        assert (self.initialized()
                and len(self.free_list) == self.num_kv_blocks)
        self.block_size = None
        self.phy_token_kv_size = None
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
        return len(self.free_list) == 0

    def num_free_blocks(self) -> int:
        return len(self.free_list)

    def get_free_blocks(self) -> List[int]:
        return self.free_list

    def get_used_blocks(self) -> List[int]:
        if not self.initialized():
            return []
        stt_idx = self.page_id * self.num_kv_blocks
        all_blocks = [stt_idx + i for i in range(self.num_kv_blocks)]
        return [block for block in all_blocks if block not in self.free_list]

    def _has_block(self, block_id: int) -> bool:
        stt_idx = self.page_id * self.num_kv_blocks
        return stt_idx <= block_id < stt_idx + self.num_kv_blocks

    def _sanity_check(self, block_id: int) -> None:
        assert self._has_block(block_id), (
            f"Block {block_id} is not in page {self.page_id}")


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
                 num_layers: int = 1,
                 shm: shared_memory.SharedMemory = None,
                 device: str = "cpu"):
        print(f"Init PageAllocator: "
              f"total_mem_size={total_mem_size//(1024*1024)}MB, "
              f"page_size={page_size//(1024*1024)}MB, "
              f"shm: {shm.name if shm else None}")

        self.num_layers = num_layers
        self.total_mem_size = total_mem_size
        self.page_size = page_size
        self.page_size_all_layers = self.page_size * self.num_layers * 2
        self.num_free_pages = total_mem_size // page_size
        self.num_total_pages = total_mem_size // page_size

        self.free_page_list: deque[int] = deque(range(self.num_free_pages))
        self.write_shm_times = []

        # Performance metrics
        self.slow_path_count = 0  # Count of alloc_page() slow path hits
        self.fast_path_count = 0  # Count of alloc_page() fast path hits
        self.reserved_at_alloc = []  # Track reserved count at alloc entry

        if shm is not None:
            self.shm = shm
            self.memory_in_use = np.ndarray((1, ),
                                            dtype=np.int64,
                                            buffer=self.shm.buf)
            self.memory_in_use[0] = 0
        else:
            self.shm = None
            self.memory_in_use = None

        # Preallocation settings
        self.min_reserved_pages = 5
        self.max_reserved_pages = 200
        self.batch_size = 16  # Suggested batch parameter
        self.reserved_page_list: List[int] = []
        self.reclaimed_page_list: List[int] = []

        # Preallocation thread management
        self.prealloc_lock = threading.Lock()
        self.prealloc_cond = threading.Condition(self.prealloc_lock)
        self.prealloc_running = False
        self.prealloc_needed = False
        self.prealloc_thd = None

        # NEW: Track how many upper-level allocs are in the "protected zone"
        self.alloc_active = 0

        # --- Refill thresholds & budgets (new) ---
        # Keep your existing semantics:
        #   min_reserved_pages -> soft low-water (L_soft)
        #   max_reserved_pages -> high-water (H)
        # Add a critical low-water (L_crit) as a hard floor.
        self.crit_reserved_pages = int(
            os.getenv("KVCACHED_L_CRIT",
                      str(max(1, min(self.min_reserved_pages // 4, 64)))))
        # Refill batching & budget
        self.refill_sleep_sec = float(
            os.getenv("KVCACHED_REFILL_SLEEP_SEC", "0.05"))  # 50ms
        self.refill_budget_ms = float(
            os.getenv("KVCACHED_REFILL_BUDGET_MS", "1.0"))  # per wake
        self.refill_bmin = int(os.getenv("KVCACHED_REFILL_BMIN", "32"))
        self.refill_bmax = int(os.getenv("KVCACHED_REFILL_BMAX", "128"))

        # Ensure threshold relationships: L_crit ≤ L_soft ≤ H
        self.min_reserved_pages = max(1, self.min_reserved_pages)
        self.max_reserved_pages = max(self.max_reserved_pages,
                                      self.min_reserved_pages)
        self.crit_reserved_pages = min(self.crit_reserved_pages,
                                       self.min_reserved_pages)

        # Start preallocation thread
        self._start_prealloc_thread()

    def __del__(self):
        # Stop preallocation thread
        self._stop_prealloc_thread()

        # Clean up shared memory
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()

    def _prealloc_worker(self):
        """Refill reserved pages with hysteresis + time budget."""
        if _is_kvcached_profile_enabled():
            logger.info("🧵[PREALLOC-WORKER] started")

        while self.prealloc_running:
            # Check whether refill is needed
            with self.prealloc_lock:
                reserved = len(self.reserved_page_list)
                free_pages = len(self.free_page_list)
                need_refill = (self.prealloc_needed
                               or (reserved < self.min_reserved_pages))

                # Condition: pause when alloc is busy and stock >= L_soft; otherwise allow refill
                should_pause = ((not need_refill)
                                or (self.alloc_active > 0
                                    and reserved >= self.min_reserved_pages))
                if should_pause:
                    # Sleep as fallback to avoid missing wakeups
                    self.prealloc_cond.wait(timeout=self.refill_sleep_sec)
                    self.prealloc_needed = False
                    continue

                # Consume external trigger
                self.prealloc_needed = False

            # Start a time-budget-limited refill round
            t_start = time.time()
            while (time.time() - t_start) * 1000.0 < self.refill_budget_ms:
                with self.prealloc_lock:
                    if not self.prealloc_running:
                        break

                    reserved = len(self.reserved_page_list)
                    free_pages = len(self.free_page_list)
                    if free_pages == 0:
                        break

                    # Target level: if alloc is busy and >= L_crit, only top up to L_soft; otherwise fill to H
                    if self.alloc_active > 0 and reserved >= self.crit_reserved_pages:
                        target = self.min_reserved_pages  # top-up only
                    else:
                        target = self.max_reserved_pages  # fill to H when idle or too low

                    deficit = max(0, target - reserved)
                    if deficit == 0:
                        break

                    take_cap = min(deficit, free_pages)
                    # Dynamic batching, clamped within [bmin, bmax]
                    take = max(self.refill_bmin, min(self.refill_bmax,
                                                     take_cap))

                    pages = []
                    for _ in range(take):
                        if self.free_page_list:
                            pages.append(self.free_page_list.popleft())
                        else:
                            break

                if not pages:
                    break

                # Map outside lock
                try:
                    map_to_kv_tensors([pid * self.page_size for pid in pages])
                except Exception as e:
                    # Return pages on failure
                    with self.prealloc_lock:
                        for pid in reversed(pages):
                            self.free_page_list.appendleft(pid)
                    if _is_kvcached_profile_enabled():
                        logger.warning(
                            f"❌[PREALLOC-MAP-FAIL] pages={len(pages)} err={e}")
                    break

                # Write back reserved list and memory stats
                with self.prealloc_lock:
                    if self.shm is not None:
                        with Timer(self.write_shm_times):
                            self.memory_in_use[0] += len(
                                pages) * self.page_size_all_layers
                    self.reserved_page_list.extend(pages)
                    self.prealloc_cond.notify_all()

            # End of refill round, next round triggered by cond or timeout

    def _start_prealloc_thread(self):
        """Start the preallocation thread"""
        if self.prealloc_thd is None:
            self.prealloc_running = True
            self.prealloc_thd = threading.Thread(target=self._prealloc_worker,
                                                 daemon=True)
            self.prealloc_thd.start()
            print("Started page preallocation thread")

            # Initial preallocation trigger
            self._trigger_preallocation()

    def _stop_prealloc_thread(self):
        """Stop the preallocation thread"""
        if self.prealloc_thd is not None:
            with self.prealloc_lock:
                self.prealloc_running = False
                self.prealloc_cond.notify_all()
            self.prealloc_thd.join()
            self.prealloc_thd = None
            print("Stopped page preallocation thread")

    def _trigger_preallocation(self):
        """Trigger the preallocation thread to fill up reserved blocks"""
        with self.prealloc_lock:
            if _is_kvcached_profile_enabled():
                current_time = time.time()
                free_pages = len(self.free_page_list)
                reserved_pages = len(self.reserved_page_list)
                msg = (f"🔔[TRIGGER-PREALLOC] [{current_time:.3f}] "
                       f"Triggering preallocation, "
                       f"prealloc_running={self.prealloc_running}, "
                       f"free_pages={free_pages}, "
                       f"reserved_pages={reserved_pages}")
                logger.info(msg)

            self.prealloc_needed = True
            self.prealloc_cond.notify()

    def alloc_page(self) -> Page:
        if _is_kvcached_profile_enabled():
            current_time = time.time()
            free_pages_before = len(self.free_page_list)
            reserved_pages_before = len(self.reserved_page_list)

        # Fast path: allocate from reserved pages (already mapped)
        page_id = None
        with self.prealloc_lock:
            if self.reserved_page_list:
                page_id = self.reserved_page_list.pop()

                # Trigger preallocation to refill reserved pool
                if (len(self.reserved_page_list) < self.min_reserved_pages):
                    self.prealloc_needed = True
                    self.prealloc_cond.notify()

        if page_id is not None:
            if _is_kvcached_profile_enabled():
                free_pages_after = len(self.free_page_list)
                reserved_pages_after = len(self.reserved_page_list)
                msg = (f"📄[ALLOC-PAGE-FAST] [{current_time:.3f}] "
                       f"page_id={page_id}, "
                       f"free_pages: {free_pages_before}->{free_pages_after}, "
                       f"reserved_pages: {reserved_pages_before}->"
                       f"{reserved_pages_after}")
                logger.info(msg)
            return Page(page_id, self.page_size)

        # Slow path: allocate from free list and map it
        with self.prealloc_lock:
            if not self.free_page_list:
                raise ValueError("No free page available")
            page_id = self.free_page_list.popleft()

        # Map the page to physical memory (slow path)
        page = Page(page_id, self.page_size)
        # Fix: use same offset calculation as original
        with nvtx_range("map_to_kv_tensors"):
            map_to_kv_tensors([page_id * self.page_size])

        with self.prealloc_lock:
            if self.shm is not None:
                with Timer(self.write_shm_times):
                    self.memory_in_use[0] += self.page_size_all_layers

        # Trigger preallocation to refill the pool
        self._trigger_preallocation()

        if _is_kvcached_profile_enabled():
            free_pages_final = len(self.free_page_list)
            reserved_pages_final = len(self.reserved_page_list)
            memory_usage = (self.memory_in_use[0]
                            if self.memory_in_use is not None else 0)
            msg = (f"📄[ALLOC-PAGE-SLOW] [{current_time:.3f}] "
                   f"page_id={page_id}, "
                   f"free_pages: {free_pages_before}->{free_pages_final}, "
                   f"reserved_pages: {reserved_pages_before}->"
                   f"{reserved_pages_final}, "
                   f"memory_in_use={memory_usage}")
            logger.info(msg)

        return page

    def free_page(self, page: Page) -> None:
        page_id = page.page_id

        if _is_kvcached_profile_enabled():
            current_time = time.time()
            free_pages_before = len(self.free_page_list)
            reserved_pages_before = len(self.reserved_page_list)
            memory_usage_before = (self.memory_in_use[0]
                                   if self.memory_in_use is not None else 0)

        # Check if we can add to reserved pool
        added_to_reserved = False
        with self.prealloc_lock:
            if len(self.reserved_page_list) < self.max_reserved_pages:
                # Fast path: add to reserved pool (keep mapped)
                self.reserved_page_list.append(page_id)
                added_to_reserved = True

        if added_to_reserved:
            if _is_kvcached_profile_enabled():
                free_pages_after = len(self.free_page_list)
                reserved_pages_after = len(self.reserved_page_list)
                memory_usage_after = (self.memory_in_use[0]
                                      if self.memory_in_use is not None else 0)
                msg = (f"📄[FREE-PAGE-FAST] [{current_time:.3f}] "
                       f"page_id={page_id}, "
                       f"free_pages: {free_pages_before}->{free_pages_after}, "
                       f"reserved_pages: {reserved_pages_before}->"
                       f"{reserved_pages_after}, "
                       f"memory_usage: {memory_usage_before}->"
                       f"{memory_usage_after}")
                logger.info(msg)
            return

        # Slow path: unmap and add to free list
        page.destroy()
        # Fix: use same offset calculation as original
        with nvtx_range("unmap_from_kv_tensors"):
            unmap_from_kv_tensors([page_id * self.page_size])
        with self.prealloc_lock:
            self.free_page_list.append(page_id)
            if self.shm is not None:
                with Timer(self.write_shm_times):
                    self.memory_in_use[0] -= self.page_size_all_layers

        if _is_kvcached_profile_enabled():
            free_pages_final = len(self.free_page_list)
            reserved_pages_final = len(self.reserved_page_list)
            memory_usage_final = (self.memory_in_use[0]
                                  if self.memory_in_use is not None else 0)
            msg = (f"📄[FREE-PAGE-SLOW] [{current_time:.3f}] "
                   f"page_id={page_id}, "
                   f"free_pages: {free_pages_before}->{free_pages_final}, "
                   f"reserved_pages: {reserved_pages_before}->"
                   f"{reserved_pages_final}, "
                   f"memory_usage: {memory_usage_before}->"
                   f"{memory_usage_final}")
            logger.info(msg)

    def free_pages(self, page_ids: List[int]) -> None:
        if not page_ids:
            return

        # Try to add pages to reserved pool first
        reserved_pages = []
        remaining_pages = []

        with self.prealloc_lock:
            # Calculate how many we can add to reserved pool
            space_in_reserved = max(
                0, self.max_reserved_pages - len(self.reserved_page_list))

            if space_in_reserved > 0:
                # Add up to space_in_reserved pages to reserved pool
                reserved_pages = page_ids[:space_in_reserved]
                self.reserved_page_list.extend(reserved_pages)

                # The rest go to remaining_pages
                remaining_pages = page_ids[space_in_reserved:]
            else:
                # No space in reserved pool, all pages go to remaining
                remaining_pages = page_ids

        # Unmap remaining pages and add to free list
        if remaining_pages:
            # Fix: use same offset calculation as original
            unmap_from_kv_tensors(
                [pid * self.page_size for pid in remaining_pages])
            with self.prealloc_lock:
                self.free_page_list.extend(remaining_pages)
                if self.shm is not None:
                    with Timer(self.write_shm_times):
                        self.memory_in_use[0] -= (len(remaining_pages) *
                                                  self.page_size_all_layers)

    def resize(self, new_mem_size: int) -> bool:
        new_num_pages = new_mem_size // self.page_size
        if new_num_pages > self.num_total_pages:
            num_pages_to_add = new_num_pages - self.num_total_pages
            for i in range(num_pages_to_add):
                self.free_page_list.append(self.num_total_pages + i)
            self.num_total_pages = new_num_pages
            self.num_free_pages = len(self.free_page_list)
            return True
        elif new_num_pages < self.num_total_pages:
            num_pages_to_remove = self.num_total_pages - new_num_pages
            available_pages = (len(self.free_page_list) +
                               len(self.reserved_page_list))
            if available_pages < num_pages_to_remove:
                return False
            for _ in range(num_pages_to_remove):
                if self.free_page_list:
                    self.free_page_list.pop()
                elif self.reserved_page_list:
                    page_id = self.reserved_page_list.pop()
                    slab_offsets = [page_id * self.page_size]
                    unmap_from_kv_tensors(slab_offsets)
                    if self.shm is not None:
                        with Timer(self.write_shm_times):
                            self.memory_in_use[0] -= (
                                self.page_size_all_layers)
            self.num_total_pages = new_num_pages
            self.num_free_pages = len(self.free_page_list)
            return True
        return True

    def trim(self):
        pages_to_unmap = []
        with self.prealloc_lock:
            pages_to_unmap = self.reserved_page_list.copy()
            self.reserved_page_list = []

        # Fix: use same offset calculation as original
        slab_offsets = [page_id * self.page_size for page_id in pages_to_unmap]
        if slab_offsets:
            unmap_from_kv_tensors(slab_offsets)
            if self.shm is not None:
                with Timer(self.write_shm_times):
                    mem_decrease = (len(slab_offsets) *
                                    self.page_size_all_layers)
                    self.memory_in_use[0] -= mem_decrease

        with self.prealloc_lock:
            self.free_page_list.extend(pages_to_unmap)

    def get_num_free_pages(self) -> int:
        with self.prealloc_lock:
            return len(self.free_page_list)

    def get_num_inuse_pages(self) -> int:
        with self.prealloc_lock:
            return (self.num_total_pages - len(self.free_page_list) -
                    len(self.reserved_page_list))

    def get_num_total_pages(self) -> int:
        return self.num_total_pages

    def get_num_reserved_pages(self) -> int:
        with self.prealloc_lock:
            return len(self.reserved_page_list)

    def get_page_id(self, block_id: int, block_mem_size: int) -> int:
        return block_id * block_mem_size // self.page_size

    def get_num_free_blocks(self, block_mem_size: int) -> int:
        return (self.get_num_free_pages() *
                self._num_blocks_per_page(block_mem_size))

    def get_num_inuse_blocks(self, block_mem_size: int) -> int:
        return (self.get_num_inuse_pages() *
                self._num_blocks_per_page(block_mem_size))

    def get_num_total_blocks(self, block_mem_size: int) -> int:
        return (self.get_num_total_pages() *
                self._num_blocks_per_page(block_mem_size))

    def _num_blocks_per_page(self, block_mem_size: int):
        assert self.page_size % block_mem_size == 0
        return self.page_size // block_mem_size


class KVCacheManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        cell_size: int,
        num_layers: int,
        shm: shared_memory.SharedMemory = None,
        return_device: Optional[str] = None,
        **kwargs,
    ):
        self.num_blocks = num_blocks
        self.block_mem_size = block_size * cell_size
        self.num_layers = num_layers

        mem_size = self.num_blocks * self.block_mem_size
        self.page_allocator = PageAllocator(mem_size,
                                            PAGE_SIZE,
                                            num_layers=num_layers,
                                            shm=shm)

        self.num_avail_blocks = 0  # Only count free blocks in avail_pages
        self.avail_pages: Dict[int, Page] = {}
        self.full_pages: Dict[int, Page] = {}

        self.reserved_blocks: List[int] = []

        self.in_shrink: bool = False
        self.target_num_blocks: Optional[int] = None

        # ---- New optional fast-path settings ----
        # Set device for returning Tensor directly from allocator, defaults
        # to reading from environment variable; if empty, keeps old
        # behavior (return list)
        self.return_device: Optional[str] = (
            return_device if return_device is not None else os.getenv(
                "KVCACHED_RETURN_DEVICE", None))
        # Whether to pin_memory CPU tensor and then async copy
        self.pin_cpu: bool = True

        # Pre-reserve all available pages for performance optimization
        available = self.available_size()
        if available > 0:
            reserved_success = self.try_to_reserve(available)
            print(f"Pre-reserved {len(self.reserved_blocks)} blocks out of "
                  f"{available} available blocks. Success: {reserved_success}")

    def alloc(self, need_size: int) -> Union[List[int], torch.Tensor]:
        if _is_kvcached_profile_enabled():
            current_time = time.time()
            num_avail_blocks_before = self.num_avail_blocks
            avail_pages_before = len(self.avail_pages)
            full_pages_before = len(self.full_pages)
            reserved_blocks_before = len(self.reserved_blocks)
            memory_usage = 0
            if hasattr(self.page_allocator, 'memory_in_use'
                       ) and self.page_allocator.memory_in_use is not None:
                memory_usage = self.page_allocator.memory_in_use[0]

        with nvtx_range(f"kv_alloc_total({need_size})"):
            if need_size == 0:
                if self.return_device:
                    dev = self.return_device
                    device_str = dev if dev.startswith("cuda") else "cpu"
                    return torch.empty(0, dtype=torch.int32, device=device_str)
                return []

            if self.available_size() < need_size:
                if _is_kvcached_profile_enabled():
                    logger.info(
                        f"🚫[KV-ALLOC-FAIL] [{current_time:.3f}] "
                        f"need_size={need_size} > available_size={self.available_size()}"
                    )
                return None

            ret_index: List[int] = []
            remaining_need = need_size

            # fast path: reserved_blocks
            with nvtx_range("alloc_reserved_fastpath"):
                # Protect reserved_blocks with the same lock to avoid races
                with self.page_allocator.prealloc_lock:
                    if self.reserved_blocks:
                        n = min(len(self.reserved_blocks), remaining_need)
                        ret_index = self.reserved_blocks[:n]
                        del self.reserved_blocks[:n]
                        remaining_need -= n

            with nvtx_range("alloc_page_loop"):
                # ENTER critical section for the whole multi-page allocation
                alloc_start_time = time.time()
                with self.page_allocator.prealloc_lock:
                    self.page_allocator.alloc_active += 1
                    if _is_kvcached_profile_enabled():
                        logger.info(
                            f"📦[ALLOC] [{alloc_start_time:.3f}] "
                            f"critical enter, pages_needed={remaining_need}, "
                            f"alloc_active={self.page_allocator.alloc_active}")
                    # Optional: wake up worker to see alloc_active>0 and go wait
                    self.page_allocator.prealloc_cond.notify_all()
                try:
                    while remaining_need > 0:
                        if not self.avail_pages:
                            with nvtx_range("alloc_page+map"):
                                page = self.page_allocator.alloc_page()
                                page.init(self.block_mem_size)
                                self.num_avail_blocks += page.num_free_blocks()
                        else:
                            _, page = self.avail_pages.popitem()

                        assert page is not None
                        free_n = page.num_free_blocks()
                        if free_n > remaining_need:
                            self.num_avail_blocks -= remaining_need
                            alloced_index = page.free_list[:remaining_need]
                            page.free_list = page.free_list[remaining_need:]
                            ret_index.extend(alloced_index)
                            remaining_need = 0
                            self.avail_pages[page.page_id] = page
                        else:
                            self.num_avail_blocks -= free_n
                            ret_index.extend(page.free_list)
                            remaining_need -= free_n
                            page.free_list = []
                            self.full_pages[page.page_id] = page
                finally:
                    # LEAVE critical section
                    with self.page_allocator.prealloc_lock:
                        self.page_allocator.alloc_active -= 1
                        if _is_kvcached_profile_enabled():
                            alloc_duration = time.time() - alloc_start_time
                            logger.info(
                                f"📦[ALLOC] [{time.time():.3f}] "
                                f"critical exit, allocated={len(ret_index)}, "
                                f"duration={alloc_duration*1000:.2f}ms, "
                                f"alloc_active={self.page_allocator.alloc_active}"
                            )
                        self.page_allocator.prealloc_cond.notify_all()

            # If tensor is not needed, return list directly
            if self.return_device is None:
                if _is_kvcached_profile_enabled():
                    num_avail_blocks_after = self.num_avail_blocks
                    avail_pages_after = len(self.avail_pages)
                    full_pages_after = len(self.full_pages)
                    reserved_blocks_after = len(self.reserved_blocks)
                    memory_usage_after = 0
                    if hasattr(
                            self.page_allocator, 'memory_in_use'
                    ) and self.page_allocator.memory_in_use is not None:
                        memory_usage_after = self.page_allocator.memory_in_use[
                            0]

                    msg = (
                        f"📦[KV-ALLOC-SUCCESS] [{current_time:.3f}] "
                        f"need_size={need_size}, allocated={len(ret_index)}, "
                        f"num_avail_blocks: {num_avail_blocks_before}->{num_avail_blocks_after}, "
                        f"avail_pages: {avail_pages_before}->{avail_pages_after}, "
                        f"full_pages: {full_pages_before}->{full_pages_after}, "
                        f"reserved_blocks: {reserved_blocks_before}->{reserved_blocks_after}, "
                        f"memory_usage: {memory_usage}->{memory_usage_after}")
                    logger.info(msg)
                return ret_index

            # list -> pinned CPU tensor -> async to GPU
            with nvtx_range("list->tensor/H2D"):
                idx_cpu = torch.as_tensor(ret_index, dtype=torch.int32)
                if self.return_device.startswith("cuda"):
                    if self.pin_cpu and not idx_cpu.is_pinned():
                        idx_cpu = idx_cpu.pin_memory()
                    idx_gpu = idx_cpu.to(self.return_device, non_blocking=True)
                    result = idx_gpu
                else:
                    result = idx_cpu

            if _is_kvcached_profile_enabled():
                num_avail_blocks_after = self.num_avail_blocks
                avail_pages_after = len(self.avail_pages)
                full_pages_after = len(self.full_pages)
                reserved_blocks_after = len(self.reserved_blocks)
                memory_usage_after = 0
                if hasattr(self.page_allocator, 'memory_in_use'
                           ) and self.page_allocator.memory_in_use is not None:
                    memory_usage_after = self.page_allocator.memory_in_use[0]

                msg = (
                    f"📦[KV-ALLOC-TENSOR] [{current_time:.3f}] "
                    f"need_size={need_size}, allocated={len(ret_index)}, "
                    f"num_avail_blocks: {num_avail_blocks_before}->{num_avail_blocks_after}, "
                    f"avail_pages: {avail_pages_before}->{avail_pages_after}, "
                    f"full_pages: {full_pages_before}->{full_pages_after}, "
                    f"reserved_blocks: {reserved_blocks_before}->{reserved_blocks_after}, "
                    f"memory_usage: {memory_usage}->{memory_usage_after}")
                logger.info(msg)

            return result

    def free(self, indices: Union[List[int], torch.Tensor, np.ndarray]):
        indices_len = len(indices) if hasattr(indices, '__len__') else 1

        if _is_kvcached_profile_enabled():
            current_time = time.time()
            num_avail_blocks_before = self.num_avail_blocks
            avail_pages_before = len(self.avail_pages)
            full_pages_before = len(self.full_pages)
            reserved_blocks_before = len(self.reserved_blocks)
            memory_usage_before = 0
            if hasattr(self.page_allocator, 'memory_in_use'
                       ) and self.page_allocator.memory_in_use is not None:
                memory_usage_before = self.page_allocator.memory_in_use[0]

        with nvtx_range(f"kv_free({indices_len})"):
            # --- Added: support for tensor / numpy array / iterable ---

            if isinstance(indices, torch.Tensor):
                if indices.is_cuda:
                    indices = indices.cpu()
                indices = indices.tolist()
            elif not isinstance(indices, list):
                # e.g., numpy array or other iterable
                indices = list(indices)

            idx_dict = defaultdict(list)
            for idx in indices:
                page_id = self.page_allocator.get_page_id(
                    idx, self.block_mem_size)
                idx_dict[page_id].append(idx)

            pages_to_free: List[int] = []
            for page_id, idxs in idx_dict.items():
                if (page_id not in self.full_pages
                        and page_id not in self.avail_pages):
                    warnings.warn(
                        f"Page {page_id} is not in avail_pages or full_pages, "
                        f"it is possible that the page is already freed.")
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

        if _is_kvcached_profile_enabled():
            num_avail_blocks_after = self.num_avail_blocks
            avail_pages_after = len(self.avail_pages)
            full_pages_after = len(self.full_pages)
            reserved_blocks_after = len(self.reserved_blocks)
            memory_usage_after = 0
            if hasattr(self.page_allocator, 'memory_in_use'
                       ) and self.page_allocator.memory_in_use is not None:
                memory_usage_after = self.page_allocator.memory_in_use[0]

            msg = (
                f"🗑️[KV-FREE] [{current_time:.3f}] "
                f"freed_size={indices_len}, pages_freed={len(pages_to_free)}, "
                f"num_avail_blocks: {num_avail_blocks_before}->{num_avail_blocks_after}, "
                f"avail_pages: {avail_pages_before}->{avail_pages_after}, "
                f"full_pages: {full_pages_before}->{full_pages_after}, "
                f"reserved_blocks: {reserved_blocks_before}->{reserved_blocks_after}, "
                f"memory_usage: {memory_usage_before}->{memory_usage_after}")
            logger.info(msg)

    def try_to_reserve(self, need_size: int) -> bool:
        if _is_kvcached_profile_enabled():
            current_time = time.time()
            reserved_blocks_before = len(self.reserved_blocks)
            available_before = self.available_size()

        if self.available_size() < need_size:
            if _is_kvcached_profile_enabled():
                logger.info(
                    f"🚫[TRY-RESERVE-FAIL] [{current_time:.3f}] "
                    f"need_size={need_size} > available_size={available_before}, "
                    f"reserved_blocks={reserved_blocks_before}")
            return False

        reserved = self.alloc(need_size)
        if reserved is None:
            if _is_kvcached_profile_enabled():
                logger.info(f"🚫[TRY-RESERVE-ALLOC-FAIL] [{current_time:.3f}] "
                            f"alloc({need_size}) failed, "
                            f"reserved_blocks={reserved_blocks_before}")
            return False

        if isinstance(reserved, torch.Tensor):
            reserved = reserved.cpu().tolist()
        with self.page_allocator.prealloc_lock:
            self.reserved_blocks.extend(reserved)

        if _is_kvcached_profile_enabled():
            reserved_blocks_after = len(self.reserved_blocks)
            available_after = self.available_size()
            logger.info(
                f"✅[TRY-RESERVE-SUCCESS] [{current_time:.3f}] "
                f"need_size={need_size}, allocated={len(reserved)}, "
                f"reserved_blocks: {reserved_blocks_before}->{reserved_blocks_after}, "
                f"available_size: {available_before}->{available_after}")
        return True

    def free_reserved(self):
        if _is_kvcached_profile_enabled():
            current_time = time.time()
            reserved_blocks_before = len(self.reserved_blocks)

        if self.reserved_blocks:
            self.free(self.reserved_blocks)
            self.reserved_blocks = []

            if _is_kvcached_profile_enabled():
                logger.info(f"🗑️[FREE-RESERVED] [{current_time:.3f}] "
                            f"freed_blocks={reserved_blocks_before}, "
                            f"reserved_blocks: {reserved_blocks_before}->0")
        else:
            if _is_kvcached_profile_enabled():
                logger.info(f"ℹ️[FREE-RESERVED-EMPTY] [{current_time:.3f}] "
                            f"no reserved blocks to free")

    def resize(self, new_num_blocks: int):
        if _is_kvcached_profile_enabled():
            current_time = time.time()
            old_num_blocks = self.num_blocks
            reserved_blocks_before = len(self.reserved_blocks)
            in_shrink_before = self.in_shrink

        new_mem_size = new_num_blocks * self.block_mem_size
        if self.page_allocator.resize(new_mem_size):
            if self.in_shrink:
                self.in_shrink = False
                self.target_num_blocks = None

            if _is_kvcached_profile_enabled():
                logger.info(f"✅[RESIZE-SUCCESS] [{current_time:.3f}] "
                            f"blocks: {old_num_blocks}->{new_num_blocks}, "
                            f"in_shrink: {in_shrink_before}->False, "
                            f"reserved_blocks={reserved_blocks_before}")
            return True  # Successfully resized.

        # Failed to resize due to too many in-use blocks.
        assert (len(self.reserved_blocks) == 0), \
            "Reserved blocks must be freed before resizing."
        self.in_shrink = True
        self.target_num_blocks = new_num_blocks
        self.free_reserved()

        if _is_kvcached_profile_enabled():
            logger.info(f"❌[RESIZE-FAIL] [{current_time:.3f}] "
                        f"blocks: {old_num_blocks}->{new_num_blocks}, "
                        f"in_shrink: {in_shrink_before}->True, "
                        f"target_num_blocks={new_num_blocks}, "
                        f"freed_reserved_blocks={reserved_blocks_before}")
        return False

    def trim(self):
        self.page_allocator.trim()

    def available_size(self) -> int:
        if _is_kvcached_profile_enabled():
            current_time = time.time()

        avail_size = self.num_avail_blocks + len(self.reserved_blocks)
        if self.in_shrink:
            free_size = 0
            virtual_free_size = 0
            physical_free_size = 0
        else:
            virtual_free_size = self.page_allocator.get_num_free_blocks(
                self.block_mem_size)
            # physical_free_size = self._physical_free_size()
            physical_free_size = float('inf')
            free_size = min(virtual_free_size, physical_free_size)

        total_available = avail_size + free_size

        # More frequent logging of key metric changes (log every 1000 calls)
        if _is_kvcached_profile_enabled():
            memory_usage = 0
            if hasattr(self.page_allocator, 'memory_in_use'
                       ) and self.page_allocator.memory_in_use is not None:
                memory_usage = self.page_allocator.memory_in_use[0]

            # Record intermediate calculation details
            if not hasattr(self, '_available_size_call_count'):
                self._available_size_call_count = 0
            self._available_size_call_count += 1

            # Log detailed status every 1000 calls, including intermediate calculations
            if self._available_size_call_count % 1000 == 0:
                free_pages = len(self.page_allocator.free_page_list)
                reserved_pages = len(self.page_allocator.reserved_page_list)
                msg = (
                    f"📊[AVAILABLE-SIZE-DETAIL] [{current_time:.3f}] "
                    f"total_available={total_available}, "
                    f"avail_size={avail_size}(num_avail_blocks={self.num_avail_blocks}+"
                    f"reserved_blocks={len(self.reserved_blocks)}), "
                    f"free_size={free_size}(virtual={virtual_free_size},"
                    f"physical={physical_free_size}), "
                    f"in_shrink={self.in_shrink}, "
                    f"avail_pages={len(self.avail_pages)}, "
                    f"full_pages={len(self.full_pages)}, "
                    f"free_pages={free_pages}, "
                    f"reserved_pages={reserved_pages}, "
                    f"memory_in_use={memory_usage}")
                logger.info(msg)

        return total_available

    def log_memory_usage_detailed(self):
        """Detailed logging of current memory usage for external calls"""
        if _is_kvcached_profile_enabled():
            current_time = time.time()

            # KVCacheManager level status
            free_pages = len(self.page_allocator.free_page_list)
            reserved_pages = len(self.page_allocator.reserved_page_list)

            # Memory usage information
            memory_usage = 0
            if hasattr(self.page_allocator, 'memory_in_use'
                       ) and self.page_allocator.memory_in_use is not None:
                memory_usage = self.page_allocator.memory_in_use[0]

            msg = (f"🔍[DETAILED-MEMORY] [{current_time:.3f}] "
                   f"num_avail_blocks={self.num_avail_blocks}, "
                   f"len(avail_pages)={len(self.avail_pages)}, "
                   f"len(full_pages)={len(self.full_pages)}, "
                   f"len(reserved_blocks)={len(self.reserved_blocks)}, "
                   f"len(free_page_list)={free_pages}, "
                   f"len(reserved_page_list)={reserved_pages}, "
                   f"prealloc_needed={self.page_allocator.prealloc_needed}, "
                   f"prealloc_running={self.page_allocator.prealloc_running}, "
                   f"memory_in_use={memory_usage}")
            logger.info(msg)

    def get_mapped_memory_size(self, unit='bytes') -> float:
        """Get memory usage in specified unit (bytes, kb, mb, gb)."""
        memory_bytes = ((self.page_allocator.get_num_inuse_pages() +
                         self.page_allocator.get_num_reserved_pages()) *
                        self.num_layers * PAGE_SIZE * 2)

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
        avail_phy_blocks = ((avail_phy_pages // self.num_layers // 2) *
                            (PAGE_SIZE // self.block_mem_size))
        return avail_phy_blocks

    def clear(self):
        raise NotImplementedError
