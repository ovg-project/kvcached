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
        error_msg = (f"❌ [NVTX-ERROR] NVTX not available but trying to use "
                     f"range '{name}'")
        print(error_msg)
        raise RuntimeError(error_msg + " - Please ensure torch.cuda.nvtx is "
                           "properly installed")

    cuda_sync_if_enabled()  # Sync before starting NVTX range
    _nvtx.range_push(name)
    try:
        yield
    finally:
        _nvtx.range_pop()
        cuda_sync_if_enabled()  # Sync after ending NVTX range


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
        self.timings.append(time.time() - self.start)


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
        self.min_reserved_pages = 100
        self.max_reserved_pages = 200
        self.reserved_page_list: List[int] = []
        self.reclaimed_page_list: List[int] = []

        # Preallocation thread management
        self.prealloc_lock = threading.Lock()
        self.prealloc_cond = threading.Condition(self.prealloc_lock)
        self.prealloc_running = False
        self.prealloc_needed = False
        self.prealloc_thd = None

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
        """Preallocates pages and maps them to physical memory."""
        while self.prealloc_running:
            with self.prealloc_lock:
                # Wait until preallocation is needed or thread is stopped
                while (not self.prealloc_needed and self.prealloc_running):
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

                # Map pages to physical memory (outside lock)
                if pages_to_reserve:
                    try:
                        # Fix: use same offset calculation as original
                        map_to_kv_tensors(
                            [pid * self.page_size for pid in pages_to_reserve])
                        if self.shm is not None:
                            with Timer(self.write_shm_times):
                                self.memory_in_use[0] += (
                                    len(pages_to_reserve) *
                                    self.page_size_all_layers)

                        self.reserved_page_list.extend(pages_to_reserve)
                    except Exception as e:
                        # If mapping fails, return pages to free list
                        self.free_page_list.extendleft(pages_to_reserve)
                        print(f"Failed to preallocate "
                              f"{len(pages_to_reserve)} pages: {e}")

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
            self.prealloc_needed = True
            self.prealloc_cond.notify()

    def alloc_page(self) -> Page:
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

        if self.shm is not None:
            with Timer(self.write_shm_times):
                self.memory_in_use[0] += self.page_size_all_layers

        # Trigger preallocation to refill the pool
        self._trigger_preallocation()

        return page

    def free_page(self, page: Page) -> None:
        page_id = page.page_id

        # Check if we can add to reserved pool
        added_to_reserved = False
        with self.prealloc_lock:
            if len(self.reserved_page_list) < self.max_reserved_pages:
                # Fast path: add to reserved pool (keep mapped)
                self.reserved_page_list.append(page_id)
                added_to_reserved = True

        if added_to_reserved:
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
                    slab_offsets = [page_id * self.page_size_all_layers]
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
        return len(self.free_page_list)

    def get_num_inuse_pages(self) -> int:
        return (self.num_total_pages - self.get_num_free_pages() -
                len(self.reserved_page_list))

    def get_num_total_pages(self) -> int:
        return self.num_total_pages

    def get_num_reserved_pages(self) -> int:
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

        # 提前预留所有可用页面以优化性能
        available = self.available_size()
        if available > 0:
            reserved_success = self.try_to_reserve(available)
            print(f"Pre-reserved {len(self.reserved_blocks)} blocks out of "
                  f"{available} available blocks. Success: {reserved_success}")

    def alloc(self, need_size: int) -> Union[List[int], torch.Tensor]:
        with nvtx_range(f"kv_alloc_total({need_size})"):
            if need_size == 0:
                if self.return_device:
                    dev = self.return_device
                    device_str = dev if dev.startswith("cuda") else "cpu"
                    return torch.empty(0, dtype=torch.int32, device=device_str)
                return []

            if self.available_size() < need_size:
                return None

            ret_index: List[int] = []
            remaining_need = need_size

            # fast path: reserved_blocks
            with nvtx_range("alloc_reserved_fastpath"):
                if self.reserved_blocks:
                    if len(self.reserved_blocks) >= remaining_need:
                        ret_index = self.reserved_blocks[:remaining_need]
                        self.reserved_blocks = self.reserved_blocks[
                            remaining_need:]
                        remaining_need = 0
                    else:
                        ret_index = self.reserved_blocks
                        remaining_need -= len(self.reserved_blocks)
                        self.reserved_blocks = []

            with nvtx_range("alloc_page_loop"):
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

            # If tensor is not needed, return list directly
            if self.return_device is None:
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

            return result

    def free(self, indices: Union[List[int], torch.Tensor, np.ndarray]):
        indices_len = len(indices) if hasattr(indices, '__len__') else 1
        with nvtx_range(f"kv_free({indices_len})"):
            # --- Added: support for tensor / numpy array / iterable ---
            if isinstance(indices, torch.Tensor):
                if indices.is_cuda:
                    indices = indices.cpu()
                indices = indices.tolist()
            elif not isinstance(indices, list):
                # e.g., numpy array or other iterable
                indices = list(indices)

            if self.reserved_blocks:
                for idx in indices:
                    if idx in self.reserved_blocks:
                        self.reserved_blocks.remove(idx)

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

    def try_to_reserve(self, need_size: int) -> bool:
        if self.available_size() < need_size:
            return False
        reserved = self.alloc(need_size)
        if reserved is None:
            return False
        if isinstance(reserved, torch.Tensor):
            reserved = reserved.cpu().tolist()
        self.reserved_blocks.extend(reserved)
        return True

    def free_reserved(self):
        if self.reserved_blocks:
            self.free(self.reserved_blocks)
            self.reserved_blocks = []

    def resize(self, new_num_blocks: int):
        new_mem_size = new_num_blocks * self.block_mem_size
        if self.page_allocator.resize(new_mem_size):
            if self.in_shrink:
                self.in_shrink = False
                self.target_num_blocks = None
            return True  # Successfully resized.
        # Failed to resize due to too many in-use blocks.
        assert (len(self.reserved_blocks) == 0), \
            "Reserved blocks must be freed before resizing."
        self.in_shrink = True
        self.target_num_blocks = new_num_blocks
        self.free_reserved()
        return False

    def trim(self):
        self.page_allocator.trim()

    def available_size(self) -> int:
        avail_size = self.num_avail_blocks + len(self.reserved_blocks)
        if self.in_shrink:
            free_size = 0
        else:
            virtual_free_size = self.page_allocator.get_num_free_blocks(
                self.block_mem_size)
            # physical_free_size = self._physical_free_size()
            physical_free_size = float('inf')
            free_size = min(virtual_free_size, physical_free_size)
        return avail_size + free_size

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
