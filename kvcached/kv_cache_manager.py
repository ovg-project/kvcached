# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
kvcached Memory Manager

This module implements a hierarchical memory management system for KV cache:
- Pages: Large memory chunks (e.g., 2MB) that are mapped/unmapped to physical memory
- Blocks: Smaller units within pages that are allocated to store KV cache data
"""

from __future__ import annotations

import functools
import threading
import time
from typing import Any, Dict, List, Optional

from kvcached.locks import NoOpLock
from kvcached.tp_ipc_util import broadcast_kv_tensors_created
from kvcached.utils import (
    CONTIGUOUS_LAYOUT,
    DEFAULT_IPC_NAME,
    PAGE_PREALLOC_ENABLED,
    PAGE_SIZE,
    SANITY_CHECK,
    KVCachedConfigError,
    get_kvcached_logger,
)
from kvcached.vmm_ops import kv_tensors_created

try:
    import kvcached.vmm_ops as kvcached_cpp
    PageAllocator = kvcached_cpp.PageAllocator
    InternalPage: Any = kvcached_cpp.InternalPage
except ImportError as e:
    raise ImportError(f"Failed to import kvcached.vmm_ops. Please ensure the C++ extension is built properly. err: {e}")

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
        world_size: int = 1,
        pp_rank: int = 0,
        async_sched: bool = False,
        reserve_null_block: bool = False,
        num_kv_buffers: int = 2,
        group_id: int = 0,
        pool_name: Optional[str] = None,
    ):
        """
        Args:
            num_blocks: Number of blocks.
            block_size: Size of each block in bytes.
            cell_size: Size of each cell in bytes.
            num_layers: Number of layers.
            world_size: Tensor parallel world size within a pipeline stage.
            async_sched: Whether asynchronous scheduling is enabled.
            reserve_null_block: Whether to reserve the first block as null block
                for padding tokens. This is required by SGLang which assumes the
                first block is always reserved as padded tokens.
            num_kv_buffers: Number of KV buffers per layer (2 for MHA K+V,
                1 for MLA combined KV).
            group_id: KV cache group identifier for hybrid attention models.
                Different groups have independent FTensors and page spaces.
            pool_name: Stable, low-cardinality name assigned by the engine
                integration when this pool is created.
        """
        self.num_blocks = num_blocks
        self.block_mem_size = block_size * cell_size
        self.num_layers = num_layers
        self.num_kv_buffers = num_kv_buffers
        self.reserve_null_block = reserve_null_block
        self.group_id = group_id
        self._pool_name = pool_name

        # The physical page size used by kvcached page allocator.
        self.page_size = PAGE_SIZE
        # A block must fit within a single page; otherwise a page holds zero
        # usable blocks, the KV pool is permanently empty, and the engine
        # silently deadlocks during warmup (available_size() stays 0). This
        # happens with hybrid linear-attention models (e.g. Qwen3.5/3.6 GDN,
        # Mamba) whose per-block recurrent state exceeds the default 2MB page.
        # Fail loudly here with the exact page size needed instead of hanging.
        if self.block_mem_size > self.page_size:
            base = 2 * 1024 * 1024  # KVCACHED_PAGE_SIZE_MB granularity
            min_page_mb = ((self.block_mem_size + base - 1) // base) * 2
            raise KVCachedConfigError(
                f"kvcached KV block size ({self.block_mem_size} bytes, "
                f"{self.block_mem_size / (1024 * 1024):.2f} MiB) is larger than the "
                f"page size ({self.page_size} bytes, "
                f"{self.page_size // (1024 * 1024)} MiB), so no block fits in a "
                f"page and the KV pool would be empty. This typically happens "
                f"with hybrid linear-attention models (e.g. Qwen3.5/3.6 GDN, "
                f"Mamba) whose per-block state is large. Re-launch with "
                f"KVCACHED_PAGE_SIZE_MB={min_page_mb} (or larger; must be a "
                f"multiple of 2).")
        # NOTE: this is the memory size of the K or V tensor in one layer
        self.mem_size = self.num_blocks * self.block_mem_size
        self.world_size = world_size
        self.pp_rank = pp_rank
        self.page_allocator = PageAllocator(
            self.num_layers,
            self.mem_size,
            self.page_size,
            self.world_size,
            pp_rank=self.pp_rank,
            async_sched=async_sched,
            contiguous_layout=CONTIGUOUS_LAYOUT,
            enable_page_prealloc=PAGE_PREALLOC_ENABLED,
            num_kv_buffers=self.num_kv_buffers,
            group_id=self.group_id,
            ipc_name=DEFAULT_IPC_NAME,
        )
        # Tell the C++ PageAllocator whether map/unmap must be broadcast to
        # worker processes over IPC, even with world_size == 1 (e.g. vLLM V1
        # EngineCore + worker in separate processes). The value is pushed once
        # rather than pulled through a Python callback: a callback would make
        # the C++ prealloc thread re-enter Python, which needs the GIL and
        # deadlocks against a caller blocked inside a binding (issue #371).
        # The decision is stable by this point -- the same-process worker flip
        # (init_kvcached(is_worker=True)) happens before KVCacheManager is
        # constructed in every integration flow.
        try:
            from kvcached.integration.vllm.interfaces import should_use_worker_ipc
            use_worker_ipc = should_use_worker_ipc()
        except ImportError:
            use_worker_ipc = False
        self.page_allocator.set_use_worker_ipc(use_worker_ipc)

        if self.world_size > 1 or use_worker_ipc:
            try:
                from kvcached.tp_ipc_util import (
                    broadcast_map_to_kv_tensors,
                    broadcast_unmap_from_kv_tensors,
                )

                # Wrap Python functions to match C++ callback signature
                def map_callback(world_size: int, offsets: List[int], pp_rank: int = 0, group_id: int = 0) -> None:
                    """Wrapper for Python broadcast function"""
                    broadcast_map_to_kv_tensors(world_size, offsets, pp_rank, group_id)

                def unmap_callback(world_size: int, offsets: List[int]) -> None:
                    """Wrapper for Python broadcast function"""
                    broadcast_unmap_from_kv_tensors(world_size, offsets, pp_rank, group_id)

                # Set the callbacks in the PageAllocator
                self.page_allocator.set_broadcast_map_callback(map_callback)
                self.page_allocator.set_broadcast_unmap_callback(unmap_callback)

                logger.info("Set up broadcast callbacks for multi-process (world_size=%d, use_worker_ipc=%s)",
                            self.world_size, use_worker_ipc)
            except ImportError as e:
                logger.warning("Failed to import tp_ipc_util module: %s. Broadcast callbacks will not be available.", e)
            except Exception as e:
                logger.warning("Failed to set up broadcast callbacks: %s. Falling back to single-process mode.", e)

        self.num_avail_blocks = 0  # Only count free blocks in avail_pages
        self.avail_pages: Dict[int, InternalPage] = {}
        self.full_pages: Dict[int, InternalPage] = {}

        self.reserved_blocks: List[int] = []
        self.null_block: Optional[list[int]] = None

        self.in_shrink: bool = False
        self.target_num_blocks: Optional[int] = None
        self._memory_limit_bytes: Optional[int] = None
        self._memory_limit_effective_bytes: Optional[int] = None
        self._memory_limit_revision = -1
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
            try:
                from kvcached.integration.vllm.interfaces import should_use_worker_ipc
                vllm_remote = should_use_worker_ipc()
            except ImportError:
                vllm_remote = False

            if self.world_size > 1 or vllm_remote:
                return broadcast_kv_tensors_created(
                    self.world_size, self.pp_rank,
                    group_id=self.group_id)
            else:
                return kv_tensors_created(group_id=self.group_id)

        try:
            total_wait = 0.0
            while not _check_kv_tensors_created():
                if total_wait >= KV_TENSOR_WAIT_TIMEOUT:
                    raise TimeoutError("KV tensors not created after "
                                       f"{KV_TENSOR_WAIT_TIMEOUT} seconds")
                time.sleep(0.001)  # 1ms
                total_wait += 0.001
            # KV tensors created now
            # Possibly reserve the first block as null block for padding tokens
            self._reserve_null_block()

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

    def _reserve_null_block(self) -> None:
        """
        Reserve the first block as null block for padding tokens.
        """
        if not self.reserve_null_block:
            self.null_block = None
            return

        wait_started = time.monotonic()
        last_log_at = 0.0
        last_reason: Optional[str] = None
        loop_count = 0
        alloc_attempts = 0

        def _log_wait(reason: str, available_before: int) -> None:
            nonlocal last_log_at, last_reason

            now = time.monotonic()
            if reason == last_reason and now - last_log_at < 10.0:
                return

            last_log_at = now
            last_reason = reason
            try:
                allocator = self.page_allocator
                allocator_state = (
                    f"available_before={available_before}, "
                    f"available_now={self.available_size()}, "
                    f"num_avail_blocks={self.num_avail_blocks}, "
                    f"reserved_blocks={len(self.reserved_blocks)}, "
                    f"avail_pages={len(self.avail_pages)}, "
                    "virtual_free_pages="
                    f"{allocator.get_num_free_pages()}, "
                    "physical_free_pages="
                    f"{allocator.get_avail_physical_pages()}, "
                    "reserved_physical_pages="
                    f"{allocator.get_num_reserved_pages()}"
                )
            except Exception as exc:
                # Diagnostics must never turn a recoverable capacity wait into
                # a startup failure.
                allocator_state = (
                    f"available_before={available_before}, "
                    f"state_snapshot_error={exc!r}"
                )

            logger.warning(
                "Waiting for physical KV capacity to reserve null block: "
                f"reason={reason}, elapsed={now - wait_started:.1f}s, "
                f"loops={loop_count}, alloc_attempts={alloc_attempts}, "
                f"group_id={getattr(self, 'group_id', None)}, "
                f"pp_rank={getattr(self, 'pp_rank', None)}, "
                f"{allocator_state}")

        while True:
            loop_count += 1
            available_before = self.available_size()
            if available_before < 1:
                _log_wait("no_effective_capacity", available_before)
                time.sleep(0.01)
                continue

            alloc_attempts += 1
            null_block = self._alloc(1, _skip_wait=True)
            if null_block is None:
                # Another colocated engine may consume the last physical page
                # between available_size() and alloc_page(), or alloc_page()
                # may fail after the capacity check. Keep the engine alive and
                # make the stalled startup state visible in pod logs.
                _log_wait("alloc_returned_none_after_capacity_check",
                          available_before)
                time.sleep(0.01)
                continue
            if null_block != [0]:
                logger.error(f"Failed to reserve null block, got {null_block}")
                raise RuntimeError("Failed to reserve null block at index 0")

            self.null_block = null_block
            elapsed = time.monotonic() - wait_started
            if loop_count > 1:
                logger.info(
                    "Reserved null block after waiting for physical KV "
                    f"capacity: elapsed={elapsed:.1f}s, loops={loop_count}, "
                    f"alloc_attempts={alloc_attempts}")
            return


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

        new_mem_size = self.page_allocator.get_resize_target()
        if new_mem_size > 0:
            self.resize(new_mem_size)

        if self.available_size() < need_size:
            logger.warning(f"available_size()={self.available_size()} < "
                           f"need_size={need_size}")
            return None

        ret_index = []
        page: Optional[InternalPage] = None

        remaining_need = need_size

        num_from_reserved = 0
        if self.reserved_blocks:  # Try to allocate from reserved blocks first
            num_from_reserved = min(len(self.reserved_blocks), remaining_need)
            # ret_index is empty before so we directly assign it
            ret_index = self.reserved_blocks[:num_from_reserved]
            self.reserved_blocks = self.reserved_blocks[num_from_reserved:]
            remaining_need -= num_from_reserved

        while remaining_need > 0:  # Allocate the remaining blocks from pages
            if not self.avail_pages:
                # Only alloc_page() is recoverable here. available_size() saw
                # enough logical capacity, but another instance sharing the
                # physical pool consumed pages before we reached this call, so
                # roll back and report an allocation miss instead of leaking
                # blocks or crashing. Anything else raising in this loop is an
                # invariant failure (e.g. InternalPage.alloc()'s "Not enough
                # free blocks in page", raised after _pick_avail_page() has
                # already removed the page from avail_pages and before its
                # blocks reach ret_index, which rollback therefore cannot
                # restore) and must stay fail-loud.
                try:
                    page = self.page_allocator.alloc_page()
                    page.init(self.block_mem_size)
                except RuntimeError as e:
                    self._rollback_partial_alloc(ret_index, num_from_reserved)
                    logger.warning(
                        f"alloc_page() failed after partially allocating "
                        f"{len(ret_index)}/{need_size} blocks; rolled back: {e}")
                    return None
                # A page may have zero usable blocks when block_mem_size is
                # large (e.g. HYBRID_LINEAR) and every aligned block would
                # straddle the page boundary. Park it in full_pages so it's
                # not re-handed-out but stays lookupable by free().
                if page.num_free_blocks() == 0:
                    self.full_pages[page.page_id] = page
                    continue
                self.num_avail_blocks += page.num_free_blocks()
            else:
                page = self._pick_avail_page(remaining_need)
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

    def _rollback_partial_alloc(self, ret_index: List[int],
                                num_from_reserved: int) -> None:
        """Return partially allocated blocks after a mid-alloc failure.

        The first ``num_from_reserved`` entries of ``ret_index`` came off the
        reservation ledger and are prepended back onto it; the rest came from
        pages and go back through the regular free() path (safe to call here:
        the lock is re-entrant).
        """
        page_blocks = ret_index[num_from_reserved:]
        if page_blocks:
            self.free(page_blocks)
        reserved_blocks = ret_index[:num_from_reserved]
        if reserved_blocks:
            self.reserved_blocks = reserved_blocks + self.reserved_blocks

    def _pick_avail_page(self, remaining_need: int) -> InternalPage:
        """Pick the available page this allocation fits into best.

        `avail_pages.popitem()` hands back the most recently touched page, and
        a partially consumed page is re-inserted at the tail, so allocation
        drains one page dry before moving on. That is fine for a single block,
        but a long prefill run then walks whatever partly-filled pages happen
        to sit at the tail and smears itself over many of them. Since a page
        only returns physical memory once every block on it is free, a request
        whose blocks are spread over a dozen pages pins all twelve for as long
        as any one of those blocks survives in the prefix cache.

        Choosing the smallest page that still holds the whole remaining run
        (and the emptiest page when none does, so the next bite is as big as
        possible) keeps a request's blocks together instead.

        This is O(len(avail_pages)) and runs once per page consumed, not per
        block; measured at ~7us for 100 available pages, with no throughput
        change on a 96-way serving workload. Bucketing pages by free-block
        count would make it independent of pool size if that ever matters.
        """
        best_id: Optional[int] = None
        best_free: Optional[int] = None
        fallback_id: Optional[int] = None
        fallback_free = -1
        for page_id, page in self.avail_pages.items():
            free = page.num_free_blocks()
            if free >= remaining_need:
                if best_free is None or free < best_free:
                    best_id, best_free = page_id, free
            elif free > fallback_free:
                fallback_id, fallback_free = page_id, free
        chosen = best_id if best_id is not None else fallback_id
        assert chosen is not None, "caller guarantees avail_pages is non-empty"
        return self.avail_pages.pop(chosen)

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

        idx_dict = self.page_allocator.group_indices_by_page(indices, self.block_mem_size)

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
            # Detach first: once off the ledger the blocks are ordinary
            # blocks, so free()'s sanity check (which forbids freeing a
            # block that is still reserved) correctly stays quiet. Freeing
            # before detaching trips that check under KVCACHED_SANITY_CHECK.
            blocks, self.reserved_blocks = self.reserved_blocks, []
            self.free(blocks)

    @synchronized
    def resize(self, new_mem_size: int):
        """
        Reset the limit of the K or V tensor in one layer.
        new_mem_size: the memory size of the K or V tensor in one layer
        """
        self._wait_post_init()
        assert new_mem_size >= 0, "new_mem_size must be non-negative"
        if self.page_allocator.resize(new_mem_size):
            if self.in_shrink:
                self.in_shrink = False
                self.target_num_blocks = None
            return True  # Successfully resized.
        # Failed to resize due to too many in-use blocks. Free any
        # outstanding reserved blocks before entering the lazy shrink wait.
        # NOTE: we can support resizing with reserved blocks, but we want to
        # enforce this check for now to ensure correctness.
        self.free_reserved()
        assert (len(self.reserved_blocks) == 0
                ), "Reserved blocks must be freed before resizing."
        self.in_shrink = True
        self.target_num_blocks = new_mem_size // self.block_mem_size
        return False

    @synchronized
    def trim(self) -> None:
        """
        Trim the reserved pages to free up physical memory.
        """
        self._wait_post_init()
        self.page_allocator.trim()

    @synchronized
    def set_memory_limit(
        self,
        limit_bytes: int,
        *,
        revision: int,
    ) -> Dict[str, Any]:
        """Apply a revisioned memory limit through the existing resize path."""
        limit_bytes = int(limit_bytes)
        revision = int(revision)
        if limit_bytes < 0:
            raise ValueError("limit_bytes must be non-negative")
        if revision < 0:
            raise ValueError("revision must be non-negative")

        current_revision = self._memory_limit_revision
        current_limit = self._memory_limit_bytes
        if revision < current_revision:
            return self._memory_limit_state(status="stale")
        if revision == current_revision and current_limit == limit_bytes:
            return self._memory_limit_state()
        if revision == current_revision:
            return self._memory_limit_state(status="conflict")

        page_bundle_bytes = self._memory_limit_page_bundle_bytes()
        max_pages = self.mem_size // self.page_size
        target_pages = min(limit_bytes // page_bundle_bytes, max_pages)
        effective_limit_bytes = target_pages * page_bundle_bytes

        self.resize(target_pages * self.page_size)
        self._memory_limit_bytes = limit_bytes
        self._memory_limit_effective_bytes = effective_limit_bytes
        self._memory_limit_revision = revision
        return self._memory_limit_state()

    def _memory_limit_page_bundle_bytes(self) -> int:
        return self.page_size * self.num_layers * self.num_kv_buffers

    def _memory_limit_state(
        self,
        *,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        page_state = self.page_allocator.get_page_state()
        page_bundle_bytes = self._memory_limit_page_bundle_bytes()
        mapped_pages = (int(page_state["inuse_pages"])
                        + int(page_state["reserved_pages"]))
        mapped_bytes = mapped_pages * page_bundle_bytes
        effective_limit_bytes = self._memory_limit_effective_bytes
        if status is None:
            status = "deferred" if self.in_shrink else "applied"
        return {
            "status": status,
            "pool_name": str(self.pool_name or ""),
            "group_id": self.group_id,
            "limit_bytes": self._memory_limit_bytes,
            "effective_limit_bytes": effective_limit_bytes,
            "current_capacity_bytes": (
                int(page_state["total_pages"]) * page_bundle_bytes
            ),
            "revision": self._memory_limit_revision,
            "mapped_bytes": mapped_bytes,
            "remaining_bytes": (
                None if effective_limit_bytes is None else
                max(0, effective_limit_bytes - mapped_bytes)
            ),
            "overage_bytes": (
                0 if effective_limit_bytes is None else
                max(0, mapped_bytes - effective_limit_bytes)
            ),
            "reason": {
                "deferred": "inuse_capacity_above_limit",
                "conflict": "revision_reused_with_different_limit",
            }.get(status, ""),
        }

    @synchronized
    def memory_limit_state(self) -> Dict[str, Any]:
        """Return the current revisioned resize limit and apply state."""
        return self._memory_limit_state()

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
            blocks_from_free_pages = free_pages * InternalPage.get_num_blocks(
                self.page_size, self.block_mem_size)
        return avail_blocks + blocks_from_free_pages

    @synchronized
    def get_page_occupancy(self, page_ids: List[int]) -> Dict[int, int]:
        """Return the number of allocated blocks on each of `page_ids`.

        A page only returns physical memory once every block on it is freed, so
        callers choosing which blocks to evict need to know how many blocks a
        page still holds. Page ids the manager does not track report 0.
        """
        occupancy: Dict[int, int] = {}
        for page_id in page_ids:
            if page_id in self.full_pages:
                page = self.full_pages[page_id]
            elif page_id in self.avail_pages:
                page = self.avail_pages[page_id]
            else:
                occupancy[page_id] = 0
                continue
            # Blocks straddling a page boundary belong to neither page, so a
            # page's capacity comes from its own block range rather than from
            # page_size // block_mem_size.
            start, end = InternalPage.get_block_range(page_id, self.page_size,
                                                      self.block_mem_size)
            occupancy[page_id] = (end - start) - page.num_free_blocks()
        return occupancy

    @synchronized
    def get_mapped_memory_size(self, unit='bytes') -> float:
        """Get memory usage in specified unit (bytes, kb, mb, gb)."""
        memory_bytes = (self.page_allocator.get_num_inuse_pages() *
                        self.num_layers * self.page_size *
                        self.num_kv_buffers)

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

    @property
    def pool_name(self) -> Optional[str]:
        """Return the stable name assigned when this pool was created."""
        return self._pool_name

    @synchronized
    def observability_snapshot(self, *, integration=None):
        """Return a read-only snapshot of this KV cache pool."""
        from kvcached.observability import build_kv_cache_pool_snapshot
        return build_kv_cache_pool_snapshot(
            self,
            integration=integration,
        )

    @synchronized
    def observability_snapshot_dict(self, *, integration=None):
        """Return a JSON-serializable read-only snapshot of this KV cache pool."""
        return self.observability_snapshot(
            integration=integration,
        ).to_dict()

    @synchronized
    def clear(self):
        """
        Free all allocated blocks and reset the allocator to initial state.
        """

        self._wait_post_init()

        # Stop the prealloc thread first — it runs on the PageAllocator's
        # lock and can grab pages between our trim/reset/reserve steps,
        # causing the null-block reservation to get a non-zero block.
        # (This used to call a `_stop_prealloc_thread(timeout=...)` binding
        # that never existed, so clear() raised AttributeError.)
        self.page_allocator.stop_prealloc_thread()

        # Clear reserved blocks
        self.free_reserved()

        # Free all blocks from avail_pages and full_pages
        pages_to_free: List[int] = []
        for page in self.avail_pages.values():
            pages_to_free.append(page.page_id)
        for page in self.full_pages.values():
            pages_to_free.append(page.page_id)
        if pages_to_free:
            self.page_allocator.free_pages(pages_to_free)
        self.avail_pages.clear()
        self.full_pages.clear()

        # Trim the page allocator to free up reserved pages
        self.trim()

        # Reset the page allocator's free list to its original sorted order.
        # After free_pages + trim, freed pages are appended to the END of
        # free_page_list, so the order is scrambled.  This matters because
        # _reserve_null_block() pops from the LEFT and expects to get page 0
        # (which yields block 0 — the null block SGLang requires).
        self.page_allocator.reset_free_page_order()

        self.target_num_blocks = None
        self.in_shrink = False
        self.num_avail_blocks = 0

        # Possibly reserve the first block as null block for padding tokens
        self._reserve_null_block()

        # Restart the prealloc thread now that null block is safely reserved.
        self.page_allocator.start_prealloc_thread()

    # Private methods
    @synchronized
    def _get_num_alloced_blocks(self) -> int:
        """Return how many blocks are currently handed out of their pages.

        Reserved blocks (``self.reserved_blocks``) are part of this count:
        try_to_reserve() obtains them via alloc(), so they have already left
        their pages. They are deliberately NOT added a second time below.
        """
        # Blocks from fully allocated pages
        blocks_from_full_pages = len(self.full_pages) * InternalPage.get_num_blocks(
            self.page_size, self.block_mem_size)
        # Blocks from partially allocated pages. num_avail_blocks is the number
        # of free blocks in the partially allocated pages so the number of
        # allocated blocks is the total number of blocks in the partially
        # allocated pages minus the number of free blocks.
        blocks_from_avail_pages = len(self.avail_pages) * InternalPage.get_num_blocks(
            self.page_size, self.block_mem_size) - self.num_avail_blocks
        return blocks_from_full_pages + blocks_from_avail_pages
