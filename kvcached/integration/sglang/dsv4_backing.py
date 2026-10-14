# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Backing leases for native SGLang SWA and full/C128 page IDs.

The manager uses one block per VMM page. Native token pages can share or
straddle those pages; neither their IDs nor SGLang's mappings are rewritten.
"""

import functools
import inspect
from collections import Counter
from typing import Any


class SWABacking:
    def __init__(self, manager: Any, *, native_pages: int, native_page_bytes: int,
                 physical_page_bytes: int) -> None:
        if min(native_pages, native_page_bytes, physical_page_bytes) <= 0:
            raise ValueError("SWA backing dimensions must be positive")
        if native_page_bytes > physical_page_bytes:
            raise ValueError("The reserved null page must cover native page zero")
        self.manager = manager
        self.native_pages = native_pages
        self.native_page_bytes = native_page_bytes
        self.physical_page_bytes = physical_page_bytes
        self.live: set[int] = set()
        self.refs: Counter[int] = Counter()

    def _span(self, page: int) -> range:
        start = page * self.native_page_bytes // self.physical_page_bytes
        end = ((page + 1) * self.native_page_bytes - 1) // self.physical_page_bytes
        # Manager block zero permanently backs padding, including any live
        # native pages sharing that physical page.
        return range(max(1, start), end + 1)

    def acquire(self, pages: set[int]) -> bool:
        if any(page <= 0 or page >= self.native_pages for page in pages):
            raise ValueError("Native SWA page is outside its backing reservation")
        # Even an empty physical claim can rely on the permanent null page.
        # Its asynchronous reservation must settle before native IDs are published.
        self.manager.wait_ready()
        additions = pages - self.live
        refs = Counter(physical for page in additions for physical in self._span(page))
        missing = sorted(refs.keys() - self.refs.keys())
        if missing and self.manager.alloc_exact(missing) is None:
            return False
        self.refs.update(refs)
        self.live.update(additions)
        return True

    def release(self, pages: set[int]) -> None:
        removals = pages & self.live
        refs = Counter(physical for page in removals for physical in self._span(page))
        unused = sorted(physical for physical, count in refs.items()
                        if self.refs[physical] == count)
        # Commit bookkeeping only after the backend succeeds. Mapping failures
        # propagate to the adapter's fatal state; they are never capacity misses.
        if unused:
            self.manager.free(unused)
        self.refs.subtract(refs)
        self.refs += Counter()  # Drop zero counts.
        self.live.difference_update(removals)

    def available_pages(self) -> int:
        # An upper bound: placement can still fail for the exact native IDs.
        physical = len(self.refs) + 1 + self.manager.available_size()
        free_bytes = physical * self.physical_page_bytes
        free_bytes -= (len(self.live) + 1) * self.native_page_bytes
        return max(0, free_bytes // self.native_page_bytes)


class _BackingMiss(Exception):
    pass


def bind_native_swa(allocator: Any, backing: SWABacking,
                    full_backing: SWABacking | None = None) -> None:
    """Bind the public alloc/free boundary of SGLang 0.5.20's native children.

    Child allocation returns before the composite publishes full-to-SWA
    mappings. An expected miss returns only newly acquired pages via native
    free_page_ids; an unexpected error poisons this adapter until teardown.
    Optional C128 backing follows full IDs independently of SWA retirement.
    No native free list is read or assigned.
    """
    import torch

    full = allocator.full_attn_allocator
    swa = allocator.swa_attn_allocator
    page_size = allocator.page_size
    for child in (full, swa):
        for name in ("alloc", "free", "free_page_ids", "free_group_end", "clear"):
            if not callable(getattr(child, name, None)):
                raise ValueError(f"Unsupported native SWA allocator: missing {name}")
        if page_size > 1:
            for name in ("alloc_extend", "alloc_decode"):
                method = getattr(child, name, None)
                if not callable(method) or "last_loc" not in inspect.signature(method).parameters:
                    raise ValueError(f"Unsupported native SWA allocator: {name} signature")
    if getattr(allocator, "swa_req_ring", False):
        raise ValueError("Per-request SWA rings must retain native storage")

    backings = {id(swa): backing}
    if full_backing is not None:
        if full.size // page_size >= full_backing.native_pages:
            raise ValueError("C128 storage cannot cover the native full allocator")
        backings[id(full)] = full_backing

    pending: list[tuple[Any, set[int]]] | None = None
    failed = False
    native_available = {id(child): child.available_size for child in (full, swa)}
    original_free_pages = {id(child): child.free_page_ids for child in (full, swa)}

    def require_healthy() -> None:
        if failed:
            raise RuntimeError("DeepSeek-V4 backing failed; tear down this allocator")

    def page_ids(indices: Any, divisor: int = page_size) -> set[int]:
        return set((indices // divisor).unique().tolist())

    def return_pages(child: Any, pages: set[int]) -> None:
        if pages:
            indices = torch.tensor(sorted(pages), dtype=torch.int64, device=child.device)
            original_free_pages[id(child)](indices)

    def wrap_child(child: Any, original: Any) -> Any:
        signature = inspect.signature(original)

        @functools.wraps(original)
        def allocate(*args: Any, **kwargs: Any) -> Any:
            require_healthy()
            if pending is None:
                raise RuntimeError("Managed SWA allocation requires the composite transaction")
            arguments = signature.bind(*args, **kwargs).arguments
            last_loc = arguments.get("last_loc")
            prior = page_ids(last_loc) if last_loc is not None else set()
            result = original(*args, **kwargs)
            if result is None:
                raise _BackingMiss
            pages = page_ids(result)
            new = pages - prior
            pending.append((child, new))
            child_backing = backings.get(id(child))
            if child_backing is not None and not child_backing.acquire(pages):
                raise _BackingMiss
            return result
        return allocate

    def wrap_composite(original: Any) -> Any:
        @functools.wraps(original)
        def allocate(*args: Any, **kwargs: Any) -> Any:
            nonlocal pending, failed
            require_healthy()
            if pending is not None:
                return original(*args, **kwargs)
            pending = []
            try:
                # All initialization must settle before resize/reclaim can
                # acquire manager locks or release another pool's backing.
                for child_backing in backings.values():
                    child_backing.manager.wait_ready()
                # Reclaim before native IDs or backing reference snapshots are
                # selected: eviction may release existing SWA backing leases.
                for child_backing in backings.values():
                    child_backing.manager.apply_resize_target()
                    child_backing.manager.reclaim_for_shrink()
                try:
                    return original(*args, **kwargs)
                except _BackingMiss:
                    for child, pages in reversed(pending):
                        return_pages(child, pages)
                        rollback_backing = backings.get(id(child))
                        if rollback_backing is not None:
                            rollback_backing.release(pages)
                    return None
            except Exception:
                # The error can be after mapping publication. Guessing a
                # rollback here could expose reused storage through live IDs.
                failed = True
                raise
            finally:
                pending = None
        return allocate

    def bind_release(child: Any, child_backing: SWABacking) -> None:
        release_depth = 0
        deferred: set[int] = set()

        def wrap_release(original: Any, divisor: int) -> Any:
            @functools.wraps(original)
            def release(indices: Any) -> Any:
                nonlocal release_depth, failed
                require_healthy()
                if release_depth:
                    return original(indices)
                pages = page_ids(indices, divisor)
                release_depth += 1
                try:
                    result = original(indices)
                    if child.free_group is None:
                        child_backing.release(pages)
                    else:
                        deferred.update(pages)
                    return result
                except Exception:
                    failed = True
                    raise
                finally:
                    release_depth -= 1
            return release

        original_group_end = child.free_group_end

        def free_group_end() -> None:
            nonlocal failed
            require_healthy()
            try:
                original_group_end()
                child_backing.release(deferred)
                deferred.clear()
            except Exception:
                failed = True
                raise

        original_clear = child.clear

        def clear() -> None:
            nonlocal failed
            require_healthy()
            try:
                original_clear()
                child_backing.release(set(child_backing.live))
                deferred.clear()
            except Exception:
                failed = True
                raise

        def available_size() -> int:
            require_healthy()
            return min(native_available[id(child)](),
                       child_backing.available_pages() * page_size)

        child.free = wrap_release(child.free, page_size)
        child.free_page_ids = wrap_release(child.free_page_ids, 1)
        child.free_group_end = free_group_end
        child.clear = clear
        child.available_size = available_size

    original_resize = getattr(allocator, "resize", None)
    if original_resize is not None:
        full_capacity = full.size
        swa_capacity = swa.size

        def resize(config: Any) -> None:
            require_healthy()
            if (int(config.full_max_total_num_tokens) > full_capacity
                    or int(config.swa_max_total_num_tokens) > swa_capacity):
                raise ValueError("SWA resize exceeds the original backing or mapping capacity")
            original_resize(config)

        allocator.resize = resize

    for child in (full, swa):
        for name in ("alloc", "alloc_extend", "alloc_decode"):
            original = getattr(child, name, None)
            if original is not None:
                setattr(child, name, wrap_child(child, original))
    for name in ("alloc", "alloc_extend", "alloc_decode", "alloc_extend_swa_tail"):
        original = getattr(allocator, name, None)
        if original is not None:
            setattr(allocator, name, wrap_composite(original))
    for child in (full, swa):
        child_backing = backings.get(id(child))
        if child_backing is not None:
            bind_release(child, child_backing)
    # Slot conservation counts logical IDs, not the current physical budget.
    allocator._conserve_swa_available_size = native_available[id(swa)]
    allocator._conserve_full_available_size = native_available[id(full)]
    # Scheduler limit discovery must reach the physical backing manager.
    # Keep it separate from kvcached_allocator: its block IDs are physical
    # pages, not the native token-page IDs used by page-aware eviction.
    allocator._kvcached_dsv4_backing = backing
    allocator._kvcached_dsv4_full_backing = full_backing
