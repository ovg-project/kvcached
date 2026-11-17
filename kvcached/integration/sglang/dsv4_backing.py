# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Backing leases for IDs allocated by SGLang's native SWA allocator.

The manager uses one block per VMM page. Native token pages can share or
straddle those pages; neither their IDs nor SGLang's mappings are rewritten.
"""

import functools
import inspect
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
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

    def can_acquire(self, pages: set[int]) -> bool:
        """Check an already validated placement without taking backing leases."""
        missing = self.missing_physical_pages(pages)
        return not missing or self.manager.can_alloc_exact(sorted(missing))

    def missing_physical_pages(self, pages: set[int]) -> set[int]:
        return {physical for page in pages - self.live for physical in self._span(page)} - self.refs.keys()

    def physical_demand(self, native_pages: int) -> int:
        """Forecast additional physical pages after using held native-page slack."""
        held_bytes = (len(self.refs) + 1) * self.physical_page_bytes
        held_bytes -= (len(self.live) + 1) * self.native_page_bytes
        missing = max(0, native_pages * self.native_page_bytes - held_bytes)
        return (missing + self.physical_page_bytes - 1) // self.physical_page_bytes

    def available_pages(self) -> int:
        # An upper bound: placement can still fail for the exact native IDs.
        physical = len(self.refs) + 1 + self.manager.available_size()
        free_bytes = physical * self.physical_page_bytes
        free_bytes -= (len(self.live) + 1) * self.native_page_bytes
        return max(0, free_bytes // self.native_page_bytes)


class _BackingMiss(Exception):
    pass


def bind_native_swa(allocator: Any, backing: SWABacking, *, on_capacity_miss: Any = None,
                    on_priority_miss: Any = None) -> None:
    """Bind the public alloc/free boundary of SGLang 0.5.20's native children.

    Child allocation returns before the composite publishes full-to-SWA
    mappings. An expected miss returns only newly acquired pages via native
    free_page_ids; an unexpected error poisons this adapter until teardown.
    Only native public allocation and page-census APIs are used. Native free
    lists are never assigned by this adapter.
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

    backings = {id(swa): (backing,), id(full): ()}

    pending: list[tuple[Any, set[int]]] | None = None
    failed = False
    native_available = {id(child): child.available_size for child in (full, swa)}
    original_free_pages = {id(child): child.free_page_ids for child in (full, swa)}
    preparation: ContextVar[Any] = ContextVar("dsv4_native_preparation", default=None)
    budget: ContextVar[Any] = ContextVar("dsv4_native_budget", default=None)
    returning: ContextVar[bool] = ContextVar("dsv4_native_return", default=False)
    decode_reserved: dict[int, set[int]] = {id(full): set(), id(swa): set()}
    placement_misses: dict[int, tuple[int, ...]] = {}

    def release_decode_reservation() -> None:
        nonlocal failed
        require_healthy()
        try:
            for child in (swa, full):
                if not decode_reserved[id(child)]:
                    continue
                for item in reversed(backings[id(child)]):
                    item.release(decode_reserved[id(child)])
                decode_reserved[id(child)].clear()
        except Exception:
            failed = True
            raise

    def ready_managers() -> None:
        for child_backings in backings.values():
            for item in child_backings:
                item.manager.wait_ready()
        for child_backings in backings.values():
            for item in child_backings:
                item.manager.apply_resize_target()
                item.manager.reclaim_for_shrink()

    original_decode_check = getattr(allocator, "check_decode_capacity", None)
    if original_decode_check is not None:
        # After this public normalization, the public census has the allocation
        # prefix used by the audited 0.5.20 native paged/token allocators.
        for child in (full, swa):
            original_merge = child.merge_and_sort_free

            def merge(*, _child=child, _original=original_merge):
                # Decode's bs-based sort is conservative: its kernel consumes
                # only the new-page count. Keep the promised native prefix even
                # when window/cache eviction appended other free pages.
                if pending is None or not decode_reserved[id(_child)]:
                    return _original()
                return None

            child.merge_and_sort_free = merge

        @functools.wraps(original_decode_check)
        def check_decode_capacity(*, num_tokens: int, **kwargs: Any) -> bool:
            nonlocal failed
            release_decode_reservation()
            ready_managers()
            if on_priority_miss is not None and on_priority_miss(kwargs.get("requests")):
                return False
            if not original_decode_check(num_tokens=num_tokens, **kwargs):
                return False
            count = (num_tokens + page_size - 1) // page_size
            try:
                for child in (full, swa):
                    child.merge_and_sort_free()
                    choices = child.get_all_free_pages()[:count].tolist()
                    if len(choices) != count:
                        release_decode_reservation()
                        return False
                    pages = set(choices)
                    decode_reserved[id(child)] = pages
                    for item in backings[id(child)]:
                        if not item.acquire(pages):
                            release_decode_reservation()
                            placement_misses[id(child)] = tuple(choices)
                            return False
                return True
            except Exception:
                try:
                    release_decode_reservation()
                finally:
                    failed = True
                raise

        allocator.check_decode_capacity = check_decode_capacity
    allocator._kvcached_release_decode_reservation = release_decode_reservation

    def move_budget(child: Any, pages: int) -> None:
        counts = budget.get()
        if counts is not None:
            counts[id(child)] += pages * page_size

    @contextmanager
    def budget_context(counts: tuple[int, int]):
        token = budget.set({id(full): counts[0], id(swa): counts[1]})
        try:
            yield
        finally:
            budget.reset(token)

    class Preparation:
        """Own only newly selected native pages during one prepare call."""

        def __init__(self) -> None:
            self.pages: dict[int, set[int]] = {id(full): set(), id(swa): set()}
            self.token: Any = None

        def __enter__(self) -> Any:
            require_healthy()
            if preparation.get() is not None:
                raise RuntimeError("Nested DSV4 preparation transactions are unsupported")
            self.token = preparation.set(self)
            return self

        def __exit__(self, *exc: Any) -> None:
            preparation.reset(self.token)

        def rollback(self) -> None:
            nonlocal failed
            require_healthy()
            try:
                # Native preparation queues index writes before the peer verdict.
                # Only the rejected path waits; never unmap their new pages early.
                if torch.device(full.device).type == "cuda":
                    torch.cuda.current_stream(full.device).synchronize()
                for child in (swa, full):
                    pages = self.pages[id(child)]
                    return_pages(child, pages)
                    for item in reversed(backings[id(child)]):
                        item.release(pages)
                    move_budget(child, len(pages))
                    pages.clear()
            except Exception:
                failed = True
                raise

    def require_healthy() -> None:
        if failed:
            raise RuntimeError("DeepSeek-V4 backing failed; tear down this allocator")

    def page_ids(indices: Any, divisor: int = page_size) -> set[int]:
        return set((indices // divisor).unique().tolist())

    def return_pages(child: Any, pages: set[int]) -> None:
        if pages:
            indices = torch.tensor(sorted(pages), dtype=torch.int64, device=child.device)
            token = returning.set(True)
            try:
                original_free_pages[id(child)](indices)
            finally:
                returning.reset(token)

    def wrap_child(child: Any, original: Any) -> Any:
        signature = inspect.signature(original)

        @functools.wraps(original)
        def allocate(*args: Any, **kwargs: Any) -> Any:
            require_healthy()
            if pending is None:
                raise RuntimeError("Managed SWA allocation requires the composite transaction")
            arguments = signature.bind(*args, **kwargs).arguments
            reserved = decode_reserved[id(child)]
            if reserved:
                if "need_size" in arguments:
                    required = arguments["need_size"] // page_size
                elif "prefix_lens_cpu" in arguments:
                    ends = arguments["seq_lens_cpu"].tolist()
                    starts = arguments["prefix_lens_cpu"].tolist()
                    required = sum((end + page_size - 1) // page_size
                                   - (start + page_size - 1) // page_size
                                   for start, end in zip(starts, ends))
                else:
                    required = sum((length - 1) % page_size == 0
                                   for length in arguments["seq_lens_cpu"].tolist())
                choices = child.get_all_free_pages()[:required].tolist()
                if required > len(reserved) or not set(choices) <= reserved:
                    raise RuntimeError("Native allocation no longer matches its decode reservation")
            last_loc = arguments.get("last_loc")
            prior = page_ids(last_loc) if last_loc is not None else set()
            result = original(*args, **kwargs)
            if result is None:
                raise _BackingMiss
            pages = page_ids(result)
            new = pages - prior
            pending.append((child, new))
            for child_backing in backings[id(child)]:
                if not child_backing.acquire(pages):
                    raise _BackingMiss(child, max(1, len(new)))
            placement_misses.pop(id(child), None)
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
                if not any(decode_reserved.values()):
                    ready_managers()
                try:
                    result = original(*args, **kwargs)
                    for child, pages in pending:
                        decode_reserved[id(child)].difference_update(pages)
                        move_budget(child, -len(pages))
                    transaction = preparation.get()
                    if transaction is not None:
                        for child, pages in pending:
                            transaction.pages[id(child)].update(pages)
                    release_decode_reservation()
                    return result
                except _BackingMiss as miss:
                    for child, pages in reversed(pending):
                        return_pages(child, pages)
                        for rollback_backing in reversed(backings[id(child)]):
                            rollback_backing.release(pages)
                        decode_reserved[id(child)].difference_update(pages)
                    release_decode_reservation()
                    if miss.args:
                        child, count = miss.args
                        census = getattr(child, "get_all_free_pages", None)
                        if census is not None:
                            normalize = getattr(child, "merge_and_sort_free", None)
                            if normalize is not None:
                                normalize()
                            placement_misses[id(child)] = tuple(census()[:count].tolist())
                    return None
            except Exception:
                # The error can be after mapping publication. Guessing a
                # rollback here could expose reused storage through live IDs.
                failed = True
                raise
            finally:
                pending = None
        @functools.wraps(original)
        def translated(*args: Any, **kwargs: Any) -> Any:
            result = allocate(*args, **kwargs)
            if result is None and on_capacity_miss is not None:
                # Translate only after rollback, outside the fatal-error guard.
                # The tagged coordinator owns its separate preparation window;
                # otherwise main's batch journal catches this typed signal.
                return on_capacity_miss(allocator, "the request", original.__name__, allocator)
            return result
        return translated

    def bind_release(child: Any, child_backings: tuple[SWABacking, ...]) -> None:
        release_depth = 0
        deferred: set[int] = set()

        def release_backing(pages: set[int]) -> None:
            if pages:
                # Sorted native frees may be staged behind the old prefix.
                # Let the next admission normalize and reconsider that progress.
                placement_misses.pop(id(child), None)
            reserved = decode_reserved[id(child)]
            if reserved:
                # Unsorted native allocators prepend freed SWA window pages.
                # Keep their existing leases until the prepared allocation
                # chooses its IDs; they may now precede the promised prefix.
                reserved.update(pages)
            else:
                for child_backing in child_backings:
                    child_backing.release(pages)

        def wrap_release(original: Any, divisor: int) -> Any:
            @functools.wraps(original)
            def release(indices: Any) -> Any:
                nonlocal release_depth, failed
                require_healthy()
                if release_depth or returning.get():
                    return original(indices)
                pages = page_ids(indices, divisor)
                release_depth += 1
                try:
                    result = original(indices)
                    transaction = preparation.get()
                    if transaction is not None:
                        transaction.pages[id(child)].difference_update(pages)
                    if child.free_group is None:
                        release_backing(pages)
                        move_budget(child, len(pages))
                        deferred.difference_update(pages)
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
                release_backing(deferred)
                move_budget(child, len(deferred))
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
                for child_backing in child_backings:
                    child_backing.release(set(child_backing.live))
                decode_reserved[id(child)].clear()
                placement_misses.pop(id(child), None)
                deferred.clear()
            except Exception:
                failed = True
                raise

        def available_size() -> int:
            require_healthy()
            counts = budget.get()
            if counts is not None:
                return max(0, min(native_available[id(child)](), counts[id(child)]))
            capacity = min([native_available[id(child)](),
                            *((item.available_pages() + len(decode_reserved[id(child)])) * page_size
                              for item in child_backings)])
            missed = placement_misses.get(id(child))
            if missed:
                prefix = tuple(child.get_all_free_pages()[:len(missed)].tolist())
                if prefix == missed:
                    # A failed large prefill must not hide a feasible smaller
                    # decode. Find the affordable prefix without claiming it.
                    low, high = 0, len(prefix)
                    while low < high:
                        middle = (low + high + 1) // 2
                        if all(item.can_acquire(set(prefix[:middle])) for item in child_backings):
                            low = middle
                        else:
                            high = middle - 1
                    if low < len(prefix):
                        return min(capacity, low * page_size)
                placement_misses.pop(id(child), None)
            return capacity

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
        bind_release(child, backings[id(child)])
    allocator._kvcached_prepare_transaction = Preparation
    allocator._kvcached_budget_context = budget_context
    allocator._kvcached_budget_counts = lambda: (full.available_size(), swa.available_size())

    def backing_demand(native_pages: int) -> list[tuple[Any, int]]:
        demands = []
        for child in (full, swa):
            missed = placement_misses.get(id(child))
            prefix: tuple[int, ...] = ()
            if missed and native_pages > 0:
                current = tuple(child.get_all_free_pages()[:len(missed)].tolist())
                if current == missed:
                    prefix = missed
            for item in backings[id(child)]:
                demands.append((item.manager, max(item.physical_demand(native_pages),
                                                 len(item.missing_physical_pages(set(prefix))))))
        return demands

    allocator._kvcached_backing_demand = backing_demand
    # Slot conservation counts logical IDs, not the current physical budget.
    allocator._conserve_swa_available_size = native_available[id(swa)]
    allocator._conserve_full_available_size = native_available[id(full)]
    # Scheduler limit discovery must reach the physical backing manager.
    # Keep it separate from kvcached_allocator: its block IDs are physical
    # pages, not the native token-page IDs used by page-aware eviction.
    allocator._kvcached_dsv4_backing = backing


def bind_native_decode_wrapper(allocator: Any) -> None:
    """Preserve an outer HiSparse limit and reserve its logical SWA backing."""
    logical: Any = getattr(allocator, "logical_attn_allocator", None)
    release = getattr(logical, "_kvcached_release_decode_reservation", None)
    if release is None or getattr(allocator, "_kvcached_decode_wrapper_bound", False):
        return
    original = allocator.check_decode_capacity

    @functools.wraps(original)
    def check_decode_capacity(*args: Any, **kwargs: Any) -> bool:
        release()
        if not original(*args, **kwargs):
            return False
        return bool(logical.check_decode_capacity(*args, **kwargs))

    allocator.check_decode_capacity = check_decode_capacity
    allocator._kvcached_release_decode_reservation = release
    allocator._kvcached_decode_wrapper_bound = True
