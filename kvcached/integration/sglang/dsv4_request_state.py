# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Request-row admission for native DeepSeek-V4 C128 compressor state."""

import functools
import types
from contextvars import ContextVar
from typing import Any

from kvcached.integration.sglang.dsv4_backing import SWABacking

request_pool_context: ContextVar[Any] = ContextVar("dsv4_state_request_pool", default=None)
request_admission_context: ContextVar[Any] = ContextVar("dsv4_request_admission", default=None)


class RequestStateBacking:
    """Back native row choices before the scheduler promises admission.

    Native alloc_rows/free_rows select and restore standby rows. Only backing
    leases are retained; the native free list and request binding stay native.
    """

    def __init__(self, pool: Any, groups: list[dict[str, Any]]) -> None:
        self.pool = pool
        self.entries: list[tuple[SWABacking, int, int]] = [
            (g["backing"], g["bank_stride"], g["bank_count"]) for g in groups
        ]
        self.active: set[int] = set()
        self.standby: set[int] = set()
        self.admission_rows = 1
        self.failed = False
        self._alloc_rows = pool.alloc_rows
        self._free_rows = pool.free_rows
        self._available = pool.available_size
        self._clear = pool.clear

    def _ready(self) -> None:
        if self.failed:
            raise RuntimeError("DeepSeek-V4 request-state backing is unavailable after a backend failure")
        if getattr(self.pool, "_on_alloc_rows", None) is not None:
            raise RuntimeError("DeepSeek-V4 request-state preflight cannot run native row callbacks")

    @staticmethod
    def _ids(rows: set[int], stride: int, banks: int) -> set[int]:
        return {row + bank * stride for bank in range(banks) for row in rows}

    def _acquire(self, rows: set[int]) -> bool:
        acquired: list[tuple[SWABacking, set[int]]] = []
        try:
            for backing, stride, banks in self.entries:
                ids = self._ids(rows, stride, banks)
                added = ids - backing.live
                if not backing.acquire(ids):
                    for previous, additions in reversed(acquired):
                        previous.release(additions)
                    return False
                acquired.append((backing, added))
        except Exception:
            self.failed = True
            for previous, additions in reversed(acquired):
                previous.release(additions)
            raise
        return True

    def _release(self, rows: set[int]) -> None:
        try:
            for backing, stride, banks in self.entries:
                backing.release(self._ids(rows, stride, banks))
        except Exception:
            self.failed = True
            raise

    def reserve(self, count: int) -> int:
        self._ready()
        for backing, _, _ in self.entries:
            backing.manager.wait_ready()
        for backing, _, _ in self.entries:
            backing.manager.apply_resize_target()
        count = min(max(0, count), int(self._available()))
        if not count:
            return 0
        rows = self._alloc_rows(count)
        if rows is None:
            return 0
        # Preflight increments native generations only for free rows. Returning
        # the same ordered vector restores native tail allocation order.
        selected: set[int] = set()
        try:
            requested = set(rows)
            if self._acquire(requested):
                selected = requested
            else:
                # Native allocation consumes the ordered tail. Find the largest
                # backed tail instead of hiding a smaller usable batch after a
                # full-window miss. Successful probes are nested and retained.
                low, high = 0, len(rows) - 1
                while low < high:
                    middle = (low + high + 1) // 2
                    candidate = set(rows[-middle:])
                    if self._acquire(candidate):
                        selected = candidate
                        low = middle
                    else:
                        high = middle - 1
            self._release(self.standby - selected)
            self.standby = selected
            return len(selected)
        except Exception:
            self._release(selected - self.standby)
            raise
        finally:
            self._free_rows(rows)

    def reclaim_on_idle(self, *, keep_standby: bool) -> None:
        """Apply request-state limits without evicting unrelated KV prefixes."""
        self._ready()
        managers = [entry[0].manager for entry in self.entries]
        for manager in managers:
            manager.wait_ready()
        for manager in managers:
            manager.apply_resize_target()
        if not keep_standby and any(manager.in_shrink for manager in managers):
            self._release(self.standby)
            self.standby.clear()
        for manager in managers:
            manager.reclaim_for_shrink()

    def bind(self) -> None:
        owner = self

        def available(_pool: Any) -> int:
            if request_admission_context.get() is owner:
                return owner.reserve(owner.admission_rows)
            return int(owner._available())

        def alloc_rows(_pool: Any, count: int) -> Any:
            owner._ready()
            rows = owner._alloc_rows(count)
            if rows is None or not rows:
                return rows
            try:
                success = owner._acquire(set(rows))
            except Exception:
                owner._free_rows(rows)
                raise
            if not success:
                owner._free_rows(rows)
                return None
            owner.active.update(rows)
            owner.standby.difference_update(rows)
            return rows

        def free_rows(_pool: Any, rows: list[int]) -> None:
            owner._ready()
            owner._free_rows(rows)
            owner.active.difference_update(rows)
            # Returned rows become the next native choices and already have
            # backing. Retain only the next admission window, using the public
            # row API to preserve its exact order while releasing surplus.
            owner.standby.update(rows)
            owner.reserve(min(owner.admission_rows, len(owner.standby)))

        def clear(_pool: Any) -> None:
            owner._ready()
            owner._clear()
            owner._release(owner.active | owner.standby)
            owner.active.clear()
            owner.standby.clear()
            owner.admission_rows = 1

        self.pool.available_size = types.MethodType(available, self.pool)
        self.pool.alloc_rows = types.MethodType(alloc_rows, self.pool)
        self.pool.free_rows = types.MethodType(free_rows, self.pool)
        self.pool.clear = types.MethodType(clear, self.pool)
        self.pool._kvcached_dsv4_request_state = self


def install_request_state_context(module: Any) -> bool:
    """Select supported native request pools before the state factory runs."""
    cls = getattr(module, "KVCacheConfigurator", None)
    original = getattr(cls, "_build_token_to_kv_pool", None)
    native_pool = getattr(module, "ReqToTokenPool", None)
    if cls is None or original is None or native_pool is None:
        return False
    if getattr(original, "_kvcached_request_state", False):
        return True

    @functools.wraps(original)
    def build(configurator: Any, *args: Any, **kwargs: Any) -> Any:
        pool = kwargs.get("req_to_token_pool")
        supported = False
        try:
            schedule = module.get_schedule()
            supported = (
                not configurator.is_draft_worker and type(pool) is native_pool
                and module.get_disagg().disaggregation_mode == "null"
                and module.get_exec().dllm.dllm_algorithm is None
                and not (schedule.enable_priority_scheduling
                         and not schedule.disable_priority_preemption)
                and pool._on_alloc_rows is None
                and pool.available_size() == pool.size
            )
        except (AttributeError, RuntimeError):
            pass
        token = request_pool_context.set(pool if supported else None)
        try:
            result = original(configurator, *args, **kwargs)
        finally:
            request_pool_context.reset(token)
        groups = getattr(result, "_kvcached_c128_state_groups", {})
        if groups:
            owner = RequestStateBacking(pool, list(groups.values()))
            owner.bind()
            result._kvcached_dsv4_request_state = owner
        return result

    build._kvcached_request_state = True  # type: ignore[attr-defined]
    cls._build_token_to_kv_pool = build
    return True


def install_request_admission(scheduler: Any) -> bool:
    original = getattr(scheduler, "get_num_allocatable_reqs", None)
    if original is None:
        return False
    if getattr(original, "_kvcached_request_state", False):
        return True

    @functools.wraps(original)
    def available(this: Any, running_bs: int, beam_width: Any = None,
                  running_batch: Any = None) -> Any:
        owner = getattr(this.req_to_token_pool, "_kvcached_dsv4_request_state", None)
        if owner is not None:
            pending = this.beam_coordinator.pending_member_rows(running_batch or this.running_batch)
            # Ask the native gate for its logical batch limit without reserving
            # physical storage, then preflight that many complete requests.
            count = original(this, running_bs, beam_width, running_batch)
            owner.admission_rows = pending + max(0, count) * (beam_width or 1)
            token = request_admission_context.set(owner)
            try:
                return original(this, running_bs, beam_width, running_batch)
            finally:
                request_admission_context.reset(token)
        return original(this, running_bs, beam_width, running_batch)

    available._kvcached_request_state = True  # type: ignore[attr-defined]
    scheduler.get_num_allocatable_reqs = available
    return True
