# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Request-row admission for native DeepSeek-V4 C128 compressor state."""

import functools
import types
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, NamedTuple

from kvcached.integration.sglang.dsv4_backing import SWABacking

request_pool_context: ContextVar[Any] = ContextVar("dsv4_state_request_pool", default=None)
request_admission_context: ContextVar[Any] = ContextVar("dsv4_request_admission", default=None)


class AdmissionReservation(NamedTuple):
    previous: frozenset[int]
    rows: tuple[int, ...]
    target: int

    @property
    def available(self) -> int:
        return len(self.rows)


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
        self.admission_budget: int | None = None
        self.capacity_override: int | None = None
        self.demand_rows: list[int] = []
        self.standby_reclaimable = lambda: True
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
        return len(self._reserve(count, retain_standby=False))

    def _reserve(self, count: int, *, retain_standby: bool) -> list[int]:
        self._ready()
        self.demand_rows.clear()
        for backing, _, _ in self.entries:
            backing.manager.wait_ready()
        for backing, _, _ in self.entries:
            backing.manager.apply_resize_target()
        count = min(max(0, count), int(self._available()))
        if not count:
            return []
        rows = self._alloc_rows(count)
        if rows is None:
            return []
        # Preflight increments native generations only for free rows. Returning
        # the same ordered vector restores native tail allocation order.
        selected: set[int] = set()
        try:
            requested = set(rows)
            if self._acquire(requested):
                selected = requested
            else:
                self.demand_rows = list(rows)
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
            if retain_standby:
                self.standby.update(selected)
            else:
                self._release(self.standby - selected)
                self.standby = selected
            return [row for row in rows if row in selected]
        except Exception:
            self._release(selected - self.standby)
            raise
        finally:
            self._free_rows(rows)

    def logical_available(self) -> int:
        return int(self._available())

    def managers(self) -> list[Any]:
        return [backing.manager for backing, _, _ in self.entries]

    def backing_demand(self, pending_rows: int) -> list[tuple[Any, int]]:
        if pending_rows <= 0:
            self.demand_rows.clear()
        # Probes cover the entire logical window. Forecast only the native
        # tail needed by actual admissions, excluding rows already consumed.
        rows = [row for row in self.demand_rows if row not in self.active]
        needed = set(rows[-pending_rows:]) if pending_rows > 0 else set()
        return [(backing.manager, len(backing.missing_physical_pages(
            self._ids(needed, stride, banks))))
            for backing, stride, banks in self.entries]

    def can_reclaim_standby(self) -> bool:
        return (self.admission_budget is None and self.capacity_override is None
                and request_admission_context.get() is not self and self.standby_reclaimable())

    def reclaim_standby(self) -> None:
        if self.can_reclaim_standby():
            self._release(self.standby)
            self.standby.clear()

    def register_reclaimers(self) -> None:
        if hasattr(self, "reclaimers"):
            return
        self.reclaimers = [RequestStateReclaimer(self, entry) for entry in self.entries]
        for reclaimer in self.reclaimers:
            manager = reclaimer.backing.manager
            manager._register_shrink_reclaimer(self.reclaim_standby)
            manager._register_cache_owner(reclaimer.count_pages, reclaimer.evict_pages)

    @contextmanager
    def capacity_query(self, count: int):
        previous = self.capacity_override
        self.capacity_override = count
        try:
            yield
        finally:
            self.capacity_override = previous

    def prepare_admission(self, count: int) -> AdmissionReservation:
        previous = frozenset(self.standby)
        rows = self._reserve(count, retain_standby=True)
        return AdmissionReservation(previous, tuple(rows), max(0, count))

    def finish_admission(self, reservation: AdmissionReservation, count: int) -> None:
        """Return only newly reserved surplus; retain a previous standby lease."""
        if not 0 <= count <= reservation.available:
            raise ValueError("Admission exceeds the locally backed native rows")
        keep = set(reservation.rows[-count:]) if count else set()
        retained = set(reservation.previous) | keep
        self._release(self.standby - retained)
        self.standby.intersection_update(retained)
        self.admission_budget = count

    def end_admission(self) -> None:
        self.admission_budget = None

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
            if owner.capacity_override is not None:
                return owner.capacity_override
            if owner.admission_budget is not None:
                return min(owner.admission_budget, int(owner._available()))
            if request_admission_context.get() is owner:
                return owner.reserve(owner.admission_rows)
            return int(owner._available())

        def alloc_rows(_pool: Any, count: int) -> Any:
            owner._ready()
            if owner.admission_budget is not None and count > owner.admission_budget:
                return None
            rows = owner._alloc_rows(count)
            if rows is None or not rows:
                return rows
            try:
                success = owner._acquire(set(rows))
            except Exception:
                owner._free_rows(rows)
                raise
            if not success:
                owner.demand_rows = list(rows)
                owner._free_rows(rows)
                return None
            owner.active.update(rows)
            owner.standby.difference_update(rows)
            if owner.admission_budget is not None:
                owner.admission_budget -= len(rows)
            return rows

        def free_rows(_pool: Any, rows: list[int]) -> None:
            owner._ready()
            owner._free_rows(rows)
            owner.active.difference_update(rows)
            # Returned rows become the next native choices and already have
            # backing. Retain only the next admission window, using the public
            # row API to preserve its exact order while releasing surplus.
            owner.standby.update(rows)
            if owner.admission_budget is None:
                owner.reserve(min(owner.admission_rows, len(owner.standby)))

        def clear(_pool: Any) -> None:
            owner._ready()
            owner._clear()
            owner._release(owner.active | owner.standby)
            owner.active.clear()
            owner.standby.clear()
            owner.demand_rows.clear()
            owner.admission_rows = 1

        self.pool.available_size = types.MethodType(available, self.pool)
        self.pool.alloc_rows = types.MethodType(alloc_rows, self.pool)
        self.pool.free_rows = types.MethodType(free_rows, self.pool)
        self.pool.clear = types.MethodType(clear, self.pool)
        self.pool._kvcached_dsv4_request_state = self


class RequestStateReclaimer:
    """Only unpromised request-row leases; never token-prefix cache entries."""

    def __init__(self, owner: RequestStateBacking, entry: tuple[SWABacking, int, int]):
        self.owner = owner
        self.backing, self.stride, self.banks = entry

    def count_pages(self) -> int:
        if not self.owner.can_reclaim_standby():
            return 0
        rows = self.owner._ids(self.owner.standby, self.stride, self.banks)
        refs = Counter(page for row in rows for page in self.backing._span(row))
        return sum(count == self.backing.refs[page] for page, count in refs.items())

    def evict_pages(self, count: int) -> int:
        before = len(self.backing.refs)
        if count > 0:
            # Coupled manager retirement enforces the shared byte allowance;
            # unused row leases need not remain promised to a future request.
            self.owner.reclaim_standby()
        return before - len(self.backing.refs)


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
            if owner.capacity_override is not None or owner.admission_budget is not None:
                return original(this, running_bs, beam_width, running_batch)
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
