# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Policy and state management for CPU-backed KV page offloading.

The GPU transfer itself is deliberately hidden behind ``PageTransferBackend``.
That keeps the eviction policy testable on CPU-only CI and lets engine-specific
CUDA implementations be added without coupling them to cache bookkeeping.

One kvcached page id represents the same byte range in every layer and KV
buffer.  Therefore an offloaded page is a *bundle* containing
``num_layers * num_kv_buffers`` page-sized payloads.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, NamedTuple, Optional, Protocol, Sequence, Tuple


class OffloadError(RuntimeError):
    """Raised when an offload or restore transaction cannot be completed."""

    def __init__(
        self,
        message: str,
        *,
        page_id: int,
        operation: str,
        evicted_page_ids: Tuple[int, ...] = (),
    ):
        super().__init__(message)
        self.page_id = page_id
        self.operation = operation
        self.evicted_page_ids = evicted_page_ids


@dataclass(frozen=True)
class PageGeometry:
    """Describe the physical data represented by one logical kvcached page."""

    page_size: int
    num_layers: int
    num_kv_buffers: int = 2

    def __post_init__(self) -> None:
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_kv_buffers <= 0:
            raise ValueError("num_kv_buffers must be positive")

    @property
    def payload_count(self) -> int:
        return self.num_layers * self.num_kv_buffers

    @property
    def logical_page_bytes(self) -> int:
        return self.page_size * self.payload_count


@dataclass(frozen=True)
class OffloadedPage:
    """CPU copy of one logical page across all KV tensors."""

    page_id: int
    payloads: Tuple[bytes, ...]

    @property
    def size_bytes(self) -> int:
        return sum(len(payload) for payload in self.payloads)


class StoreResult(NamedTuple):
    """Result of inserting one page into the bounded CPU store."""

    stored: bool
    evicted_page_ids: Tuple[int, ...]


class PageTransferBackend(Protocol):
    """Data-plane operations required by :class:`CPUOffloadManager`.

    A CUDA implementation should copy every layer/KV-buffer slice to pinned
    host memory in ``read_gpu_page`` and perform the inverse copy in
    ``write_gpu_page``.  ``release_gpu_page`` must only run after the host copy
    has completed successfully.
    """

    def read_gpu_page(self, page_id: int, geometry: PageGeometry) -> Sequence[bytes]:
        ...

    def release_gpu_page(self, page_id: int) -> None:
        ...

    def allocate_gpu_page(self, page_id: int) -> None:
        ...

    def write_gpu_page(
        self,
        page_id: int,
        payloads: Sequence[bytes],
        geometry: PageGeometry,
    ) -> None:
        ...

    def rollback_gpu_page(self, page_id: int) -> None:
        ...


class CPUOffloadStore:
    """Bounded LRU store for CPU-resident KV pages.

    The oldest page is evicted first when ``max_bytes`` would be exceeded.
    Reading a page refreshes its recency.  The store owns immutable ``bytes``
    objects so callers cannot mutate a cached page behind its bookkeeping.
    """

    def __init__(self, geometry: PageGeometry, max_bytes: int):
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.geometry = geometry
        self.max_bytes = max_bytes
        self._pages: "OrderedDict[int, OffloadedPage]" = OrderedDict()
        self._used_bytes = 0

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def __len__(self) -> int:
        return len(self._pages)

    def __contains__(self, page_id: object) -> bool:
        return page_id in self._pages

    def page_ids(self) -> Tuple[int, ...]:
        """Return page ids from least to most recently used."""

        return tuple(self._pages)

    def _normalize_payloads(self, payloads: Iterable[bytes]) -> Tuple[bytes, ...]:
        normalized = tuple(bytes(payload) for payload in payloads)
        if len(normalized) != self.geometry.payload_count:
            raise ValueError(
                "offloaded page has "
                f"{len(normalized)} payloads; expected {self.geometry.payload_count}"
            )
        invalid_sizes = [
            len(payload)
            for payload in normalized
            if len(payload) != self.geometry.page_size
        ]
        if invalid_sizes:
            raise ValueError(
                "every payload must be exactly "
                f"{self.geometry.page_size} bytes; got invalid sizes {invalid_sizes}"
            )
        return normalized

    def put(self, page_id: int, payloads: Iterable[bytes]) -> StoreResult:
        if page_id < 0:
            raise ValueError("page_id must be non-negative")

        normalized = self._normalize_payloads(payloads)
        page = OffloadedPage(page_id=page_id, payloads=normalized)
        if page.size_bytes > self.max_bytes:
            return StoreResult(stored=False, evicted_page_ids=())

        evicted: List[int] = []
        old_page = self._pages.pop(page_id, None)
        if old_page is not None:
            self._used_bytes -= old_page.size_bytes

        while self._pages and self._used_bytes + page.size_bytes > self.max_bytes:
            evicted_id, evicted_page = self._pages.popitem(last=False)
            self._used_bytes -= evicted_page.size_bytes
            evicted.append(evicted_id)

        self._pages[page_id] = page
        self._used_bytes += page.size_bytes
        return StoreResult(stored=True, evicted_page_ids=tuple(evicted))

    def get(self, page_id: int) -> Optional[OffloadedPage]:
        page = self._pages.get(page_id)
        if page is None:
            return None
        self._pages.move_to_end(page_id)
        return page

    def remove(self, page_id: int) -> Optional[OffloadedPage]:
        page = self._pages.pop(page_id, None)
        if page is not None:
            self._used_bytes -= page.size_bytes
        return page

    def clear(self) -> None:
        self._pages.clear()
        self._used_bytes = 0


@dataclass(frozen=True)
class OffloadResult:
    """Observable result of one GPU-to-CPU offload transaction."""

    page_id: int
    stored: bool
    evicted_page_ids: Tuple[int, ...] = ()


@dataclass(frozen=True)
class OffloadCandidate:
    """Runtime evidence used to decide whether a GPU page may be offloaded."""

    page_id: int
    active_blocks: int
    cached_blocks: int
    total_blocks: int
    last_access_tick: int
    estimated_recompute_ms: float

    def __post_init__(self) -> None:
        if self.page_id < 0:
            raise ValueError("page_id must be non-negative")
        if min(self.active_blocks, self.cached_blocks, self.total_blocks) < 0:
            raise ValueError("block counts must be non-negative")
        if self.active_blocks + self.cached_blocks > self.total_blocks:
            raise ValueError("active and cached blocks exceed page capacity")
        if self.last_access_tick < 0:
            raise ValueError("last_access_tick must be non-negative")
        if self.estimated_recompute_ms < 0:
            raise ValueError("estimated_recompute_ms must be non-negative")

    @property
    def eligible(self) -> bool:
        """Only inactive pages containing reusable cache are worth offloading."""

        return self.active_blocks == 0 and self.cached_blocks > 0


@dataclass(frozen=True)
class OffloadPlan:
    selected_page_ids: Tuple[int, ...]
    bytes_to_offload: int
    eligible_pages: int
    skipped_active_pages: int
    limited_by_cpu_capacity: bool


class PageOffloadPlanner:
    """Select cold, fully inactive pages without splitting page ownership."""

    def __init__(self, geometry: PageGeometry):
        self.geometry = geometry

    def plan(
        self,
        candidates: Iterable[OffloadCandidate],
        pages_needed: int,
        cpu_available_bytes: int,
    ) -> OffloadPlan:
        if pages_needed < 0:
            raise ValueError("pages_needed must be non-negative")
        if cpu_available_bytes < 0:
            raise ValueError("cpu_available_bytes must be non-negative")

        materialized = list(candidates)
        page_ids = [candidate.page_id for candidate in materialized]
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("offload candidates contain duplicate page ids")
        eligible = [candidate for candidate in materialized if candidate.eligible]
        eligible.sort(
            key=lambda candidate: (
                candidate.last_access_tick,
                -candidate.estimated_recompute_ms,
                candidate.page_id,
            )
        )

        capacity_pages = cpu_available_bytes // self.geometry.logical_page_bytes
        selected_count = min(pages_needed, capacity_pages, len(eligible))
        selected = tuple(
            candidate.page_id for candidate in eligible[:selected_count]
        )
        return OffloadPlan(
            selected_page_ids=selected,
            bytes_to_offload=selected_count * self.geometry.logical_page_bytes,
            eligible_pages=len(eligible),
            skipped_active_pages=sum(
                candidate.active_blocks > 0 for candidate in materialized
            ),
            limited_by_cpu_capacity=(
                pages_needed > 0
                and capacity_pages < min(pages_needed, len(eligible))
            ),
        )

    def estimated_transfer_ms(self, bandwidth_gbps: float) -> float:
        if bandwidth_gbps <= 0:
            raise ValueError("bandwidth_gbps must be positive")
        bytes_per_second = bandwidth_gbps * 1_000_000_000
        return self.geometry.logical_page_bytes / bytes_per_second * 1000

    def restore_beats_recompute(
        self,
        estimated_recompute_ms: float,
        bandwidth_gbps: float,
    ) -> bool:
        if estimated_recompute_ms < 0:
            raise ValueError("estimated_recompute_ms must be non-negative")
        return self.estimated_transfer_ms(bandwidth_gbps) < estimated_recompute_ms


class CPUOffloadManager:
    """Coordinate transactional page movement between GPU and CPU.

    The CPU copy is committed before GPU memory is released.  During restore,
    the CPU copy remains available until allocation and copy-back both finish.
    These orderings avoid losing the only valid copy after a partial failure.
    """

    def __init__(self, store: CPUOffloadStore, backend: PageTransferBackend):
        self.store = store
        self.backend = backend

    def offload(self, page_id: int) -> OffloadResult:
        payloads = self.backend.read_gpu_page(page_id, self.store.geometry)
        result = self.store.put(page_id, payloads)
        if not result.stored:
            return OffloadResult(page_id=page_id, stored=False)

        try:
            self.backend.release_gpu_page(page_id)
        except Exception as exc:
            raise OffloadError(
                f"CPU copy for page {page_id} succeeded, but GPU release failed; "
                "the CPU copy was retained for recovery",
                page_id=page_id,
                operation="release",
                evicted_page_ids=result.evicted_page_ids,
            ) from exc

        return OffloadResult(
            page_id=page_id,
            stored=True,
            evicted_page_ids=result.evicted_page_ids,
        )

    def restore(self, page_id: int) -> bool:
        page = self.store.get(page_id)
        if page is None:
            return False

        try:
            self.backend.allocate_gpu_page(page_id)
            self.backend.write_gpu_page(
                page_id,
                page.payloads,
                self.store.geometry,
            )
        except Exception as exc:
            try:
                self.backend.rollback_gpu_page(page_id)
            except Exception:
                pass
            raise OffloadError(
                f"failed to restore page {page_id} to GPU",
                page_id=page_id,
                operation="restore",
            ) from exc

        self.store.remove(page_id)
        return True

    def discard(self, page_id: int) -> bool:
        return self.store.remove(page_id) is not None

    def stats(self) -> Dict[str, int]:
        return {
            "offloaded_pages": len(self.store),
            "used_bytes": self.store.used_bytes,
            "capacity_bytes": self.store.max_bytes,
        }
