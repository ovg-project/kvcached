# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Process-local accounting for memory owned by serving runtimes.

Registration reports facts; it does not allocate memory, resize KV pools, or
deduct bytes from an engine's budget. The integration consuming these facts
owns that decision and must avoid counting already-profiled memory twice.
"""

from __future__ import annotations

import operator
import re
import threading
import weakref
from typing import Any, Dict, List, Optional, Tuple

from kvcached.observability import SCHEMA_VERSION, RuntimeReservationSnapshot
from kvcached.utils import normalize_gpu_device

_Key = Tuple[str, str, str, int]
_reservations: Dict[_Key, Tuple[weakref.ReferenceType[Any], int]] = {}
_lock = threading.RLock()


def _device_key(device: str) -> str:
    # A bare "cuda" changes meaning with the current device. Require an
    # explicit ordinal rather than querying CUDA from a reporting API.
    if not isinstance(device, str):
        raise TypeError("device must be a string with an explicit ordinal")
    normalized = normalize_gpu_device(device.strip().lower())
    match = re.fullmatch(r"([a-z][a-z0-9_]*):([0-9]+)", normalized)
    if match is None:
        raise ValueError("device must include an explicit ordinal, e.g. cuda:0")
    return f"{match[1]}:{int(match[2])}"


def _label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    return value


def register_runtime_owned_reservation(
    device: str,
    pool_name: str,
    num_bytes: int,
    *,
    owner: Any,
    integration: str,
) -> None:
    """Replace one live owner's reported bytes for a device and category.

    ``pool_name`` is an integration-defined, low-cardinality category, such
    as ``workspace`` or ``swa_kv``. Reuse common names for generic buffers;
    prefix feature-specific categories, e.g. ``dsv4.workspace``.
    The owner must support weak references;
    it is tracked by identity, not equality, and is never kept alive here.
    Different owners of the same category add together. Zero removes only
    this owner's entry. Report only runtime-owned bytes, excluding backing
    already managed by kvcached. Aliased buffers must be deduplicated by the
    caller before registration. Register after successful construction.
    """
    device = _device_key(device)
    integration = _label(integration, "integration")
    pool_name = _label(pool_name, "pool_name")
    if isinstance(num_bytes, bool):
        raise TypeError("num_bytes must be an integer, not bool")
    num_bytes = operator.index(num_bytes)
    if num_bytes < 0:
        raise ValueError("num_bytes must be nonnegative")
    key = (integration, device, pool_name, id(owner))

    def remove(reference: weakref.ReferenceType[Any]) -> None:
        with _lock:
            current = _reservations.get(key)
            if current is not None and current[0] is reference:
                _reservations.pop(key, None)

    reference = weakref.ref(owner, remove)
    with _lock:
        if num_bytes == 0:
            _reservations.pop(key, None)
        else:
            _reservations[key] = (reference, num_bytes)


def clear_runtime_owned_reservations(*, integration: Optional[str] = None) -> None:
    """Forget reports after the integration's resources have stopped.

    An incomplete shutdown must retain reports for its retry. Omitting the
    integration clears all process-local records, for whole-process teardown.
    """
    if integration is not None:
        integration = _label(integration, "integration")
    with _lock:
        keys = [key for key in _reservations.copy()
                if integration is None or key[0] == integration]
        for key in keys:
            _reservations.pop(key, None)


def get_runtime_reservation_snapshots(
    *,
    integration: Optional[str] = None,
    device: Optional[str] = None,
) -> List[RuntimeReservationSnapshot]:
    """Return immutable totals per integration, device and pool category.

    The snapshot includes only live owners at collection time. It is not a
    history, a global device-memory census, or a cross-process transport.
    """
    if integration is not None:
        integration = _label(integration, "integration")
    if device is not None:
        device = _device_key(device)
    totals: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    with _lock:
        # Copy the dictionary before Python iteration: allocating item tuples
        # can trigger cyclic GC, whose callbacks mutate the original registry
        # reentrantly. Callbacks never mutate this copy.
        for key, (reference, num_bytes) in _reservations.copy().items():
            if integration is not None and key[0] != integration:
                continue
            if device is not None and key[1] != device:
                continue
            if reference() is None:
                continue
            group = key[:3]
            total, count = totals.get(group, (0, 0))
            totals[group] = (total + num_bytes, count + 1)
    return [
        RuntimeReservationSnapshot(
            schema_version=SCHEMA_VERSION, integration=engine, device=dev,
            pool_name=pool, num_bytes=total, owner_count=count)
        for (engine, dev, pool), (total, count) in sorted(totals.items())
    ]


def get_runtime_reservation_snapshot_dicts(
    *,
    integration: Optional[str] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return JSON-serializable runtime reservation totals."""
    return [snapshot.to_dict() for snapshot in get_runtime_reservation_snapshots(
        integration=integration, device=device)]


def get_runtime_owned_reservation_breakdown(
    device: str,
    *,
    integration: str,
) -> Dict[str, int]:
    """Return the reported bytes per pool category on one device."""
    return {snapshot.pool_name: snapshot.num_bytes
            for snapshot in get_runtime_reservation_snapshots(
                integration=integration, device=device)}


def get_runtime_owned_reservation_bytes(device: str, *, integration: str) -> int:
    """Return the sum of reported bytes on one device for an integration."""
    return sum(get_runtime_owned_reservation_breakdown(
        device, integration=integration).values())
