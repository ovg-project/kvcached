# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Process-local registry for live kvcached-backed KV pools."""

from __future__ import annotations

import threading
import weakref
from typing import Any, Dict, List, Optional, Tuple

_registered_pools: Dict[int, Tuple[Any, Optional[str]]] = {}
_registered_pools_lock = threading.RLock()


def register_kv_cache_pool(
    manager: Any,
    *,
    integration: Optional[str] = None,
) -> None:
    """Register a live KV pool without extending its lifetime."""

    manager_id = id(manager)

    def _remove(reference: Any) -> None:
        with _registered_pools_lock:
            current = _registered_pools.get(manager_id)
            if current is not None and current[0] is reference:
                _registered_pools.pop(manager_id, None)

    reference = weakref.ref(manager, _remove)
    with _registered_pools_lock:
        _registered_pools[manager_id] = (reference, integration)


def clear_registered_kv_cache_pools(*, integration: Optional[str] = None) -> None:
    """Forget registered pools, optionally for one engine integration."""

    with _registered_pools_lock:
        if integration is None:
            _registered_pools.clear()
            return
        stale_ids = [
            manager_id
            for manager_id, (_, registered_integration) in _registered_pools.items()
            if registered_integration == integration
        ]
        for manager_id in stale_ids:
            _registered_pools.pop(manager_id, None)


def get_registered_kv_cache_pools(
    *,
    integration: Optional[str] = None,
) -> List[Tuple[Any, Optional[str]]]:
    """Return live KV managers and their engine integration metadata."""

    with _registered_pools_lock:
        entries = list(_registered_pools.values())

    pools = []
    for reference, registered_integration in entries:
        if integration is not None and registered_integration != integration:
            continue
        manager = reference()
        if manager is not None:
            pools.append((manager, registered_integration))
    return pools
