# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
FastAPI router that exposes KV cache memory management endpoints.

These endpoints are injected into vLLM's HTTP server via autopatch so that
each vLLM instance can manage its own kvcached memory through the same
host:port it already uses for inference (e.g. localhost:12346).

The router reads/writes POSIX shared-memory segments in /dev/shm that are
created by kvcached's MemInfoTracker.  This is the same mechanism used by
the ``kvctl`` CLI tool.

Safety:
    A minimum KV cache floor is enforced to prevent users from accidentally
    setting the limit so low that the model cannot process any requests.
    The floor defaults to 512 MB or 2% of GPU memory (whichever is larger)
    and can be overridden via the ``KVCACHED_MIN_KV_CACHE_MB`` environment
    variable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from kvcached.cli.kvtop import _detect_kvcache_ipc_names
from kvcached.cli.utils import (
    _format_size,
    get_kv_cache_limit,
    get_total_gpu_memory,
    update_kv_cache_limit,
)
from kvcached.utils import DEFAULT_IPC_NAME, get_kvcached_logger

logger = get_kvcached_logger()

# ---------------------------------------------------------------------------
# Safety floor: minimum KV cache size (bytes).
#
# The primary safety floor is auto-calculated from model parameters
# (num_layers × num_kv_buffers × page_size × MIN_SAFE_PAGES_PER_LAYER)
# and stored in shared memory by MemInfoTracker at init time.
#
# The static fallback below is used only when the shared-memory value is
# unavailable (e.g. segment not yet created).  It can be overridden via
# KVCACHED_MIN_KV_CACHE_MB env var.
# ---------------------------------------------------------------------------
_FALLBACK_MIN_KV_CACHE_MB = int(os.getenv("KVCACHED_MIN_KV_CACHE_MB", "512"))
_FALLBACK_MIN_KV_CACHE_BYTES = _FALLBACK_MIN_KV_CACHE_MB * 1024 * 1024


def _get_min_kv_cache_bytes(ipc_name: Optional[str] = None) -> int:
    """Return the effective minimum KV cache size in bytes.

    Reads the model-aware ``min_safe_limit`` from shared memory first.
    Falls back to max(static_floor, 2% of GPU memory) when the shared
    memory value is unavailable or zero.
    """
    # Try to read model-calculated min_safe_limit from shared memory
    if ipc_name is not None:
        info = get_kv_cache_limit(ipc_name)
        if info is not None and info.min_safe_limit > 0:
            return info.min_safe_limit

    # Fallback: static floor or 2% of GPU
    gpu_total = get_total_gpu_memory()
    gpu_floor = int(gpu_total * 0.02) if gpu_total > 0 else 0
    return max(_FALLBACK_MIN_KV_CACHE_BYTES, gpu_floor)


# Size suffix table shared with kvctl
_SIZE_SUFFIXES = {
    'b': 1,
    'k': 1024,
    'kb': 1024,
    'm': 1024**2,
    'mb': 1024**2,
    'g': 1024**3,
    'gb': 1024**3,
}


def _parse_size(size_str: str) -> int:
    """Convert human-friendly size strings (e.g. ``512M``, ``2G``) to bytes."""
    s = size_str.strip().lower().replace(',', '').replace('_', '')
    for suf, mul in sorted(_SIZE_SUFFIXES.items(), key=lambda kv: -len(kv[0])):
        if s.endswith(suf):
            num_part = s[:-len(suf)] or "0"
            return int(float(num_part) * mul)
    return int(float(s))


def _build_segment_info(ipc_name: str) -> Optional[Dict[str, Any]]:
    """Return a dict describing one IPC segment, or None if not found."""
    info = get_kv_cache_limit(ipc_name)
    if info is None:
        return None
    usage_pct = (info.used_size / info.total_size * 100
                 ) if info.total_size > 0 else 0.0
    free = max(info.total_size - info.used_size - info.prealloc_size, 0)
    physical_occupied = info.used_size + info.prealloc_size
    # Read model-aware safety floor from shared memory
    min_floor = _get_min_kv_cache_bytes(ipc_name)
    # Minimum limit that takes effect immediately (>= used_size succeeds;
    # < used_size enters shrink mode which waits for requests to finish).
    min_effective_limit = info.used_size
    return {
        "ipc_name": ipc_name,
        "total_size": info.total_size,
        "total_size_human": _format_size(info.total_size),
        "used_size": info.used_size,
        "used_size_human": _format_size(info.used_size),
        "prealloc_size": info.prealloc_size,
        "prealloc_size_human": _format_size(info.prealloc_size),
        "free_size": free,
        "free_size_human": _format_size(free),
        "usage_percent": round(usage_pct, 2),
        "physical_occupied": physical_occupied,
        "physical_occupied_human": _format_size(physical_occupied),
        "min_effective_limit": min_effective_limit,
        "min_effective_limit_human": _format_size(min_effective_limit),
        "min_safe_limit": info.min_safe_limit,
        "min_safe_limit_human": _format_size(info.min_safe_limit),
        "safety_floor": min_floor,
        "safety_floor_human": _format_size(min_floor),
    }


def _apply_limit(ipc_name: str, size_bytes: int,
                 force: bool = False) -> Dict[str, Any]:
    """Core logic for applying a KV cache limit with safety checks.

    Args:
        ipc_name: Target IPC segment name.
        size_bytes: Desired limit in bytes.
        force: If True, allow setting below safety floor (still enforces
               absolute minimum of 64 MB to prevent total starvation).

    Returns:
        A dict suitable for JSONResponse content.

    Raises:
        ValueError: If the segment is not found or size is invalid.
    """
    current_info = get_kv_cache_limit(ipc_name)
    if current_info is None:
        raise ValueError(f"IPC segment '{ipc_name}' not found")

    if size_bytes <= 0:
        raise ValueError("Size must be positive")

    # --- Safety floor enforcement ---
    min_floor = _get_min_kv_cache_bytes(ipc_name)
    # Absolute hard minimum: 64 MB (even with force=True)
    hard_min = 64 * 1024 * 1024
    effective_min = hard_min if force else min_floor

    clamped = False
    original_request = size_bytes
    if size_bytes < effective_min:
        size_bytes = effective_min
        clamped = True

    previous_total = current_info.total_size
    previous_used = current_info.used_size

    update_kv_cache_limit(ipc_name, size_bytes)

    immediate = previous_used <= size_bytes
    if immediate:
        message = (
            f"Limit updated {_format_size(previous_total)} -> "
            f"{_format_size(size_bytes)}. "
            f"Free pages will be reclaimed immediately.")
    else:
        message = (
            f"Limit updated {_format_size(previous_total)} -> "
            f"{_format_size(size_bytes)}. "
            f"Usage ({_format_size(previous_used)}) exceeds new limit"
            f" — shrink mode active, memory reclaimed as requests complete.")

    if clamped:
        message += (
            f" (Requested {_format_size(original_request)} was below "
            f"safety floor {_format_size(effective_min)}, clamped up.)")

    result: Dict[str, Any] = {
        "ipc_name": ipc_name,
        "success": True,
        "previous_total_size": previous_total,
        "previous_total_size_human": _format_size(previous_total),
        "new_total_size": size_bytes,
        "new_total_size_human": _format_size(size_bytes),
        "current_used_size": previous_used,
        "current_used_size_human": _format_size(previous_used),
        "immediate_reclaim": immediate,
        "safety_floor": min_floor,
        "safety_floor_human": _format_size(min_floor),
        "clamped": clamped,
        "message": message,
    }
    if clamped:
        result["requested_size"] = original_request
        result["requested_size_human"] = _format_size(original_request)
    return result


# ------------------------------------------------------------------
# Starlette route handlers
# ------------------------------------------------------------------

def create_router():
    """Create and return Starlette Route objects for kvcache management.

    Uses raw Starlette routes instead of FastAPI's decorator-based routing
    to completely bypass vLLM's custom request-validation middleware.
    """
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def kvcache_status(request: Request) -> JSONResponse:
        """Return memory usage for all active kvcached IPC segments."""
        try:
            ipc_names = _detect_kvcache_ipc_names()
            segments: List[Dict[str, Any]] = []
            for name in ipc_names:
                seg = _build_segment_info(name)
                if seg is not None:
                    segments.append(seg)

            gpu_total = get_total_gpu_memory()
            try:
                import torch
                gpu_free, _ = torch.cuda.mem_get_info()
            except Exception:
                gpu_free = 0

            return JSONResponse(content={
                "self_ipc_name": DEFAULT_IPC_NAME,
                "segments": segments,
                "gpu_total_memory": gpu_total,
                "gpu_total_memory_human":
                    _format_size(gpu_total) if gpu_total else "N/A",
                "gpu_free_memory": gpu_free,
                "gpu_free_memory_human":
                    _format_size(gpu_free) if gpu_free else "N/A",
            })
        except Exception as e:
            logger.error("Error in /kvcache/status: %s", e)
            return JSONResponse(status_code=500,
                                content={"error": str(e)})

    async def kvcache_limit(request: Request) -> JSONResponse:
        """Set an absolute KV cache limit for an IPC segment.

        Body (JSON):
            ipc_name   – (optional) target IPC segment; defaults to self.
            size       – human-readable size, e.g. "5G", "512M"
            size_bytes – exact byte count (alternative to *size*)
            force      – (optional, default false) bypass safety floor
                         (still enforces 64 MB hard minimum)
        """
        try:
            body = await request.json()

            ipc_name = body.get("ipc_name") or DEFAULT_IPC_NAME
            force = bool(body.get("force", False))

            size_bytes: Optional[int] = None
            if 'size_bytes' in body:
                size_bytes = int(body['size_bytes'])
            elif 'size' in body:
                size_bytes = _parse_size(str(body['size']))
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error":
                        "Body must contain 'size' (e.g. '5G') or 'size_bytes'"
                    })

            result = _apply_limit(ipc_name, size_bytes, force=force)
            return JSONResponse(content=result)
        except ValueError as e:
            return JSONResponse(status_code=400,
                                content={"error": str(e)})
        except Exception as e:
            logger.error("Error in /kvcache/limit: %s", e)
            return JSONResponse(status_code=500,
                                content={"error": str(e)})

    async def kvcache_limit_percent(request: Request) -> JSONResponse:
        """Set KV cache limit as a percentage of total GPU memory.

        Body (JSON):
            ipc_name – (optional) defaults to self.
            percent  – float, e.g. 30 means 30% of total GPU memory.
            force    – (optional) bypass safety floor.
        """
        try:
            body = await request.json()

            ipc_name = body.get("ipc_name") or DEFAULT_IPC_NAME
            force = bool(body.get("force", False))

            percent = body.get('percent')
            if percent is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Body must contain 'percent'"})

            percent = float(percent)
            if percent <= 0 or percent > 100:
                return JSONResponse(
                    status_code=400,
                    content={"error": "percent must be between 0 and 100"})

            gpu_total = get_total_gpu_memory()
            if gpu_total <= 0:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error":
                        "CUDA unavailable; cannot compute size from percentage"
                    })

            size_bytes = int(gpu_total * percent / 100.0)
            result = _apply_limit(ipc_name, size_bytes, force=force)
            result["gpu_total_memory"] = gpu_total
            result["gpu_total_memory_human"] = _format_size(gpu_total)
            result["percent"] = percent
            return JSONResponse(content=result)
        except ValueError as e:
            return JSONResponse(status_code=400,
                                content={"error": str(e)})
        except Exception as e:
            logger.error("Error in /kvcache/limit_percent: %s", e)
            return JSONResponse(status_code=500,
                                content={"error": str(e)})

    async def kvcache_trim(request: Request) -> JSONResponse:
        """Force-release reserved pages and prefix cache to reclaim GPU memory.

        This endpoint:
        1. Evicts all prefix-cached blocks (safe: only affects completed
           requests' cached KV data, not in-flight inference).
        2. Sets the KV cache limit to current used_size so that reserved
           (pre-allocated) pages are released on the next alloc cycle.

        The result is that physical GPU memory drops to the minimum needed
        for currently active requests, freeing everything else for other
        Docker instances on the same GPU.

        Body (JSON, all optional):
            ipc_name – target segment (defaults to self).
            target   – (optional) desired limit after trim, e.g. "1G".
                       If omitted, trims to current active usage.
                       Subject to safety floor unless force=true.
            force    – bypass safety floor (still enforces 64 MB hard min).
        """
        try:
            # Parse optional body (trim can be called with no body)
            try:
                body = await request.json()
            except Exception:
                body = {}

            ipc_name = body.get("ipc_name") or DEFAULT_IPC_NAME
            force = bool(body.get("force", False))

            current_info = get_kv_cache_limit(ipc_name)
            if current_info is None:
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": f"IPC segment '{ipc_name}' not found",
                        "available_segments": _detect_kvcache_ipc_names(),
                    })

            previous_total = current_info.total_size
            previous_used = current_info.used_size
            previous_prealloc = current_info.prealloc_size

            # Step 1: Determine target limit
            if 'target' in body:
                target_bytes = _parse_size(str(body['target']))
            elif 'target_bytes' in body:
                target_bytes = int(body['target_bytes'])
            else:
                # Default: trim to current used_size (release everything idle)
                target_bytes = previous_used

            # Step 2: Apply safety floor
            min_floor = _get_min_kv_cache_bytes(ipc_name)
            hard_min = 64 * 1024 * 1024
            effective_min = hard_min if force else min_floor
            clamped = False
            original_target = target_bytes
            if target_bytes < effective_min:
                target_bytes = effective_min
                clamped = True

            # Step 3: Write the new limit to shared memory
            update_kv_cache_limit(ipc_name, target_bytes)

            freed_estimate = max(previous_total - target_bytes, 0)
            message = (
                f"Trim applied: limit {_format_size(previous_total)} -> "
                f"{_format_size(target_bytes)}. "
                f"Estimated {_format_size(freed_estimate)} reclaimable. "
                f"Prefix cache blocks will be evicted as new requests arrive.")
            if clamped:
                message += (
                    f" (Requested target {_format_size(original_target)} "
                    f"clamped to safety floor {_format_size(effective_min)}.)")

            result: Dict[str, Any] = {
                "success": True,
                "ipc_name": ipc_name,
                "previous_total": previous_total,
                "previous_total_human": _format_size(previous_total),
                "previous_used": previous_used,
                "previous_used_human": _format_size(previous_used),
                "previous_prealloc": previous_prealloc,
                "previous_prealloc_human": _format_size(previous_prealloc),
                "new_limit": target_bytes,
                "new_limit_human": _format_size(target_bytes),
                "freed_estimate": freed_estimate,
                "freed_estimate_human": _format_size(freed_estimate),
                "safety_floor": min_floor,
                "safety_floor_human": _format_size(min_floor),
                "clamped": clamped,
                "message": message,
            }
            return JSONResponse(content=result)
        except Exception as e:
            logger.error("Error in /kvcache/trim: %s", e)
            return JSONResponse(status_code=500,
                                content={"error": str(e)})

    async def kvcache_safety_floor(request: Request) -> JSONResponse:
        """Return the current safety floor configuration."""
        try:
            # Try to read model-aware min_safe_limit from the first IPC segment
            ipc_names = _detect_kvcache_ipc_names()
            first_ipc = ipc_names[0] if ipc_names else None
            min_floor = _get_min_kv_cache_bytes(first_ipc)

            # Also read the raw min_safe_limit from shared memory
            model_min_safe = 0
            if first_ipc is not None:
                info = get_kv_cache_limit(first_ipc)
                if info is not None:
                    model_min_safe = info.min_safe_limit

            gpu_total = get_total_gpu_memory()
            return JSONResponse(content={
                "safety_floor": min_floor,
                "safety_floor_human": _format_size(min_floor),
                "model_min_safe_limit": model_min_safe,
                "model_min_safe_limit_human": _format_size(model_min_safe) if model_min_safe > 0 else "N/A (not calculated)",
                "fallback_configured_min_mb": _FALLBACK_MIN_KV_CACHE_MB,
                "gpu_percent_floor": int(gpu_total * 0.02) if gpu_total > 0 else 0,
                "gpu_percent_floor_human": _format_size(int(gpu_total * 0.02)) if gpu_total > 0 else "N/A",
                "hard_minimum": 64 * 1024 * 1024,
                "hard_minimum_human": "64.00 MB",
                "env_vars": {
                    "KVCACHED_MIN_KV_CACHE_MB": "Fallback static floor (default 512)",
                    "KVCACHED_MIN_SAFE_PAGES_PER_LAYER": "Pages per layer for model-aware calculation (default 2)",
                },
                "description": (
                    "The safety floor prevents setting KV cache limits too low. "
                    "It is auto-calculated from model parameters "
                    "(num_layers × num_kv_buffers × page_size × min_safe_pages_per_layer). "
                    "Falls back to max(KVCACHED_MIN_KV_CACHE_MB, 2%% of GPU memory) "
                    "if model parameters are unavailable. "
                    "Use force=true in limit requests to bypass (64 MB hard min still applies)."
                ),
            })
        except Exception as e:
            logger.error("Error in /kvcache/safety_floor: %s", e)
            return JSONResponse(status_code=500,
                                content={"error": str(e)})

    routes = [
        Route("/kvcache/status", kvcache_status, methods=["GET"]),
        Route("/kvcache/limit", kvcache_limit, methods=["POST"]),
        Route("/kvcache/limit_percent", kvcache_limit_percent, methods=["POST"]),
        Route("/kvcache/trim", kvcache_trim, methods=["POST"]),
        Route("/kvcache/safety_floor", kvcache_safety_floor, methods=["GET"]),
    ]
    return routes


def attach_to_app(app) -> None:
    """Attach the kvcache management routes to a FastAPI/Starlette app.

    Called by the autopatch mechanism after vLLM builds its FastAPI app.
    Uses Starlette Route objects mounted directly on the app's router to
    completely bypass vLLM's custom request-validation middleware.
    Safe to call multiple times (idempotent).
    """
    if getattr(app.state, '_kvcached_router_attached', False):
        return
    routes = create_router()
    for route in routes:
        app.routes.append(route)
    app.state._kvcached_router_attached = True
    logger.info(
        "kvcached KV cache management endpoints registered "
        "(/kvcache/status, /kvcache/limit, /kvcache/limit_percent, "
        "/kvcache/trim, /kvcache/safety_floor)")
