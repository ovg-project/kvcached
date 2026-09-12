# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""REST API for inspecting and adjusting kvcached KV-cache limits.

Exposes over HTTP what ``kvctl`` and ``kvtop`` already expose on the terminal:
the per-IPC-segment KV-cache accounting kept in ``/dev/shm``, plus the limit
adjustments ``kvctl limit`` performs. Start it with ``kvweb`` or ``kvctl web``.

The server binds to loopback by default and has no authentication unless
``KVCACHED_WEB_API_KEY`` is set, because the mutating endpoints can shrink or
delete the KV cache of a running engine.
"""

import argparse
import asyncio
import json
import os
import secrets
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from kvcached import __version__
from kvcached.cli.kvctl import parse_size
from kvcached.cli.kvtop import _detect_kvcache_ipc_names
from kvcached.cli.utils import (
    _format_size,
    delete_kv_cache_segment,
    get_ipc_name,
    get_kv_cache_limit,
    get_total_gpu_memory,
    update_kv_cache_limit,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

#: Requests must carry this key when the variable is set, as an ``X-API-Key``
#: header. Only :data:`STREAM_PATH` also accepts it as an ``api_key`` query
#: parameter, for clients such as ``EventSource`` that cannot set headers.
API_KEY_ENV_VAR = "KVCACHED_WEB_API_KEY"

STREAM_PATH = "/api/stream"

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class LimitSizeRequest(BaseModel):
    size: str = Field(...,
                      description="Size string, e.g. '512M', '2G', or a plain "
                      "byte count")


class LimitPercentRequest(BaseModel):
    percent: float = Field(...,
                           ge=0.0,
                           le=100.0,
                           description="Percentage of total GPU memory")


# ---------------------------------------------------------------------------
# Status collection
# ---------------------------------------------------------------------------


def get_gpu_info() -> Dict[str, Any]:
    """Return total/used/free GPU memory, or zeros when CUDA is unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            return {
                "available": True,
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": free,
                "usage_percent": round(used / total * 100, 1) if total else 0.0,
                "total_human": _format_size(total),
                "used_human": _format_size(used),
                "free_human": _format_size(free),
            }
    except Exception:  # pragma: no cover - best-effort probe
        pass

    # No CUDA runtime: report capacity if we can still read it, nothing else.
    total = get_total_gpu_memory()
    return {
        "available": False,
        "total_bytes": total,
        "used_bytes": 0,
        "free_bytes": total,
        "usage_percent": 0.0,
        "total_human": _format_size(total),
        "used_human": _format_size(0),
        "free_human": _format_size(total),
    }


def get_ipc_details(name: str) -> Optional[Dict[str, Any]]:
    """Return the accounting for one IPC segment, or None if it is absent."""
    clean_name = get_ipc_name(name)
    mem_info = get_kv_cache_limit(clean_name)
    if mem_info is None:
        return None

    total = int(mem_info.total_size)
    used = int(mem_info.used_size)
    prealloc = int(mem_info.prealloc_size)
    free = max(total - used - prealloc, 0)

    return {
        "name": clean_name,
        "total_bytes": total,
        "used_bytes": used,
        "prealloc_bytes": prealloc,
        "free_bytes": free,
        "usage_percent": round(
            (used + prealloc) / total * 100, 1) if total else 0.0,
        "total_human": _format_size(total),
        "used_human": _format_size(used),
        "prealloc_human": _format_size(prealloc),
        "free_human": _format_size(free),
    }


def get_all_status() -> Dict[str, Any]:
    """Return GPU memory plus the accounting for every detected segment."""
    ipcs: List[Dict[str, Any]] = []
    total_managed = 0
    total_used = 0

    for name in _detect_kvcache_ipc_names():
        details = get_ipc_details(name)
        if details is None:
            # The segment was deleted between detection and readout.
            continue
        ipcs.append(details)
        total_managed += details["total_bytes"]
        total_used += details["used_bytes"] + details["prealloc_bytes"]

    return {
        "timestamp": time.time(),
        "gpu": get_gpu_info(),
        "summary": {
            "ipc_count": len(ipcs),
            "total_managed_bytes": total_managed,
            "total_managed_human": _format_size(total_managed),
            "total_used_bytes": total_used,
            "total_used_human": _format_size(total_used),
        },
        "ipcs": ipcs,
    }


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def require_api_key(request: Request,
                    x_api_key: Optional[str] = Header(default=None)) -> None:
    """Reject the request unless it carries the configured API key.

    A no-op when ``KVCACHED_WEB_API_KEY`` is unset, which keeps the default
    loopback-only deployment usable without any client-side setup.

    Applied to the whole app rather than per route, so a route added later
    cannot be left unauthenticated by accident.
    """
    expected = os.environ.get(API_KEY_ENV_VAR)
    if not expected:
        return

    presented = x_api_key
    if presented is None and request.url.path == STREAM_PATH:
        # EventSource cannot set headers. Query strings end up in access logs
        # and proxy logs, so only the read-only stream accepts a key this way.
        presented = request.query_params.get("api_key")

    if presented is None or not secrets.compare_digest(presented, expected):
        raise HTTPException(status_code=401,
                            detail="Missing or invalid API key")


app = FastAPI(
    title="kvcached control API",
    description=__doc__,
    version=__version__,
    dependencies=[Depends(require_api_key)],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", summary="Service metadata and documentation links")
def api_root() -> Dict[str, Any]:
    return {
        "name": "kvcached control API",
        "version": __version__,
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
    }


@app.get("/api/status", summary="GPU memory and every KV cache segment")
def api_get_status() -> Dict[str, Any]:
    return get_all_status()


@app.get("/api/ipcs", summary="List the names of all detected segments")
def api_list_ipcs() -> Dict[str, List[str]]:
    return {"ipcs": _detect_kvcache_ipc_names()}


@app.get("/api/ipcs/{name}", summary="Accounting for a single segment")
def api_get_ipc(name: str) -> Dict[str, Any]:
    details = get_ipc_details(name)
    if details is None:
        raise HTTPException(status_code=404,
                            detail=f"IPC segment '{name}' not found")
    return details


@app.post("/api/ipcs/{name}/limit", summary="Set a segment's limit in bytes")
def api_set_limit(name: str, req: LimitSizeRequest) -> Dict[str, Any]:
    clean_name = get_ipc_name(name)
    try:
        size_bytes = parse_size(req.size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OverflowError as exc:
        raise HTTPException(status_code=400,
                            detail=f"Size '{req.size}' is too large") from exc

    if size_bytes < 0:
        raise HTTPException(status_code=400,
                            detail=f"Size must not be negative: '{req.size}'")

    # update_kv_cache_limit() takes a write lock, and RwLockedShm creates the
    # backing file when one is missing. Without this check a typo would leave
    # behind a segment that no engine owns but that kvtop still reports.
    if get_kv_cache_limit(clean_name) is None:
        raise HTTPException(status_code=404,
                            detail=f"IPC segment '{clean_name}' not found")

    update_kv_cache_limit(clean_name, size_bytes)

    return {
        "message":
        f"Set the limit of '{clean_name}' to {_format_size(size_bytes)}",
        "ipc": get_ipc_details(clean_name),
    }


@app.post("/api/ipcs/{name}/limit-percent",
          summary="Set a segment's limit as a share of total GPU memory")
def api_set_limit_percent(name: str,
                          req: LimitPercentRequest) -> Dict[str, Any]:
    clean_name = get_ipc_name(name)

    # update_kv_cache_limit() takes a write lock, and RwLockedShm creates the
    # backing file when one is missing. Without this check a typo would leave
    # behind a segment that no engine owns but that kvtop still reports.
    if get_kv_cache_limit(clean_name) is None:
        raise HTTPException(status_code=404,
                            detail=f"IPC segment '{clean_name}' not found")

    gpu_total = get_total_gpu_memory()
    if gpu_total <= 0:
        raise HTTPException(
            status_code=503,
            detail="Total GPU memory is unknown, so a percentage cannot be "
            "converted to bytes")

    size_bytes = int(gpu_total * req.percent / 100.0)
    update_kv_cache_limit(clean_name, size_bytes)

    return {
        "message":
        f"Set the limit of '{clean_name}' to {req.percent}% of GPU memory "
        f"({_format_size(size_bytes)})",
        "ipc":
        get_ipc_details(clean_name),
    }


@app.delete("/api/ipcs/{name}", summary="Delete a segment and its backing file")
def api_delete_ipc(name: str) -> Dict[str, str]:
    clean_name = get_ipc_name(name)
    if not delete_kv_cache_segment(clean_name):
        raise HTTPException(status_code=404,
                            detail=f"IPC segment '{clean_name}' not found")
    return {"message": f"Deleted IPC segment '{clean_name}'"}


@app.get(STREAM_PATH, summary="Server-sent event stream of /api/status")
async def api_stream_status(
    interval: float = Query(1.0, ge=0.2, le=10.0),
    # Consumed by require_api_key, declared here so that /docs shows it.
    api_key: Optional[str] = Query(
        default=None,
        description=f"Alternative to the X-API-Key header, for clients that "
        f"cannot set headers. Only checked when {API_KEY_ENV_VAR} is set."),
) -> StreamingResponse:

    async def event_generator():
        while True:
            yield f"data: {json.dumps(get_all_status())}\n\n"
            await asyncio.sleep(interval)

    return StreamingResponse(event_generator(),
                             media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve(host: str = DEFAULT_HOST,
          port: int = DEFAULT_PORT,
          cors_origins: Optional[List[str]] = None) -> None:
    """Run the API server, warning first if the bind address is reachable."""
    if cors_origins:
        app.add_middleware(CORSMiddleware,
                           allow_origins=list(cors_origins),
                           allow_credentials=True,
                           allow_methods=["*"],
                           allow_headers=["*"])

    if host not in _LOOPBACK_HOSTS and not os.environ.get(API_KEY_ENV_VAR):
        print(f"WARNING: binding to {host} exposes endpoints that can shrink "
              f"or delete the KV cache of a running engine, and no API key is "
              f"configured. Set {API_KEY_ENV_VAR} to require one.")

    print(f"kvcached control API listening on http://{host}:{port} "
          f"(docs at /docs)")
    uvicorn.run(app, host=host, port=port)


def build_arg_parser(
        parser: Optional[argparse.ArgumentParser] = None
) -> argparse.ArgumentParser:
    """Add the server options to *parser*, or to a fresh one."""
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Serve the kvcached control API.")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind address (default: {DEFAULT_HOST}). Binding to a reachable "
        f"address is only safe with {API_KEY_ENV_VAR} set.")
    parser.add_argument("--port",
                        type=int,
                        default=DEFAULT_PORT,
                        help=f"Bind port (default: {DEFAULT_PORT}).")
    parser.add_argument(
        "--cors-origin",
        action="append",
        dest="cors_origins",
        metavar="ORIGIN",
        help="Allow browser requests from ORIGIN. Repeatable. Omit unless a "
        "separate front-end needs to call this API.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    serve(args.host, args.port, args.cors_origins)


if __name__ == "__main__":
    main()
