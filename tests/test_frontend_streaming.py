# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the controller proxy's streaming error path (#610).

Once ``StreamResponse.prepare()`` has written the status line and headers, a
second HTTP response can no longer be sent. Returning a fresh
``web.Response(status=500)`` after an upstream stream failure appends a new
status line and headers to the unfinished chunked body, which clients fail to
parse: aiohttp C-parser clients hang until their own timeout and Python-parser
clients report invalid chunk framing. The proxy must instead truncate the
committed stream and let the client observe the failure.
"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "controller"))

import frontend as frontend_module  # noqa: E402

COMPLETION_ENDPOINTS = ["/v1/completions", "/v1/chat/completions"]


class _FakeContent:
    """Stand-in for ``aiohttp.StreamReader`` exposed by a backend response."""

    def __init__(self, chunks, error=None):
        self._chunks = list(chunks)
        self._error = error

    async def iter_chunked(self, size):
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error


class _FakeUpstreamResponse:
    """Minimal aiohttp ``ClientResponse`` the router would hand back."""

    status = 200
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self, chunks, error=None):
        self.content = _FakeContent(chunks, error)
        self.released = False

    async def release(self):
        self.released = True


class _FakeRouter:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.calls = []

    async def route_request(self, model, data, path):
        self.calls.append((model, path))
        if self._error is not None:
            raise self._error
        return self._result


@asynccontextmanager
async def _running_frontend(router):
    """Serve the two completion handlers over a real aiohttp socket."""
    frontend = object.__new__(frontend_module.MultiLLMFrontend)
    frontend.router = router

    app = web.Application()
    app.router.add_post("/v1/completions", frontend.handle_completion)
    app.router.add_post("/v1/chat/completions", frontend.handle_chat_completion)

    server = TestServer(app)
    await server.start_server()
    try:
        yield server.port
    finally:
        await server.close()


async def _post_raw(port, path, payload, timeout=5.0):
    """POST with a hand-written HTTP request and return the raw response bytes."""
    body = json.dumps(payload).encode()
    head = (f"POST {path} HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: application/json\r\n"
            "Connection: close\r\n"
            f"Content-Length: {len(body)}\r\n\r\n").encode()

    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(head + body)
    await writer.drain()

    raw = b""
    try:
        while True:
            chunk = await asyncio.wait_for(reader.read(65536), timeout=timeout)
            if not chunk:
                break
            raw += chunk
    except (asyncio.TimeoutError, ConnectionResetError):
        pass
    writer.close()
    return raw


@pytest.mark.parametrize("path", COMPLETION_ENDPOINTS)
async def test_upstream_failure_after_headers_emits_one_response(path):
    """A failed upstream stream must not append a second HTTP response."""
    upstream = _FakeUpstreamResponse([b'data: {"choices": []}\n\n'],
                                     error=ConnectionResetError("upstream closed"))
    async with _running_frontend(_FakeRouter(upstream)) as port:
        raw = await _post_raw(port, path, {"model": "m", "prompt": "hi"})

    assert raw.count(b"HTTP/1.1 ") == 1, raw
    assert b"200 OK" in raw
    assert b"500 Internal Server Error" not in raw
    assert b'data: {"choices": []}' in raw
    # The body stays truncated: completing the chunked encoding would make
    # clients mistake a partial answer for a finished one.
    assert not raw.endswith(b"0\r\n\r\n"), raw
    assert upstream.released, "the upstream response must be released"


@pytest.mark.parametrize("path", COMPLETION_ENDPOINTS)
async def test_successful_stream_is_relayed_and_terminated(path):
    """The healthy path still relays every chunk and ends the chunked body."""
    upstream = _FakeUpstreamResponse([b"data: 1\n\n", b"data: 2\n\n"])
    async with _running_frontend(_FakeRouter(upstream)) as port:
        raw = await _post_raw(port, path, {"model": "m", "prompt": "hi"})

    assert raw.count(b"HTTP/1.1 ") == 1, raw
    assert b"data: 1" in raw and b"data: 2" in raw
    assert raw.endswith(b"0\r\n\r\n"), raw
    assert upstream.released


async def test_failure_before_headers_still_returns_json_error():
    """Errors raised before prepare() keep the ordinary JSON error response."""
    router = _FakeRouter(error=RuntimeError("routing failed"))
    async with _running_frontend(router) as port:
        raw = await _post_raw(port, "/v1/completions", {"model": "m", "prompt": "hi"})

    assert raw.count(b"HTTP/1.1 ") == 1, raw
    assert b"500 Internal Server Error" in raw
    assert b"routing failed" in raw


async def test_missing_model_is_rejected_before_routing():
    """A request without a model never reaches the router or a stream."""
    router = _FakeRouter(_FakeUpstreamResponse([]))
    async with _running_frontend(router) as port:
        raw = await _post_raw(port, "/v1/completions", {"prompt": "hi"})

    assert b"400 Bad Request" in raw
    assert b"model parameter is required" in raw
    assert router.calls == []
