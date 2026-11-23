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

import aiohttp
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


# --- Request lifetime for streaming responses -------------------------------
#
# ``session.post()`` resolves when the response headers arrive, not when the
# body ends. A streaming request is therefore only half over when the router
# returns it, and the model is still generating while the proxy relays. The
# request must stay in flight until the relay finishes: the sleep manager
# nominates idle models for a sleep call, and vLLM's sleep in abort mode kills
# whatever is running.

from router import LLMRouter  # noqa: E402
from sleep_manager import SleepConfig, SleepManager  # noqa: E402
from traffic_monitor import TrafficMonitor  # noqa: E402


class _ObservingContent:
    """Streams chunks while reporting the monitor's view of the request."""

    def __init__(self, chunks, observe, error=None):
        self._chunks = chunks
        self._observe = observe
        self._error = error

    async def iter_chunked(self, size):
        for chunk in self._chunks:
            self._observe()
            yield chunk
        if self._error is not None:
            raise self._error
        self._observe()


class _FakeBackendResponse(_FakeUpstreamResponse):
    """A backend response whose body is the observing content above."""

    def __init__(self, content):
        self.content = content
        self.released = False


class _FakeSession:
    """Stands in for ``aiohttp.ClientSession`` so the router wraps its own
    ``StreamedResponse`` around a body we control chunk by chunk."""

    closed = False

    def __init__(self, response):
        self._response = response

    async def post(self, url, json=None, timeout=None):
        return self._response


@asynccontextmanager
async def _live_router_frontend(monitor, backend_response):
    """A real LLMRouter and the real proxy, over a fake backend connection."""
    sleep_manager = SleepManager(SleepConfig(idle_threshold_seconds=0),
                                 traffic_monitor=monitor)
    router = LLMRouter(
        {"models": {"m": {"endpoint": {"host": "127.0.0.1", "port": 1}}}},
        sleep_manager=sleep_manager,
        traffic_monitor=monitor,
    )
    real_session = router.session
    router.session = _FakeSession(backend_response)

    frontend = object.__new__(frontend_module.MultiLLMFrontend)
    frontend.router = router
    app = web.Application()
    app.router.add_post("/v1/completions", frontend.handle_completion)
    server = TestServer(app)
    await server.start_server()
    try:
        yield router, server.port, sleep_manager
    finally:
        await server.close()
        router.session = real_session
        await router.close()


async def _post_stream(port, path="/v1/completions"):
    """POST a streaming request; a truncated body is reported, not raised."""
    url = f"http://127.0.0.1:{port}{path}"
    chunks = []
    error = None
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"model": "m", "stream": True}) as resp:
            status = resp.status
            try:
                async for chunk in resp.content.iter_chunked(1024):
                    chunks.append(chunk)
            except aiohttp.ClientPayloadError as exc:
                error = exc
    return status, b"".join(chunks), error


async def test_streaming_request_stays_in_flight_until_the_relay_ends():
    """While the body is relayed the model is busy; only the relay ends it."""
    monitor = TrafficMonitor(idle_threshold_seconds=0)
    seen: "list[tuple[int, int]]" = []

    def observe():
        stats = monitor.get_model_stats("m")
        # (-1, -1) means the model is not tracked at all.
        seen.append((-1, -1) if stats is None
                    else (stats.inflight_requests, stats.total_requests))

    content = _ObservingContent([b"data: 1\n\n", b"data: 2\n\n"], observe)
    backend = _FakeBackendResponse(content)

    async with _live_router_frontend(monitor, backend) as (_, port, _sleep):
        status, body, error = await _post_stream(port)

    assert status == 200
    assert error is None
    assert body == b"data: 1\n\ndata: 2\n\n"

    # Every mid-relay observation: in flight, not yet completed. Recording the
    # end at header time would show (0, 1) here.
    assert seen, "the body must have been relayed in chunks"
    assert set(seen) == {(1, 0)}, seen

    stats = monitor.get_model_stats("m")
    assert stats.inflight_requests == 0
    assert stats.total_requests == 1
    assert stats.successful_requests == 1
    assert stats.failed_requests == 0
    assert backend.released, "the upstream connection must be released"


async def test_a_sleep_candidate_is_never_a_streaming_model():
    """The candidate list stays empty for as long as a stream is running."""
    monitor = TrafficMonitor(idle_threshold_seconds=0)
    candidates = []

    def observe():
        candidates.append(monitor.get_idle_models(0))

    content = _ObservingContent([b"data: 1\n\n", b"data: 2\n\n"], observe)
    backend = _FakeBackendResponse(content)

    async with _live_router_frontend(monitor, backend) as (_, port, sleep_manager):
        await _post_stream(port)
        # A zero-second threshold: the model becomes a candidate immediately
        # after the stream finishes, and only then.
        assert sleep_manager.get_sleep_candidates() == ["m"]

    assert candidates and all(c == [] for c in candidates), candidates


async def test_failed_stream_records_a_failure_and_clears_the_in_flight_count():
    """A stream that breaks mid-body is a failed request, not a silent one."""
    monitor = TrafficMonitor(idle_threshold_seconds=0)

    def observe():
        pass

    content = _ObservingContent([b"data: 1\n\n"], observe,
                                error=ConnectionResetError("upstream closed"))
    backend = _FakeBackendResponse(content)

    async with _live_router_frontend(monitor, backend) as (_, port, _):
        status, body, error = await _post_stream(port)

    stats = monitor.get_model_stats("m")
    assert stats.inflight_requests == 0
    assert stats.total_requests == 1
    assert stats.successful_requests == 0
    assert stats.failed_requests == 1
    assert backend.released
    # Headers were already committed, so the client observes a truncated body
    # rather than a second HTTP response.
    assert status == 200
    assert body == b"data: 1\n\n"
    assert error is not None


async def test_a_cancelled_relay_is_recorded_as_a_failure():
    """Cancelling the relay must not look like a completed request."""
    monitor = TrafficMonitor(idle_threshold_seconds=0)

    def observe():
        pass

    content = _ObservingContent([b"data: 1\n\n"], observe,
                                error=asyncio.CancelledError())
    backend = _FakeBackendResponse(content)

    async with _live_router_frontend(monitor, backend) as (_, port, _):
        status, body, error = await _post_stream(port)

    stats = monitor.get_model_stats("m")
    assert stats.inflight_requests == 0
    assert stats.total_requests == 1
    assert stats.successful_requests == 0
    assert stats.failed_requests == 1
    assert backend.released
    assert status == 200
    assert body == b"data: 1\n\n"
    assert error is not None


async def test_a_relay_that_never_starts_leaves_no_request_in_flight(
        monkeypatch):
    """A failure between routing and the relay must still release the stream."""
    monitor = TrafficMonitor(idle_threshold_seconds=0)

    def boom(*args, **kwargs):
        raise RuntimeError("downstream header construction failed")

    backend = _FakeBackendResponse(_ObservingContent([b"data: 1\n\n"], lambda: None))
    monkeypatch.setattr(frontend_module.web, "StreamResponse", boom)

    async with _live_router_frontend(monitor, backend) as (_, port, sleep_manager):
        status, _body, _error = await _post_stream(port)
        # Released, so the model is not pinned busy for the rest of its life.
        assert sleep_manager.get_sleep_candidates() == ["m"]

    assert status == 500
    stats = monitor.get_model_stats("m")
    assert stats.inflight_requests == 0
    assert backend.released


async def test_an_aborted_generation_is_recorded_as_a_failure():
    """``finish_reason: "abort"`` is a killed generation, not a success."""
    monitor = TrafficMonitor(idle_threshold_seconds=0)
    chunks = [
        b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n',
        b'data: {"choices": [{"delta": {}, "finish_reason": "abort"}]}\n\n',
    ]
    backend = _FakeBackendResponse(_ObservingContent(chunks, lambda: None))

    async with _live_router_frontend(monitor, backend) as (_, port, _):
        status, body, error = await _post_stream(port)

    assert status == 200
    assert error is None
    # The client still gets exactly what the engine sent.
    assert body == b"".join(chunks)

    stats = monitor.get_model_stats("m")
    assert stats.inflight_requests == 0
    assert stats.total_requests == 1
    assert stats.successful_requests == 0
    assert stats.failed_requests == 1


async def test_a_split_abort_event_is_still_recorded_as_a_failure():
    """The abort marker may straddle two reads; detect it on the full event."""
    monitor = TrafficMonitor(idle_threshold_seconds=0)
    chunks = [
        b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n',
        b'data: {"choices": [{"delta": {}, "finish_reas',
        b'on": "abort"}]}\n\n',
    ]
    backend = _FakeBackendResponse(_ObservingContent(chunks, lambda: None))

    async with _live_router_frontend(monitor, backend) as (_, port, _):
        status, body, error = await _post_stream(port)

    assert status == 200
    assert error is None
    assert body == b"".join(chunks)

    stats = monitor.get_model_stats("m")
    assert stats.inflight_requests == 0
    assert stats.total_requests == 1
    assert stats.successful_requests == 0
    assert stats.failed_requests == 1


async def test_abort_mentioned_inside_generated_text_is_not_a_failure():
    """Only the structural field counts; escaped content is just text."""
    monitor = TrafficMonitor(idle_threshold_seconds=0)
    chunks = [
        b'data: {"choices": [{"delta": {"content": '
        b'"\\"finish_reason\\": \\"abort\\""}}]}\n\n',
        b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n',
    ]
    backend = _FakeBackendResponse(_ObservingContent(chunks, lambda: None))

    async with _live_router_frontend(monitor, backend) as (_, port, _):
        await _post_stream(port)

    stats = monitor.get_model_stats("m")
    assert stats.successful_requests == 1
    assert stats.failed_requests == 0
