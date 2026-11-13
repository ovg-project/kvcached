# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for controller error translation.

Before this change every routing failure collapsed into the same 500
``{"error": "Failed to route request for model ..."}`` response. A bad
sampling parameter, an unknown model and an unreachable engine were
indistinguishable to clients, and the engine's own OpenAI-shaped error body
was dropped on the floor. The frontend now forwards the upstream status and
payload, and classifies router-side failures (404/503/504).
"""

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
from router import (  # noqa: E402
    BackendUnavailableError,
    GatewayTimeoutError,
    LLMRouter,
    ModelNotFoundError,
    UpstreamError,
)
from sleep_manager import SleepConfig, SleepManager  # noqa: E402
from traffic_monitor import TrafficMonitor  # noqa: E402

ENDPOINTS = ["/v1/completions", "/v1/chat/completions"]
BAD_REQUEST_BODY = {
    "error": {
        "message": "temperature must be non-negative, got -5.0",
        "type": "BadRequestError",
        "param": "temperature",
        "code": 400,
    }
}


class _FailingRouter:
    """Router stub that raises the exception under test."""

    def __init__(self, error):
        self._error = error

    async def route_request(self, model, data, path):
        raise self._error


@asynccontextmanager
async def _frontend(router):
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


async def _post(port, path, payload=None, raw=None):
    url = f"http://127.0.0.1:{port}{path}"
    headers = {"Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        if raw is not None:
            async with session.post(url, data=raw, headers=headers) as response:
                return response.status, await response.text()
        async with session.post(url, json=payload, headers=headers) as response:
            return response.status, await response.text()


@pytest.mark.parametrize("path", ENDPOINTS)
@pytest.mark.parametrize("error,status,error_type", [
    (ModelNotFoundError("Model 'nope' is not configured"), 404, "NotFoundError"),
    (BackendUnavailableError("engine is unreachable"), 503,
     "ServiceUnavailableError"),
    (GatewayTimeoutError("engine timed out"), 504, "TimeoutError"),
])
async def test_router_failures_keep_their_status(path, error, status, error_type):
    async with _frontend(_FailingRouter(error)) as port:
        code, body = await _post(port, path, {"model": "m", "prompt": "hi"})

    assert code == status, body
    payload = json.loads(body)
    assert payload["error"]["type"] == error_type
    assert payload["error"]["code"] == status


@pytest.mark.parametrize("path", ENDPOINTS)
async def test_upstream_error_body_is_forwarded_verbatim(path):
    error = UpstreamError(400, BAD_REQUEST_BODY)
    async with _frontend(_FailingRouter(error)) as port:
        code, body = await _post(port, path, {"model": "m", "temperature": -5})

    assert code == 400
    assert json.loads(body) == BAD_REQUEST_BODY


async def test_upstream_429_is_not_rewritten_to_500():
    error = UpstreamError(429, {"error": {"message": "slow down", "code": 429}})
    async with _frontend(_FailingRouter(error)) as port:
        code, body = await _post(port, "/v1/completions", {"model": "m"})

    assert code == 429
    assert json.loads(body)["error"]["message"] == "slow down"


@pytest.mark.parametrize("path", ENDPOINTS)
async def test_malformed_json_body_is_a_400(path):
    async with _frontend(_FailingRouter(RuntimeError("must not be called"))) as port:
        code, body = await _post(port, path, raw=b"{not json")

    assert code == 400, body
    assert "Invalid JSON body" in json.loads(body)["error"]["message"]


async def test_unexpected_failure_stays_a_500():
    async with _frontend(_FailingRouter(RuntimeError("boom"))) as port:
        code, body = await _post(port, "/v1/completions", {"model": "m"})

    assert code == 500
    payload = json.loads(body)
    assert payload["error"]["type"] == "InternalServerError"
    assert "boom" in payload["error"]["message"]


@asynccontextmanager
async def _upstream():
    """Fake engine that answers differently per model name."""

    async def dispatch(request):
        data = await request.json()
        model = data.get("model")
        if model == "ok":
            return web.json_response({"id": "cmpl-1", "choices": []})
        if model == "bad":
            return web.json_response(BAD_REQUEST_BODY, status=400)
        if model == "busy":
            return web.Response(text="engine overloaded", status=503,
                                content_type="text/plain")
        return web.json_response({"error": {"message": "no such model"}}, status=404)

    app = web.Application()
    app.router.add_post("/v1/completions", dispatch)
    server = TestServer(app)
    await server.start_server()
    try:
        yield server.port
    finally:
        await server.close()


def _router(port, names):
    models = {"models": {name: {"endpoint": {"host": "127.0.0.1", "port": port}}
                         for name in names}}
    return LLMRouter(models,
                     sleep_manager=SleepManager(SleepConfig(),
                                                traffic_monitor=TrafficMonitor()),
                     traffic_monitor=TrafficMonitor())


async def test_router_keeps_engine_status_and_payload():
    async with _upstream() as port:
        router = _router(port, ["ok", "bad", "busy"])
        try:
            assert await router.route_request("ok", {"model": "ok"}) == {
                "id": "cmpl-1", "choices": []
            }

            with pytest.raises(UpstreamError) as bad:
                await router.route_request("bad", {"model": "bad"})
            assert bad.value.status == 400
            assert bad.value.payload() == BAD_REQUEST_BODY

            with pytest.raises(UpstreamError) as busy:
                await router.route_request("busy", {"model": "busy"})
            assert busy.value.status == 503
            assert "engine overloaded" in busy.value.payload()["error"]["message"]
        finally:
            await router.close()


async def test_unknown_model_is_a_404():
    async with _upstream() as port:
        router = _router(port, ["ok"])
        try:
            with pytest.raises(ModelNotFoundError):
                await router.route_request("nope", {"model": "nope"})
        finally:
            await router.close()


async def test_unreachable_engine_is_a_503():
    # Bind and release a port so nothing is listening there.
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    dead_port = sock.getsockname()[1]
    sock.close()

    router = _router(dead_port, ["dead"])
    try:
        with pytest.raises(BackendUnavailableError):
            await router.route_request("dead", {"model": "dead"})
    finally:
        await router.close()
