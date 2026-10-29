# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Controller lifecycle contracts over controlled HTTP backends, without a GPU."""

import asyncio
import socket
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientTimeout, web
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).parent.parent / "controller"))

from frontend import MultiLLMFrontend
from sleep_manager import SleepConfig, SleepManager
from traffic_monitor import TrafficMonitor


async def eventually(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(wait(), 2)


@pytest.fixture(params=[False, True], ids=["write-detects-disconnect", "handler-cancellation"])
async def service(monkeypatch, request):
    monkeypatch.setattr("frontend.set_ulimit", lambda: None)
    entered, finish, extra = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def backend(request):
        data = await request.json()
        if data.get("before_headers"):
            entered.set()
            await finish.wait()
        if not data.get("stream"):
            return web.json_response({"ok": True})
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"text":"one"}\n\n')
        entered.set()
        await extra.wait()
        await response.write(b'data: {"text":"two"}\n\n')
        await finish.wait()
        if data.get("truncate"):
            request.transport.close()
            return response
        await response.write(b'data: [DONE]\n\n')
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post('/{tail:.*}', backend)
    async with TestServer(app) as upstream:
        config = SleepConfig(idle_threshold_seconds=0, min_sleep_duration=0)
        config.vllm_models_config = {"model": {"host": upstream.host, "port": str(upstream.port)}}
        frontend = MultiLLMFrontend(0, {"model": {"host": upstream.host, "port": upstream.port}}, config)
        manager = frontend.sleep_manager
        manager._call_vllm_sleep_api = AsyncMock(return_value=True)
        manager._call_vllm_wakeup_api = AsyncMock(return_value=True)
        async with TestClient(TestServer(frontend.app, handler_cancellation=request.param)) as client:
            try:
                yield frontend, client, entered, finish, extra
            finally:
                finish.set()
                extra.set()
                await frontend.router.close()


def test_active_request_and_idempotent_end(monkeypatch):
    now = 100.0
    monkeypatch.setattr("traffic_monitor.time.time", lambda: now)
    monitor = TrafficMonitor()
    request = monitor.record_request_start("model", "/generate")
    now = 1000.0
    assert monitor.get_idle_models() == []
    monitor.record_request_end(request)
    monitor.record_request_end(request, success=False)
    stats = monitor.get_model_stats("model")
    assert stats.active_requests == 0
    assert stats.total_requests == stats.successful_requests == 1
    assert stats.failed_requests == 0
    assert monitor.get_idle_models() == []
    now = 1301.0
    assert monitor.get_idle_models() == ["model"]


@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
async def test_stream_remains_active_until_body_completion(service, endpoint):
    frontend, client, entered, finish, extra = service
    response = await client.post(endpoint, json={"model": "model", "stream": True})
    await entered.wait()
    await response.content.readline()
    stats = frontend.traffic_monitor.get_model_stats("model")
    assert stats.total_requests == 0
    assert frontend.sleep_manager.get_sleep_candidates() == []
    assert not await frontend.sleep_manager.put_model_to_sleep("model")
    frontend.sleep_manager._call_vllm_sleep_api.assert_not_called()
    extra.set()
    finish.set()
    assert b'[DONE]' in await response.read()
    await eventually(lambda: stats.total_requests == 1)
    assert stats.active_requests == 0
    assert stats.successful_requests == 1
    assert await frontend.sleep_manager.put_model_to_sleep("model")


async def test_disconnected_client_drains_upstream_before_sleep(service):
    frontend, client, entered, finish, extra = service
    response = await client.post('/v1/completions', json={"model": "model", "stream": True})
    await entered.wait()
    await response.content.readline()
    response.close()
    await asyncio.sleep(0.01)
    extra.set()  # Default aiohttp handlers discover disconnect on the next write.
    await eventually(lambda: bool(frontend.router._drains))
    stats = frontend.traffic_monitor.get_model_stats("model")
    assert stats.active_requests == 1
    assert not await frontend.sleep_manager.put_model_to_sleep("model")
    finish.set()
    await eventually(lambda: stats.active_requests == 0)
    assert not stats.backend_state_unknown
    assert stats.failed_requests == 1
    assert not frontend.router._drains
    assert await frontend.sleep_manager.put_model_to_sleep("model")


async def test_cancellation_before_headers_is_not_claimed_drained(service):
    frontend, _, entered, finish, _ = service
    task = asyncio.create_task(frontend.router.route_request("model", {"before_headers": True}))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    stats = frontend.traffic_monitor.get_model_stats("model")
    assert stats.active_requests == 0
    assert stats.total_requests == 1
    assert stats.backend_state_unknown
    assert not await frontend.sleep_manager.put_model_to_sleep("model")
    finish.set()


async def test_nonstreaming_request_is_active_while_waiting_for_headers(service):
    frontend, _, entered, finish, _ = service
    task = asyncio.create_task(frontend.router.route_request("model", {"before_headers": True}))
    await entered.wait()
    assert frontend.sleep_manager.get_sleep_candidates() == []
    assert not await frontend.sleep_manager.put_model_to_sleep("model")
    finish.set()
    assert await task == {"ok": True}
    stats = frontend.traffic_monitor.get_model_stats("model")
    assert stats.active_requests == 0
    assert stats.successful_requests == 1
    assert not stats.backend_state_unknown


async def test_truncated_upstream_releases_count_but_blocks_sleep(service):
    frontend, _, entered, finish, extra = service
    response = await frontend.router.route_request("model", {"stream": True, "truncate": True})
    await entered.wait()
    frontend.router.finish_stream(response, success=False)
    frontend.router.finish_stream(response, success=False)
    extra.set()
    finish.set()
    stats = frontend.traffic_monitor.get_model_stats("model")
    await eventually(lambda: stats.active_requests == 0)
    assert stats.total_requests == 1
    assert stats.backend_state_unknown
    assert frontend.sleep_manager.get_sleep_candidates() == []


async def test_shutdown_before_drain_task_starts_does_not_leak(service):
    frontend, _, entered, _, _ = service
    response = await frontend.router.route_request("model", {"stream": True})
    await entered.wait()
    frontend.router.finish_stream(response, success=False)
    await frontend.router.close()
    stats = frontend.traffic_monitor.get_model_stats("model")
    assert stats.active_requests == 0
    assert stats.total_requests == 1
    assert not frontend.router._streams
    assert not frontend.router._drains


async def test_drain_retains_original_upstream_deadline(service, monkeypatch):
    frontend, _, entered, _, _ = service
    post = frontend.router.session.post

    def short_deadline(*args, **kwargs):
        kwargs["timeout"] = ClientTimeout(total=0.25)
        return post(*args, **kwargs)

    monkeypatch.setattr(frontend.router.session, "post", short_deadline)
    response = await frontend.router.route_request("model", {"stream": True})
    await entered.wait()
    frontend.router.finish_stream(response, success=False)
    stats = frontend.traffic_monitor.get_model_stats("model")
    await eventually(lambda: stats.active_requests == 0)
    assert stats.backend_state_unknown
    assert not frontend.router._streams
    assert not frontend.router._drains


async def test_sleep_in_progress_blocks_admission_then_wakes_once(service):
    frontend, _, entered, finish, extra = service
    manager = frontend.sleep_manager
    sleeping, release = asyncio.Event(), asyncio.Event()

    async def sleep(*args, **kwargs):
        sleeping.set()
        await release.wait()
        return True

    manager._call_vllm_sleep_api = AsyncMock(side_effect=sleep)
    sleep_task = asyncio.create_task(manager.put_model_to_sleep("model", manual=True))
    await sleeping.wait()
    requests = [asyncio.create_task(frontend.router.route_request("model", {"stream": True})) for _ in range(12)]
    await asyncio.sleep(0)
    assert not entered.is_set()
    release.set()
    assert await sleep_task
    responses = await asyncio.gather(*requests)
    assert all(response is not None for response in responses)
    manager._call_vllm_wakeup_api.assert_awaited_once()
    assert not manager.is_model_sleeping("model")
    assert frontend.traffic_monitor.get_model_stats("model").active_requests == 12
    for response in responses:
        frontend.router.finish_stream(response, success=False)
    extra.set()
    finish.set()
    await eventually(lambda: frontend.traffic_monitor.get_model_stats("model").active_requests == 0)
    assert frontend.traffic_monitor.get_model_stats("model").total_requests == 12


async def test_repeated_concurrent_drains_return_to_quiescence(service):
    frontend, _, entered, finish, extra = service
    for cycle in range(20):
        entered.clear()
        finish.clear()
        extra.clear()
        responses = await asyncio.gather(*(
            frontend.router.route_request("model", {"stream": True}) for _ in range(8)
        ))
        stats = frontend.traffic_monitor.get_model_stats("model")
        assert stats.active_requests == 8
        for response in responses:
            frontend.router.finish_stream(response, success=False)
        extra.set()
        finish.set()
        await eventually(lambda: stats.active_requests == 0)
        assert stats.total_requests == (cycle + 1) * 8
        assert not stats.backend_state_unknown
        assert not frontend.router._streams
        assert not frontend.router._drains


async def test_stale_idle_candidate_is_rechecked_under_transition_lock(monkeypatch):
    now = 100.0
    monkeypatch.setattr("traffic_monitor.time.time", lambda: now)
    monitor = TrafficMonitor()
    manager = SleepManager(SleepConfig(idle_threshold_seconds=10), traffic_monitor=monitor)
    request = monitor.record_request_start("model", "/generate")
    monitor.record_request_end(request)
    now = 200.0
    assert manager.get_sleep_candidates() == ["model"]
    async with manager.request_admission("model"):
        pending_sleep = asyncio.create_task(manager.put_model_to_sleep("model"))
        request = monitor.record_request_start("model", "/generate")
        monitor.record_request_end(request)
    assert not await pending_sleep
    assert not manager.is_model_sleeping("model")


@pytest.mark.parametrize("error", [asyncio.TimeoutError(), ConnectionError("lost response")])
async def test_upstream_failure_finishes_once_and_marks_unknown(service, error):
    frontend, _, _, _, _ = service
    frontend.router.session.post = AsyncMock(side_effect=error)
    assert await frontend.router.route_request("model", {}) is None
    stats = frontend.traffic_monitor.get_model_stats("model")
    assert stats.active_requests == 0
    assert stats.total_requests == stats.failed_requests == 1
    assert stats.backend_state_unknown
    assert frontend.sleep_manager.get_sleep_candidates() == []


async def test_refused_connection_does_not_poison_later_idle_sleep(service):
    frontend, client, _, _, _ = service
    endpoint = frontend.router.models["model"].endpoint
    original_port = endpoint.port
    # Reserve a real non-listening TCP port: no request can reach an engine.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        endpoint.port = reserved.getsockname()[1]
        try:
            response = await client.post('/v1/completions', json={"model": "model"})
            assert response.status == 500
            await response.read()
        finally:
            endpoint.port = original_port
    stats = frontend.traffic_monitor.get_model_stats("model")
    assert stats.active_requests == 0
    assert stats.total_requests == stats.failed_requests == 1
    assert not stats.backend_state_unknown
    response = await client.post('/v1/completions', json={"model": "model"})
    assert response.status == 200
    assert await response.json() == {"ok": True}
    assert await frontend.sleep_manager.put_model_to_sleep("model")


@pytest.mark.parametrize("mode", ["abort", "wait"])
async def test_manual_vllm_sleep_keeps_native_drain_contract(mode):
    seen = []

    async def sleep(request):
        seen.append(dict(request.query))
        return web.Response()

    app = web.Application()
    app.router.add_post('/sleep', sleep)
    async with TestServer(app) as server:
        monitor = TrafficMonitor()
        manager = SleepManager(SleepConfig(
            vllm_sleep_mode=mode,
            vllm_models_config={"model": {"host": server.host, "port": str(server.port)}}
        ), traffic_monitor=monitor)
        request = monitor.record_request_start("model", "/v1/completions")
        assert not await manager.put_model_to_sleep("model")
        assert not seen
        assert await manager.put_model_to_sleep("model", manual=True)
        assert seen == [{"level": "1", "mode": mode}]
        # Native sleep owns abort/wait; local accounting still completes once.
        assert monitor.get_model_stats("model").active_requests == 1
        monitor.record_request_end(request)
        assert monitor.get_model_stats("model").active_requests == 0


async def test_manual_sglang_release_still_requires_drained_requests():
    monitor = TrafficMonitor()
    manager = SleepManager(SleepConfig(
        sglang_models_config={"model": {"host": "127.0.0.1", "port": "1"}}
    ), traffic_monitor=monitor)
    manager._call_sglang_release_api = AsyncMock(return_value=True)
    request = monitor.record_request_start("model", "/v1/completions")
    assert not await manager.put_model_to_sleep("model", manual=True)
    manager._call_sglang_release_api.assert_not_called()
    monitor.record_request_end(request)
    assert await manager.put_model_to_sleep("model", manual=True)
    manager._call_sglang_release_api.assert_awaited_once()
