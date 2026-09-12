# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Endpoint-level tests for the kvcached control API."""

import json
import os

import pytest

# Guard on the module under test rather than on one dependency, so that a
# missing piece of the `web` extra skips instead of failing collection.
kvweb = pytest.importorskip("kvcached.cli.kvweb",
                            reason="needs the `web` extra")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from kvcached.cli.utils import (  # noqa: E402
    delete_kv_cache_segment,
    get_ipc_path,
    init_kv_cache_limit,
)

IPC_NAME = "kvweb-test-ipc"
TOTAL_MEM = 10_000_000  # 10 MB


@pytest.fixture(autouse=True)
def no_configured_api_key(monkeypatch):
    """Authentication is opt-in, so keep an ambient key out of every test.

    A key in the environment would otherwise 401 every request in this module,
    not just the tests that are about authentication.
    """
    monkeypatch.delenv(kvweb.API_KEY_ENV_VAR, raising=False)


@pytest.fixture
def client():
    return TestClient(kvweb.app)


@pytest.fixture
def segment():
    """Create a KV cache segment for the test and remove it afterwards."""
    # Clear anything an interrupted run left behind.
    delete_kv_cache_segment(IPC_NAME)
    init_kv_cache_limit(IPC_NAME, TOTAL_MEM)
    yield IPC_NAME
    delete_kv_cache_segment(IPC_NAME)


def test_get_ipc_reports_the_configured_limit(client, segment):
    body = client.get(f"/api/ipcs/{segment}").json()

    assert body["name"] == segment
    assert body["total_bytes"] == TOTAL_MEM
    assert body["free_bytes"] == TOTAL_MEM - body["used_bytes"] - body[
        "prealloc_bytes"]


def test_get_ipc_404s_for_an_unknown_segment(client):
    res = client.get("/api/ipcs/no-such-segment")

    assert res.status_code == 404


def test_set_limit_accepts_a_human_readable_size(client, segment):
    res = client.post(f"/api/ipcs/{segment}/limit", json={"size": "2M"})

    assert res.status_code == 200
    assert res.json()["ipc"]["total_bytes"] == 2 * 1024**2
    assert client.get(f"/api/ipcs/{segment}").json()["total_bytes"] == (2 *
                                                                       1024**2)


@pytest.mark.parametrize("size", ["dummy", "1e999G"])
def test_set_limit_rejects_an_unusable_size(client, segment, size):
    res = client.post(f"/api/ipcs/{segment}/limit", json={"size": size})

    assert res.status_code == 400


def test_set_limit_rejects_a_negative_size(client, segment):
    """A negative limit hides the segment instead of shrinking it.

    total_size is a signed int64 and _detect_kvcache_ipc_names() skips
    anything <= 0, so writing one would drop a live engine's segment out of
    kvtop and out of this API.
    """
    res = client.post(f"/api/ipcs/{segment}/limit", json={"size": "-1G"})

    assert res.status_code == 400
    assert client.get(f"/api/ipcs/{segment}").json()["total_bytes"] == TOTAL_MEM


def test_set_limit_404s_instead_of_creating_a_segment(client):
    """The API must never bring a segment into existence.

    ``init_kv_cache_limit`` zeroes the used/prealloc counters, so writing to a
    name that no engine owns would either invent a phantom segment or corrupt
    the accounting of a live one.
    """
    res = client.post("/api/ipcs/no-such-segment/limit", json={"size": "1G"})

    assert res.status_code == 404
    assert not os.path.exists(get_ipc_path("no-such-segment"))


def test_delete_removes_the_segment(client, segment):
    assert client.delete(f"/api/ipcs/{segment}").status_code == 200

    assert not os.path.exists(get_ipc_path(segment))
    assert client.get(f"/api/ipcs/{segment}").status_code == 404


def test_delete_404s_for_an_unknown_segment(client):
    assert client.delete("/api/ipcs/no-such-segment").status_code == 404


def test_status_lists_detected_segments(client, segment):
    body = client.get("/api/status").json()

    assert segment in [ipc["name"] for ipc in body["ipcs"]]
    assert body["summary"]["ipc_count"] == len(body["ipcs"])


def test_requests_are_unauthenticated_when_no_key_is_configured(client):
    assert client.get("/api/ipcs").status_code == 200


def test_a_configured_key_is_required(client, monkeypatch):
    monkeypatch.setenv(kvweb.API_KEY_ENV_VAR, "s3cret")

    assert client.get("/api/ipcs").status_code == 401
    assert client.get("/api/ipcs",
                      headers={
                          "X-API-Key": "wrong"
                      }).status_code == 401
    assert client.get("/api/ipcs", headers={
        "X-API-Key": "s3cret"
    }).status_code == 200


def test_the_query_parameter_fallback_is_limited_to_the_stream(
        client, monkeypatch):
    """Query strings get logged, so only the read-only stream accepts one."""
    monkeypatch.setenv(kvweb.API_KEY_ENV_VAR, "s3cret")

    assert client.get("/api/ipcs?api_key=s3cret").status_code == 401
    assert client.delete("/api/ipcs/anything?api_key=s3cret").status_code == 401


def test_root_reports_service_metadata(client):
    body = client.get("/").json()

    assert body["name"] == "kvcached control API"
    assert body["version"] == kvweb.__version__
    assert body["docs_url"] == "/docs"
    assert body["openapi_url"] == "/openapi.json"


def test_list_ipcs_includes_the_segment(client, segment):
    assert segment in client.get("/api/ipcs").json()["ipcs"]


def test_set_limit_percent_converts_against_total_gpu_memory(
        client, segment, monkeypatch):
    monkeypatch.setattr(kvweb, "get_total_gpu_memory", lambda: 8 * 1024**3)

    res = client.post(f"/api/ipcs/{segment}/limit-percent", json={"percent": 50})

    assert res.status_code == 200
    assert res.json()["ipc"]["total_bytes"] == 4 * 1024**3


def test_set_limit_percent_404s_for_an_unknown_segment(client, monkeypatch):
    monkeypatch.setattr(kvweb, "get_total_gpu_memory", lambda: 8 * 1024**3)

    res = client.post("/api/ipcs/no-such-segment/limit-percent",
                      json={"percent": 50})

    assert res.status_code == 404
    assert not os.path.exists(get_ipc_path("no-such-segment"))


def test_set_limit_percent_503s_when_gpu_memory_is_unknown(
        client, segment, monkeypatch):
    monkeypatch.setattr(kvweb, "get_total_gpu_memory", lambda: 0)

    res = client.post(f"/api/ipcs/{segment}/limit-percent", json={"percent": 50})

    assert res.status_code == 503


@pytest.mark.parametrize("percent", [-1, 101])
def test_set_limit_percent_rejects_a_value_outside_0_100(client, segment,
                                                        percent):
    res = client.post(f"/api/ipcs/{segment}/limit-percent",
                      json={"percent": percent})

    assert res.status_code == 422


def test_set_limit_requires_a_size(client, segment):
    assert client.post(f"/api/ipcs/{segment}/limit", json={}).status_code == 422


@pytest.mark.parametrize("interval", [0.1, 11])
def test_stream_rejects_an_interval_outside_its_bounds(client, interval):
    # Validation happens before the generator starts, so this cannot hang.
    res = client.get(f"/api/stream?interval={interval}")

    assert res.status_code == 422


async def test_stream_emits_the_status_payload_as_server_sent_events(segment):
    """Exercise the response object rather than the endpoint.

    The generator never returns, so pulling it through TestClient blocks
    forever; taking one chunk from the body iterator is the same code path
    without the server loop.
    """
    response = await kvweb.api_stream_status(interval=0.2)

    assert response.media_type == "text/event-stream"

    chunk = await response.body_iterator.__anext__()
    assert chunk.startswith("data: ") and chunk.endswith("\n\n")

    payload = json.loads(chunk[len("data: "):])
    assert segment in [ipc["name"] for ipc in payload["ipcs"]]


def test_stream_accepts_the_api_key_as_a_query_parameter(client, monkeypatch):
    """The fallback exists for EventSource, which cannot send headers."""
    monkeypatch.setenv(kvweb.API_KEY_ENV_VAR, "s3cret")

    assert client.get("/api/stream?interval=1").status_code == 401

    # A deliberately out-of-range interval: 422 rather than 401 shows the key
    # was accepted and the request reached validation, and stops the endless
    # generator from starting and hanging the test.
    assert client.get("/api/stream?interval=99&api_key=s3cret").status_code == 422
