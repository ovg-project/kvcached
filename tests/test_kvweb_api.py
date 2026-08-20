# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Endpoint-level tests for the kvcached control API."""

import os

import pytest

# Guard on the module under test rather than on one dependency, so that a
# missing piece of the `web` extra skips instead of failing collection.
kvweb = pytest.importorskip("kvcached.cli.kvweb",
                            reason="needs the `web` extra")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from kvcached.cli.utils import (  # noqa: E402
    get_ipc_path,
    init_kv_cache_limit,
)

IPC_NAME = "kvweb-test-ipc"
TOTAL_MEM = 10_000_000  # 10 MB


@pytest.fixture
def client():
    return TestClient(kvweb.app)


@pytest.fixture
def segment():
    """Create a KV cache segment for the test and remove it afterwards."""
    init_kv_cache_limit(IPC_NAME, TOTAL_MEM)
    yield IPC_NAME
    path = get_ipc_path(IPC_NAME)
    if os.path.exists(path):
        os.remove(path)


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


def test_set_limit_rejects_an_unparsable_size(client, segment):
    res = client.post(f"/api/ipcs/{segment}/limit", json={"size": "banana"})

    assert res.status_code == 400


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
    assert kvweb.API_KEY_ENV_VAR not in os.environ
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


def test_a_configured_key_is_accepted_as_a_query_parameter(client, monkeypatch):
    """EventSource cannot set headers, so /api/stream needs this fallback."""
    monkeypatch.setenv(kvweb.API_KEY_ENV_VAR, "s3cret")

    assert client.get("/api/ipcs?api_key=wrong").status_code == 401
    assert client.get("/api/ipcs?api_key=s3cret").status_code == 200
