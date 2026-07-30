# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import socket
import sys
import time
import types

import pytest


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),  # type: ignore[attr-defined]
    reason="requires Unix sockets",
)
def test_worker_meminfo_uses_bound_device(monkeypatch, tmp_path):
    import kvcached.tp_ipc_util as tp_ipc_util

    monkeypatch.setattr(tp_ipc_util, "SOCKET_DIR", str(tmp_path))
    selected_devices = []

    class DeviceContext:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            selected_devices.append(self.device)

        def __exit__(self, *args):
            return False

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        device=DeviceContext,
        mem_get_info=lambda: (1234, 5678),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    tp_ipc_util.start_worker_listener_thread(
        rank=0, pp_rank=0, device_index=2)
    socket_path = tmp_path / "w0.sock"
    deadline = time.monotonic() + 2.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists(), "worker listener socket was not created"

    assert tp_ipc_util.query_worker_cuda_mem_get_info(
        tp_size=1,
        pp_rank=0,
        timeout=1.0,
    ) == (1234, 5678)
    assert selected_devices == [2]


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),  # type: ignore[attr-defined]
    reason="requires Unix sockets",
)
def test_worker_listener_survives_disconnected_meminfo_client(monkeypatch,
                                                              tmp_path):
    import kvcached.tp_ipc_util as tp_ipc_util

    monkeypatch.setattr(tp_ipc_util, "SOCKET_DIR", str(tmp_path))
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        mem_get_info=lambda: (1234, 5678))
    monkeypatch.setitem(sys.modules, "torch", torch)

    tp_ipc_util.start_worker_listener_thread(rank=0, pp_rank=0)
    socket_path = tmp_path / "w0.sock"
    deadline = time.monotonic() + 2.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists(), "worker listener socket was not created"

    with socket.socket(
            socket.AF_UNIX,  # type: ignore[attr-defined]
            socket.SOCK_STREAM) as sock:
        sock.connect(str(socket_path))
        tp_ipc_util.send_msg(sock, {"cmd": "cuda_mem_get_info"})

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            result = tp_ipc_util.query_worker_cuda_mem_get_info(
                tp_size=1, pp_rank=0, timeout=0.5)
            break
        except (ConnectionError, ConnectionRefusedError, TimeoutError):
            time.sleep(0.01)
    else:
        raise AssertionError("worker listener did not recover after disconnect")

    assert result == (1234, 5678)


def test_worker_meminfo_queries_representative_rank_zero(monkeypatch):
    import kvcached.tp_ipc_util as tp_ipc_util

    queried_ranks = []

    async def fake_send(rank, message, pp_rank):
        queried_ranks.append((pp_rank, rank))
        return {
            "status": "success",
            "free_bytes": 4000 - rank * 100,
            "total_bytes": 8000 + rank * 100,
        }

    monkeypatch.setattr(tp_ipc_util, "_send_and_receive_message", fake_send)

    assert asyncio.run(tp_ipc_util._query_worker_cuda_mem_get_info(
        tp_size=4, pp_rank=2)) == (4000, 8000)
    assert queried_ranks == [(2, 0)]


def test_worker_meminfo_error_identifies_pp_and_rank(monkeypatch):
    import kvcached.tp_ipc_util as tp_ipc_util

    async def fake_send(rank, message, pp_rank):
        raise ConnectionError("listener unavailable")

    monkeypatch.setattr(tp_ipc_util, "_send_and_receive_message", fake_send)

    with pytest.raises(RuntimeError,
                       match="Worker pp3/rank0.*listener unavailable"):
        asyncio.run(tp_ipc_util._query_worker_cuda_mem_get_info(
            tp_size=2, pp_rank=3))


def test_worker_meminfo_sync_wrapper_waits_for_async_response(monkeypatch):
    import kvcached.tp_ipc_util as tp_ipc_util

    async def fake_query(tp_size, pp_rank):
        await asyncio.sleep(0)
        return 1234, 5678

    monkeypatch.setattr(tp_ipc_util, "_query_worker_cuda_mem_get_info",
                        fake_query)

    assert tp_ipc_util.query_worker_cuda_mem_get_info(
        tp_size=4, pp_rank=1, timeout=0.5) == (1234, 5678)


def test_worker_meminfo_sync_wrapper_times_out(monkeypatch):
    import kvcached.tp_ipc_util as tp_ipc_util

    async def never_finishes(tp_size, pp_rank):
        await asyncio.sleep(1)
        return 1234, 5678

    monkeypatch.setattr(tp_ipc_util, "_query_worker_cuda_mem_get_info",
                        never_finishes)

    with pytest.raises(asyncio.TimeoutError):
        tp_ipc_util.query_worker_cuda_mem_get_info(
            tp_size=4, pp_rank=1, timeout=0.01)


def test_meminfo_provider_retries_worker_runtime_error(monkeypatch):
    import kvcached.meminfo_provider as meminfo_provider

    responses = [RuntimeError("workers are starting"), (1234, 5678)]

    def fake_query(*args, **kwargs):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(meminfo_provider, "query_worker_cuda_mem_get_info",
                        fake_query)
    monkeypatch.setattr(meminfo_provider.time, "sleep", lambda _: None)

    assert meminfo_provider.query_mem_info(2) == (1234, 5678)
    assert responses == []


def test_meminfo_provider_queries_tp0_each_refresh(monkeypatch):
    import kvcached.meminfo_provider as meminfo_provider

    queries = []
    results = [(3000, 8000), (2900, 8000)]

    def query(tp_size, pp_rank, timeout):
        queries.append((tp_size, pp_rank, timeout))
        return results.pop(0)

    monkeypatch.setattr(meminfo_provider, "query_worker_cuda_mem_get_info",
                        query)

    assert meminfo_provider.query_mem_info(4, pp_rank=3) == (3000, 8000)
    assert meminfo_provider.query_mem_info(4, pp_rank=3) == (2900, 8000)
    assert [(tp, pp) for tp, pp, _ in queries] == [(4, 3), (4, 3)]


def test_meminfo_provider_fails_closed_with_dfx_context(monkeypatch):
    import kvcached.meminfo_provider as meminfo_provider

    clock = iter([0.0, 0.0, 0.02, 0.02, 0.02])
    errors = []

    class Logger:
        def debug(self, *args, **kwargs):
            pass

        def error(self, message, *args):
            errors.append(message % args)

    def query(tp_size, pp_rank, timeout):
        raise RuntimeError("rank2 listener unavailable")

    monkeypatch.setattr(meminfo_provider, "query_worker_cuda_mem_get_info",
                        query)
    monkeypatch.setattr(meminfo_provider.time, "sleep", lambda _: None)
    monkeypatch.setattr(meminfo_provider.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(meminfo_provider, "MEMINFO_QUERY_TIMEOUT", 0.01)
    monkeypatch.setattr(meminfo_provider, "logger", Logger())

    with pytest.raises(TimeoutError, match="worker CUDA memory info"):
        meminfo_provider.query_mem_info(4, pp_rank=2)

    assert "pp=2 tp=4 attempts=1" in errors[0]
    assert "rank2 listener unavailable" in errors[0]
    assert "refusing new physical mappings" in errors[0]
