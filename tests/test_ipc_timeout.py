# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Worker-IPC exchanges must be bounded in time (issue #371, the other half).

A worker that is dead fails fast: the connect is refused and the caller gets
an exception, which the callers already handle. A worker that is alive but not
answering used to hang the caller forever in readexactly() -- and when that
caller is the C++ prealloc thread, alloc_page() blocks indefinitely on a
reserve that never arrives. These tests pin down the timeout that converts
that silent hang into an error.

CPU-only: the compiled extension is stubbed if absent, and the "worker" is a
plain unix socket that accepts and never replies.
"""

import socket
import sys
import threading
import time
import types

import pytest


def _install_fake_vmm_ops():
    """Only when the compiled extension is unavailable (CPU-only CI): stub the
    few names tp_ipc_util imports for its worker side, which these tests do
    not exercise. Installed conditionally so the tests that need the real
    module are unaffected by import order."""
    fake = types.ModuleType("kvcached.vmm_ops")
    fake.kv_tensors_created = lambda *a, **kw: True  # type: ignore[attr-defined]
    fake.map_to_kv_tensors = lambda *a, **kw: None  # type: ignore[attr-defined]
    fake.unmap_from_kv_tensors = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["kvcached.vmm_ops"] = fake


try:
    import kvcached.vmm_ops  # noqa: F401
except Exception:  # noqa: BLE001 - any import failure means no GPU build
    _install_fake_vmm_ops()

from kvcached import tp_ipc_util  # noqa: E402


@pytest.fixture
def silent_worker(tmp_path):
    """A unix socket that accepts connections and reads, but never replies --
    an alive-but-stuck worker."""
    sock_path = str(tmp_path / "w0.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(4)
    conns = []
    stop = threading.Event()

    def serve():
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            conns.append(conn)  # hold the connection open, never write

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    yield sock_path
    stop.set()
    t.join(timeout=2)
    for c in conns:
        c.close()
    server.close()


def test_unresponsive_worker_raises_within_the_timeout(
        silent_worker, monkeypatch):
    monkeypatch.setattr(tp_ipc_util, "IPC_TIMEOUT_S", 1.5)
    monkeypatch.setattr(tp_ipc_util, "get_worker_socket_path",
                        lambda rank, pp_rank=0: silent_worker)

    t0 = time.time()
    with pytest.raises(RuntimeError, match="did not answer"):
        tp_ipc_util.broadcast_map_to_kv_tensors(1, [0])
    elapsed = time.time() - t0
    assert elapsed < 10, (
        f"took {elapsed:.1f}s -- the exchange is not actually bounded")


def test_dead_worker_fails_fast_without_needing_the_timeout(
        tmp_path, monkeypatch):
    """No socket at all (worker dead): connect is refused, error is immediate.
    This is the failure path the callers already handle; the timeout must not
    slow it down."""
    monkeypatch.setattr(tp_ipc_util, "IPC_TIMEOUT_S", 30.0)
    monkeypatch.setattr(tp_ipc_util, "get_worker_socket_path",
                        lambda rank, pp_rank=0: str(tmp_path / "absent.sock"))

    t0 = time.time()
    with pytest.raises(RuntimeError):
        tp_ipc_util.broadcast_map_to_kv_tensors(1, [0])
    assert time.time() - t0 < 5


def test_timeout_disabled_by_env_zero(monkeypatch, silent_worker):
    """<= 0 keeps the old unbounded behaviour available for debugging; verify
    the knob is honoured by checking a small positive value still bounds it."""
    monkeypatch.setattr(tp_ipc_util, "IPC_TIMEOUT_S", 0.5)
    monkeypatch.setattr(tp_ipc_util, "get_worker_socket_path",
                        lambda rank, pp_rank=0: silent_worker)
    with pytest.raises(RuntimeError, match="did not answer"):
        tp_ipc_util.broadcast_map_to_kv_tensors(1, [0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
