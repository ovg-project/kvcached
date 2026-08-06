# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import threading
import types
from typing import Any

import pytest


class _FakeConnection:
    def close(self):
        pass


class _FakeServerSocket:
    def __init__(self):
        self._accepted = False
        self._stop = threading.Event()

    def bind(self, path):
        self.path = path

    def listen(self):
        pass

    def accept(self):
        if not self._accepted:
            self._accepted = True
            return _FakeConnection(), None
        self._stop.wait()
        raise RuntimeError("test listener stopped")


@pytest.mark.parametrize("command", ["map_to_kv_tensors", "unmap_from_kv_tensors"])
def test_worker_reports_vmm_boolean_failures(monkeypatch, command):
    torch: Any = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        synchronize=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    vmm_ops: Any = types.ModuleType("kvcached.vmm_ops")
    vmm_ops.kv_tensors_created = lambda group_id=0: True
    vmm_ops.map_to_kv_tensors = lambda offsets, group_id=0: False
    vmm_ops.map_to_kv_tensors_with_result = (
        lambda offsets, group_id=0: (False, [])
    )
    vmm_ops.unmap_from_kv_tensors = lambda offsets, group_id=0: False
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", vmm_ops)
    import kvcached

    monkeypatch.setattr(kvcached, "vmm_ops", vmm_ops, raising=False)

    tp_ipc_util = importlib.import_module("kvcached.tp_ipc_util")
    tp_ipc_util = importlib.reload(tp_ipc_util)

    server = _FakeServerSocket()
    response = {}
    completed = threading.Event()

    monkeypatch.setattr(tp_ipc_util.socket, "AF_UNIX", 1, raising=False)
    monkeypatch.setattr(tp_ipc_util.socket, "SOCK_STREAM", 1, raising=False)
    monkeypatch.setattr(tp_ipc_util.socket, "socket", lambda *args: server)
    monkeypatch.setattr(tp_ipc_util.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(tp_ipc_util.os.path, "exists", lambda path: False)
    monkeypatch.setattr(
        tp_ipc_util,
        "recv_msg",
        lambda conn: {"cmd": command, "offsets": [7], "group_id": 3},
    )

    def capture_response(conn, message):
        response.update(message)
        completed.set()

    monkeypatch.setattr(tp_ipc_util, "send_msg", capture_response)

    tp_ipc_util.start_worker_listener_thread(rank=0)
    assert completed.wait(timeout=1)
    assert response["status"] == "error"
    assert "group_id=3" in response["message"]


@pytest.mark.parametrize(
    ("command", "expected_status"),
    [
        ("prepare_unmap_from_kv_tensors", "prepared"),
        ("commit_unmap_from_kv_tensors", "committed"),
        ("abort_unmap_from_kv_tensors", "aborted"),
    ],
)
def test_worker_executes_unmap_transaction_phases(monkeypatch, command, expected_status):
    torch: Any = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        synchronize=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    calls: list[tuple[Any, ...]] = []

    def prepare(offsets, transaction_id, group_id=0):
        calls.append(("prepare", offsets, transaction_id, group_id))
        return True

    def commit(transaction_id, group_id=0):
        calls.append(("commit", transaction_id, group_id))
        return True

    def abort(transaction_id, group_id=0):
        calls.append(("abort", transaction_id, group_id))
        return True

    vmm_ops: Any = types.ModuleType("kvcached.vmm_ops")
    vmm_ops.kv_tensors_created = lambda group_id=0: True
    vmm_ops.map_to_kv_tensors = lambda offsets, group_id=0: True
    vmm_ops.unmap_from_kv_tensors = lambda offsets, group_id=0: True
    vmm_ops.prepare_unmap_from_kv_tensors = prepare
    vmm_ops.commit_unmap_from_kv_tensors = commit
    vmm_ops.abort_unmap_from_kv_tensors = abort
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", vmm_ops)
    import kvcached

    monkeypatch.setattr(kvcached, "vmm_ops", vmm_ops, raising=False)

    tp_ipc_util = importlib.reload(importlib.import_module("kvcached.tp_ipc_util"))
    server = _FakeServerSocket()
    response = {}
    completed = threading.Event()

    monkeypatch.setattr(tp_ipc_util.socket, "AF_UNIX", 1, raising=False)
    monkeypatch.setattr(tp_ipc_util.socket, "SOCK_STREAM", 1, raising=False)
    monkeypatch.setattr(tp_ipc_util.socket, "socket", lambda *args: server)
    monkeypatch.setattr(tp_ipc_util.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(tp_ipc_util.os.path, "exists", lambda path: False)
    monkeypatch.setattr(
        tp_ipc_util,
        "recv_msg",
        lambda conn: {
            "cmd": command,
            "offsets": [7],
            "transaction_id": "txn-1",
            "group_id": 3,
        },
    )

    def capture_response(conn, message):
        response.update(message)
        completed.set()

    monkeypatch.setattr(tp_ipc_util, "send_msg", capture_response)
    tp_ipc_util.start_worker_listener_thread(rank=0)

    assert completed.wait(timeout=1)
    assert response == {"status": expected_status}
    assert calls
