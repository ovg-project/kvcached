# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""TP worker socket directories must not outlive the worker (issue #476).

Every engine launch created /tmp/kvcached-tp-<ipc>-<hash>/w<rank>.sock, at
TP=1 too, and nothing removed it: the listener served the socket until the
process died and kvctl delete only knew about /dev/shm. These tests pin down
the fix from three sides: stopping the listener unlinks the socket and the
directory, vLLM's Worker.shutdown() stops it, and kvctl delete removes the
directory derived from an IPC name.

CPU-only: the compiled extension is stubbed if absent.
"""

import importlib.util
import os
import pickle
import shutil
import socket
import sys
import tempfile
import threading
import types
from pathlib import Path
from unittest import mock

import pytest


def _install_fake_vmm_ops():
    """Only when the compiled extension is unavailable (CPU-only CI): stub the
    names tp_ipc_util imports for its worker side."""
    fake = types.ModuleType("kvcached.vmm_ops")
    fake.kv_tensors_created = lambda *a, **kw: True  # type: ignore[attr-defined]
    fake.map_to_kv_tensors = lambda *a, **kw: None  # type: ignore[attr-defined]
    fake.unmap_from_kv_tensors = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["kvcached.vmm_ops"] = fake


try:
    import kvcached.vmm_ops  # noqa: F401
except Exception:  # noqa: BLE001 - any import failure means no GPU build
    _install_fake_vmm_ops()

import kvcached.utils  # noqa: E402
from kvcached import tp_ipc_util  # noqa: E402
from kvcached.cli import utils as cli_utils  # noqa: E402
from kvcached.utils import get_tp_socket_dir  # noqa: E402

IPC_NAME = "t"


@pytest.fixture
def socket_root(monkeypatch):
    """A throwaway stand-in for /tmp. Deliberately not tmp_path: unix socket
    paths are limited to ~104 characters on macOS."""
    root = tempfile.mkdtemp(prefix="kvcached-test-",
                            dir="/tmp" if os.path.isdir("/tmp") else None)
    monkeypatch.setattr(kvcached.utils, "TP_SOCKET_DIR_ROOT", root)
    monkeypatch.setattr(tp_ipc_util, "SOCKET_DIR", get_tp_socket_dir(IPC_NAME))
    # Other test files start listeners through faked sockets and threads and
    # never stop them; drop those so stop() here only meets this file's own.
    tp_ipc_util._listeners.clear()
    yield root
    tp_ipc_util.stop_worker_listener_threads()
    shutil.rmtree(root, ignore_errors=True)


def _ask(socket_path, msg):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(5)
        s.connect(socket_path)
        tp_ipc_util.send_msg(s, msg)
        return tp_ipc_util.recv_msg(s)


def test_socket_dir_is_derived_from_the_ipc_name(monkeypatch):
    monkeypatch.setattr(kvcached.utils, "TP_SOCKET_DIR_ROOT", "/tmp")
    first = get_tp_socket_dir("kvcached_vLLM_1234")
    assert first.startswith("/tmp/kvcached-tp-kvcached_vLLM_1234-")
    assert len(os.path.basename(first)) == len("kvcached-tp-kvcached_vLLM_1234-") + 8
    assert get_tp_socket_dir("kvcached_vLLM_1234") == first
    assert get_tp_socket_dir("kvcached_vLLM_1235") != first


def test_stop_unlinks_socket_and_removes_directory(socket_root):
    tp_ipc_util.start_worker_listener_thread(0)
    path = tp_ipc_util.get_worker_socket_path(0)
    listener = tp_ipc_util._listeners[(0, 0)]
    assert os.path.exists(path)
    assert _ask(path, {"cmd": "kv_tensors_created"})["status"] == "success"

    tp_ipc_util.stop_worker_listener_threads()

    assert listener.thread is not None and not listener.thread.is_alive()
    assert not os.path.exists(path)
    assert not os.path.exists(tp_ipc_util.SOCKET_DIR)
    assert os.listdir(socket_root) == []


def test_stop_removes_pp_stage_subdirectory(socket_root):
    tp_ipc_util.start_worker_listener_thread(0, pp_rank=1)
    path = tp_ipc_util.get_worker_socket_path(0, 1)
    assert os.path.dirname(path) == os.path.join(tp_ipc_util.SOCKET_DIR, "pp1")
    assert os.path.exists(path)

    tp_ipc_util.stop_worker_listener_threads()

    assert not os.path.exists(os.path.dirname(path))
    assert not os.path.exists(tp_ipc_util.SOCKET_DIR)


def test_stop_keeps_directory_while_another_worker_socket_remains(socket_root):
    tp_ipc_util.start_worker_listener_thread(0)
    other = tp_ipc_util.get_worker_socket_path(1)
    open(other, "w").close()  # rank 1 lives in another process

    tp_ipc_util.stop_worker_listener_threads()

    assert not os.path.exists(tp_ipc_util.get_worker_socket_path(0))
    assert os.path.exists(other)
    assert os.path.isdir(tp_ipc_util.SOCKET_DIR)


def test_restart_replaces_the_previous_listener(socket_root):
    tp_ipc_util.start_worker_listener_thread(0)
    first = tp_ipc_util._listeners[(0, 0)]
    tp_ipc_util.start_worker_listener_thread(0)
    second = tp_ipc_util._listeners[(0, 0)]

    assert second is not first
    assert first.thread is not None and not first.thread.is_alive()
    assert _ask(tp_ipc_util.get_worker_socket_path(0),
                {"cmd": "kv_tensors_created"})["status"] == "success"


def test_stop_is_safe_without_listeners(socket_root):
    tp_ipc_util.stop_worker_listener_threads()
    tp_ipc_util.stop_worker_listener_threads()
    assert os.listdir(socket_root) == []


def _delayed_map_request():
    """A map_to_kv_tensors request split into its length header and body."""
    body = pickle.dumps({"cmd": "map_to_kv_tensors", "offsets": [],
                         "group_id": 0})
    return len(body).to_bytes(4, "big"), body


def test_stop_cancels_a_read_stalled_in_recv_msg(socket_root, monkeypatch):
    """A connection accepted before stop() must not dispatch afterwards.

    Repro from review: send the length header, withhold the body, call
    stop_worker_listener_threads() while the listener sits in recv_msg().
    The timed join expired, only the listening socket was closed, and
    delivering the body then ran map_to_kv_tensors() after stop had
    returned. Stop must instead cancel the read and reap the thread.
    """
    mapped = threading.Event()

    def fake_map(*a, **kw):
        mapped.set()
        return True, []

    monkeypatch.setattr(tp_ipc_util, "_map_to_kv_tensors_with_result", fake_map)
    in_recv = threading.Event()
    real_recv = tp_ipc_util.recv_msg

    def recv_and_signal(sock):
        in_recv.set()
        return real_recv(sock)

    monkeypatch.setattr(tp_ipc_util, "recv_msg", recv_and_signal)

    tp_ipc_util.start_worker_listener_thread(0)
    path = tp_ipc_util.get_worker_socket_path(0)
    listener = tp_ipc_util._listeners[(0, 0)]
    header, body = _delayed_map_request()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(path)
        client.sendall(header)  # recv_msg() now blocks waiting for the body
        assert in_recv.wait(5)

        tp_ipc_util.stop_worker_listener_threads()

        try:
            client.sendall(body)
        except OSError:
            pass  # stop already shut this connection down
        thread = listener.thread
        assert thread is not None
        thread.join(2)
        assert not mapped.is_set(), \
            "map_to_kv_tensors dispatched after stop returned"
        assert not thread.is_alive(), \
            "stop returned while the listener was still alive"
    assert not os.path.exists(path)
    assert not os.path.exists(tp_ipc_util.SOCKET_DIR)


def test_stop_drains_a_handler_executing_a_backend_operation(
        socket_root, monkeypatch):
    """stop() must not return while a handler is inside a VMM operation.

    The old stop joined for one second and then tore down unconditionally,
    so an operation slower than that was still touching the allocator when
    the integrations moved on to _shutdown_kvcached_impl().
    """
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_map(*a, **kw):
        entered.set()
        release.wait(10)
        finished.set()
        return True, []

    monkeypatch.setattr(tp_ipc_util, "_map_to_kv_tensors_with_result", slow_map)

    tp_ipc_util.start_worker_listener_thread(0)
    path = tp_ipc_util.get_worker_socket_path(0)
    header, body = _delayed_map_request()
    outcome = {}

    def call_stop():
        tp_ipc_util.stop_worker_listener_threads()
        outcome["map_finished_first"] = finished.is_set()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(path)
        client.sendall(header + body)
        assert entered.wait(5)

        stopper = threading.Thread(target=call_stop)
        stopper.start()
        # Longer than the old 1.0 s join timeout: the old stop had already
        # returned by now, the draining stop is still waiting.
        stopper.join(1.5)
        assert stopper.is_alive(), \
            "stop returned while a handler was still executing"

        release.set()
        stopper.join(5)
        assert not stopper.is_alive()
        assert outcome["map_finished_first"] is True
    assert not os.path.exists(path)
    assert not os.path.exists(tp_ipc_util.SOCKET_DIR)


def test_incomplete_stop_retains_the_listener_and_a_retry_finishes(
        socket_root, monkeypatch):
    """If draining times out, stop must keep the listener state for a retry
    instead of unlinking the socket under a still-running handler."""
    entered = threading.Event()
    release = threading.Event()

    def stuck_map(*a, **kw):
        entered.set()
        release.wait(10)
        return True, []

    monkeypatch.setattr(tp_ipc_util, "_map_to_kv_tensors_with_result", stuck_map)

    tp_ipc_util.start_worker_listener_thread(0)
    path = tp_ipc_util.get_worker_socket_path(0)
    listener = tp_ipc_util._listeners[(0, 0)]
    header, body = _delayed_map_request()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(path)
        client.sendall(header + body)
        assert entered.wait(5)

        assert tp_ipc_util.stop_worker_listener_threads(
            drain_timeout_s=0.1) is False
        assert tp_ipc_util._listeners[(0, 0)] is listener
        assert os.path.exists(path)  # no teardown under a live handler

        release.set()
        assert tp_ipc_util.stop_worker_listener_threads() is True
    assert listener.thread is not None and not listener.thread.is_alive()
    assert not os.path.exists(path)
    assert not os.path.exists(tp_ipc_util.SOCKET_DIR)
    assert tp_ipc_util._listeners == {}


def _load_shutdown_interfaces(monkeypatch, integration):
    manager_module = types.ModuleType("kvcached.kv_cache_manager")
    setattr(manager_module, "KVCacheManager", object)
    backend = types.ModuleType("kvcached.vmm_ops")
    for name in ("create_kv_tensors", "init_kvcached", "shutdown_kvcached"):
        setattr(backend, name, mock.Mock())
    monkeypatch.setitem(sys.modules, "kvcached.kv_cache_manager", manager_module)
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", backend)
    monkeypatch.setitem(sys.modules, "kvcached.tp_ipc_util", tp_ipc_util)
    path = Path(__file__).parents[1] / "kvcached/integration" / integration / "interfaces.py"
    spec = importlib.util.spec_from_file_location(f"_cleanup_{integration}", path)
    assert spec is not None and spec.loader is not None
    interfaces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interfaces)
    return interfaces


@pytest.mark.parametrize("integration", ["vllm", "sglang"])
def test_integration_shutdown_defers_allocator_teardown_until_drained(
        socket_root, monkeypatch, integration):
    interfaces = _load_shutdown_interfaces(monkeypatch, integration)
    for name, value in (("_kvcached_initialized", True), ("_kvcached_device", "cuda:0"),
                        ("_async_sched", True), ("_is_worker", True)):
        setattr(interfaces, name, value)
    clear = mock.Mock()
    monkeypatch.setattr(interfaces, "clear_registered_kv_cache_pools", clear)
    monkeypatch.setattr(interfaces, "stop_worker_listener_threads",
                        lambda: tp_ipc_util.stop_worker_listener_threads(0.05))
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def parked_map(*a, **kw):
        entered.set()
        release.wait(10)
        finished.set()
        return True, []

    teardown = mock.Mock(side_effect=lambda: finished.is_set())
    monkeypatch.setattr(interfaces, "_shutdown_kvcached_impl", teardown)
    monkeypatch.setattr(tp_ipc_util, "_map_to_kv_tensors_with_result", parked_map)
    tp_ipc_util.start_worker_listener_thread(0)
    listener = tp_ipc_util._listeners[(0, 0)]
    header, body = _delayed_map_request()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(listener.socket_path)
            client.sendall(header + body)
            assert entered.wait(5)
            assert interfaces.shutdown_kvcached() is False
            teardown.assert_not_called()
            clear.assert_not_called()
            assert interfaces._kvcached_initialized is True
            assert interfaces._kvcached_device == "cuda:0"
            assert interfaces._async_sched is True
            assert interfaces._is_worker is True
            assert tp_ipc_util._listeners[(0, 0)] is listener
            assert listener.thread is not None and listener.thread.is_alive()
            release.set()
            listener.thread.join(5)
            assert not listener.thread.is_alive()
            assert interfaces.shutdown_kvcached() is True
            teardown.assert_called_once_with()
            clear.assert_called_once_with(integration=integration)
            assert interfaces._kvcached_initialized is False
            assert interfaces._kvcached_device is None
            assert interfaces._async_sched is False
            if integration == "vllm":
                assert interfaces._is_worker is False
            assert not os.path.exists(listener.socket_path)
            assert interfaces.shutdown_kvcached() is True
            teardown.assert_called_once_with()
    finally:
        release.set()


def test_cleanup_exception_keeps_listener_registered_for_retry(socket_root, monkeypatch):
    tp_ipc_util.start_worker_listener_thread(0)
    listener = tp_ipc_util._listeners[(0, 0)]
    with monkeypatch.context() as patch:
        patch.setattr(listener, "stop", mock.Mock(side_effect=OSError("cleanup failed")))
        assert tp_ipc_util.stop_worker_listener_threads() is False
        assert tp_ipc_util._listeners[(0, 0)] is listener
        assert os.path.exists(listener.socket_path)
    assert tp_ipc_util.stop_worker_listener_threads() is True
    assert tp_ipc_util._listeners == {}


@pytest.mark.parametrize("cleanup_error", [False, True])
def test_restart_preserves_listener_when_stop_cannot_finish(
        socket_root, monkeypatch, cleanup_error):
    tp_ipc_util.start_worker_listener_thread(0)
    listener = tp_ipc_util._listeners[(0, 0)]
    inode = os.stat(listener.socket_path).st_ino
    with monkeypatch.context() as patch:
        stop = mock.Mock(return_value=False)
        if cleanup_error:
            stop.side_effect = OSError("cleanup failed")
        patch.setattr(listener, "stop", stop)
        with pytest.raises((RuntimeError, OSError)):
            tp_ipc_util.start_worker_listener_thread(0)
        assert tp_ipc_util._listeners[(0, 0)] is listener
        assert os.stat(listener.socket_path).st_ino == inode
        assert listener.thread is not None and listener.thread.is_alive()
    tp_ipc_util.start_worker_listener_thread(0)
    assert tp_ipc_util._listeners[(0, 0)] is not listener
    assert not listener.thread.is_alive()


def test_concurrent_stop_does_not_report_success_while_another_is_draining(
        socket_root, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def parked_map(*a, **kw):
        entered.set()
        release.wait(10)
        return True, []

    monkeypatch.setattr(tp_ipc_util, "_map_to_kv_tensors_with_result", parked_map)
    tp_ipc_util.start_worker_listener_thread(0)
    listener = tp_ipc_util._listeners[(0, 0)]
    header, body = _delayed_map_request()
    results = []
    first = threading.Thread(target=lambda: results.append(
        tp_ipc_util.stop_worker_listener_threads()))
    second = threading.Thread(target=lambda: results.append(
        tp_ipc_util.stop_worker_listener_threads()))
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(listener.socket_path)
            client.sendall(header + body)
            assert entered.wait(5)
            first.start()
            assert listener.stop_event.wait(5)
            second.start()
            second.join(0.1)
            assert second.is_alive()
            assert results == []
    finally:
        release.set()
        if first.ident is not None:
            first.join(5)
        if second.ident is not None:
            second.join(5)
    assert results == [True, True]
    assert not first.is_alive() and not second.is_alive()


@pytest.mark.parametrize("phase", ["prepare", "commit"])
def test_combined_unmap_preserves_typed_error_and_phase(socket_root, monkeypatch, phase):
    from kvcached.errors import StateConsistencyError

    monkeypatch.setattr(tp_ipc_util, "_sync_before_unmap", lambda: None)
    prepare = mock.Mock(return_value=True)
    commit = mock.Mock(return_value=True)
    failing = prepare if phase == "prepare" else commit
    failing.side_effect = StateConsistencyError("injected transaction failure")
    monkeypatch.setattr(tp_ipc_util, "prepare_unmap_from_kv_tensors", prepare)
    monkeypatch.setattr(tp_ipc_util, "commit_unmap_from_kv_tensors", commit)
    monkeypatch.setattr(tp_ipc_util, "kv_tensors_created", lambda **kw: True)
    tp_ipc_util.start_worker_listener_thread(0)
    path = tp_ipc_util.get_worker_socket_path(0)

    response = _ask(path, {
        "cmd": "prepare_unmap_from_kv_tensors", "offsets": [0],
        "transaction_id": "cleanup-test", "commit": True,
    })

    assert response["status"] == "error"
    assert response["error_type"] == "state_consistency"
    assert response["phase"] == phase
    if phase == "prepare":
        commit.assert_not_called()
    assert _ask(path, {"cmd": "kv_tensors_created"})["status"] == "success"
    assert tp_ipc_util.stop_worker_listener_threads() is True


def test_disconnected_error_reply_does_not_kill_listener(socket_root, monkeypatch):
    from kvcached.errors import MapQuarantinedError

    failed = mock.Mock(side_effect=MapQuarantinedError("injected map failure"))
    monkeypatch.setattr(tp_ipc_util, "_map_to_kv_tensors_with_result", failed)
    monkeypatch.setattr(tp_ipc_util, "kv_tensors_created", lambda **kw: True)
    real_send = tp_ipc_util.send_msg
    errors = []

    def send_with_disconnected_error(sock, message):
        if message.get("status") == "error":
            errors.append(message)
            raise BrokenPipeError("injected disconnected peer")
        real_send(sock, message)

    monkeypatch.setattr(tp_ipc_util, "send_msg", send_with_disconnected_error)
    tp_ipc_util.start_worker_listener_thread(0)
    path = tp_ipc_util.get_worker_socket_path(0)
    with pytest.raises(ConnectionError):
        _ask(path, {"cmd": "map_to_kv_tensors", "offsets": [0]})

    assert len(errors) == 1
    assert errors[0]["error_type"] == "map_quarantined"
    assert _ask(path, {"cmd": "kv_tensors_created"})["status"] == "success"
    assert tp_ipc_util.stop_worker_listener_threads() is True


def test_worker_shutdown_stops_the_listener(monkeypatch):
    from kvcached.integration.vllm import patches

    monkeypatch.setitem(sys.modules, "kvcached.tp_ipc_util", tp_ipc_util)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    stop = mock.Mock()
    monkeypatch.setattr(tp_ipc_util, "stop_worker_listener_threads", stop)
    calls = []

    class Worker:
        def shutdown(self):
            stop.assert_not_called()  # vLLM's own teardown runs first
            calls.append("shutdown")

    gpu_worker_mod = types.ModuleType("mock_gpu_worker")
    setattr(gpu_worker_mod, "Worker", Worker)
    assert patches.GPUWorkerPatch().patch_worker_shutdown(gpu_worker_mod)
    assert patches.GPUWorkerPatch().patch_worker_shutdown(gpu_worker_mod)  # idempotent

    Worker().shutdown()

    assert calls == ["shutdown"]
    stop.assert_called_once_with()


def test_worker_shutdown_stops_the_listener_even_if_vllm_teardown_raises(monkeypatch):
    from kvcached.integration.vllm import patches

    monkeypatch.setitem(sys.modules, "kvcached.tp_ipc_util", tp_ipc_util)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    stop = mock.Mock()
    monkeypatch.setattr(tp_ipc_util, "stop_worker_listener_threads", stop)

    class Worker:
        def shutdown(self):
            raise RuntimeError("kv transfer teardown failed")

    gpu_worker_mod = types.ModuleType("mock_gpu_worker")
    setattr(gpu_worker_mod, "Worker", Worker)
    assert patches.GPUWorkerPatch().patch_worker_shutdown(gpu_worker_mod)

    with pytest.raises(RuntimeError, match="kv transfer"):
        Worker().shutdown()
    stop.assert_called_once_with()


def test_worker_shutdown_patch_is_inert_when_kvcached_is_disabled(monkeypatch):
    from kvcached.integration.vllm import patches

    monkeypatch.setitem(sys.modules, "kvcached.tp_ipc_util", tp_ipc_util)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: False)
    stop = mock.Mock()
    monkeypatch.setattr(tp_ipc_util, "stop_worker_listener_threads", stop)

    class Worker:
        def shutdown(self):
            pass

    gpu_worker_mod = types.ModuleType("mock_gpu_worker")
    setattr(gpu_worker_mod, "Worker", Worker)
    assert patches.GPUWorkerPatch().patch_worker_shutdown(gpu_worker_mod)

    Worker().shutdown()
    stop.assert_not_called()


def test_worker_without_shutdown_is_left_alone():
    from kvcached.integration.vllm import patches

    class Worker:
        pass

    gpu_worker_mod = types.ModuleType("mock_gpu_worker")
    setattr(gpu_worker_mod, "Worker", Worker)
    assert patches.GPUWorkerPatch().patch_worker_shutdown(gpu_worker_mod)
    assert not hasattr(Worker, "shutdown")


def test_delete_tp_socket_dir_removes_the_directory_by_ipc_name(socket_root):
    socket_dir = get_tp_socket_dir(IPC_NAME)
    os.makedirs(os.path.join(socket_dir, "pp1"))
    open(os.path.join(socket_dir, "w0.sock"), "w").close()
    open(os.path.join(socket_dir, "pp1", "w0.sock"), "w").close()

    assert cli_utils.delete_tp_socket_dir(IPC_NAME) is True
    assert not os.path.exists(socket_dir)
    assert cli_utils.delete_tp_socket_dir(IPC_NAME) is False


def test_delete_tp_socket_dir_accepts_a_shm_path(socket_root):
    socket_dir = get_tp_socket_dir(IPC_NAME)
    os.makedirs(socket_dir)

    assert cli_utils.delete_tp_socket_dir(f"/dev/shm/{IPC_NAME}") is True
    assert not os.path.exists(socket_dir)


def test_kvctl_delete_reports_the_socket_dir(socket_root, monkeypatch, capsys):
    from kvcached.cli import kvctl

    monkeypatch.setattr(cli_utils, "delete_kv_cache_segment", lambda name: False)
    os.makedirs(get_tp_socket_dir(IPC_NAME))

    kvctl.cmd_delete(IPC_NAME)

    out = capsys.readouterr()
    assert "Removed worker socket dir" in out.out
    assert "not found" not in out.err
    assert not os.path.exists(get_tp_socket_dir(IPC_NAME))


def test_kvctl_delete_reports_not_found_when_nothing_exists(socket_root, monkeypatch, capsys):
    from kvcached.cli import kvctl

    monkeypatch.setattr(cli_utils, "delete_kv_cache_segment", lambda name: False)

    kvctl.cmd_delete(IPC_NAME)

    assert "not found" in capsys.readouterr().err


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
