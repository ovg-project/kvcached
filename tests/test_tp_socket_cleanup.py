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

import os
import shutil
import socket
import sys
import tempfile
import types
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


def test_worker_shutdown_stops_the_listener(monkeypatch):
    from kvcached.integration.vllm import patches

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
