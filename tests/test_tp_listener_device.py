# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from unittest import mock

import pytest


def _mock_torch(monkeypatch, current_device=0):
    torch = mock.MagicMock()
    torch.__version__ = "2.6.0"
    torch.__spec__ = ModuleSpec("torch", loader=None)
    torch.cuda.current_device.return_value = current_device

    def parse_device(device):
        value = str(device)
        index = int(value.split(":", 1)[1]) if ":" in value else None
        return SimpleNamespace(index=index)

    torch.device.side_effect = parse_device
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


@pytest.mark.parametrize("backend", ["cuda", "xpu"])
def test_listener_thread_restores_its_own_device(monkeypatch, tmp_path, backend):
    """The listener must set its device through get_device_module(), not
    torch.cuda.

    On a +xpu build torch.cuda.set_device raises AttributeError, and the
    listener runs in a daemon thread, so that kills the listener without
    failing anything: it just never answers, and the caller waits on a reply
    that will not come. Mocking torch wholesale hides this, so the xpu case
    below gives torch.cuda the AttributeError a real +xpu build raises.
    """
    torch = _mock_torch(monkeypatch)
    if backend == "xpu":
        torch.cuda.set_device.side_effect = AttributeError(
            "module 'torch._C' has no attribute '_cuda_setDevice'"
        )
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())

    import kvcached.tp_ipc_util as tp_ipc_util

    device_module = torch.xpu if backend == "xpu" else torch.cuda
    monkeypatch.setattr(
        tp_ipc_util, "get_device_module", lambda device=None: device_module
    )

    thread_target = None

    class FakeSocket:
        def bind(self, path):
            pass

        def listen(self):
            pass

        def accept(self):
            raise RuntimeError("stop listener")

    class FakeThread:
        def __init__(self, target, daemon):
            nonlocal thread_target
            thread_target = target

        def start(self):
            pass

    monkeypatch.setattr(tp_ipc_util, "SOCKET_DIR", str(tmp_path))
    monkeypatch.setattr(tp_ipc_util.socket, "AF_UNIX", 1, raising=False)
    monkeypatch.setattr(
        tp_ipc_util.socket, "socket", lambda *args, **kwargs: FakeSocket()
    )
    monkeypatch.setattr(tp_ipc_util.threading, "Thread", FakeThread)

    tp_ipc_util.start_worker_listener_thread(2, 0, device_index=3)

    assert thread_target is not None
    # Reaching accept() means the device was restored without raising.
    with pytest.raises(RuntimeError, match="stop listener"):
        thread_target()
    device_module.set_device.assert_called_once_with(3)


@pytest.mark.parametrize("integration", ["vllm", "sglang"])
@pytest.mark.parametrize("device", ["cuda:3", "hip:3"])
def test_worker_listener_uses_explicit_device(monkeypatch, integration, device):
    _mock_torch(monkeypatch, current_device=0)
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())

    module_name = f"kvcached.integration.{integration}.interfaces"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    interfaces = importlib.import_module(module_name)
    listener = mock.Mock()
    monkeypatch.setattr(interfaces, "start_worker_listener_thread", listener)

    kwargs = {
        "tp_rank": 2,
        "world_size": 4,
        "pp_rank": 1,
        "device": device,
    }
    if integration == "vllm":
        kwargs["is_worker"] = True
    interfaces.init_kvcached(**kwargs)

    listener.assert_called_once_with(2, 1, device_index=3)


def test_vllm_reinit_listener_uses_initialized_device(monkeypatch):
    _mock_torch(monkeypatch, current_device=0)
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())

    module_name = "kvcached.integration.vllm.interfaces"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    interfaces = importlib.import_module(module_name)
    listener = mock.Mock()
    monkeypatch.setattr(interfaces, "start_worker_listener_thread", listener)
    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    monkeypatch.setattr(interfaces, "_kvcached_device", "cuda:3")
    monkeypatch.setattr(interfaces, "_is_worker", False)

    interfaces.init_kvcached(
        tp_rank=2,
        world_size=4,
        pp_rank=1,
        is_worker=True,
        device="cuda:1",
    )

    listener.assert_called_once_with(2, 1, device_index=3)
