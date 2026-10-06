# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest

sys.modules.setdefault("kvcached.vmm_ops", mock.MagicMock())


@pytest.mark.parametrize("worker", [SimpleNamespace(device="cuda:3"), SimpleNamespace(local_rank=3)])
def test_completed_worker_batch_is_synchronized_on_assigned_device(monkeypatch, worker):
    synchronize = mock.Mock()
    device = mock.Mock(return_value=nullcontext())
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=synchronize,
            device=device,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    from kvcached.integration.vllm.patches import _worker_physical_release_barrier

    assert _worker_physical_release_barrier(worker)

    device.assert_called_once_with(getattr(worker, "device", 3))
    synchronize.assert_called_once_with()


def test_release_barrier_rejects_unknown_worker_device():
    from kvcached.integration.vllm.patches import _worker_physical_release_barrier

    with pytest.raises(RuntimeError, match="worker CUDA device"):
        _worker_physical_release_barrier(SimpleNamespace())



def test_listener_release_barrier_uses_assigned_device(monkeypatch):
    synchronize = mock.Mock()
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, synchronize=synchronize)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    from kvcached.tp_ipc_util import _sync_before_unmap

    _sync_before_unmap(2)
    synchronize.assert_called_once_with(2)
