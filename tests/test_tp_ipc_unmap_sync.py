# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest

sys.modules.setdefault("kvcached.vmm_ops", mock.MagicMock())


@pytest.mark.parametrize("backend", ["cuda", "xpu"])
@pytest.mark.parametrize("worker", [SimpleNamespace(device="dev:3"), SimpleNamespace(local_rank=3)])
def test_completed_worker_batch_is_synchronized_on_assigned_device(monkeypatch, worker, backend):
    synchronize = mock.Mock()
    device = mock.Mock(return_value=nullcontext())
    device_module = SimpleNamespace(
        is_available=lambda: True,
        synchronize=synchronize,
        device=device,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(**{backend: device_module}))

    from kvcached.integration.vllm import patches

    # The barrier must go through get_device_module(): torch.cuda has no XPU
    # devices, so a hardcoded torch.cuda barrier raises on an Intel worker.
    monkeypatch.setattr(patches, "get_device_module", lambda dev=None: device_module)

    assert patches._worker_physical_release_barrier(worker)

    device.assert_called_once_with(getattr(worker, "device", 3))
    synchronize.assert_called_once_with()


def test_release_barrier_rejects_unknown_worker_device():
    from kvcached.integration.vllm.patches import _worker_physical_release_barrier

    with pytest.raises(RuntimeError, match="worker device"):
        _worker_physical_release_barrier(SimpleNamespace())
