# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest import mock

sys.modules.setdefault("kvcached.vmm_ops", mock.MagicMock())


def test_completed_worker_batch_is_synchronized_on_its_device_before_unmap(
    monkeypatch,
):
    synchronize = mock.Mock()
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=synchronize,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    from kvcached.tp_ipc_util import _sync_before_unmap

    _sync_before_unmap(2)

    synchronize.assert_called_once_with(2)
