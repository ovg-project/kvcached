# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types


def test_explicit_ipc_name_is_exact_without_probing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    utils = importlib.import_module("kvcached.utils")

    monkeypatch.setenv("KVCACHED_IPC_NAME", "shared-instance-ipc")

    def fail_if_probed(name):
        raise AssertionError(f"explicit IPC name should not probe {name}")

    monkeypatch.setattr(utils, "_ipc_segment_exists", fail_if_probed)

    assert utils._obtain_default_ipc_name() == "shared-instance-ipc"
