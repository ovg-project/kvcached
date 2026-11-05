# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import sys
import types

import pytest

# Inode of Linux's initial PID namespace, PID_NS_INIT_INO in <linux/nsfs.h>.
HOST_PID_NAMESPACE = 0xEFFFFFFC


def _utils(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    return importlib.import_module("kvcached.utils")


def test_explicit_ipc_name_is_exact_without_probing(monkeypatch):
    utils = _utils(monkeypatch)

    monkeypatch.setenv("KVCACHED_IPC_NAME", "shared-instance-ipc")

    def fail_if_probed(name):
        raise AssertionError(f"explicit IPC name should not probe {name}")

    monkeypatch.setattr(utils, "_ipc_segment_exists", fail_if_probed)

    assert utils._obtain_default_ipc_name() == "shared-instance-ipc"


@pytest.mark.parametrize("pgid,pid_namespace,expected", [
    # runc and crun start each container's init with setsid(), so an engine
    # launched as the entrypoint has PGID 1 in every container.
    (1, 4026532461, "kvcached_vLLM_pidns4026532461_1"),
    (1, 4026532789, "kvcached_vLLM_pidns4026532789_1"),
    (4242, HOST_PID_NAMESPACE, "kvcached_vLLM_4242"),
    (4242, None, "kvcached_vLLM_4242"),  # no /proc/self/ns/pid
])
def test_default_name(monkeypatch, pgid, pid_namespace, expected):
    utils = _utils(monkeypatch)
    monkeypatch.delenv("KVCACHED_IPC_NAME", raising=False)
    monkeypatch.setattr(utils, "_detect_engine_tag", lambda: "vLLM")
    monkeypatch.setattr(utils, "_ipc_segment_exists", lambda name: False)
    monkeypatch.setattr(utils.os, "getpgid", lambda pid: pgid)
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if path != "/proc/self/ns/pid":
            return real_stat(path, *args, **kwargs)
        if pid_namespace is None:
            raise FileNotFoundError(path)
        return types.SimpleNamespace(st_ino=pid_namespace)

    monkeypatch.setattr(utils.os, "stat", stat)
    assert utils._obtain_default_ipc_name() == expected
