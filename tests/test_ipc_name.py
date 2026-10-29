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


@pytest.fixture
def derive(monkeypatch):
    """Derive a default name in a process with the given PGID and PID namespace.

    A pid_namespace of None stands for a system without /proc/self/ns/pid.
    """
    utils = _utils(monkeypatch)
    monkeypatch.delenv("KVCACHED_IPC_NAME", raising=False)
    monkeypatch.setattr(utils, "_detect_engine_tag", lambda: "vLLM")
    monkeypatch.setattr(utils, "_ipc_segment_exists", lambda name: False)
    real_stat = os.stat

    def run(pgid, pid_namespace):
        def stat(path, *args, **kwargs):
            if path != "/proc/self/ns/pid":
                return real_stat(path, *args, **kwargs)
            if pid_namespace is None:
                raise FileNotFoundError(path)
            return types.SimpleNamespace(st_ino=pid_namespace)

        monkeypatch.setattr(utils.os, "getpgid", lambda pid: pgid)
        monkeypatch.setattr(utils.os, "stat", stat)
        return utils._obtain_default_ipc_name()

    return run


def test_engines_in_separate_containers_get_distinct_names(derive):
    # runc and crun start each container's init with setsid(), so an engine
    # launched as the entrypoint has PGID 1 in every container.
    first = derive(pgid=1, pid_namespace=4026532461)
    second = derive(pgid=1, pid_namespace=4026532789)
    assert first != second


def test_processes_of_one_container_agree_on_the_name(derive):
    assert derive(1, 4026532461) == derive(1, 4026532461)


@pytest.mark.parametrize("pid_namespace", [HOST_PID_NAMESPACE, None])
def test_host_processes_keep_the_pgid_name(derive, pid_namespace):
    assert derive(4242, pid_namespace) == "kvcached_vLLM_4242"
