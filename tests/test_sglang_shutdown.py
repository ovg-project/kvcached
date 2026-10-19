# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Readiness keeps child/signal ownership and sockets, never control-file metadata."""

import os
import types
from typing import Callable
from unittest.mock import Mock

import pytest

from kvcached.integration.sglang import shutdown


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("children", [True, False])
def test_readiness_keeps_native_payload_and_registers_only_owned_children(
    monkeypatch, enabled, children
):
    monkeypatch.setenv("ENABLE_KVCACHED", str(enabled).lower())
    callbacks: list[Callable[[], None]] = []
    monkeypatch.setattr(shutdown, "register_owner_sigterm_cleanup", callbacks.append)
    payload = {"status": "ready", "max_req_input_len": 512}

    class Scheduler:
        def get_init_info(self):
            return payload

    class InitResult:
        def __init__(self):
            self.scheduler_infos = [payload]
            self.all_child_pids = [os.getpid()] if children else []
            self.wait_for_ready = lambda: "ready"

    kill_calls = Mock()

    def kill(parent_pid, include_parent=True, skip_pid=None, wait_timeout=60):
        kill_calls(parent_pid, include_parent=include_parent)

    monkeypatch.setattr(shutdown, "_bound_worker_sockets", list)
    controller = Mock()
    engine = types.SimpleNamespace(
        Engine=type("Engine", (), {}),
        SchedulerInitResult=InitResult,
        kill_process_tree=kill,
        run_data_parallel_controller_process=controller,
    )
    scheduler = types.SimpleNamespace(Scheduler=Scheduler)
    tokenizer = types.SimpleNamespace(kill_process_tree=kill)
    patch = shutdown.SGLangShutdownPatch()
    assert patch.patch_shutdown(engine, scheduler, tokenizer)
    assert patch.patch_shutdown(engine, scheduler, tokenizer)
    result = InitResult()
    assert result.wait_for_ready() == "ready"
    assert Scheduler().get_init_info() is payload
    assert result.scheduler_infos == [payload]
    assert "_kvcached_ipc_names" not in payload
    assert payload.get(shutdown._SOCKETS_KEY) == ([] if enabled else None)
    assert engine.run_data_parallel_controller_process is shutdown._run_data_parallel_controller_process
    assert len(callbacks) == int(enabled and children)
    if callbacks:
        callbacks[0]()
        kill_calls.assert_called_once_with(os.getpid(), include_parent=False)


def test_inherited_owner_never_stops_child_process_tree(monkeypatch):
    kill = Mock()
    owner = shutdown._ShutdownOwner((), kill)
    monkeypatch.setattr(shutdown.os, "getpid", lambda: owner.pid + 1)
    owner.shutdown()
    kill.assert_not_called()


def test_completed_owner_cannot_stop_later_children():
    process = Mock()
    process.is_running.return_value = False
    kill = Mock()
    owner = shutdown._ShutdownOwner((process,), kill)
    owner.shutdown()
    kill.assert_not_called()
