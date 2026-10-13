# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Readiness keeps child/signal ownership, never control-file metadata."""

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

    kill = Mock()
    controller = Mock()
    engine = types.SimpleNamespace(
        Engine=type("Engine", (), {}),
        SchedulerInitResult=InitResult,
        kill_process_tree=kill,
        run_data_parallel_controller_process=controller,
    )
    scheduler = types.SimpleNamespace(Scheduler=Scheduler)
    tokenizer = types.SimpleNamespace(kill_process_tree=kill)
    original_info = Scheduler.get_init_info
    patch = shutdown.SGLangShutdownPatch()
    assert patch.patch_shutdown(engine, scheduler, tokenizer)
    assert patch.patch_shutdown(engine, scheduler, tokenizer)
    result = InitResult()
    assert result.wait_for_ready() == "ready"
    assert Scheduler.get_init_info is original_info
    assert Scheduler().get_init_info() is payload
    assert result.scheduler_infos == [payload]
    assert engine.run_data_parallel_controller_process is controller
    assert engine.kill_process_tree is kill and tokenizer.kill_process_tree is kill
    assert len(callbacks) == int(enabled and children)
    if callbacks:
        callbacks[0]()
        kill.assert_called_once_with(os.getpid(), include_parent=False)


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
