# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
import types
from importlib.machinery import ModuleSpec
from typing import Any

import pytest


def _import_tp_ipc_util(monkeypatch):
    torch = types.ModuleType("torch")
    torch.__spec__ = ModuleSpec("torch", loader=None)
    monkeypatch.setitem(sys.modules, "torch", torch)

    vmm_ops: Any = types.ModuleType("kvcached.vmm_ops")
    vmm_ops.__spec__ = ModuleSpec(vmm_ops.__name__, loader=None)
    vmm_ops.kv_tensors_created = lambda group_id=0: True
    vmm_ops.map_to_kv_tensors = lambda offsets, group_id=0: True
    vmm_ops.unmap_from_kv_tensors = lambda offsets, group_id=0: True
    monkeypatch.setitem(sys.modules, vmm_ops.__name__, vmm_ops)

    from kvcached import tp_ipc_util

    return tp_ipc_util


def test_target_pp_ranks_uses_single_explicit_stage(monkeypatch):
    tp_ipc_util = _import_tp_ipc_util(monkeypatch)
    monkeypatch.setenv("KVCACHED_PP_SIZE", "4")

    assert list(tp_ipc_util._target_pp_ranks(2)) == [2]


def test_target_pp_ranks_expands_coordinator_marker(monkeypatch):
    tp_ipc_util = _import_tp_ipc_util(monkeypatch)
    monkeypatch.setenv("KVCACHED_PP_SIZE", "3")

    assert list(tp_ipc_util._target_pp_ranks(-1)) == [0, 1, 2]


def test_broadcast_map_fans_out_to_all_pp_stages(monkeypatch):
    tp_ipc_util = _import_tp_ipc_util(monkeypatch)
    calls = []

    async def fake_send(rank, message, pp_rank=0):
        calls.append((pp_rank, rank, message))
        return {"status": "success"}

    monkeypatch.setenv("KVCACHED_PP_SIZE", "2")
    monkeypatch.setattr(tp_ipc_util, "_send_and_receive_message", fake_send)

    asyncio.run(
        tp_ipc_util._broadcast_map_to_kv_tensors(3, [7, 11], pp_rank=-1)
    )

    assert [(pp_rank, rank) for pp_rank, rank, _ in calls] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert all(call[2]["group_id"] == 0 for call in calls)


def test_broadcast_error_reports_pp_and_rank(monkeypatch):
    tp_ipc_util = _import_tp_ipc_util(monkeypatch)

    async def fake_send(rank, message, pp_rank=0):
        if pp_rank == 1 and rank == 0:
            return {"status": "error", "message": "boom"}
        return {"status": "success", "created": True}

    monkeypatch.setenv("KVCACHED_PP_SIZE", "2")
    monkeypatch.setattr(tp_ipc_util, "_send_and_receive_message", fake_send)

    with pytest.raises(RuntimeError, match="pp1/rank0"):
        asyncio.run(
            tp_ipc_util._broadcast_kv_tensors_created(2, pp_rank=-1)
        )
