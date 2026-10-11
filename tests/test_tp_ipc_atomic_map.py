# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import sys
import types
from typing import Any

import pytest

from kvcached.errors import MapQuarantinedError, StateConsistencyError


def _load_tp_ipc_util(monkeypatch):
    fake_torch: Any = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        synchronize=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_vmm_ops: Any = types.ModuleType("kvcached.vmm_ops")
    fake_vmm_ops.kv_tensors_created = lambda *args, **kwargs: True
    fake_vmm_ops.map_to_kv_tensors = lambda *args, **kwargs: True
    fake_vmm_ops.map_to_kv_tensors_with_result = lambda offsets, group_id=0: (True, list(offsets))
    fake_vmm_ops.unmap_from_kv_tensors = lambda *args, **kwargs: True
    fake_vmm_ops.prepare_unmap_from_kv_tensors = lambda *args, **kwargs: True
    fake_vmm_ops.commit_unmap_from_kv_tensors = lambda *args, **kwargs: True
    fake_vmm_ops.abort_unmap_from_kv_tensors = lambda *args, **kwargs: True
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", fake_vmm_ops)
    import kvcached

    monkeypatch.setattr(kvcached, "vmm_ops", fake_vmm_ops, raising=False)

    module = importlib.import_module("kvcached.tp_ipc_util")
    return importlib.reload(module)


def test_partial_pp_map_rolls_back_only_new_offsets(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)
    calls = []

    async def fake_send(rank, message, pp_rank=0):
        calls.append((pp_rank, rank, message["cmd"], tuple(message["offsets"])))
        if message["cmd"] == "unmap_from_kv_tensors":
            return {"status": "success"}
        if (pp_rank, rank) == (0, 1):
            return {"status": "error", "message": "map failed"}
        newly_mapped = {
            (0, 0): [0],
            (1, 0): [0, 2],
            (1, 1): [],
        }[(pp_rank, rank)]
        return {
            "status": "success",
            "newly_mapped_offsets": newly_mapped,
        }

    monkeypatch.setenv("KVCACHED_PP_SIZE", "2")
    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)

    with pytest.raises(RuntimeError, match="pp0/rank1"):
        asyncio.run(
            module._broadcast_map_to_kv_tensors(
                tp_size=2,
                offsets=[0, 2],
                pp_rank=-1,
                group_id=7,
            )
        )

    rollback_calls = [call for call in calls if call[2] == "unmap_from_kv_tensors"]
    assert rollback_calls == [
        (0, 0, "unmap_from_kv_tensors", (0,)),
        (1, 0, "unmap_from_kv_tensors", (0, 2)),
    ]


def test_transactional_map_fails_closed_with_legacy_extension(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)
    legacy_calls = []

    monkeypatch.delattr(module.vmm_ops, "map_to_kv_tensors_with_result")
    monkeypatch.setattr(
        module,
        "map_to_kv_tensors",
        lambda offsets, group_id=0: legacy_calls.append((offsets, group_id)),
    )

    with pytest.raises(RuntimeError, match="does not support transactional map"):
        module._map_to_kv_tensors_with_result([0, 2], group_id=7)

    assert legacy_calls == []


def test_partial_map_reports_rollback_failure(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)

    async def fake_send(rank, message, pp_rank=0):
        if message["cmd"] == "unmap_from_kv_tensors":
            return {"status": "error", "message": "rollback failed"}
        if rank == 1:
            return {"status": "error", "message": "map failed"}
        return {"status": "success", "newly_mapped_offsets": [0]}

    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)

    with pytest.raises(StateConsistencyError, match="rollback failures.*rollback failed"):
        asyncio.run(module._broadcast_map_to_kv_tensors(2, [0]))


def test_lost_map_response_is_reported_as_unknown_state(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)

    async def fake_send(rank, message, pp_rank=0):
        if message["cmd"] == "map_to_kv_tensors" and rank == 1:
            raise ConnectionError("response lost")
        return {"status": "success", "newly_mapped_offsets": [0]}

    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)

    with pytest.raises(StateConsistencyError, match="state_consistency_unknown.*pp0/rank1"):
        asyncio.run(module._broadcast_map_to_kv_tensors(2, [0]))


def test_successful_map_does_not_issue_rollback(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)
    commands = []

    async def fake_send(rank, message, pp_rank=0):
        commands.append(message["cmd"])
        return {"status": "success", "newly_mapped_offsets": [0]}

    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)
    asyncio.run(module._broadcast_map_to_kv_tensors(2, [0]))

    assert commands == ["map_to_kv_tensors", "map_to_kv_tensors"]


def test_unmap_commits_only_after_every_pp_rank_prepares(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)
    calls = []

    async def fake_send(rank, message, pp_rank=0):
        calls.append((pp_rank, rank, message["cmd"], message["transaction_id"]))
        status = {
            "prepare_unmap_from_kv_tensors": "prepared",
            "commit_unmap_from_kv_tensors": "committed",
        }[message["cmd"]]
        return {"status": status}

    monkeypatch.setenv("KVCACHED_PP_SIZE", "2")
    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)
    asyncio.run(module._broadcast_unmap_from_kv_tensors(2, [0], pp_rank=-1))

    assert [call[2] for call in calls] == [
        "prepare_unmap_from_kv_tensors",
        "prepare_unmap_from_kv_tensors",
        "prepare_unmap_from_kv_tensors",
        "prepare_unmap_from_kv_tensors",
        "commit_unmap_from_kv_tensors",
        "commit_unmap_from_kv_tensors",
        "commit_unmap_from_kv_tensors",
        "commit_unmap_from_kv_tensors",
    ]
    assert len({call[3] for call in calls}) == 1


def test_unmap_prepare_failure_aborts_every_target(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)
    calls = []

    async def fake_send(rank, message, pp_rank=0):
        calls.append((pp_rank, rank, message["cmd"]))
        if message["cmd"] == "prepare_unmap_from_kv_tensors" and rank == 1:
            return {"status": "error", "message": "prepare failed"}
        if message["cmd"] == "abort_unmap_from_kv_tensors":
            return {"status": "aborted"}
        return {"status": "prepared"}

    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)
    with pytest.raises(RuntimeError, match="prepare failed"):
        asyncio.run(module._broadcast_unmap_from_kv_tensors(2, [0]))

    assert [call[2] for call in calls] == [
        "prepare_unmap_from_kv_tensors",
        "prepare_unmap_from_kv_tensors",
        "abort_unmap_from_kv_tensors",
        "abort_unmap_from_kv_tensors",
    ]


def test_unmap_abort_failure_reports_unknown_state(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)

    async def fake_send(rank, message, pp_rank=0):
        if message["cmd"] == "prepare_unmap_from_kv_tensors" and rank == 1:
            raise ConnectionError("prepare response lost")
        if message["cmd"] == "abort_unmap_from_kv_tensors" and rank == 1:
            raise ConnectionError("worker unreachable")
        if message["cmd"] == "abort_unmap_from_kv_tensors":
            return {"status": "aborted"}
        return {"status": "prepared"}

    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)
    with pytest.raises(StateConsistencyError, match="state_consistency_unknown.*worker unreachable"):
        asyncio.run(module._broadcast_unmap_from_kv_tensors(2, [0]))


def test_unmap_commit_retries_only_unconfirmed_targets(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)
    commit_calls = []

    async def fake_send(rank, message, pp_rank=0):
        if message["cmd"] == "prepare_unmap_from_kv_tensors":
            return {"status": "prepared"}
        commit_calls.append(rank)
        if rank == 1 and commit_calls.count(1) == 1:
            raise ConnectionError("commit response lost")
        return {"status": "committed"}

    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)
    asyncio.run(module._broadcast_unmap_from_kv_tensors(2, [0]))
    assert commit_calls == [0, 1, 1]


def test_unmap_commit_failure_never_attempts_abort(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)
    commands = []

    async def fake_send(rank, message, pp_rank=0):
        commands.append(message["cmd"])
        if message["cmd"] == "prepare_unmap_from_kv_tensors":
            return {"status": "prepared"}
        raise ConnectionError("commit unconfirmed")

    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)
    with pytest.raises(StateConsistencyError, match="state_consistency_unknown.*commit unconfirmed"):
        asyncio.run(module._broadcast_unmap_from_kv_tensors(2, [0]))
    assert "abort_unmap_from_kv_tensors" not in commands


@pytest.mark.parametrize("kind,expected", [
    ("map_quarantined", MapQuarantinedError),
    ("state_consistency", StateConsistencyError),
])
def test_worker_error_type_survives_broadcast(monkeypatch, kind, expected):
    module = _load_tp_ipc_util(monkeypatch)

    async def fake_send(rank, message, pp_rank=0):
        if rank == 1 and message["cmd"] == "map_to_kv_tensors":
            return {"status": "error", "message": "same message", "error_type": kind}
        return {"status": "success", "newly_mapped_offsets": [0]}

    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)
    with pytest.raises(expected, match="same message"):
        asyncio.run(module._broadcast_map_to_kv_tensors(2, [0]))


def test_unmap_prepare_fatal_is_not_hidden_by_successful_abort(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)

    async def fake_send(rank, message, pp_rank=0):
        if message["cmd"] == "prepare_unmap_from_kv_tensors":
            return {"status": "error", "message": "restore failed", "error_type": "state_consistency", "phase": "prepare"}
        return {"status": "aborted"}

    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)
    with pytest.raises(StateConsistencyError, match="restore failed"):
        asyncio.run(module._broadcast_unmap_from_kv_tensors(2, [0]))


@pytest.mark.parametrize("pp_rank", [0, 3, -1])
def test_single_target_unmap_uses_one_exchange(monkeypatch, pp_rank):
    module = _load_tp_ipc_util(monkeypatch)
    calls = []

    async def send(rank, message, stage=0):
        calls.append((rank, stage, message))
        return {"status": "committed"}

    monkeypatch.setenv("KVCACHED_PP_SIZE", "1")
    monkeypatch.setattr(module, "_send_and_receive_message", send)
    asyncio.run(module._broadcast_unmap_from_kv_tensors(1, [0, 2], pp_rank, 7))
    assert len(calls) == 1
    rank, stage, message = calls[0]
    assert (rank, stage) == (0, max(pp_rank, 0))
    assert message["cmd"] == "prepare_unmap_from_kv_tensors"
    assert message["commit"] is True
    assert message["group_id"] == 7
    assert message["offsets"] == [0, 2]


def test_tp_one_pp_two_keeps_prepare_barrier(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)
    calls = []

    async def send(rank, message, stage=0):
        assert not message.get("commit")
        calls.append((stage, message["cmd"]))
        return {"status": "prepared" if len(calls) <= 2 else "committed"}

    monkeypatch.setenv("KVCACHED_PP_SIZE", "2")
    monkeypatch.setattr(module, "_send_and_receive_message", send)
    asyncio.run(module._broadcast_unmap_from_kv_tensors(1, [0], -1))
    assert calls == [
        (0, "prepare_unmap_from_kv_tensors"),
        (1, "prepare_unmap_from_kv_tensors"),
        (0, "commit_unmap_from_kv_tensors"),
        (1, "commit_unmap_from_kv_tensors"),
    ]


@pytest.mark.parametrize("response", [
    ConnectionError("reply lost"),
    {"status": "error", "phase": "commit", "message": "release failed"},
    None,
])
def test_single_target_unconfirmed_result_retries_commit_only(monkeypatch, response):
    module = _load_tp_ipc_util(monkeypatch)
    calls = []

    async def send(rank, message, stage=0):
        calls.append(message)
        if len(calls) == 1:
            if isinstance(response, Exception):
                raise response
            return response
        return {"status": "committed"}

    monkeypatch.setattr(module, "_send_and_receive_message", send)
    asyncio.run(module._broadcast_unmap_from_kv_tensors(1, [0]))
    assert [m["cmd"] for m in calls] == [
        "prepare_unmap_from_kv_tensors", "commit_unmap_from_kv_tensors",
    ]
    assert len({m["transaction_id"] for m in calls}) == 1


def test_single_target_unknown_outcome_fails_closed_without_abort(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)
    calls = []

    async def send(rank, message, stage=0):
        calls.append(message)
        raise ConnectionError("reply lost")

    monkeypatch.setattr(module, "_send_and_receive_message", send)
    with pytest.raises(StateConsistencyError, match="commit could not be confirmed"):
        asyncio.run(module._broadcast_unmap_from_kv_tensors(1, [0]))
    assert [m["cmd"] for m in calls] == [
        "prepare_unmap_from_kv_tensors",
        "commit_unmap_from_kv_tensors", "commit_unmap_from_kv_tensors",
    ]


@pytest.mark.parametrize("fatal", [False, True])
def test_single_target_prepare_rejection_can_abort(monkeypatch, fatal):
    module = _load_tp_ipc_util(monkeypatch)
    calls = []

    async def send(rank, message, stage=0):
        calls.append(message["cmd"])
        if len(calls) == 1:
            return {"status": "error", "phase": "prepare", "message": "prepare failed",
                    "error_type": "state_consistency" if fatal else "operation_failed"}
        return {"status": "aborted"}

    monkeypatch.setattr(module, "_send_and_receive_message", send)
    with pytest.raises(StateConsistencyError if fatal else RuntimeError, match="prepare failed"):
        asyncio.run(module._broadcast_unmap_from_kv_tensors(1, [0]))
    assert calls == ["prepare_unmap_from_kv_tensors", "abort_unmap_from_kv_tensors"]
