# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
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

    import kvcached

    monkeypatch.setattr(kvcached, "vmm_ops", vmm_ops, raising=False)
    return importlib.reload(importlib.import_module("kvcached.tp_ipc_util"))


def test_target_pp_ranks_uses_single_explicit_stage(monkeypatch):
    tp_ipc_util = _import_tp_ipc_util(monkeypatch)
    monkeypatch.setenv("KVCACHED_PP_SIZE", "4")

    assert tp_ipc_util._target_pp_ranks(2) == [2]


def test_target_pp_ranks_expands_coordinator_marker(monkeypatch):
    tp_ipc_util = _import_tp_ipc_util(monkeypatch)
    monkeypatch.setenv("KVCACHED_PP_SIZE", "3")

    assert tp_ipc_util._target_pp_ranks(-1) == [0, 1, 2]


def test_broadcast_map_fans_out_to_all_pp_stages(monkeypatch):
    tp_ipc_util = _import_tp_ipc_util(monkeypatch)
    calls = []

    async def fake_send(rank, message, pp_rank=0):
        calls.append((pp_rank, rank, message))
        states = {
            "prepare_map_to_kv_tensors": "reserved",
            "commit_prepared_map": "prepared",
            "finalize_map_to_kv_tensors": "committed",
        }
        return {
            "status": "success",
            "transaction_state": states.get(message["cmd"]),
        }

    monkeypatch.setenv("KVCACHED_PP_SIZE", "2")
    tp_ipc_util._PHYSICAL_DEVICE_ID_CACHE.update(
        {(pp, rank): f"gpu-{rank}" for pp in range(2) for rank in range(3)}
    )
    monkeypatch.setattr(tp_ipc_util, "_send_and_receive_message", fake_send)

    asyncio.run(tp_ipc_util._broadcast_map_to_kv_tensors(3, [7, 11], pp_rank=-1))

    prepare_calls = [
        call
        for call in calls
        if call[2].get("cmd") == "prepare_map_to_kv_tensors"
        and call[2].get("offsets") == [7, 11]
    ]
    assert [(pp_rank, rank) for pp_rank, rank, _ in prepare_calls] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert all(call[2]["group_id"] == 0 for call in prepare_calls)
    assert sum(call[2].get("cmd") == "commit_prepared_map" for call in calls) == 6


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


@pytest.mark.parametrize("layers", [(8, 16), (16, 8)])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("buffers", [1, 2])
def test_uneven_pp_maps_and_retires_the_same_logical_pages(
    monkeypatch, layers, contiguous, buffers
):
    ipc = _import_tp_ipc_util(monkeypatch)
    page = 2 * 1024 * 1024
    strides = [page * n * buffers if contiguous else page for n in layers]
    resident = [{0}, {0}]
    ipc._PHYSICAL_DEVICE_ID_CACHE.update({(pp, 0): f"gpu-{pp}" for pp in range(2)})
    prepared = {}
    calls = []

    async def send(rank, message, pp_rank=0):
        assert rank == 0
        monkeypatch.setattr(ipc.vmm_ops, "kv_mapping_page_size",
                            lambda group_id=0: strides[pp_rank], raising=False)
        command = message["cmd"]
        calls.append((pp_rank, command))
        if command in ("prepare_map_to_kv_tensors", "prepare_unmap_from_kv_tensors"):
            assert message["group_id"] == 3
            offsets = ipc._worker_offsets(message, 3)
            assert offsets == [strides[pp_rank], 2 * strides[pp_rank]]
            prepared[pp_rank] = offsets
            if command == "prepare_map_to_kv_tensors":
                return {"status": "success", "transaction_state": "reserved"}
            return {"status": "prepared"}
        if command in ("commit_prepared_map", "finalize_map_to_kv_tensors"):
            assert ipc._worker_offsets(message, 3) == prepared[pp_rank]
            resident[pp_rank].update(prepared[pp_rank])
            state = "prepared" if command == "commit_prepared_map" else "committed"
            return {"status": "success", "transaction_state": state}
        assert command == "commit_unmap_from_kv_tensors"
        resident[pp_rank].difference_update(prepared.pop(pp_rank))
        return {"status": "committed"}

    monkeypatch.setenv("KVCACHED_PP_SIZE", "2")
    monkeypatch.setattr(ipc, "_send_and_receive_message", send)
    offsets = [strides[0], 2 * strides[0]]
    asyncio.run(ipc._broadcast_map_to_kv_tensors(
        1, offsets, pp_rank=-1, group_id=3, offset_stride=strides[0]))
    assert resident == [{0, strides[0], 2 * strides[0]},
                        {0, strides[1], 2 * strides[1]}]
    asyncio.run(ipc._broadcast_unmap_from_kv_tensors(
        1, offsets, pp_rank=-1, group_id=3, offset_stride=strides[0]))
    assert resident == [{0}, {0}]
    assert calls[-2:] == [(0, "commit_unmap_from_kv_tensors"),
                         (1, "commit_unmap_from_kv_tensors")]


@pytest.mark.parametrize("failure", ["prepare", "commit", "lost_reply"])
def test_uneven_pp_recovery_preserves_worker_local_offsets(monkeypatch, failure):
    ipc = _import_tp_ipc_util(monkeypatch)
    strides = [32, 64]
    messages = []
    ipc._PHYSICAL_DEVICE_ID_CACHE.update({(pp, 0): f"gpu-{pp}" for pp in range(2)})

    async def send(rank, message, pp_rank=0):
        messages.append((pp_rank, message["cmd"]))
        monkeypatch.setattr(ipc.vmm_ops, "kv_mapping_page_size",
                            lambda group_id=0: strides[pp_rank], raising=False)
        assert ipc._worker_offsets(message, 0) == [0, strides[pp_rank]]
        command = message["cmd"]
        states = {
            "prepare_map_to_kv_tensors": "reserved",
            "commit_prepared_map": "prepared",
            "finalize_map_to_kv_tensors": "committed",
            "get_map_transaction_state": "prepared",
            "abort_prepared_map": "aborted",
            "orphan_map_transaction": "orphaned",
        }
        state = states[command]
        if pp_rank == 0:
            if failure == "prepare" and command == "prepare_map_to_kv_tensors":
                return {"status": "error", "transaction_state": "not_prepared"}
            if command == "commit_prepared_map":
                if failure == "commit":
                    return {"status": "error", "transaction_state": "partial"}
                if failure == "lost_reply":
                    raise RuntimeError("lost reply")
        return {"status": "success", "transaction_state": state}

    monkeypatch.setenv("KVCACHED_PP_SIZE", "2")
    monkeypatch.setattr(ipc, "_send_and_receive_message", send)
    operation = ipc._broadcast_map_to_kv_tensors(
        1, [0, strides[0]], pp_rank=-1, offset_stride=strides[0])
    if failure == "lost_reply":
        asyncio.run(operation)
        assert (0, "get_map_transaction_state") in messages
    else:
        with pytest.raises(RuntimeError):
            asyncio.run(operation)
        cleanup = "abort_prepared_map" if failure == "prepare" else "orphan_map_transaction"
        assert (1, cleanup) in messages
    assert not any("unmap" in command for _, command in messages)


@pytest.mark.parametrize("stride, offsets", [(0, [0]), (-1, [0]), (32, [16]),
                                            (32, [-32])])
def test_worker_rejects_invalid_offset_geometry(monkeypatch, stride, offsets):
    ipc = _import_tp_ipc_util(monkeypatch)
    monkeypatch.setattr(ipc.vmm_ops, "kv_mapping_page_size",
                        lambda group_id=0: 64, raising=False)
    with pytest.raises(ValueError, match="offset"):
        ipc._worker_offsets({"offset_stride": stride, "offsets": offsets}, 0)
