# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import pickle
import sys
import types
from typing import Any

import pytest


def _install_fake_vmm_ops(monkeypatch):
    fake_torch = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    fake = types.ModuleType("kvcached.vmm_ops")
    setattr(fake, "kv_tensors_created", lambda *args, **kwargs: True)
    setattr(fake, "map_to_kv_tensors", lambda *args, **kwargs: True)
    setattr(fake, "unmap_from_kv_tensors", lambda *args, **kwargs: True)
    setattr(
        fake,
        "prepare_map_to_kv_tensors",
        lambda *args, **kwargs: {"success": True},
    )
    setattr(
        fake,
        "commit_prepared_map",
        lambda *args, **kwargs: {"success": True},
    )
    setattr(fake, "abort_prepared_map", lambda *args, **kwargs: True)
    setattr(fake, "has_prepared_map", lambda *args, **kwargs: False)
    setattr(fake, "current_device_pci_bus_id", lambda: "0000:00:00.0")
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", fake)


def _import_tp_ipc_util(monkeypatch):
    previous = sys.modules.pop("kvcached.tp_ipc_util", None)
    _install_fake_vmm_ops(monkeypatch)
    module = importlib.import_module("kvcached.tp_ipc_util")
    module._PHYSICAL_DEVICE_ID_CACHE.clear()
    module._UNRESOLVED_PHYSICAL_GROWTH_TRANSACTIONS.clear()
    return module, previous


def _restore_tp_ipc_util(previous):
    sys.modules.pop("kvcached.tp_ipc_util", None)
    if previous is not None:
        sys.modules["kvcached.tp_ipc_util"] = previous


def _success_response(command: str) -> dict[str, Any]:
    states = {
        "prepare_map_to_kv_tensors": "reserved",
        "commit_prepared_map": "prepared",
        "finalize_map_to_kv_tensors": "committed",
        "abort_prepared_map": "aborted",
        "orphan_map_transaction": "orphaned",
    }
    response: dict[str, Any] = {"status": "success"}
    if command in states:
        response["transaction_state"] = states[command]
    return response


def test_distributed_map_reserves_per_device_before_concurrent_commit(monkeypatch):
    tp_ipc_util, previous = _import_tp_ipc_util(monkeypatch)
    events: list[tuple[str, int]] = []

    async def exchange(rank, message, pp_rank=0):
        events.append((message["cmd"], rank))
        if message["cmd"] == "get_physical_device_id":
            return {"status": "success", "physical_device_id": f"gpu-{rank}"}
        return _success_response(message["cmd"])

    monkeypatch.setattr(tp_ipc_util, "_send_and_receive_message", exchange)
    try:
        tp_ipc_util.broadcast_map_to_kv_tensors(2, [0], group_id=7)
        assert events == [
            ("get_physical_device_id", 0),
            ("get_physical_device_id", 1),
            ("prepare_map_to_kv_tensors", 0),
            ("prepare_map_to_kv_tensors", 1),
            ("commit_prepared_map", 0),
            ("commit_prepared_map", 1),
            ("finalize_map_to_kv_tensors", 0),
            ("finalize_map_to_kv_tensors", 1),
        ]
    finally:
        _restore_tp_ipc_util(previous)


def test_physical_device_ids_are_cached_between_transactions(monkeypatch):
    tp_ipc_util, previous = _import_tp_ipc_util(monkeypatch)
    commands: list[str] = []

    async def exchange(rank, message, pp_rank=0):
        commands.append(message["cmd"])
        if message["cmd"] == "get_physical_device_id":
            return {"status": "success", "physical_device_id": f"gpu-{rank}"}
        return _success_response(message["cmd"])

    monkeypatch.setattr(tp_ipc_util, "_send_and_receive_message", exchange)
    try:
        tp_ipc_util.broadcast_map_to_kv_tensors(2, [0])
        tp_ipc_util.broadcast_map_to_kv_tensors(2, [16])
        assert commands.count("get_physical_device_id") == 2
        assert commands.count("prepare_map_to_kv_tensors") == 4
    finally:
        _restore_tp_ipc_util(previous)


def test_physical_capacity_signal_changes_after_notification(monkeypatch, tmp_path):
    tp_ipc_util, previous = _import_tp_ipc_util(monkeypatch)
    tp_ipc_util._PHYSICAL_DEVICE_ID_CACHE.update(
        {(0, 0): "gpu-0", (0, 1): "gpu-1"}
    )
    monkeypatch.setenv("KVCACHED_PHYSICAL_GROWTH_LOCK_DIR", str(tmp_path))
    try:
        before = tp_ipc_util.physical_growth_capacity_epoch(2)

        assert tp_ipc_util.notify_physical_growth_capacity_changed(2)

        after = tp_ipc_util.physical_growth_capacity_epoch(2)
        assert before == (0, 0)
        assert after is not None
        assert after != before
    finally:
        _restore_tp_ipc_util(previous)


def test_prepare_failure_aborts_unmapped_reservations_without_commit(monkeypatch):
    tp_ipc_util, previous = _import_tp_ipc_util(monkeypatch)
    tp_ipc_util._PHYSICAL_DEVICE_ID_CACHE.update({(0, 0): "gpu-0", (0, 1): "gpu-1"})
    events: list[tuple[str, int]] = []

    async def exchange(rank, message, pp_rank=0):
        events.append((message["cmd"], rank))
        if message["cmd"] == "prepare_map_to_kv_tensors" and rank == 1:
            return {
                "status": "error",
                "message": "capacity_exhausted",
                "transaction_state": "not_prepared",
            }
        return _success_response(message["cmd"])

    monkeypatch.setattr(tp_ipc_util, "_send_and_receive_message", exchange)
    try:
        with pytest.raises(RuntimeError, match="capacity_exhausted"):
            tp_ipc_util.broadcast_map_to_kv_tensors(2, [0], group_id=7)
        assert events == [
            ("prepare_map_to_kv_tensors", 0),
            ("prepare_map_to_kv_tensors", 1),
            ("abort_prepared_map", 0),
        ]
        assert not any("unmap" in command for command, _ in events)
    finally:
        _restore_tp_ipc_util(previous)


def test_prepare_failure_retains_adopted_orphan(monkeypatch):
    tp_ipc_util, previous = _import_tp_ipc_util(monkeypatch)
    tp_ipc_util._PHYSICAL_DEVICE_ID_CACHE.update({(0, 0): "gpu-0", (0, 1): "gpu-1"})
    events: list[tuple[str, int]] = []

    async def exchange(rank, message, pp_rank=0):
        events.append((message["cmd"], rank))
        if message["cmd"] == "prepare_map_to_kv_tensors":
            if rank == 0:
                return {"status": "success", "transaction_state": "prepared"}
            return {
                "status": "error",
                "message": "capacity_exhausted",
                "transaction_state": "not_prepared",
            }
        return _success_response(message["cmd"])

    monkeypatch.setattr(tp_ipc_util, "_send_and_receive_message", exchange)
    try:
        with pytest.raises(RuntimeError, match="capacity_exhausted"):
            tp_ipc_util.broadcast_map_to_kv_tensors(2, [0], group_id=7)
        assert events[-1] == ("orphan_map_transaction", 0)
        assert not any("unmap" in command for command, _ in events)
    finally:
        _restore_tp_ipc_util(previous)


def test_unknown_prepare_blocks_later_local_growth_fail_closed(monkeypatch):
    tp_ipc_util, previous = _import_tp_ipc_util(monkeypatch)
    tp_ipc_util._PHYSICAL_DEVICE_ID_CACHE[(0, 0)] = "gpu-0"
    prepare_calls = 0

    async def exchange(rank, message, pp_rank=0):
        nonlocal prepare_calls
        if message["cmd"] == "prepare_map_to_kv_tensors":
            prepare_calls += 1
            return RuntimeError("response lost")
        return {"status": "error", "message": "listener unavailable"}

    monkeypatch.setattr(tp_ipc_util, "_send_and_receive_message", exchange)
    try:
        with pytest.raises(
            tp_ipc_util.MapTransactionOutcomeUnknownError,
            match="physical growth is blocked fail-closed",
        ):
            tp_ipc_util.broadcast_map_to_kv_tensors(1, [0])
        assert len(tp_ipc_util._UNRESOLVED_PHYSICAL_GROWTH_TRANSACTIONS) == 1
        with pytest.raises(
            tp_ipc_util.MapTransactionOutcomeUnknownError,
            match="unresolved KV map transaction",
        ):
            tp_ipc_util.broadcast_map_to_kv_tensors(1, [16])
        assert prepare_calls == 1
    finally:
        _restore_tp_ipc_util(previous)


def test_commit_failure_retains_mapped_pages_without_online_unmap(monkeypatch):
    tp_ipc_util, previous = _import_tp_ipc_util(monkeypatch)
    tp_ipc_util._PHYSICAL_DEVICE_ID_CACHE.update({(0, 0): "gpu-0", (0, 1): "gpu-1"})
    events: list[tuple[str, int]] = []

    async def exchange(rank, message, pp_rank=0):
        events.append((message["cmd"], rank))
        if message["cmd"] == "commit_prepared_map" and rank == 1:
            return {
                "status": "error",
                "message": "map failed after partial commit",
                "transaction_state": "mapped",
            }
        return _success_response(message["cmd"])

    monkeypatch.setattr(tp_ipc_util, "_send_and_receive_message", exchange)
    try:
        with pytest.raises(RuntimeError, match="map failed after partial commit"):
            tp_ipc_util.broadcast_map_to_kv_tensors(2, [0])
        assert ("orphan_map_transaction", 0) in events
        assert ("orphan_map_transaction", 1) in events
        assert not any("unmap" in command for command, _ in events)
    finally:
        _restore_tp_ipc_util(previous)


def test_commit_response_loss_reconciles_before_finalize(monkeypatch):
    tp_ipc_util, previous = _import_tp_ipc_util(monkeypatch)
    tp_ipc_util._PHYSICAL_DEVICE_ID_CACHE.update({(0, 0): "gpu-0", (0, 1): "gpu-1"})
    events: list[tuple[str, int]] = []

    async def exchange(rank, message, pp_rank=0):
        events.append((message["cmd"], rank))
        if message["cmd"] == "commit_prepared_map" and rank == 1:
            return {"status": "error", "message": "success response lost"}
        if message["cmd"] == "get_map_transaction_state":
            return {"status": "success", "transaction_state": "prepared"}
        return _success_response(message["cmd"])

    monkeypatch.setattr(tp_ipc_util, "_send_and_receive_message", exchange)
    try:
        tp_ipc_util.broadcast_map_to_kv_tensors(2, [0])
        assert ("get_map_transaction_state", 1) in events
        assert events[-2:] == [
            ("finalize_map_to_kv_tensors", 0),
            ("finalize_map_to_kv_tensors", 1),
        ]
    finally:
        _restore_tp_ipc_util(previous)


def test_physical_growth_stats_include_worker_wait_and_both_phases(monkeypatch):
    tp_ipc_util, previous = _import_tp_ipc_util(monkeypatch)
    try:
        responses = [
            {
                "physical_growth": {
                    "reserve_us": 30,
                    "ticket_wait_us": 4,
                    "targets_count": 24,
                    "capacity_checks": 1,
                    "required_bytes": 1024,
                }
            },
            {
                "physical_growth": {
                    "reserve_us": 40,
                    "ticket_wait_us": 6,
                    "targets_count": 24,
                    "capacity_checks": 1,
                    "capacity_rejections": 1,
                    "required_bytes": 1024,
                    "free_bytes": 900,
                    "headroom_bytes": 100,
                    "usable_bytes": 800,
                    "shortfall_bytes": 224,
                }
            },
            {"physical_growth": {"map_us": 50, "targets_count": 24}},
            {"physical_growth": {"map_us": 60, "targets_count": 24}},
        ]
        counters = tp_ipc_util._build_physical_growth_operation_counters(
            [0, 64], responses
        )
        assert counters["physical_growth_transactions_total"] == 1
        assert counters["physical_growth_worker_operations_total"] == 4
        assert counters["physical_growth_ticket_wait_us_total"] == 10
        assert counters["physical_growth_reserve_us_total"] == 70
        assert counters["physical_growth_map_us_total"] == 110
        assert counters["physical_growth_worker_targets_total"] == 96
        assert counters["physical_growth_capacity_checks_total"] == 2
        assert counters["physical_growth_capacity_rejections_total"] == 1
        assert counters["physical_growth_required_bytes_total"] == 2048
        assert counters["physical_growth_required_bytes_max"] == 1024
        assert counters["physical_growth_rejected_free_bytes_total"] == 900
        assert counters["physical_growth_rejected_headroom_bytes_max"] == 100
        assert counters["physical_growth_rejected_usable_bytes_total"] == 800
        assert counters["physical_growth_rejected_shortfall_bytes_max"] == 224
    finally:
        _restore_tp_ipc_util(previous)


def test_orphan_can_be_adopted_across_repeated_same_payload_retries(monkeypatch):
    tp_ipc_util, previous = _import_tp_ipc_util(monkeypatch)
    try:
        registry = tp_ipc_util._MapTransactionRegistry()
        registry.record_reserved("failed", 7, [64, 0])
        registry.mark_mapped("failed", 7, [64, 0])
        registry.mark_prepared("failed", 7, [64, 0])
        assert registry.mark_orphan("failed", 7, [64, 0]) == "orphaned"
        assert registry.adopt_orphan("retry", 7, [0, 64]) == "failed"
        registry.mark_prepared("retry", 7, [0, 64])
        assert registry.finalize("retry", 7, [0, 64]) == "committed"
    finally:
        _restore_tp_ipc_util(previous)


def test_reserved_transaction_can_only_abort(monkeypatch):
    tp_ipc_util, previous = _import_tp_ipc_util(monkeypatch)
    try:
        registry = tp_ipc_util._MapTransactionRegistry()
        registry.record_reserved("reserved", 7, [0])
        with pytest.raises(RuntimeError, match="cannot be orphaned"):
            registry.mark_orphan("reserved", 7, [0])
        assert registry.abort_reserved("reserved", 7, [0]) == "aborted"
    finally:
        _restore_tp_ipc_util(previous)


@pytest.mark.parametrize("phase", ["prepare_map_to_kv_tensors", "commit_prepared_map"])
def test_transient_state_query_failure_retries_only_unknown_rank(monkeypatch, phase):
    module, previous = _import_tp_ipc_util(monkeypatch)
    module._PHYSICAL_DEVICE_ID_CACHE.update({(0, 0): "gpu-0", (0, 1): "gpu-1"})
    events = []

    async def exchange(rank, message, pp_rank=0):
        command = message["cmd"]
        events.append((command, rank))
        if command == phase and rank == 1:
            raise ConnectionError("response lost")
        if command == "get_map_transaction_state":
            if events.count((command, rank)) == 1:
                raise ConnectionError("query response lost")
            return _success_response(phase)
        return _success_response(command)

    monkeypatch.setattr(module, "_send_and_receive_message", exchange)
    try:
        module.broadcast_map_to_kv_tensors(2, [0])
        assert events.count(("get_map_transaction_state", 1)) == 2
        assert ("get_map_transaction_state", 0) not in events
        assert events.count((phase, 1)) == 1
        assert not module._UNRESOLVED_PHYSICAL_GROWTH_TRANSACTIONS
        assert events[-1] == ("finalize_map_to_kv_tensors", 1)
    finally:
        _restore_tp_ipc_util(previous)


@pytest.mark.parametrize("phase", ["prepare_map_to_kv_tensors", "commit_prepared_map"])
def test_unreconciled_outcome_is_terminal_without_cleanup_or_retry(monkeypatch, phase):
    module, previous = _import_tp_ipc_util(monkeypatch)
    module._PHYSICAL_DEVICE_ID_CACHE.update({(0, 0): "gpu-0", (0, 1): "gpu-1"})
    events = []

    async def exchange(rank, message, pp_rank=0):
        command = message["cmd"]
        events.append((command, rank))
        if (command == phase and rank == 1) or command == "get_map_transaction_state":
            raise ConnectionError("listener unavailable")
        return _success_response(command)

    monkeypatch.setattr(module, "_send_and_receive_message", exchange)
    try:
        with pytest.raises(module.MapTransactionOutcomeUnknownError, match="restart"):
            module.broadcast_map_to_kv_tensors(2, [0], group_id=7)
        assert events.count(("get_map_transaction_state", 1)) == 2
        assert not any(
            "unmap" in cmd or cmd in {"abort_prepared_map", "orphan_map_transaction"}
            for cmd, _ in events
        )
        calls = len(events)
        with pytest.raises(module.MapTransactionOutcomeUnknownError):
            module.broadcast_map_to_kv_tensors(2, [16], group_id=8)
        assert len(events) == calls
        assert not issubclass(module.MapTransactionOutcomeUnknownError, RuntimeError)
    finally:
        _restore_tp_ipc_util(previous)


@pytest.mark.parametrize("cleanup", ["abort_prepared_map", "orphan_map_transaction"])
def test_unconfirmed_cleanup_latches_terminal_failure(monkeypatch, cleanup):
    module, previous = _import_tp_ipc_util(monkeypatch)
    module._PHYSICAL_DEVICE_ID_CACHE.update({(0, 0): "gpu-0", (0, 1): "gpu-1"})

    async def exchange(rank, message, pp_rank=0):
        command = message["cmd"]
        if command == "prepare_map_to_kv_tensors" and rank == 1:
            return {"status": "error", "transaction_state": "not_prepared"}
        if command == "prepare_map_to_kv_tensors" and cleanup == "orphan_map_transaction":
            return {"status": "success", "transaction_state": "prepared"}
        if command == cleanup:
            raise ConnectionError("cleanup response lost")
        return _success_response(command)

    monkeypatch.setattr(module, "_send_and_receive_message", exchange)
    try:
        with pytest.raises(module.MapTransactionOutcomeUnknownError, match="cleanup"):
            module.broadcast_map_to_kv_tensors(2, [0])
        with pytest.raises(module.MapTransactionOutcomeUnknownError, match="cleanup"):
            module.raise_if_physical_growth_unresolved()
    finally:
        _restore_tp_ipc_util(previous)


def test_state_queries_have_a_deadline_and_cancel_stalled_reads(monkeypatch):
    module, previous = _import_tp_ipc_util(monkeypatch)
    cancelled = []

    async def exchange(rank, message, pp_rank=0):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(rank)

    monkeypatch.setattr(module, "_send_and_receive_message", exchange)
    monkeypatch.setattr(module, "IPC_TIMEOUT_S", 0.01)
    try:
        states, failures = asyncio.run(module._reconcile_map_transaction_states(
            [(0, 0)], [ConnectionError("lost")], [0], 0, "stalled"
        ))
        assert states == {}
        assert len(failures) == 1
        assert cancelled == [0, 0]
    finally:
        _restore_tp_ipc_util(previous)


def test_readiness_timeout_cancels_stalled_exchange(monkeypatch):
    module, previous = _import_tp_ipc_util(monkeypatch)
    cancelled = []

    async def stalled(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    monkeypatch.setattr(module, "_broadcast_kv_tensors_created", stalled)
    try:
        with pytest.raises(asyncio.TimeoutError):
            module.broadcast_kv_tensors_created(2, timeout_s=0.01)
        assert cancelled == [True]
    finally:
        _restore_tp_ipc_util(previous)


@pytest.mark.parametrize("bad_payload", [b"", b"\x00", b"\x00" * 4, None])
def test_listener_survives_malformed_first_message(monkeypatch, tmp_path, bad_payload):
    module, previous = _import_tp_ipc_util(monkeypatch)
    torch = sys.modules["torch"]
    monkeypatch.setattr(torch, "cuda", types.SimpleNamespace(
        current_device=lambda: 0, set_device=lambda device: None,
    ), raising=False)
    targets = []

    def wire(message):
        payload = pickle.dumps(message)
        return len(payload).to_bytes(4, "big") + payload

    class Connection:
        def __init__(self, data):
            self.data = data
            self.response = None
            self.closed = False

        def recv(self, size):
            chunk, self.data = self.data[:size], self.data[size:]
            return chunk

        def sendall(self, data):
            self.response = pickle.loads(data[4:])

        def close(self):
            self.closed = True

    bad = Connection(wire([]) if bad_payload is None else bad_payload)
    good = Connection(wire({"cmd": "get_physical_device_id"}))
    connections = iter([bad, good])

    class Server:
        def bind(self, path):
            pass

        def listen(self):
            pass

        def accept(self):
            return next(connections), None

    class Thread:
        def __init__(self, target, daemon):
            targets.append(target)

        def start(self):
            pass

    monkeypatch.setattr(module, "SOCKET_DIR", str(tmp_path))
    monkeypatch.setattr(module.socket, "AF_UNIX", 1, raising=False)
    monkeypatch.setattr(module.socket, "socket", lambda *args: Server())
    monkeypatch.setattr(module.threading, "Thread", Thread)
    try:
        module.start_worker_listener_thread(0)
        with pytest.raises(StopIteration):
            targets[0]()
        assert bad.response is not None
        assert bad.response["status"] == "error"
        assert "transaction_state" not in bad.response
        assert good.response == {
            "status": "success", "physical_device_id": "0000:00:00.0",
        }
        assert bad.closed and good.closed
    finally:
        _restore_tp_ipc_util(previous)
