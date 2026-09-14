# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import pickle
import socket
import threading
import uuid
from collections import OrderedDict
from typing import Any, Callable, Dict, NoReturn, Optional, cast

from kvcached import vmm_ops
from kvcached.utils import DEFAULT_IPC_NAME, normalize_gpu_device

kv_tensors_created = vmm_ops.kv_tensors_created
prepare_unmap_from_kv_tensors = getattr(
    vmm_ops, "prepare_unmap_from_kv_tensors", None
)
commit_unmap_from_kv_tensors = getattr(vmm_ops, "commit_unmap_from_kv_tensors", None)
abort_unmap_from_kv_tensors = getattr(vmm_ops, "abort_unmap_from_kv_tensors", None)

try:
    from kvcached.vmm_ops import (
        abort_prepared_map,
        commit_prepared_map,
        current_device_pci_bus_id,
        prepare_map_to_kv_tensors,
    )
except ImportError:
    abort_prepared_map = None
    commit_prepared_map = None
    current_device_pci_bus_id = None
    prepare_map_to_kv_tensors = None


def _get_socket_dir_name() -> str:
    """
    Build a human-readable, IPC-name-based directory with a short hash suffix.

    This keeps the original text-based IPC name visible while adding a hash
    for extra uniqueness. The hash is deterministic so all workers in the same
    engine instance agree on the directory.
    """
    # Deterministic short hash derived from the base name
    suffix = uuid.uuid5(uuid.NAMESPACE_DNS, DEFAULT_IPC_NAME).hex[:8]
    return f"kvcached-tp-{DEFAULT_IPC_NAME}-{suffix}"


# Socket directory for tensor parallel (TP) worker communication.
# Unix domain socket paths are limited to 108 characters on Linux, so we keep
# the directory name short and validate the final socket path length below.
SOCKET_DIR = os.path.join("/tmp", _get_socket_dir_name())
IPC_TIMEOUT_S = float(os.getenv("KVCACHED_IPC_TIMEOUT", "60"))
_MAX_TERMINAL_MAP_TRANSACTIONS = 4096


def _build_physical_growth_operation_counters(
    offsets: list[int],
    responses: list[Any],
) -> dict[str, int]:
    counters = {
        "physical_growth_transactions_total": 1,
        "physical_growth_offsets_total": len(offsets),
        "physical_growth_offsets_max": len(offsets),
    }
    for response in responses:
        if not isinstance(response, dict):
            continue
        stats = response.get("physical_growth")
        if not isinstance(stats, dict):
            continue
        counters["physical_growth_worker_operations_total"] = (
            counters.get("physical_growth_worker_operations_total", 0) + 1
        )
        for field in ("ticket_wait_us", "admission_us", "reserve_us", "map_us"):
            value = int(stats.get(field, 0))
            total_name = f"physical_growth_{field}_total"
            max_name = f"physical_growth_{field}_max"
            counters[total_name] = counters.get(total_name, 0) + value
            counters[max_name] = max(counters.get(max_name, 0), value)

        capacity_checks = int(stats.get("capacity_checks", 0))
        capacity_rejections = int(stats.get("capacity_rejections", 0))
        counters["physical_growth_capacity_checks_total"] = (
            counters.get("physical_growth_capacity_checks_total", 0)
            + capacity_checks
        )
        counters["physical_growth_capacity_rejections_total"] = (
            counters.get("physical_growth_capacity_rejections_total", 0)
            + capacity_rejections
        )
        if capacity_checks:
            required_bytes = int(stats.get("required_bytes", 0))
            counters["physical_growth_required_bytes_total"] = (
                counters.get("physical_growth_required_bytes_total", 0)
                + required_bytes
            )
            counters["physical_growth_required_bytes_max"] = max(
                counters.get("physical_growth_required_bytes_max", 0),
                required_bytes,
            )
        if capacity_rejections:
            for field in (
                "free_bytes",
                "headroom_bytes",
                "usable_bytes",
                "shortfall_bytes",
            ):
                value = int(stats.get(field, 0))
                total_name = f"physical_growth_rejected_{field}_total"
                max_name = f"physical_growth_rejected_{field}_max"
                counters[total_name] = counters.get(total_name, 0) + value
                counters[max_name] = max(counters.get(max_name, 0), value)

        targets = int(stats.get("targets_count", 0))
        counters["physical_growth_worker_targets_total"] = (
            counters.get("physical_growth_worker_targets_total", 0) + targets
        )
        counters["physical_growth_worker_targets_max"] = max(
            counters.get("physical_growth_worker_targets_max", 0), targets
        )
    return counters


class MapTransactionOutcomeUnknownError(Exception):
    """Terminal failure, deliberately distinct from recoverable RuntimeError."""


_PHYSICAL_DEVICE_ID_CACHE: dict[tuple[int, int], str] = {}
_UNRESOLVED_PHYSICAL_GROWTH_TRANSACTIONS: dict[str, Any] = {}


def raise_if_physical_growth_unresolved() -> None:
    """Surface failures discovered by native preallocation on the caller thread."""
    if _UNRESOLVED_PHYSICAL_GROWTH_TRANSACTIONS:
        failure = next(iter(_UNRESOLVED_PHYSICAL_GROWTH_TRANSACTIONS.values()))
        raise MapTransactionOutcomeUnknownError(failure["message"])


def _fail_unresolved_map_transaction(
    transaction_id: str,
    offsets: list[int],
    group_id: int,
    phase: str,
    failures: list[str],
) -> NoReturn:
    message = (
        f"unresolved KV map transaction {transaction_id} ({phase}, group={group_id}, "
        f"offsets={offsets}); physical growth is blocked fail-closed. "
        "Stop and restart this engine and its workers; unconfirmed reservations "
        "and mappings are retained until worker teardown, not retried or unmapped "
        "online: " + "; ".join(failures)
    )
    _UNRESOLVED_PHYSICAL_GROWTH_TRANSACTIONS[transaction_id] = {
        "offsets": list(offsets),
        "group_id": group_id,
        "phase": phase,
        "message": message,
    }
    raise MapTransactionOutcomeUnknownError(message)


class _MapTransactionRegistry:
    """Track mappings created by distributed map transactions on one worker."""

    def __init__(
        self,
        max_terminal: int = _MAX_TERMINAL_MAP_TRANSACTIONS,
    ):
        self._transactions: dict[str, tuple[str, int, tuple[int, ...]]] = {}
        self._orphans: dict[tuple[int, tuple[int, ...]], str] = {}
        self._terminal: OrderedDict[str, None] = OrderedDict()
        self._max_terminal = max_terminal
        self._lock = threading.RLock()

    @staticmethod
    def _payload(group_id: int, offsets: list[int]) -> tuple[int, tuple[int, ...]]:
        return group_id, tuple(sorted(offsets))

    def state(
        self, transaction_id: str, group_id: int, offsets: list[int]
    ) -> Optional[str]:
        with self._lock:
            return self._state_unlocked(transaction_id, group_id, offsets)

    def _state_unlocked(
        self, transaction_id: str, group_id: int, offsets: list[int]
    ) -> Optional[str]:
        transaction = self._transactions.get(transaction_id)
        if transaction is None:
            return None
        state, recorded_group_id, recorded_offsets = transaction
        if (recorded_group_id, recorded_offsets) != self._payload(group_id, offsets):
            raise RuntimeError(
                f"KV map transaction {transaction_id} payload does not match its original request"
            )
        return state

    def record_reserved(
        self, transaction_id: str, group_id: int, offsets: list[int]
    ) -> None:
        with self._lock:
            if self._state_unlocked(transaction_id, group_id, offsets) is not None:
                raise RuntimeError(
                    f"KV map transaction {transaction_id} is already registered"
                )
            self._transactions[transaction_id] = (
                "reserved",
                group_id,
                self._payload(group_id, offsets)[1],
            )

    def mark_mapped(
        self, transaction_id: str, group_id: int, offsets: list[int]
    ) -> None:
        with self._lock:
            state = self._state_unlocked(transaction_id, group_id, offsets)
            if state != "reserved":
                raise RuntimeError(
                    f"KV map transaction {transaction_id} cannot be mapped from state {state}"
                )
            self._transactions[transaction_id] = (
                "mapped",
                group_id,
                self._payload(group_id, offsets)[1],
            )

    def mark_reserved_after_orphan_adoption(
        self, transaction_id: str, group_id: int, offsets: list[int]
    ) -> None:
        with self._lock:
            state = self._state_unlocked(transaction_id, group_id, offsets)
            if state != "mapped":
                raise RuntimeError(
                    f"KV map transaction {transaction_id} cannot reserve from state {state}"
                )
            self._transactions[transaction_id] = (
                "reserved",
                group_id,
                self._payload(group_id, offsets)[1],
            )

    def abort_reserved(
        self, transaction_id: str, group_id: int, offsets: list[int]
    ) -> str:
        with self._lock:
            state = self._state_unlocked(transaction_id, group_id, offsets)
            if state is None or state == "aborted":
                return "aborted"
            if state != "reserved":
                raise RuntimeError(
                    f"KV map transaction {transaction_id} cannot be aborted from state {state}"
                )
            self._transactions[transaction_id] = (
                "aborted",
                group_id,
                self._payload(group_id, offsets)[1],
            )
            self._remember_terminal(transaction_id)
            return "aborted"

    def adopt_orphan(
        self, transaction_id: str, group_id: int, offsets: list[int]
    ) -> Optional[str]:
        """Transfer a retained mapping to a retry without mapping it again."""
        with self._lock:
            if self._state_unlocked(transaction_id, group_id, offsets) is not None:
                raise RuntimeError(
                    f"KV map transaction {transaction_id} is already registered"
                )
            payload = self._payload(group_id, offsets)
            orphan_id = self._orphans.pop(payload, None)
            if orphan_id is None:
                return None
            orphan_state = self._state_unlocked(orphan_id, group_id, offsets)
            if orphan_state != "orphaned":
                raise RuntimeError(
                    f"KV map orphan {orphan_id} has invalid state {orphan_state}"
                )
            self._transactions[orphan_id] = (
                "adopted",
                group_id,
                payload[1],
            )
            self._remember_terminal(orphan_id)
            self._transactions[transaction_id] = (
                "mapped",
                group_id,
                payload[1],
            )
            return orphan_id

    def mark_prepared(
        self, transaction_id: str, group_id: int, offsets: list[int]
    ) -> None:
        with self._lock:
            state = self._state_unlocked(transaction_id, group_id, offsets)
            if state != "mapped":
                raise RuntimeError(
                    f"KV map transaction {transaction_id} cannot be prepared from state {state}"
                )
            self._transactions[transaction_id] = (
                "prepared",
                group_id,
                self._payload(group_id, offsets)[1],
            )

    def finalize(self, transaction_id: str, group_id: int, offsets: list[int]) -> str:
        with self._lock:
            state = self._state_unlocked(transaction_id, group_id, offsets)
            if state is None:
                raise RuntimeError(
                    f"KV map transaction {transaction_id} was not prepared"
                )
            if state == "committed":
                return "committed"
            if state != "prepared":
                raise RuntimeError(
                    f"KV map transaction {transaction_id} was not prepared: {state}"
                )
            self._transactions[transaction_id] = (
                "committed",
                group_id,
                self._payload(group_id, offsets)[1],
            )
            self._remember_terminal(transaction_id)
            return "committed"

    def mark_orphan(
        self,
        transaction_id: str,
        group_id: int,
        offsets: list[int],
    ) -> str:
        with self._lock:
            state = self._state_unlocked(transaction_id, group_id, offsets)
            if state is None or state == "adopted":
                return "not_prepared"
            if state == "committed":
                raise RuntimeError(
                    f"Refusing to orphan committed KV map transaction {transaction_id}"
                )
            if state == "mapped":
                raise RuntimeError(
                    f"KV map transaction {transaction_id} was mapped but not synchronized"
                )
            if state == "orphaned":
                return "orphaned"
            if state != "prepared":
                raise RuntimeError(
                    f"KV map transaction {transaction_id} cannot be orphaned from state {state}"
                )
            payload = self._payload(group_id, offsets)
            existing_orphan = self._orphans.get(payload)
            if existing_orphan is not None and existing_orphan != transaction_id:
                raise RuntimeError(
                    f"KV map payload is already retained by transaction {existing_orphan}"
                )
            self._transactions[transaction_id] = (
                "orphaned",
                group_id,
                payload[1],
            )
            self._orphans[payload] = transaction_id
            return "orphaned"

    def _remember_terminal(self, transaction_id: str) -> None:
        self._terminal[transaction_id] = None
        self._terminal.move_to_end(transaction_id)
        while len(self._terminal) > self._max_terminal:
            stale_id, _ = self._terminal.popitem(last=False)
            self._transactions.pop(stale_id, None)


def _target_pp_ranks(pp_rank: int) -> list[int]:
    if pp_rank >= 0:
        return [pp_rank]

    pp_size = int(os.getenv("KVCACHED_PP_SIZE", "1") or "1")
    return list(range(max(pp_size, 1)))


def _env_falsey(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"0", "false", "no", "off"}


def _sync_before_unmap(device_index: Optional[int] = None) -> None:
    """Match PageAllocator's single-process async unmap safety in TP mode."""
    if _env_falsey("KVCACHED_SYNC_BEFORE_TP_UNMAP"):
        return
    _sync_cuda(device_index=device_index, suppress_errors=False)


def _sync_after_map(device_index: Optional[int] = None) -> None:
    """Make newly mapped TP pages visible before schedulers touch them."""
    if _env_falsey("KVCACHED_SYNC_AFTER_TP_MAP"):
        return
    _sync_cuda(device_index=device_index, suppress_errors=False)


def _sync_cuda(*, device_index: Optional[int], suppress_errors: bool) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(device_index)
    except Exception:
        if not suppress_errors:
            raise


def get_worker_socket_path(rank: int, pp_rank: int = 0) -> str:
    """
    Get the path for the worker socket, namespaced by pp_rank.
    Each PP stage uses its own subdirectory to avoid EADDRINUSE races
    when multiple stages start simultaneously (SGLang PP behaviour).

    The full path is guaranteed to be <= 108 characters (Unix domain socket limit).
    """
    if pp_rank > 0:
        socket_path = os.path.join(SOCKET_DIR, f"pp{pp_rank}", f"w{rank}.sock")
    else:
        socket_path = os.path.join(SOCKET_DIR, f"w{rank}.sock")

    if len(socket_path) > 108:
        raise RuntimeError(
            f"Socket path too long ({len(socket_path)} chars, max 108): {socket_path}"
        )

    return socket_path


# NOTE: All messages exchanged through the IPC layer are dictionaries with
# string keys and arbitrary JSON-serialisable (picklable) values.
Message = Dict[str, Any]


def send_msg(sock: socket.socket, msg: Message) -> None:
    """
    Send a message through the socket.
    The message is serialized using pickle.
    """
    data = pickle.dumps(msg)
    sock.sendall(len(data).to_bytes(4, 'big') + data)


# The receive side mirrors *send_msg* and therefore also returns a *Message*.
def recv_msg(sock: socket.socket) -> Message:
    """
    Receive a message from the socket.
    The message is deserialized using pickle.
    """
    length_bytes = sock.recv(4)
    if not length_bytes:
        raise ConnectionError("Socket connection closed")
    if not len(length_bytes) == 4:
        raise ValueError("Received incomplete length bytes from socket")
    length = int.from_bytes(length_bytes, 'big')
    if length <= 0:
        raise ValueError("Received invalid length for message")
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError(
                "Socket connection closed while receiving data")
        data += chunk
    if len(data) != length:
        raise ValueError("Received data length does not match expected length")
    return cast(Message, pickle.loads(data))


def resolve_gpu_device_index(device: Optional[str]) -> int:
    """Resolve an integration device string to the CUDA runtime device index."""
    import torch

    if device is not None:
        device_index = torch.device(normalize_gpu_device(device)).index
        if device_index is not None:
            return int(device_index)
    return int(torch.cuda.current_device())


def start_worker_listener_thread(
    rank: int,
    pp_rank: int = 0,
    device_index: Optional[int] = None,
):
    """
    Start a thread that listens for messages on the worker socket.
    pp_rank is used to create a PP-stage-specific subdirectory so that
    concurrent SGLang PP stages do not bind the same socket path.
    """
    import torch

    # CUDA's current device is thread-local. Capture it on the initialized
    # worker thread instead of querying again from the listener thread, whose
    # default device would otherwise be cuda:0 for every TP rank.
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    else:
        torch.cuda.set_device(device_index)
    if current_device_pci_bus_id is not None:
        physical_device_id = str(current_device_pci_bus_id())
    else:
        properties = torch.cuda.get_device_properties(device_index)
        physical_device_id = str(getattr(properties, "pci_bus_id", ""))
        if not physical_device_id:
            raise RuntimeError("Cannot resolve the worker physical GPU identifier")
    socket_dir = os.path.join(SOCKET_DIR, f"pp{pp_rank}") if pp_rank > 0 else SOCKET_DIR
    os.makedirs(socket_dir, exist_ok=True)
    socket_path = get_worker_socket_path(rank, pp_rank)

    if os.path.exists(socket_path):
        try:
            os.remove(socket_path)
        except OSError as e:
            print(f"Error removing existing socket file {socket_path}: {e}")

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(socket_path)
    server_sock.listen()
    map_transactions = _MapTransactionRegistry()

    def listen_loop():
        # CUDA's current device is thread-local. Bind the listener before any
        # map, unmap, synchronization, or memory query reaches CUDA.
        torch.cuda.set_device(device_index)
        print(f"Worker {rank} IPC listener started at {socket_path}")
        while True:
            conn, _ = server_sock.accept()
            physical_growth_stats = None
            msg: Message = {}
            try:
                received = recv_msg(conn)
                if not isinstance(received, dict):
                    raise ValueError("Worker IPC message must be a dictionary")
                msg = received
                # print(f"Worker {rank} received message: {msg}")
                group_id: int = msg.get("group_id", 0)
                if msg["cmd"] == "get_physical_device_id":
                    send_msg(
                        conn,
                        {
                            "status": "success",
                            "physical_device_id": physical_device_id,
                        },
                    )
                elif msg["cmd"] == "prepare_map_to_kv_tensors":
                    transaction_id = msg.get("transaction_id")
                    if transaction_id is None:
                        raise RuntimeError("KV map prepare is missing transaction_id")
                    transaction_state = map_transactions.state(
                        transaction_id, group_id, msg["offsets"]
                    )
                    if transaction_state is None:
                        orphan_id = map_transactions.adopt_orphan(
                            transaction_id, group_id, msg["offsets"]
                        )
                        if prepare_map_to_kv_tensors is None:
                            raise RuntimeError(
                                "native prepared-map support is unavailable"
                            )
                        physical_growth_stats = dict(
                            prepare_map_to_kv_tensors(
                                transaction_id,
                                msg["offsets"],
                                group_id=group_id,
                            )
                        )
                        prepare_success = bool(
                            physical_growth_stats.pop("success", False)
                        )
                        if prepare_success:
                            if orphan_id is None:
                                map_transactions.record_reserved(
                                    transaction_id, group_id, msg["offsets"]
                                )
                            else:
                                map_transactions.mark_reserved_after_orphan_adoption(
                                    transaction_id, group_id, msg["offsets"]
                                )
                                print(
                                    "[kvcached-map-transaction] event=orphan_adopted "
                                    f"old_transaction={orphan_id} "
                                    f"new_transaction={transaction_id} "
                                    f"pp_rank={pp_rank} rank={rank} group_id={group_id}",
                                    flush=True,
                                )
                            transaction_state = "reserved"
                        else:
                            if orphan_id is not None:
                                map_transactions.mark_prepared(
                                    transaction_id, group_id, msg["offsets"]
                                )
                                map_transactions.mark_orphan(
                                    transaction_id, group_id, msg["offsets"]
                                )
                                transaction_state = "orphaned"
                            raise RuntimeError(
                                f"Failed to prepare KV tensors for group_id={group_id}"
                            )
                    if transaction_state not in {"reserved", "prepared"}:
                        raise RuntimeError(
                            f"KV map transaction {transaction_id} cannot prepare from "
                            f"state {transaction_state}"
                        )
                    send_msg(
                        conn,
                        {
                            "status": "success",
                            "transaction_id": transaction_id,
                            "transaction_state": transaction_state,
                            "physical_growth": physical_growth_stats,
                        },
                    )
                elif msg["cmd"] == "commit_prepared_map":
                    transaction_id = msg.get("transaction_id")
                    if transaction_id is None:
                        raise RuntimeError("KV map commit is missing transaction_id")
                    transaction_state = map_transactions.state(
                        transaction_id, group_id, msg["offsets"]
                    )
                    if transaction_state == "reserved":
                        if commit_prepared_map is None:
                            raise RuntimeError(
                                "native prepared-map support is unavailable"
                            )
                        physical_growth_stats = dict(
                            commit_prepared_map(transaction_id, group_id=group_id)
                        )
                        commit_success = bool(
                            physical_growth_stats.pop("success", False)
                        )
                        map_transactions.mark_mapped(
                            transaction_id, group_id, msg["offsets"]
                        )
                        transaction_state = "mapped"
                        if not commit_success:
                            raise RuntimeError(
                                f"Failed to commit prepared KV tensors for group_id={group_id}"
                            )
                    if transaction_state == "mapped":
                        _sync_after_map(device_index)
                        map_transactions.mark_prepared(
                            transaction_id, group_id, msg["offsets"]
                        )
                        transaction_state = "prepared"
                    if transaction_state != "prepared":
                        raise RuntimeError(
                            f"KV map transaction {transaction_id} cannot commit from "
                            f"state {transaction_state}"
                        )
                    send_msg(
                        conn,
                        {
                            "status": "success",
                            "transaction_id": transaction_id,
                            "transaction_state": transaction_state,
                            "physical_growth": physical_growth_stats,
                        },
                    )
                elif msg["cmd"] == "abort_prepared_map":
                    transaction_id = msg.get("transaction_id")
                    if transaction_id is None:
                        raise RuntimeError("KV map abort is missing transaction_id")
                    transaction_state = map_transactions.state(
                        transaction_id, group_id, msg["offsets"]
                    )
                    if transaction_state == "reserved":
                        if abort_prepared_map is None or not abort_prepared_map(
                            transaction_id, group_id=group_id
                        ):
                            raise RuntimeError(
                                f"Failed to abort prepared KV tensors for group_id={group_id}"
                            )
                        transaction_state = map_transactions.abort_reserved(
                            transaction_id, group_id, msg["offsets"]
                        )
                    elif transaction_state is None:
                        transaction_state = "not_prepared"
                    send_msg(
                        conn,
                        {
                            "status": "success",
                            "transaction_id": transaction_id,
                            "transaction_state": transaction_state,
                        },
                    )
                elif msg["cmd"] == "prepare_unmap_from_kv_tensors":
                    if prepare_unmap_from_kv_tensors is None:
                        raise RuntimeError(
                            "VMM extension does not support transactional unmap"
                        )
                    _sync_before_unmap(device_index)
                    if not prepare_unmap_from_kv_tensors(
                        msg["offsets"],
                        msg["transaction_id"],
                        group_id=group_id,
                    ):
                        raise RuntimeError(
                            f"Failed to prepare KV unmap for group_id={group_id}"
                        )
                    send_msg(conn, {"status": "prepared"})
                elif msg["cmd"] == "commit_unmap_from_kv_tensors":
                    if commit_unmap_from_kv_tensors is None:
                        raise RuntimeError(
                            "VMM extension does not support transactional unmap"
                        )
                    if not commit_unmap_from_kv_tensors(
                        msg["transaction_id"], group_id=group_id
                    ):
                        raise RuntimeError(
                            f"Failed to commit KV unmap for group_id={group_id}"
                        )
                    send_msg(conn, {"status": "committed"})
                elif msg["cmd"] == "abort_unmap_from_kv_tensors":
                    if abort_unmap_from_kv_tensors is None:
                        raise RuntimeError(
                            "VMM extension does not support transactional unmap"
                        )
                    if not abort_unmap_from_kv_tensors(
                        msg["transaction_id"], group_id=group_id
                    ):
                        raise RuntimeError(
                            f"Failed to abort KV unmap for group_id={group_id}"
                        )
                    send_msg(conn, {"status": "aborted"})
                elif msg["cmd"] == "orphan_map_transaction":
                    transaction_id = msg.get("transaction_id")
                    if transaction_id is None:
                        raise RuntimeError("KV map orphan request is missing transaction_id")
                    transaction_state = map_transactions.state(
                        transaction_id, group_id, msg["offsets"]
                    )
                    if transaction_state == "reserved":
                        if abort_prepared_map is None or not abort_prepared_map(
                            transaction_id, group_id=group_id
                        ):
                            raise RuntimeError(
                                f"Failed to abort prepared KV tensors for group_id={group_id}"
                            )
                        map_transactions.abort_reserved(
                            transaction_id, group_id, msg["offsets"]
                        )
                        transaction_state = "not_prepared"
                    elif transaction_state == "mapped":
                        _sync_after_map(device_index)
                        map_transactions.mark_prepared(
                            transaction_id, group_id, msg["offsets"]
                        )
                    if transaction_state != "not_prepared":
                        transaction_state = map_transactions.mark_orphan(
                            transaction_id, group_id, msg["offsets"]
                        )
                    if transaction_state == "orphaned":
                        print(
                            "[kvcached-map-transaction] event=orphan_retained "
                            f"transaction={transaction_id} pp_rank={pp_rank} "
                            f"rank={rank} group_id={group_id}",
                            flush=True,
                        )
                    send_msg(
                        conn,
                        {
                            "status": "success",
                            "transaction_id": transaction_id,
                            "transaction_state": transaction_state,
                        },
                    )
                elif msg["cmd"] == "get_map_transaction_state":
                    transaction_id = msg.get("transaction_id")
                    if transaction_id is None:
                        raise RuntimeError("KV map state query is missing transaction_id")
                    transaction_state = map_transactions.state(
                        transaction_id, group_id, msg["offsets"]
                    )
                    send_msg(
                        conn,
                        {
                            "status": "success",
                            "transaction_id": transaction_id,
                            "transaction_state": transaction_state or "not_prepared",
                        },
                    )
                elif msg["cmd"] == "finalize_map_to_kv_tensors":
                    transaction_id = msg.get("transaction_id")
                    if transaction_id is None:
                        raise RuntimeError("KV map finalize is missing transaction_id")
                    transaction_state = map_transactions.finalize(
                        transaction_id, group_id, msg["offsets"]
                    )
                    send_msg(
                        conn,
                        {
                            "status": "success",
                            "transaction_id": transaction_id,
                            "transaction_state": transaction_state,
                        },
                    )
                elif msg["cmd"] == "kv_tensors_created":
                    created: bool = kv_tensors_created(group_id=group_id)
                    send_msg(conn, {"status": "success", "created": created})
                elif msg["cmd"] == "cuda_mem_get_info":
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                    send_msg(conn, {
                        "status": "success",
                        "free_bytes": int(free_bytes),
                        "total_bytes": int(total_bytes),
                        "device": device_index,
                    })
                else:
                    send_msg(conn, {
                        "status": "error",
                        "message": "Unknown command"
                    })
            except Exception as e:
                capacity_rejection = bool(
                    physical_growth_stats
                    and physical_growth_stats.get("capacity_rejections", 0)
                )
                if not capacity_rejection:
                    print(
                        f"Worker {rank} error processing {msg.get('cmd', '?')} "
                        f"transaction={msg.get('transaction_id', '-')} "
                        f"group_id={msg.get('group_id', 0)}: {e}",
                        flush=True,
                    )
                try:
                    error_response: Message = {"status": "error", "message": str(e)}
                    if msg.get("cmd") in {
                        "prepare_map_to_kv_tensors",
                        "commit_prepared_map",
                    }:
                        error_response["physical_growth"] = physical_growth_stats
                        transaction_id = msg.get("transaction_id")
                        if transaction_id is not None:
                            try:
                                transaction_state = map_transactions.state(
                                    transaction_id,
                                    msg.get("group_id", 0),
                                    msg.get("offsets", []),
                                )
                            except RuntimeError:
                                # A malformed or mismatched request has an
                                # unknown outcome and must be reconciled by the
                                # caller instead of claiming not_prepared.
                                transaction_state = None
                            else:
                                error_response["transaction_state"] = (
                                    transaction_state or "not_prepared"
                                )
                    send_msg(conn, error_response)
                except (BrokenPipeError, ConnectionError, OSError):
                    # A bounded meminfo requester may time out while this worker
                    # is busy. Keep the listener alive for subsequent clients.
                    pass
            finally:
                conn.close()

    t = threading.Thread(target=listen_loop, daemon=True)
    t.start()


async def _send_and_receive_message(rank: int, message: Message, pp_rank: int = 0) -> Message:
    """
    Send a message to the worker and receive a response asynchronously.
    """

    async def exchange() -> Message:
        socket_path = get_worker_socket_path(rank, pp_rank)
        reader, writer = await asyncio.open_unix_connection(socket_path)

        try:
            data = pickle.dumps(message)
            writer.write(len(data).to_bytes(4, "big") + data)
            await writer.drain()
            length_bytes = await reader.readexactly(4)
            length = int.from_bytes(length_bytes, "big")
            data = await reader.readexactly(length)
            return cast(Message, pickle.loads(data))
        finally:
            writer.close()
            await writer.wait_closed()

    if IPC_TIMEOUT_S <= 0:
        return await exchange()
    try:
        return await asyncio.wait_for(exchange(), timeout=IPC_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"worker {rank} (pp_rank={pp_rank}) did not answer "
            f"{message.get('cmd', '?')} within {IPC_TIMEOUT_S:g}s"
        ) from None


async def _physical_device_ids(
    targets: list[tuple[int, int]],
) -> list[str]:
    missing = [target for target in targets if target not in _PHYSICAL_DEVICE_ID_CACHE]
    if missing:
        responses = await asyncio.gather(
            *[
                _send_and_receive_message(
                    rank, {"cmd": "get_physical_device_id"}, pp
                )
                for pp, rank in missing
            ],
            return_exceptions=True,
        )
        for (pp, rank), response in zip(missing, responses):
            if (
                isinstance(response, Exception)
                or not isinstance(response, dict)
                or response.get("status") != "success"
                or not response.get("physical_device_id")
            ):
                raise RuntimeError(
                    f"Worker pp{pp}/rank{rank} did not report its physical GPU: "
                    f"{response}"
                )
            _PHYSICAL_DEVICE_ID_CACHE[(pp, rank)] = str(
                response["physical_device_id"]
            )
    return sorted({_PHYSICAL_DEVICE_ID_CACHE[target] for target in targets})


async def _reconcile_map_transaction_states(
    targets: list[tuple[int, int]],
    responses: list[Any],
    offsets: list[int],
    group_id: int,
    transaction_id: str,
) -> tuple[dict[tuple[int, int], str], list[str]]:
    states: dict[tuple[int, int], str] = {}
    unknown_targets: list[tuple[int, int]] = []
    valid_states = {
        "not_prepared",
        "reserved",
        "mapped",
        "prepared",
        "committed",
        "aborted",
        "orphaned",
    }
    for target, response in zip(targets, responses):
        state = response.get("transaction_state") if isinstance(response, dict) else None
        if isinstance(state, str) and state in valid_states:
            states[target] = state
        else:
            unknown_targets.append(target)

    if not unknown_targets:
        return states, []

    message = {
        "cmd": "get_map_transaction_state",
        "offsets": offsets,
        "group_id": group_id,
        "transaction_id": transaction_id,
    }
    failures: list[str] = []
    # Retry only read-only reconciliation, never prepare/commit. Keep this
    # bounded even when ordinary IPC has its timeout disabled.
    query_timeout = min(IPC_TIMEOUT_S, 60.0) if IPC_TIMEOUT_S > 0 else 60.0
    for _ in range(2):
        state_responses = await asyncio.gather(
            *[
                asyncio.wait_for(
                    _send_and_receive_message(rank, message, pp),
                    timeout=query_timeout,
                )
                for pp, rank in unknown_targets
            ],
            return_exceptions=True,
        )
        failures = []
        pending = []
        for (pp, rank), response in zip(unknown_targets, state_responses):
            state = response.get("transaction_state") if isinstance(response, dict) else None
            if (
                isinstance(response, Exception)
                or not isinstance(response, dict)
                or response.get("status") != "success"
                or not isinstance(state, str)
                or state not in valid_states
            ):
                failures.append(
                    f"Worker pp{pp}/rank{rank} transaction outcome is unknown: {response}"
                )
                pending.append((pp, rank))
            else:
                states[(pp, rank)] = state
        unknown_targets = pending
        if not unknown_targets:
            break
    return states, failures


async def _broadcast_abort_prepared_map(
    targets: list[tuple[int, int]],
    offsets: list[int],
    group_id: int,
    transaction_id: str,
) -> list[str]:
    if not targets:
        return []
    message = {
        "cmd": "abort_prepared_map",
        "offsets": offsets,
        "group_id": group_id,
        "transaction_id": transaction_id,
    }
    responses = await asyncio.gather(
        *[_send_and_receive_message(rank, message, pp) for pp, rank in targets],
        return_exceptions=True,
    )
    failures = []
    for (pp, rank), response in zip(targets, responses):
        if (
            isinstance(response, Exception)
            or not isinstance(response, dict)
            or response.get("status") != "success"
            or response.get("transaction_state") not in {"aborted", "not_prepared"}
        ):
            failures.append(
                f"Worker pp{pp}/rank{rank} failed to abort reserved transaction "
                f"{transaction_id}: {response}"
            )
    return failures


async def _broadcast_map_to_kv_tensors(
    tp_size: int,
    offsets: list[int],
    pp_rank: int = 0,
    group_id: int = 0,
    record_stats: Optional[Callable[[dict[str, int]], None]] = None,
) -> None:
    """Reserve on each GPU, then map only after every worker succeeds."""
    targets = [(pp, rank) for pp in _target_pp_ranks(pp_rank) for rank in range(tp_size)]
    raise_if_physical_growth_unresolved()

    transaction_id = uuid.uuid4().hex
    await _physical_device_ids(targets)
    prepare_message = {
        "cmd": "prepare_map_to_kv_tensors",
        "offsets": offsets,
        "group_id": group_id,
        "transaction_id": transaction_id,
    }
    # Different GPUs reserve concurrently under their own short process-shared
    # admission guards. Workers sharing one physical GPU reserve in rounds so
    # every later capacity check observes earlier handles from this transaction.
    targets_by_device: dict[str, list[tuple[int, int]]] = {}
    for target in targets:
        targets_by_device.setdefault(_PHYSICAL_DEVICE_ID_CACHE[target], []).append(target)
    prepare_by_target: dict[tuple[int, int], Any] = {}
    while any(targets_by_device.values()):
        round_targets = [
            device_targets.pop(0)
            for device_targets in targets_by_device.values()
            if device_targets
        ]
        round_responses = await asyncio.gather(
            *[
                _send_and_receive_message(rank, prepare_message, pp)
                for pp, rank in round_targets
            ],
            return_exceptions=True,
        )
        prepare_by_target.update(zip(round_targets, round_responses))
    prepare_responses = [prepare_by_target[target] for target in targets]
    prepare_states, reconcile_failures = await _reconcile_map_transaction_states(
        targets, prepare_responses, offsets, group_id, transaction_id
    )
    if reconcile_failures:
        # A prepare may still be running after the IPC timeout. Preserve its
        # worker-side reservation and block further local growth rather than
        # guessing whether physical capacity was consumed.
        if record_stats is not None:
            record_stats(
                _build_physical_growth_operation_counters(
                    offsets, prepare_responses
                )
            )
        _fail_unresolved_map_transaction(
            transaction_id, offsets, group_id, "prepare", reconcile_failures
        )

    prepare_failures: list[str] = []
    for (pp, rank), response in zip(targets, prepare_responses):
        state = prepare_states[(pp, rank)]
        if state in {"reserved", "prepared", "committed"}:
            continue
        prepare_failures.append(
            f"Worker pp{pp}/rank{rank} failed to reserve: {response}; "
            f"transaction_state={state}"
        )

    if prepare_failures:
        reserved_targets = [
            target for target, state in prepare_states.items() if state == "reserved"
        ]
        retained_targets = [
            target
            for target, state in prepare_states.items()
            if state in {"mapped", "prepared"}
        ]
        cleanup_failures = await _broadcast_abort_prepared_map(
            reserved_targets, offsets, group_id, transaction_id
        )
        cleanup_failures.extend(
            await _broadcast_orphan_map_transaction(
                retained_targets, offsets, group_id, transaction_id
            )
            if retained_targets
            else []
        )
        if record_stats is not None:
            record_stats(
                _build_physical_growth_operation_counters(
                    offsets, prepare_responses
                )
            )
        if cleanup_failures:
            _fail_unresolved_map_transaction(
                transaction_id, offsets, group_id, "prepare cleanup",
                prepare_failures + cleanup_failures,
            )
        raise RuntimeError(
            f"KV map transaction {transaction_id} reservation failed: "
            + "; ".join(prepare_failures)
        )

    # Physical capacity has now been reserved on every target GPU. Per-device
    # admission guards have already been released; CUDA mapping and
    # synchronization therefore stay outside the cross-instance critical path.
    commit_message = {
        "cmd": "commit_prepared_map",
        "offsets": offsets,
        "group_id": group_id,
        "transaction_id": transaction_id,
    }
    commit_responses = await asyncio.gather(
        *[
            _send_and_receive_message(rank, commit_message, pp)
            for pp, rank in targets
        ],
        return_exceptions=True,
    )
    commit_states, reconcile_failures = await _reconcile_map_transaction_states(
        targets, commit_responses, offsets, group_id, transaction_id
    )
    if record_stats is not None:
        record_stats(
            _build_physical_growth_operation_counters(
                offsets,
                prepare_responses + commit_responses,
            )
        )

    if reconcile_failures:
        _fail_unresolved_map_transaction(
            transaction_id, offsets, group_id, "commit", reconcile_failures
        )

    commit_failures: list[str] = []
    retain_targets: list[tuple[int, int]] = []
    abort_targets: list[tuple[int, int]] = []
    for (pp, rank), response in zip(targets, commit_responses):
        state = commit_states[(pp, rank)]
        if state in {"prepared", "committed"}:
            continue
        if state == "mapped":
            retain_targets.append((pp, rank))
        elif state == "reserved":
            abort_targets.append((pp, rank))
        commit_failures.append(
            f"Worker pp{pp}/rank{rank} failed to commit: {response}; "
            f"transaction_state={state}"
        )

    if commit_failures:
        cleanup_failures = await _broadcast_abort_prepared_map(
            abort_targets, offsets, group_id, transaction_id
        )
        prepared_targets = [
            target
            for target, state in commit_states.items()
            if state == "prepared"
        ]
        cleanup_failures.extend(
            await _broadcast_orphan_map_transaction(
                sorted(set(retain_targets + prepared_targets)),
                offsets,
                group_id,
                transaction_id,
            )
        )
        if cleanup_failures:
            _fail_unresolved_map_transaction(
                transaction_id, offsets, group_id, "commit cleanup",
                commit_failures + cleanup_failures,
            )
        raise RuntimeError(
            f"KV map transaction {transaction_id} failed; mapped pages were "
            "retained for a same-offset retry: "
            + "; ".join(commit_failures)
        )

    finalize_message = {
        "cmd": "finalize_map_to_kv_tensors",
        "offsets": offsets,
        "group_id": group_id,
        "transaction_id": transaction_id,
    }
    finalize_responses = await asyncio.gather(
        *[_send_and_receive_message(rank, finalize_message, pp) for pp, rank in targets],
        return_exceptions=True,
    )
    for (pp, rank), response in zip(targets, finalize_responses):
        if (
            isinstance(response, Exception)
            or not isinstance(response, dict)
            or response.get("status") != "success"
        ):
            print(
                "KVCached warning: worker "
                f"pp{pp}/rank{rank} did not finalize committed KV map "
                f"transaction {transaction_id}: {response}",
                flush=True,
            )


async def _broadcast_orphan_map_transaction(
    targets: list[tuple[int, int]],
    offsets: list[int],
    group_id: int,
    transaction_id: str,
) -> list[str]:
    """Retain partial mappings so a same-offset retry can adopt them safely."""
    orphan_message = {
        "cmd": "orphan_map_transaction",
        "offsets": offsets,
        "group_id": group_id,
        "transaction_id": transaction_id,
    }
    tasks = [_send_and_receive_message(rank, orphan_message, pp) for pp, rank in targets]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    failures: list[str] = []
    for (pp, rank), response in zip(targets, responses):
        if (
            isinstance(response, Exception)
            or not isinstance(response, dict)
            or response.get("status") != "success"
            or response.get("transaction_state") not in {"orphaned", "not_prepared"}
        ):
            failures.append(
                f"Worker pp{pp}/rank{rank} failed to retain transaction "
                f"{transaction_id}: {response}"
            )
    return failures


async def _broadcast_unmap_from_kv_tensors(
    tp_size: int,
    offsets: list[int],
    pp_rank: int = 0,
    group_id: int = 0,
) -> None:
    """Unmap all TP/PP workers as one recoverable transaction."""
    transaction_id = uuid.uuid4().hex
    targets = [
        (target_pp_rank, rank)
        for target_pp_rank in _target_pp_ranks(pp_rank)
        for rank in range(tp_size)
    ]
    prepare_message = {
        "cmd": "prepare_unmap_from_kv_tensors",
        "offsets": offsets,
        "transaction_id": transaction_id,
        "group_id": group_id,
    }
    prepare_responses = await asyncio.gather(
        *[
            _send_and_receive_message(rank, prepare_message, target_pp_rank)
            for target_pp_rank, rank in targets
        ],
        return_exceptions=True,
    )
    prepare_failures = [
        f"pp{target_pp_rank}/rank{rank}: {response}"
        for (target_pp_rank, rank), response in zip(targets, prepare_responses)
        if isinstance(response, Exception)
        or not isinstance(response, dict)
        or response.get("status") != "prepared"
    ]

    if prepare_failures:
        abort_message = {
            "cmd": "abort_unmap_from_kv_tensors",
            "transaction_id": transaction_id,
            "group_id": group_id,
        }
        abort_responses = await asyncio.gather(
            *[
                _send_and_receive_message(rank, abort_message, target_pp_rank)
                for target_pp_rank, rank in targets
            ],
            return_exceptions=True,
        )
        abort_failures = [
            f"pp{target_pp_rank}/rank{rank}: {response}"
            for (target_pp_rank, rank), response in zip(targets, abort_responses)
            if isinstance(response, Exception)
            or not isinstance(response, dict)
            or response.get("status") != "aborted"
        ]
        message = "KV unmap prepare failed: " + "; ".join(prepare_failures)
        if abort_failures:
            message += "; state_consistency_unknown after abort failures: "
            message += "; ".join(abort_failures)
        raise RuntimeError(message)

    commit_message = {
        "cmd": "commit_unmap_from_kv_tensors",
        "transaction_id": transaction_id,
        "group_id": group_id,
    }
    pending_targets = targets
    commit_failures: list[str] = []
    for _attempt in range(2):
        commit_responses = await asyncio.gather(
            *[
                _send_and_receive_message(rank, commit_message, target_pp_rank)
                for target_pp_rank, rank in pending_targets
            ],
            return_exceptions=True,
        )
        failed_targets = []
        commit_failures = []
        for (target_pp_rank, rank), response in zip(
            pending_targets, commit_responses
        ):
            if (
                isinstance(response, Exception)
                or not isinstance(response, dict)
                or response.get("status") != "committed"
            ):
                failed_targets.append((target_pp_rank, rank))
                commit_failures.append(
                    f"pp{target_pp_rank}/rank{rank}: {response}"
                )
        if not failed_targets:
            return
        pending_targets = failed_targets

    raise RuntimeError(
        "state_consistency_unknown: KV unmap commit could not be confirmed "
        "after retry: " + "; ".join(commit_failures)
    )


async def _broadcast_kv_tensors_created(tp_size: int,
                                        pp_rank: int = 0,
                                        group_id: int = 0) -> bool:
    """
    Broadcast the "kv_tensors_created" operation to all workers concurrently.
    Returns True if all workers report that KV tensors are created, False otherwise.
    """
    check_message = {"cmd": "kv_tensors_created", "group_id": group_id}
    targets = [(pp, rank) for pp in _target_pp_ranks(pp_rank) for rank in range(tp_size)]
    tasks = [
        _send_and_receive_message(rank, check_message, pp)
        for pp, rank in targets
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    all_created = True
    for (pp, rank), response in zip(targets, responses):
        if isinstance(response, Exception):
            raise RuntimeError(
                f"Worker pp{pp}/rank{rank} failed to check KV tensors created: {response}"
            )
        elif not isinstance(response,
                            dict) or response.get("status") != "success":
            raise RuntimeError(
                f"Worker pp{pp}/rank{rank} failed to check KV tensors created: {response}"
            )
        elif not response.get("created", False):
            all_created = False

    return all_created


# Wrapper functions to call the async function from sync code
def broadcast_map_to_kv_tensors(
    tp_size: int,
    offsets: list[int],
    pp_rank: int = 0,
    group_id: int = 0,
    record_stats: Optional[Callable[[dict[str, int]], None]] = None,
) -> None:
    asyncio.run(
        _broadcast_map_to_kv_tensors(
            tp_size,
            offsets,
            pp_rank,
            group_id,
            record_stats,
        )
    )


def broadcast_unmap_from_kv_tensors(tp_size: int, offsets: list[int],
                                    pp_rank: int = 0,
                                    group_id: int = 0) -> None:
    asyncio.run(_broadcast_unmap_from_kv_tensors(tp_size, offsets, pp_rank,
                                                 group_id))


def broadcast_kv_tensors_created(tp_size: int, pp_rank: int = 0,
                                 group_id: int = 0,
                                 timeout_s: Optional[float] = None) -> bool:
    return asyncio.run(asyncio.wait_for(
        _broadcast_kv_tensors_created(tp_size, pp_rank, group_id), timeout_s
    ))


def query_worker_cuda_mem_get_info(rank: int,
                                   pp_rank: int = 0,
                                   timeout: float = 0.1) -> tuple[int, int]:
    socket_path = get_worker_socket_path(rank, pp_rank)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(socket_path)
        send_msg(sock, {"cmd": "cuda_mem_get_info"})
        response = recv_msg(sock)
    if response.get("status") != "success":
        raise RuntimeError(f"worker cuda_mem_get_info failed: {response}")
    return int(response["free_bytes"]), int(response["total_bytes"])
