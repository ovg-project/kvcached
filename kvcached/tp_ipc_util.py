# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import os
import pickle
import socket
import threading
import uuid
from typing import Any, Dict, Optional, Tuple, cast

from kvcached import vmm_ops
from kvcached.errors import MapQuarantinedError, StateConsistencyError
from kvcached.utils import get_tp_socket_dir, normalize_gpu_device

kv_tensors_created = vmm_ops.kv_tensors_created
map_to_kv_tensors = vmm_ops.map_to_kv_tensors
unmap_from_kv_tensors = vmm_ops.unmap_from_kv_tensors
prepare_unmap_from_kv_tensors = getattr(vmm_ops, "prepare_unmap_from_kv_tensors", None)
commit_unmap_from_kv_tensors = getattr(vmm_ops, "commit_unmap_from_kv_tensors", None)
abort_unmap_from_kv_tensors = getattr(vmm_ops, "abort_unmap_from_kv_tensors", None)


def _map_to_kv_tensors_with_result(offsets: list[int], group_id: int) -> tuple[bool, list[int]]:
    operation = getattr(vmm_ops, "map_to_kv_tensors_with_result", None)
    if operation is None:
        raise RuntimeError("VMM extension does not support transactional map")
    success, newly_mapped = operation(offsets, group_id=group_id)
    return bool(success), [int(offset) for offset in newly_mapped]


def _sync_before_unmap() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Socket directory for tensor parallel (TP) worker communication:
# /tmp/kvcached-tp-<ipc_name>-<hash>. Unix domain socket paths are limited to
# 108 characters on Linux, so the name is kept short and the final socket path
# length is validated below.
SOCKET_DIR = get_tp_socket_dir()


def _target_pp_ranks(pp_rank: int) -> list[int]:
    if pp_rank >= 0:
        return [pp_rank]
    pp_size = int(os.getenv("KVCACHED_PP_SIZE", "1") or "1")
    return list(range(max(pp_size, 1)))


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
    sock.sendall(len(data).to_bytes(4, "big") + data)


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
    length = int.from_bytes(length_bytes, "big")
    if length <= 0:
        raise ValueError("Received invalid length for message")
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError("Socket connection closed while receiving data")
        data += chunk
    if len(data) != length:
        raise ValueError("Received data length does not match expected length")
    return cast(Message, pickle.loads(data))


def resolve_gpu_device_index(device: Optional[str]) -> int:
    """Resolve an explicitly indexed GPU device without consulting thread state."""
    import torch

    if device is not None:
        parsed_device = torch.device(normalize_gpu_device(device))
        if parsed_device.type == "cuda" and parsed_device.index is not None:
            return int(parsed_device.index)
    raise ValueError(f"Expected an explicitly indexed GPU device, got {device!r}")


# How long stop() waits for the listener thread to finish in-flight work.
# Successful stop means the thread has exited, so no handler can still be
# inside a VMM call when the integrations go on to shut the allocator down;
# on timeout nothing is torn down and the listener is kept for a retry.
STOP_DRAIN_TIMEOUT_S: float = 5.0


class _WorkerListener:
    """One worker's IPC listener: its bound socket, the directory holding it,
    the thread serving it, and the connections that thread has accepted."""

    def __init__(self, rank: int, pp_rank: int, root_dir: str, socket_dir: str,
                 socket_path: str, server_sock: socket.socket) -> None:
        self.rank = rank
        self.pp_rank = pp_rank
        self.root_dir = root_dir
        self.socket_dir = socket_dir
        self.socket_path = socket_path
        self.server_sock = server_sock
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self._conns: set[socket.socket] = set()
        self._conn_lock = threading.Lock()
        self._stopped = False

    def track_conn(self, conn: socket.socket) -> bool:
        """Register an accepted connection so stop() can cancel its read.
        Refused once stop has begun: stop() no longer sees the connection,
        so the listener must drop it without reading."""
        with self._conn_lock:
            if self.stop_event.is_set():
                return False
            self._conns.add(conn)
            return True

    def untrack_conn(self, conn: socket.socket) -> None:
        with self._conn_lock:
            self._conns.discard(conn)
        conn.close()

    def stop(self, drain_timeout_s: float = STOP_DRAIN_TIMEOUT_S) -> bool:
        """Stop serving, then unlink the socket and remove the directory
        once no other worker's socket is left in it. Returns True when the
        listener thread has exited and everything is cleaned up.

        Setting stop_event first means no new dispatch can start: reads of
        already accepted connections are cancelled below, and a connection
        accepted from now on is dropped unread (track_conn refuses it).
        The join then drains a handler that is already inside a VMM call,
        so a successful stop guarantees no handler touches the allocator
        afterwards. If the thread does not exit in time, nothing is torn
        down and the caller keeps the listener for a retry.
        """
        if self._stopped:
            return True
        with self._conn_lock:
            self.stop_event.set()
            conns = list(self._conns)
        for conn in conns:
            # shutdown, not close: the handler owns the socket and closes it
            # itself, so its descriptor cannot be reused under the handler.
            # A blocked recv_msg() wakes with a connection error.
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # the handler is already past this connection
        # accept() only returns on a connection, so make one to let an idle
        # loop observe stop_event.
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as wake:
                wake.settimeout(1.0)
                wake.connect(self.socket_path)
        except OSError:
            pass
        thread = self.thread
        if thread is threading.current_thread():
            return False  # A handler cannot drain itself.
        if thread is not None:
            thread.join(timeout=drain_timeout_s)
            if thread.is_alive():
                print(f"Worker {self.rank} IPC listener still busy after "
                      f"{drain_timeout_s:g}s; keeping it for a retry")
                return False
        self.server_sock.close()
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        _remove_dir_if_empty(self.socket_dir)
        if self.socket_dir != self.root_dir:
            _remove_dir_if_empty(self.root_dir)
        self._stopped = True
        print(f"Worker {self.rank} IPC listener stopped, removed {self.socket_path}")
        return True


def _remove_dir_if_empty(path: str) -> None:
    try:
        os.rmdir(path)
    except OSError:
        # Still holds another worker's socket, or already gone.
        pass


_listeners: Dict[Tuple[int, int], _WorkerListener] = {}
_listeners_lock = threading.Lock()
_atexit_registered = False


def stop_worker_listener_threads(
        drain_timeout_s: float = STOP_DRAIN_TIMEOUT_S) -> bool:
    """Stop every worker IPC listener started in this process: cancel reads
    on its accepted connections, drain a handler that is already executing,
    then unlink the socket and remove the per-instance socket directory
    (issue #476). Safe to call repeatedly and when nothing was started.

    Returns True when every listener fully stopped. A listener whose thread
    is still busy after drain_timeout_s stays registered with nothing torn
    down, so a later call (or the interpreter-exit hook) retries it; that
    case returns False.
    """
    with _listeners_lock:
        all_stopped = True
        # Keep entries visible until drained, including after cleanup errors.
        # Serializing lifecycle calls also prevents another stop from seeing
        # an empty registry while this call is still draining its handlers.
        for key, listener in list(_listeners.items()):
            try:
                stopped = listener.stop(drain_timeout_s)
            except Exception as e:
                print(f"Worker {listener.rank} IPC listener cleanup failed: {e}")
                all_stopped = False
                continue
            if stopped:
                del _listeners[key]
            else:
                all_stopped = False
        return all_stopped


def start_worker_listener_thread(
    rank: int,
    pp_rank: int = 0,
    device_index: Optional[int] = None,
):
    """Start a thread that listens for messages on the worker socket.

    ``pp_rank`` selects a PP-stage-specific socket directory so concurrent
    stages do not bind the same path. When ``device_index`` is provided, the
    listener restores that CUDA device inside the new thread before executing
    CUDA-backed map or unmap operations because CUDA's current device is
    thread-local.

    The listener is registered so that stop_worker_listener_threads() (called
    from the integrations' shutdown paths and at interpreter exit) can unlink
    the socket and remove the directory again.
    """
    with _listeners_lock:
        _start_worker_listener_thread(rank, pp_rank, device_index)


def _start_worker_listener_thread(
    rank: int, pp_rank: int, device_index: Optional[int],
) -> None:
    """Start or replace a listener while holding the lifecycle lock."""
    global _atexit_registered
    key = (rank, pp_rank)
    previous = _listeners.get(key)
    if previous is not None:
        if not previous.stop():
            raise RuntimeError(
                f"Cannot replace worker {rank} IPC listener while it is still active"
            )
        del _listeners[key]

    root_dir = SOCKET_DIR
    socket_dir = os.path.join(root_dir, f"pp{pp_rank}") if pp_rank > 0 else root_dir
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
    listener = _WorkerListener(rank, pp_rank, root_dir, socket_dir, socket_path,
                               server_sock)

    def listen_loop():
        if device_index is not None:
            import torch

            # CUDA's current device is thread-local, so restore the worker device.
            torch.cuda.set_device(device_index)
        print(f"Worker {rank} IPC listener started at {socket_path}")
        while True:
            try:
                conn, _ = server_sock.accept()
            except OSError:
                break  # socket closed by stop()
            if not listener.track_conn(conn):
                # stop() began after this accept and cannot see the
                # connection any more, so drop it without reading.
                conn.close()
                break
            unmap_phase = None
            try:
                msg: Message = recv_msg(conn)
                # print(f"Worker {rank} received message: {msg}")
                if listener.stop_event.is_set():
                    break  # stop() is waiting; do not start a dispatch
                group_id: int = msg.get("group_id", 0)
                if msg["cmd"] == "map_to_kv_tensors":
                    success, newly_mapped = _map_to_kv_tensors_with_result(msg["offsets"], group_id)
                    if not success:
                        raise RuntimeError(f"Failed to map KV tensors for group_id={group_id}")
                    send_msg(
                        conn,
                        {
                            "status": "success",
                            "newly_mapped_offsets": newly_mapped,
                        },
                    )
                elif msg["cmd"] == "unmap_from_kv_tensors":
                    _sync_before_unmap()
                    if not unmap_from_kv_tensors(msg["offsets"], group_id=group_id):
                        raise RuntimeError(f"Failed to unmap KV tensors for group_id={group_id}")
                    send_msg(conn, {"status": "success"})
                elif msg["cmd"] == "prepare_unmap_from_kv_tensors":
                    unmap_phase = "prepare"
                    if prepare_unmap_from_kv_tensors is None:
                        raise RuntimeError("VMM extension does not support transactional unmap")
                    if msg.get("commit") and commit_unmap_from_kv_tensors is None:
                        raise RuntimeError("VMM extension does not support transactional unmap")
                    _sync_before_unmap()
                    if not prepare_unmap_from_kv_tensors(
                        msg["offsets"], msg["transaction_id"], group_id=group_id
                    ):
                        raise RuntimeError(f"Failed to prepare KV unmap for group_id={group_id}")
                    if msg.get("commit"):
                        # One target needs no cross-worker prepare barrier. Keep
                        # the native transaction ID so a lost reply is retryable.
                        assert commit_unmap_from_kv_tensors is not None
                        unmap_phase = "commit"
                        if not commit_unmap_from_kv_tensors(msg["transaction_id"], group_id=group_id):
                            raise RuntimeError(f"Failed to commit KV unmap for group_id={group_id}")
                        send_msg(conn, {"status": "committed"})
                    else:
                        send_msg(conn, {"status": "prepared"})
                elif msg["cmd"] == "commit_unmap_from_kv_tensors":
                    if commit_unmap_from_kv_tensors is None:
                        raise RuntimeError("VMM extension does not support transactional unmap")
                    if not commit_unmap_from_kv_tensors(msg["transaction_id"], group_id=group_id):
                        raise RuntimeError(f"Failed to commit KV unmap for group_id={group_id}")
                    send_msg(conn, {"status": "committed"})
                elif msg["cmd"] == "abort_unmap_from_kv_tensors":
                    if abort_unmap_from_kv_tensors is None:
                        raise RuntimeError("VMM extension does not support transactional unmap")
                    if not abort_unmap_from_kv_tensors(msg["transaction_id"], group_id=group_id):
                        raise RuntimeError(f"Failed to abort KV unmap for group_id={group_id}")
                    send_msg(conn, {"status": "aborted"})
                elif msg["cmd"] == "kv_tensors_created":
                    created: bool = kv_tensors_created(group_id=group_id)
                    send_msg(conn, {"status": "success", "created": created})
                else:
                    send_msg(conn, {"status": "error", "message": "Unknown command"})
            except Exception as e:
                if listener.stop_event.is_set():
                    break  # read cancelled by stop()
                print(f"Worker {rank} error processing message: {e}")
                error_type = (
                    "state_consistency" if isinstance(e, StateConsistencyError) else
                    "map_quarantined" if isinstance(e, MapQuarantinedError) else
                    "operation_failed"
                )
                response = {"status": "error", "message": str(e), "error_type": error_type}
                if unmap_phase is not None:
                    response["phase"] = unmap_phase
                try:
                    send_msg(conn, response)
                except OSError:
                    pass  # peer went away; nothing to answer any more
            finally:
                listener.untrack_conn(conn)

    t = threading.Thread(target=listen_loop, daemon=True)
    listener.thread = t
    t.start()
    _listeners[key] = listener
    if not _atexit_registered:
        atexit.register(stop_worker_listener_threads)
        _atexit_registered = True


# How long one worker-IPC exchange may take before it is treated as a failure.
# Without a bound, a worker that is alive but not answering (its serial
# listener stuck on an earlier operation) parks the caller in readexactly()
# forever. For the C++ prealloc thread that wait is fatal: alloc_page() blocks
# indefinitely on the reserve it will never receive (issue #371). A timeout
# converts the silent hang into an exception, which the callers already handle
# (the prealloc worker returns the in-flight pages and logs; a foreground
# caller propagates the error). <= 0 disables the bound.
IPC_TIMEOUT_S: float = float(os.getenv("KVCACHED_IPC_TIMEOUT", "60"))


async def _send_and_receive_message(rank: int, message: Message, pp_rank: int = 0) -> Message:
    """
    Send a message to the worker and receive a response asynchronously.

    Raises RuntimeError naming the worker rank if the exchange does not
    complete within IPC_TIMEOUT_S.
    """

    async def exchange() -> Message:
        socket_path = get_worker_socket_path(rank, pp_rank)
        reader, writer = await asyncio.open_unix_connection(socket_path)

        try:
            # Send map command
            data = pickle.dumps(message)
            writer.write(len(data).to_bytes(4, 'big') + data)
            await writer.drain()

            # Read the length of the response from worker
            length_bytes = await reader.readexactly(4)
            length = int.from_bytes(length_bytes, 'big')

            # Read the actual response data
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
            f"{message.get('cmd', '?')} within {IPC_TIMEOUT_S:g}s "
            "(KVCACHED_IPC_TIMEOUT); the worker process is alive but its "
            "IPC listener is not responding"
        ) from None


async def _broadcast_map_to_kv_tensors(
    tp_size: int, offsets: list[int], pp_rank: int = 0, group_id: int = 0
) -> None:
    """
    Broadcast the "map_to_kv_tensors" operation to all workers concurrently.
    """
    map_message = {"cmd": "map_to_kv_tensors", "offsets": offsets, "group_id": group_id}
    targets = [
        (target_pp_rank, rank)
        for target_pp_rank in _target_pp_ranks(pp_rank)
        for rank in range(tp_size)
    ]
    tasks = [
        _send_and_receive_message(rank, map_message, target_pp_rank)
        for target_pp_rank, rank in targets
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    failures: list[str] = []
    unknown_targets: list[str] = []
    rollback_targets: list[tuple[int, int, list[int]]] = []
    for (target_pp_rank, rank), response in zip(targets, responses):
        if isinstance(response, Exception):
            target = f"pp{target_pp_rank}/rank{rank}"
            failures.append(f"Worker {target} failed to map: {response}")
            unknown_targets.append(target)
        elif not isinstance(response, dict) or response.get("status") != "success":
            failures.append(f"Worker pp{target_pp_rank}/rank{rank} failed to map: {response}")
        else:
            newly_mapped = response.get("newly_mapped_offsets")
            if newly_mapped is None:
                newly_mapped = []
            rollback_targets.append(
                (
                    target_pp_rank,
                    rank,
                    [int(offset) for offset in newly_mapped],
                )
            )

    if not failures:
        return

    rollback_tasks = []
    rollback_task_targets = []
    for target_pp_rank, rank, newly_mapped in rollback_targets:
        if not newly_mapped:
            continue
        rollback_task_targets.append((target_pp_rank, rank))
        rollback_tasks.append(
            _send_and_receive_message(
                rank,
                {
                    "cmd": "unmap_from_kv_tensors",
                    "offsets": newly_mapped,
                    "group_id": group_id,
                },
                target_pp_rank,
            )
        )

    rollback_failures: list[str] = []
    if rollback_tasks:
        rollback_responses = await asyncio.gather(*rollback_tasks, return_exceptions=True)
        for (target_pp_rank, rank), response in zip(rollback_task_targets, rollback_responses):
            if isinstance(response, Exception):
                rollback_failures.append(f"pp{target_pp_rank}/rank{rank}: {response}")
            elif not isinstance(response, dict) or response.get("status") != "success":
                rollback_failures.append(f"pp{target_pp_rank}/rank{rank}: {response}")

    message = "; ".join(failures)
    if unknown_targets:
        message += "; state_consistency_unknown for workers with lost responses: " + ", ".join(
            unknown_targets
        )
    if rollback_failures:
        message += "; rollback failures: " + "; ".join(rollback_failures)
    # Unknown outcomes cannot establish that every affected address is contained.
    if unknown_targets or rollback_failures or any(
        isinstance(response, dict) and response.get("error_type") == "state_consistency"
        for response in responses
    ):
        raise StateConsistencyError(message)
    if any(
        isinstance(response, dict) and response.get("error_type") == "map_quarantined"
        for response in responses
    ):
        raise MapQuarantinedError(message)
    raise RuntimeError(message)


async def _broadcast_unmap_from_kv_tensors(
    tp_size: int, offsets: list[int], pp_rank: int = 0, group_id: int = 0
) -> None:
    """
    Broadcast the "unmap_from_kv_tensors" operation to all workers concurrently.
    """
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
    single_target = len(targets) == 1
    if single_target:
        prepare_message["commit"] = True
    prepare_tasks = [
        _send_and_receive_message(rank, prepare_message, target_pp_rank)
        for target_pp_rank, rank in targets
    ]
    prepare_responses = await asyncio.gather(*prepare_tasks, return_exceptions=True)
    prepare_failures = [
        f"pp{target_pp_rank}/rank{rank}: {response}"
        for (target_pp_rank, rank), response in zip(targets, prepare_responses)
        if isinstance(response, Exception)
        or not isinstance(response, dict)
        or response.get("status") != "prepared"
    ]

    if single_target:
        response = prepare_responses[0]
        if isinstance(response, dict) and response.get("status") == "committed":
            return
        # After an ambiguous reply, commit may already have released handles.
        # Never abort then: retry commit, which is idempotent in the worker.
        if not (
            isinstance(response, dict)
            and response.get("status") == "error"
            and response.get("phase") == "prepare"
        ):
            prepare_failures = []

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
        if abort_failures or any(
            isinstance(response, dict) and response.get("error_type") == "state_consistency"
            for response in prepare_responses
        ):
            raise StateConsistencyError(message)
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
        for (target_pp_rank, rank), response in zip(pending_targets, commit_responses):
            if (
                isinstance(response, Exception)
                or not isinstance(response, dict)
                or response.get("status") != "committed"
            ):
                failed_targets.append((target_pp_rank, rank))
                commit_failures.append(f"pp{target_pp_rank}/rank{rank}: {response}")
        if not failed_targets:
            return
        pending_targets = failed_targets

    raise StateConsistencyError(
        "state_consistency_unknown: KV unmap commit could not be confirmed after retry: "
        + "; ".join(commit_failures)
    )


async def _broadcast_kv_tensors_created(tp_size: int, pp_rank: int = 0, group_id: int = 0) -> bool:
    """
    Broadcast the "kv_tensors_created" operation to all workers concurrently.
    Returns True if all workers report that KV tensors are created, False otherwise.
    """
    check_message = {"cmd": "kv_tensors_created", "group_id": group_id}
    targets = [
        (target_pp_rank, rank)
        for target_pp_rank in _target_pp_ranks(pp_rank)
        for rank in range(tp_size)
    ]
    tasks = [
        _send_and_receive_message(rank, check_message, target_pp_rank)
        for target_pp_rank, rank in targets
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    all_created = True
    for (target_pp_rank, rank), response in zip(targets, responses):
        if isinstance(response, Exception):
            raise RuntimeError(
                f"Worker pp{target_pp_rank}/rank{rank} failed to check "
                f"KV tensors created: {response}"
            )
        elif not isinstance(response, dict) or response.get("status") != "success":
            raise RuntimeError(
                f"Worker pp{target_pp_rank}/rank{rank} failed to check "
                f"KV tensors created: {response}"
            )
        elif not response.get("created", False):
            all_created = False

    return all_created


# Wrapper functions to call the async function from sync code
def broadcast_map_to_kv_tensors(
    tp_size: int, offsets: list[int], pp_rank: int = 0, group_id: int = 0
) -> None:
    asyncio.run(_broadcast_map_to_kv_tensors(tp_size, offsets, pp_rank, group_id))


def broadcast_unmap_from_kv_tensors(
    tp_size: int, offsets: list[int], pp_rank: int = 0, group_id: int = 0
) -> None:
    asyncio.run(_broadcast_unmap_from_kv_tensors(tp_size, offsets, pp_rank, group_id))


def broadcast_kv_tensors_created(tp_size: int, pp_rank: int = 0, group_id: int = 0) -> bool:
    return asyncio.run(_broadcast_kv_tensors_created(tp_size, pp_rank, group_id))
