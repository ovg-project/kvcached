# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import os
import pickle
import socket
import threading
from typing import Any, Dict, Optional, Tuple, cast

from kvcached.utils import get_tp_socket_dir
from kvcached.vmm_ops import kv_tensors_created, map_to_kv_tensors, unmap_from_kv_tensors

# Socket directory for tensor parallel (TP) worker communication:
# /tmp/kvcached-tp-<ipc_name>-<hash>. Unix domain socket paths are limited to
# 108 characters on Linux, so the name is kept short and the final socket path
# length is validated below.
SOCKET_DIR = get_tp_socket_dir()


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


class _WorkerListener:
    """One worker's IPC listener: its bound socket, the directory holding it,
    and the thread serving it."""

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

    def stop(self) -> None:
        """Stop serving, unlink the socket, and remove the directory once no
        other worker's socket is left in it."""
        self.stop_event.set()
        # accept() only returns on a connection, so make one to let the loop
        # observe stop_event. If that fails the daemon thread simply dies
        # with the process; the socket file is unlinked either way.
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as wake:
                wake.settimeout(1.0)
                wake.connect(self.socket_path)
        except OSError:
            pass
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.server_sock.close()
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        _remove_dir_if_empty(self.socket_dir)
        if self.socket_dir != self.root_dir:
            _remove_dir_if_empty(self.root_dir)
        print(f"Worker {self.rank} IPC listener stopped, removed {self.socket_path}")


def _remove_dir_if_empty(path: str) -> None:
    try:
        os.rmdir(path)
    except OSError:
        # Still holds another worker's socket, or already gone.
        pass


_listeners: Dict[Tuple[int, int], _WorkerListener] = {}
_listeners_lock = threading.Lock()
_atexit_registered = False


def stop_worker_listener_threads() -> None:
    """Stop every worker IPC listener started in this process: close its
    socket, unlink it, and remove the per-instance socket directory
    (issue #476). Safe to call repeatedly and when nothing was started.
    """
    with _listeners_lock:
        listeners = list(_listeners.values())
        _listeners.clear()
    for listener in listeners:
        listener.stop()


def start_worker_listener_thread(rank: int, pp_rank: int = 0):
    """
    Start a thread that listens for messages on the worker socket.
    pp_rank is used to create a PP-stage-specific subdirectory so that
    concurrent SGLang PP stages do not bind the same socket path.

    The listener is registered so that stop_worker_listener_threads() (called
    from the integrations' shutdown paths and at interpreter exit) can unlink
    the socket and remove the directory again.
    """
    global _atexit_registered
    with _listeners_lock:
        previous = _listeners.pop((rank, pp_rank), None)
    if previous is not None:
        previous.stop()

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
        print(f"Worker {rank} IPC listener started at {socket_path}")
        while True:
            try:
                conn, _ = server_sock.accept()
            except OSError:
                break  # socket closed by stop()
            if listener.stop_event.is_set():
                conn.close()
                break
            try:
                msg: Message = recv_msg(conn)
                # print(f"Worker {rank} received message: {msg}")
                group_id: int = msg.get("group_id", 0)
                if msg["cmd"] == "map_to_kv_tensors":
                    map_to_kv_tensors(msg["offsets"], group_id=group_id)
                    send_msg(conn, {"status": "success"})
                elif msg["cmd"] == "unmap_from_kv_tensors":
                    unmap_from_kv_tensors(msg["offsets"], group_id=group_id)
                    send_msg(conn, {"status": "success"})
                elif msg["cmd"] == "kv_tensors_created":
                    created: bool = kv_tensors_created(group_id=group_id)
                    send_msg(conn, {"status": "success", "created": created})
                else:
                    send_msg(conn, {
                        "status": "error",
                        "message": "Unknown command"
                    })
            except Exception as e:
                print(f"Worker {rank} error processing message: {e}")
                send_msg(conn, {"status": "error", "message": str(e)})
            finally:
                conn.close()

    t = threading.Thread(target=listen_loop, daemon=True)
    listener.thread = t
    t.start()
    with _listeners_lock:
        _listeners[(rank, pp_rank)] = listener
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


async def _broadcast_map_to_kv_tensors(tp_size: int,
                                       offsets: list[int],
                                       pp_rank: int = 0,
                                       group_id: int = 0) -> None:
    """
    Broadcast the "map_to_kv_tensors" operation to all workers concurrently.
    """
    map_message = {"cmd": "map_to_kv_tensors", "offsets": offsets,
                   "group_id": group_id}
    tasks = [
        _send_and_receive_message(rank, map_message, pp_rank) for rank in range(tp_size)
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for rank, response in enumerate(responses):
        if isinstance(response, Exception):
            raise RuntimeError(f"Worker {rank} failed to map: {response}")
        elif not isinstance(response,
                            dict) or response.get("status") != "success":
            raise RuntimeError(f"Worker {rank} failed to map: {response}")


async def _broadcast_unmap_from_kv_tensors(tp_size: int,
                                           offsets: list[int],
                                           pp_rank: int = 0,
                                           group_id: int = 0) -> None:
    """
    Broadcast the "unmap_from_kv_tensors" operation to all workers concurrently.
    """
    unmap_message = {"cmd": "unmap_from_kv_tensors", "offsets": offsets,
                     "group_id": group_id}
    tasks = [
        _send_and_receive_message(rank, unmap_message, pp_rank)
        for rank in range(tp_size)
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for rank, response in enumerate(responses):
        if isinstance(response, Exception):
            raise RuntimeError(f"Worker {rank} failed to unmap: {response}")
        elif not isinstance(response,
                            dict) or response.get("status") != "success":
            raise RuntimeError(f"Worker {rank} failed to unmap: {response}")


async def _broadcast_kv_tensors_created(tp_size: int,
                                        pp_rank: int = 0,
                                        group_id: int = 0) -> bool:
    """
    Broadcast the "kv_tensors_created" operation to all workers concurrently.
    Returns True if all workers report that KV tensors are created, False otherwise.
    """
    check_message = {"cmd": "kv_tensors_created", "group_id": group_id}
    tasks = [
        _send_and_receive_message(rank, check_message, pp_rank)
        for rank in range(tp_size)
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    all_created = True
    for rank, response in enumerate(responses):
        if isinstance(response, Exception):
            raise RuntimeError(
                f"Worker {rank} failed to check KV tensors created: {response}"
            )
        elif not isinstance(response,
                            dict) or response.get("status") != "success":
            raise RuntimeError(
                f"Worker {rank} failed to check KV tensors created: {response}"
            )
        elif not response.get("created", False):
            all_created = False

    return all_created


# Wrapper functions to call the async function from sync code
def broadcast_map_to_kv_tensors(tp_size: int, offsets: list[int],
                                pp_rank: int = 0,
                                group_id: int = 0) -> None:
    asyncio.run(_broadcast_map_to_kv_tensors(tp_size, offsets, pp_rank,
                                             group_id))


def broadcast_unmap_from_kv_tensors(tp_size: int, offsets: list[int],
                                    pp_rank: int = 0,
                                    group_id: int = 0) -> None:
    asyncio.run(_broadcast_unmap_from_kv_tensors(tp_size, offsets, pp_rank,
                                                 group_id))


def broadcast_kv_tensors_created(tp_size: int, pp_rank: int = 0,
                                 group_id: int = 0) -> bool:
    return asyncio.run(_broadcast_kv_tensors_created(tp_size, pp_rank,
                                                     group_id))
