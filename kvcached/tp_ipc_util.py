# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import pickle
import socket
import threading
import uuid
from typing import Any, Dict, Optional, cast

from kvcached.utils import DEFAULT_IPC_NAME, normalize_gpu_device
from kvcached.vmm_ops import kv_tensors_created, map_to_kv_tensors, unmap_from_kv_tensors


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
    """Start a thread that listens for messages on the worker socket.

    ``pp_rank`` selects a PP-stage-specific socket directory so concurrent
    stages do not bind the same path. When ``device_index`` is provided, the
    listener restores that CUDA device inside the new thread before executing
    CUDA-backed map or unmap operations because CUDA's current device is
    thread-local.
    """
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

    def listen_loop():
        if device_index is not None:
            import torch

            # CUDA's current device is thread-local, so restore the worker device.
            torch.cuda.set_device(device_index)
        print(f"Worker {rank} IPC listener started at {socket_path}")
        while True:
            conn, _ = server_sock.accept()
            try:
                msg: Message = recv_msg(conn)
                # print(f"Worker {rank} received message: {msg}")
                group_id: int = msg.get("group_id", 0)
                if msg["cmd"] == "map_to_kv_tensors":
                    if not map_to_kv_tensors(msg["offsets"], group_id=group_id):
                        raise RuntimeError(
                            f"Failed to map KV tensors for group_id={group_id}"
                        )
                    send_msg(conn, {"status": "success"})
                elif msg["cmd"] == "unmap_from_kv_tensors":
                    if not unmap_from_kv_tensors(msg["offsets"], group_id=group_id):
                        raise RuntimeError(
                            f"Failed to unmap KV tensors for group_id={group_id}"
                        )
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
    t.start()


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
    for (target_pp_rank, rank), response in zip(targets, responses):
        if isinstance(response, Exception):
            raise RuntimeError(
                f"Worker pp{target_pp_rank}/rank{rank} failed to map: {response}"
            )
        elif not isinstance(response,
                            dict) or response.get("status") != "success":
            raise RuntimeError(
                f"Worker pp{target_pp_rank}/rank{rank} failed to map: {response}"
            )


async def _broadcast_unmap_from_kv_tensors(tp_size: int,
                                           offsets: list[int],
                                           pp_rank: int = 0,
                                           group_id: int = 0) -> None:
    """
    Broadcast the "unmap_from_kv_tensors" operation to all workers concurrently.
    """
    unmap_message = {"cmd": "unmap_from_kv_tensors", "offsets": offsets,
                     "group_id": group_id}
    targets = [
        (target_pp_rank, rank)
        for target_pp_rank in _target_pp_ranks(pp_rank)
        for rank in range(tp_size)
    ]
    tasks = [
        _send_and_receive_message(rank, unmap_message, target_pp_rank)
        for target_pp_rank, rank in targets
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for (target_pp_rank, rank), response in zip(targets, responses):
        if isinstance(response, Exception):
            raise RuntimeError(
                f"Worker pp{target_pp_rank}/rank{rank} failed to unmap: {response}"
            )
        elif not isinstance(response,
                            dict) or response.get("status") != "success":
            raise RuntimeError(
                f"Worker pp{target_pp_rank}/rank{rank} failed to unmap: {response}"
            )


async def _broadcast_kv_tensors_created(tp_size: int,
                                        pp_rank: int = 0,
                                        group_id: int = 0) -> bool:
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
        elif not isinstance(response,
                            dict) or response.get("status") != "success":
            raise RuntimeError(
                f"Worker pp{target_pp_rank}/rank{rank} failed to check "
                f"KV tensors created: {response}"
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
