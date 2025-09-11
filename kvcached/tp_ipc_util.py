# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import pickle
import socket
import threading
from typing import Any, Dict, cast

from kvcached.vmm_ops import kv_tensors_created, map_to_kv_tensors, unmap_from_kv_tensors

SOCKET_DIR = "/tmp/kvcached-ipc"


def get_worker_socket_path(rank: int) -> str:
    """
    Get the path for the worker socket.
    The socket is used for inter-process communication.
    """
    return os.path.join(SOCKET_DIR, f"worker_{rank}.sock")


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


def start_worker_listener_thread(rank: int):
    """
    Start a thread that listens for messages on the worker socket.
    The callback is called with the received message.
    """
    os.makedirs(SOCKET_DIR, exist_ok=True)
    socket_path = get_worker_socket_path(rank)

    if os.path.exists(socket_path):
        try:
            os.remove(socket_path)
        except OSError as e:
            print(f"Error removing existing socket file {socket_path}: {e}")

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(socket_path)
    server_sock.listen()

    def listen_loop():
        print(f"Worker {rank} IPC listener started at {socket_path}")
        while True:
            conn, _ = server_sock.accept()
            try:
                msg: Message = recv_msg(conn)
                # print(f"Worker {rank} received message: {msg}")
                if msg["cmd"] == "map_to_kv_tensors":
                    map_to_kv_tensors(msg["offsets"])
                    send_msg(conn, {"status": "success"})
                elif msg["cmd"] == "unmap_from_kv_tensors":
                    unmap_from_kv_tensors(msg["offsets"])
                    send_msg(conn, {"status": "success"})
                elif msg["cmd"] == "kv_tensors_created":
                    created: bool = kv_tensors_created()
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


async def _send_and_receive_message(rank: int, message: Message) -> Message:
    """
    Send a message to the worker and receive a response asynchronously.
    """
    socket_path = get_worker_socket_path(rank)
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


async def _broadcast_map_to_kv_tensors(tp_size: int,
                                       offsets: list[int]) -> None:
    """
    Broadcast the "map_to_kv_tensors" operation to all workers concurrently.
    """
    map_message = {"cmd": "map_to_kv_tensors", "offsets": offsets}
    tasks = [
        _send_and_receive_message(rank, map_message) for rank in range(tp_size)
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for rank, response in enumerate(responses):
        if isinstance(response, Exception):
            raise RuntimeError(f"Worker {rank} failed to map: {response}")
        elif not isinstance(response,
                            dict) or response.get("status") != "success":
            raise RuntimeError(f"Worker {rank} failed to map: {response}")


async def _broadcast_unmap_from_kv_tensors(tp_size: int,
                                           offsets: list[int]) -> None:
    """
    Broadcast the "unmap_from_kv_tensors" operation to all workers concurrently.
    """
    unmap_message = {"cmd": "unmap_from_kv_tensors", "offsets": offsets}
    tasks = [
        _send_and_receive_message(rank, unmap_message)
        for rank in range(tp_size)
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for rank, response in enumerate(responses):
        if isinstance(response, Exception):
            raise RuntimeError(f"Worker {rank} failed to unmap: {response}")
        elif not isinstance(response,
                            dict) or response.get("status") != "success":
            raise RuntimeError(f"Worker {rank} failed to unmap: {response}")


async def _broadcast_kv_tensors_created(tp_size: int) -> bool:
    """
    Broadcast the "kv_tensors_created" operation to all workers concurrently.
    Returns True if all workers report that KV tensors are created, False otherwise.
    """
    check_message = {"cmd": "kv_tensors_created"}
    tasks = [
        _send_and_receive_message(rank, check_message)
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
def broadcast_map_to_kv_tensors(tp_size: int, offsets: list[int]) -> None:
    asyncio.run(_broadcast_map_to_kv_tensors(tp_size, offsets))


def broadcast_unmap_from_kv_tensors(tp_size: int, offsets: list[int]) -> None:
    asyncio.run(_broadcast_unmap_from_kv_tensors(tp_size, offsets))


def broadcast_kv_tensors_created(tp_size: int) -> bool:
    return asyncio.run(_broadcast_kv_tensors_created(tp_size))
