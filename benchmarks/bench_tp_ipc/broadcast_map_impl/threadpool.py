import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

from kvcached.tp_ipc_util import get_worker_socket_path, recv_msg, send_msg

Message = Dict[str, Any]


def send_map_cmd_to_worker(rank, offsets):
    socket_path = get_worker_socket_path(rank)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(socket_path)
        send_msg(sock, {"cmd": "map_to_kv_tensors", "offsets": offsets})
        response: Message = recv_msg(sock)
        if response.get("status") != "success":
            raise RuntimeError(f"Worker {rank} failed to map: {response}")
    finally:
        sock.close()


def broadcast_map_to_kv_tensors_to_workers(tp_size: int, offsets: list[int]) -> None:
    """Fire-and-forget broadcast: return only when **all** workers succeed."""

    def _send(rank: int):
        send_map_cmd_to_worker(rank, offsets)

    with ThreadPoolExecutor(max_workers=tp_size) as executor:
        futures = [executor.submit(_send, rank) for rank in range(tp_size)]
        # Wait for completion and raise any worker-side exception
        for future in as_completed(futures):
            future.result()
