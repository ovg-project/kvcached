import socket
from typing import Any, Dict

from kvcached.tp_ipc_util import get_worker_socket_path, recv_msg, send_msg

Message = Dict[str, Any]


def broadcast_map_to_kv_tensors(tp_size: int, offsets: list[int]) -> None:
    for rank in range(tp_size):
        socket_path = get_worker_socket_path(rank)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        try:
            send_msg(sock, {"cmd": "map_to_kv_tensors", "offsets": offsets})
            response: Message = recv_msg(sock)
            if response.get("status") != "success":
                raise RuntimeError(f"Worker {rank} failed to map: {response}")
        finally:
            sock.close()
