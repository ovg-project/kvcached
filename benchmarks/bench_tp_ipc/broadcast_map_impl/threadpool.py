# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import socket

from kvcached.tp_ipc_util import IPC_TIMEOUT_S, Message, get_worker_socket_path, recv_msg, send_msg


def exchange(rank: int, message: Message, pp_rank: int = 0) -> Message:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(IPC_TIMEOUT_S if IPC_TIMEOUT_S > 0 else None)
        sock.connect(get_worker_socket_path(rank, pp_rank))
        send_msg(sock, message)
        return recv_msg(sock)


async def send_and_receive_message(rank: int, message: Message, pp_rank: int = 0) -> Message:
    return await asyncio.to_thread(exchange, rank, message, pp_rank)
