# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import pickle

from kvcached.tp_ipc_util import IPC_TIMEOUT_S, Message, get_worker_socket_path


async def send_and_receive_message(rank: int, message: Message, pp_rank: int = 0) -> Message:
    async def exchange():
        reader, writer = await asyncio.open_unix_connection(get_worker_socket_path(rank, pp_rank))
        try:
            data = pickle.dumps(message)
            writer.write(len(data).to_bytes(4, "big") + data)
            await writer.drain()
            length = int.from_bytes(await reader.readexactly(4), "big")
            return pickle.loads(await reader.readexactly(length))
        finally:
            writer.close()
            await writer.wait_closed()

    if IPC_TIMEOUT_S > 0:
        return await asyncio.wait_for(exchange(), timeout=IPC_TIMEOUT_S)
    return await exchange()
