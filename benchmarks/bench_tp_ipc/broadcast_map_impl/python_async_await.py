# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import pickle

from kvcached.tp_ipc_util import Message, get_worker_socket_path


async def send_map_cmd_to_worker_async(rank: int, offsets: list[int]):
    socket_path = get_worker_socket_path(rank)

    reader, writer = await asyncio.open_unix_connection(socket_path)

    try:
        # Serialize and send the command (with 4-byte length prefix)
        msg: Message = {"cmd": "map_to_kv_tensors", "offsets": offsets}
        data = pickle.dumps(msg)
        writer.write(len(data).to_bytes(4, "big") + data)
        await writer.drain()

        # Read 4-byte length prefix
        length_bytes = await reader.readexactly(4)
        length = int.from_bytes(length_bytes, "big")

        # Read full response
        data = await reader.readexactly(length)
        response: Message = pickle.loads(data)

        if response.get("status") != "success":
            raise RuntimeError(f"Worker {rank} failed to map: {response}")
    finally:
        writer.close()
        await writer.wait_closed()


async def broadcast_map_to_kv_tensors(tp_size: int, offsets: list[int]) -> None:
    """
    Async version of broadcast_map_to_kv_tensors.
    Sends 'map_to_kv_tensors' to all TP workers concurrently via asyncio.
    """

    async def send_to_rank(rank):
        await send_map_cmd_to_worker_async(rank, offsets)

    # Launch async tasks for all ranks
    tasks = [send_to_rank(rank) for rank in range(tp_size)]
    await asyncio.gather(*tasks)
