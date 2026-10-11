# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Compare unmap protocols over real worker sockets and CUDA VMM."""

import argparse
import asyncio
import json
import math
import multiprocessing as mp
import os
import statistics
import time
import uuid
from pathlib import Path

PAGE_SIZE = 2 * 1024 * 1024
MODES = ("one-phase-control", "two-phase", "automatic")


def worker(rank, device, layers, pages, pipe):
    import torch

    from kvcached import tp_ipc_util as ipc, vmm_ops

    torch.cuda.set_device(device)
    vmm_ops.init_kvcached(f"cuda:{device}", PAGE_SIZE, False)
    tensors = vmm_ops.create_kv_tensors(2 * pages * PAGE_SIZE, 2, f"cuda:{device}", layers)
    ipc.start_worker_listener_thread(rank, device_index=device)
    pipe.send({"ready": True, "gpu": torch.cuda.get_device_name(device)})
    try:
        while pipe.recv() == "check":
            for tensor in tensors:
                flat = tensor.view(-1)
                for half in (0, flat.numel() // 2):
                    for page in range(pages):
                        start = half + page * PAGE_SIZE // flat.element_size()
                        flat[start:start + 128].fill_(7)
                        assert bool((flat[start:start + 128] == 7).all())
            torch.cuda.synchronize(device)
            pipe.send("checked")
    finally:
        torch.cuda.synchronize(device)
        vmm_ops.shutdown_kvcached()
        pipe.close()


async def unmap(mode, count, offsets):
    from kvcached import tp_ipc_util as ipc

    if mode == "automatic":
        await ipc._broadcast_unmap_from_kv_tensors(count, offsets)
        return
    transaction_id = uuid.uuid4().hex
    phases = [("unmap_from_kv_tensors", "success")] if mode == "one-phase-control" else [
        ("prepare_unmap_from_kv_tensors", "prepared"),
        ("commit_unmap_from_kv_tensors", "committed"),
    ]
    for command, expected in phases:
        replies = await asyncio.gather(*[
            ipc._send_and_receive_message(rank, {
                "cmd": command, "offsets": offsets, "group_id": 0,
                "transaction_id": transaction_id,
            }) for rank in range(count)
        ])
        assert all(reply.get("status") == expected for reply in replies), replies


def summary(samples):
    ordered = sorted(samples)
    return {"mean_ms": statistics.mean(samples), "p50_ms": statistics.median(samples),
            "p95_ms": ordered[math.ceil(len(ordered) * .95) - 1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0", help="One CUDA index per worker, e.g. 0,1")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    devices = [int(d) for d in args.devices.split(",")]
    if min(args.layers, args.pages, args.iterations, args.rounds) <= 0 or args.warmup < 0:
        parser.error("sizes, iterations and rounds must be positive; warmup must be nonnegative")
    os.environ["KVCACHED_IPC_NAME"] = f"ub{os.getpid()}"
    os.environ["ENABLE_KVCACHED"] = "false"
    os.environ["KVCACHED_AUTOPATCH"] = "0"
    from kvcached import tp_ipc_util as ipc

    context = mp.get_context("spawn")
    workers, pipes, hardware = [], [], []
    offsets = [i * PAGE_SIZE for i in range(args.pages)]
    results = []
    try:
        for rank, device in enumerate(devices):
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(rank, device, args.layers, args.pages, child))
            process.start()
            child.close()
            workers.append(process)
            pipes.append(parent)
            if not parent.poll(90):
                raise TimeoutError(f"worker {rank} did not initialize")
            hardware.append(parent.recv())
        for round_index in range(args.rounds):
            # Rotate the order to reduce systematic cold/order bias.
            shift = round_index % len(MODES)
            for mode in MODES[shift:] + MODES[:shift]:
                maps, unmaps = [], []
                for iteration in range(args.warmup + args.iterations):
                    start = time.perf_counter_ns()
                    ipc.broadcast_map_to_kv_tensors(len(devices), offsets)
                    mapped = time.perf_counter_ns()
                    if iteration in (0, args.warmup + args.iterations - 1):
                        for pipe in pipes:
                            pipe.send("check")
                        for pipe in pipes:
                            if not pipe.poll(30) or pipe.recv() != "checked":
                                raise RuntimeError("worker data check failed")
                    before_unmap = time.perf_counter_ns()
                    asyncio.run(unmap(mode, len(devices), offsets))
                    end = time.perf_counter_ns()
                    if iteration >= args.warmup:
                        maps.append((mapped - start) / 1e6)
                        unmaps.append((end - before_unmap) / 1e6)
                result = {"round": round_index + 1, "mode": mode,
                          "map": summary(maps), "unmap": summary(unmaps)}
                results.append(result)
                print(json.dumps(result), flush=True)
    finally:
        for pipe, process in zip(pipes, workers):
            if process.is_alive():
                try:
                    pipe.send("stop")
                except (BrokenPipeError, EOFError):
                    pass
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(10)
            pipe.close()
        for rank in range(len(workers)):
            Path(ipc.get_worker_socket_path(rank)).unlink(missing_ok=True)
        if Path(ipc.SOCKET_DIR).exists():
            Path(ipc.SOCKET_DIR).rmdir()
    assert all(p.exitcode == 0 for p in workers), [p.exitcode for p in workers]
    print(json.dumps({"configuration": vars(args), "hardware": hardware, "results": results,
                      "worker_exitcodes": [p.exitcode for p in workers]}), flush=True)


if __name__ == "__main__":
    main()
