# SPDX-FileCopyrightText: 2025 contributors to the kvcached project
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import torch

# kvcached bits
from kvcached.integration.vllm.interfaces import shutdown_kvcached
from kvcached.vmm_ops import unmap_from_kv_tensors

PAGE_SIZE = 2 * 1024 * 1024  # 2MB, typical and for benchmarking purposes

# ----------------------------- Worker ------------------------------------- #


def _worker(
    rank: int,
    procs: int,
    device_mode: str,
    pages_per_proc: int,
    iters: int,
    base_page: int,
    block_size: int,
    num_layers: int,
    start_evt: mp.Event,
    ready_q: mp.Queue,
    result_q: mp.Queue,
    iter_barrier: mp.Barrier,
    verbose: bool,
    contiguous_layout: bool,
):
    """
    One process (optionally one GPU) that:
      - initializes kvcached once
      - allocates virtual KV tensors
      - waits for start_evt, then iteratively maps (and unmaps) its pages
      - reports per-iteration map time via result_q
    """
    os.environ["KVCACHED_CONTIGUOUS_LAYOUT"] = "true" if contiguous_layout else "false"
    try:
        # Device placement
        dev_id = rank if device_mode == "per-rank" else 0
        torch.cuda.set_device(dev_id)
        device = f"cuda:{dev_id}"

        # Init kvcached
        from kvcached.integration.vllm.interfaces import init_kvcached

        init_kvcached(tp_rank=rank, tp_size=procs, is_worker=False, async_sched=False)

        # This shape is not critical for timing the memory mapping ops
        kvcache_shape = (2, 30000, 16, 2, 128)
        from kvcached.integration.vllm.interfaces import alloc_kv_cache

        alloc_kv_cache(
            kvcache_shape=kvcache_shape,
            block_size=block_size,
            dtype=torch.float16,
            device=device,
            num_layers=num_layers,
        )

        from kvcached.vmm_ops import kv_tensors_created

        while True:
            created = kv_tensors_created()
            if created:
                break
            time.sleep(1)

        if verbose:
            print(
                f"[worker {rank}] ready on {device}; pages_per_proc={pages_per_proc}",
                flush=True,
            )

        # Tell parent we're ready; then wait for global start
        ready_q.put(rank)
        start_evt.wait()

        from kvcached.vmm_ops import map_to_kv_tensors

        for it in range(iters):
            try:
                iter_barrier.wait()  # sync all ranks before starting the iteration
            except mp.BrokenBarrierError:
                print(
                    f"[worker {rank}] BrokenBarrierError on iteration {it}; exiting.",
                    flush=True,
                )
                return

            if verbose:
                print(
                    f"[worker {rank}] starting iteration {it} on {device}", flush=True
                )

            # Assign a disjoint page range to each rank
            start_page = base_page + rank * pages_per_proc
            page_ids = [start_page + i for i in range(pages_per_proc)]
            if contiguous_layout:
                offsets = [pid * PAGE_SIZE * num_layers * 2 for pid in page_ids]
            else:
                offsets = [pid * PAGE_SIZE for pid in page_ids]

            t0 = time.time()
            map_to_kv_tensors(offsets)
            t1 = time.time()
            # sleep for seconds for probing GPU utilization
            time.sleep(2)
            # Optional cleanup to keep memory mapping balanced across iters
            unmap_from_kv_tensors(offsets)

            result_q.put((rank, it, t1 - t0, dev_id))

            try:
                iter_barrier.wait()  # sync all ranks after the iteration
            except mp.BrokenBarrierError:
                print(
                    f"[worker {rank}] BrokenBarrierError on iteration {it}; exiting.",
                    flush=True,
                )
                return

            if verbose:
                print(
                    f"[worker {rank}] finished iteration {it} in {1000 * (t1 - t0):.2f} ms",
                    flush=True,
                )

    except KeyboardInterrupt:
        pass
    finally:
        shutdown_kvcached()


# ----------------------------- Harness ------------------------------------ #


@dataclass
class RunStats:
    per_iter_wall: List[float]
    per_iter_rank_times: List[List[float]]  # [iter][rank→time]
    mean: float
    p95: float
    max_t: float
    min_t: float


def _aggregate_parallel_results(procs: int, iters: int, result_q: mp.Queue) -> RunStats:
    # Gather all (rank, iter, dt, dev) tuples
    bucket: dict[int, dict[int, float]] = defaultdict(dict)  # iter -> {rank: dt}
    for _ in range(procs * iters):
        rank, it, dt, _dev = result_q.get()
        bucket[it][rank] = dt

    per_iter_wall = []
    per_iter_rank_times = []
    for it in range(iters):
        times = [bucket[it][r] for r in range(procs)]
        per_iter_rank_times.append(times)
        per_iter_wall.append(max(times))  # wall time dominated by the slowest rank

    arr = per_iter_wall
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    p95 = arr_sorted[int(0.95 * (n - 1))] if n > 1 else arr_sorted[0]
    return RunStats(
        per_iter_wall=per_iter_wall,
        per_iter_rank_times=per_iter_rank_times,
        mean=sum(arr) / len(arr),
        p95=p95,
        max_t=max(arr),
        min_t=min(arr),
    )


def _run_parallel_case(
    pages_total: int,
    procs: int,
    iters: int,
    device_mode: str,
    block_size: int,
    num_layers: int,
    base_page: int,
    verbose: bool,
    contiguous_layout: bool,
) -> RunStats:
    ctx = mp.get_context("spawn")
    start_evt = ctx.Event()
    ready_q: mp.Queue = ctx.Queue()
    result_q: mp.Queue = ctx.Queue()

    iter_barrier = ctx.Barrier(procs)

    # Split pages across ranks; distribute remainder to rank 0..r-1
    base_pp = pages_total // procs
    rem = pages_total % procs
    pages_for = [base_pp + (1 if r < rem else 0) for r in range(procs)]

    procs_list: List[mp.Process] = []
    for r in range(procs):
        p = ctx.Process(
            target=_worker,
            args=(
                r,
                procs,
                device_mode,
                pages_for[r],
                iters,
                base_page,
                block_size,
                num_layers,
                start_evt,
                ready_q,
                result_q,
                iter_barrier,
                verbose,
                contiguous_layout,
            ),
            daemon=True,
        )
        p.start()
        procs_list.append(p)

    # Wait until all workers report readiness
    ready = set()
    while len(ready) < procs:
        ready.add(ready_q.get())

    # Start the timed section for each iteration inside workers
    start_evt.set()

    # Collect results and compute wall-clock per iteration as max(rank times)
    stats = _aggregate_parallel_results(procs, iters, result_q)

    # Clean up workers
    for p in procs_list:
        p.join(timeout=3)  # wait for completion, but don't block indefinitely
    for p in procs_list:
        if p.is_alive():
            print(
                f"Warning: worker {p.pid} is still alive after join; terminating it.",
                flush=True,
            )
            p.terminate()
            p.join()

    return stats


def _run_serial_case(
    pages_total: int,
    iters: int,
    device_mode: str,
    block_size: int,
    num_layers: int,
    base_page: int,
    verbose: bool,
    contiguous_layout: bool,
) -> RunStats:
    # Serial case = just reuse the same worker function with procs=1
    return _run_parallel_case(
        pages_total=pages_total,
        procs=1,
        iters=iters,
        device_mode=device_mode,  # 'same' and 'per-rank' both map to cuda:0 here
        block_size=block_size,
        num_layers=num_layers,
        base_page=base_page,
        verbose=verbose,
        contiguous_layout=contiguous_layout,
    )


def _print_table(
    title: str, stats: RunStats, pages_total: int, procs: int, iters: int, unit="ms"
):
    factor = 1e3 if unit == "ms" else (1e6 if unit == "us" else 1.0)
    colw = (15, 11)
    sep_colw = (17, 13)  # column widths for the separator
    sep = "+" + "+".join("-" * w for w in sep_colw) + "+"
    fmt_head = "| {:{w1}} | {:{w2}} |".format
    fmt_row = "| {:{w1}} | {:>{w2}.3f} |".format

    print(f"\n========== {title} ==========")
    print(f"Processes           : {procs}")
    print(f"Iterations          : {iters}")
    print(f"Pages total / iter  : {pages_total}")
    print(f"Pages per proc / iter: {pages_total // procs} \n")

    print(sep)
    print(fmt_head("Metric", unit, w1=colw[0], w2=colw[1]))
    print(sep)
    print(fmt_row("mean wall", stats.mean * factor, w1=colw[0], w2=colw[1]))
    print(fmt_row("p95 wall", stats.p95 * factor, w1=colw[0], w2=colw[1]))
    print(fmt_row("max wall", stats.max_t * factor, w1=colw[0], w2=colw[1]))
    print(fmt_row("min wall", stats.min_t * factor, w1=colw[0], w2=colw[1]))
    print(sep)


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark parallel vs serial mapping using kvcached VMM ops."
    )
    ap.add_argument(
        "--pages-total", type=int, default=1, help="Total pages to map per iteration."
    )
    ap.add_argument(
        "--procs", type=int, default=4, help="Number of processes in the parallel case."
    )
    ap.add_argument("--iters", type=int, default=1, help="Iterations.")
    ap.add_argument(
        "--device-mode",
        choices=["per-rank", "same"],
        default="per-rank",
        help="'per-rank' → rank i uses cuda:i; 'same' → all use cuda:0",
    )
    ap.add_argument(
        "--block-size", type=int, default=16, help="Model block size (tokens)."
    )
    ap.add_argument(
        "--num-layers",
        type=int,
        default=32,
        help="Dummy layer count for alloc_kv_cache().",
    )
    ap.add_argument(
        "--base-page",
        type=int,
        default=0,
        help="Starting page id (kept disjoint across ranks/iters).",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--not-contiguous",
        action="store_true",
        help="Simulated kvcache tensor layout, contiguous by default.",
    )
    args = ap.parse_args()

    os.environ["KVCACHED_CONTIGUOUS_LAYOUT"] = (
        "false" if args.not_contiguous else "true"
    )
    contiguous_layout = (
        os.getenv("KVCACHED_CONTIGUOUS_LAYOUT", "true").lower() == "true"
    )
    print(f"Using layout contiguous={contiguous_layout}")

    if args.verbose:
        os.environ["KVCACHED_BENCH_VERBOSE"] = "1"

    # Case X: Single Process (1 proc, N/P pages)
    single = _run_serial_case(
        pages_total=args.pages_total // args.procs,
        iters=args.iters,
        device_mode=args.device_mode,
        block_size=args.block_size,
        num_layers=args.num_layers,
        base_page=args.base_page,
        verbose=args.verbose,
        contiguous_layout=contiguous_layout,
    )
    _print_table(
        "Single Process case (1 proc maps N/P pages)",
        single,
        args.pages_total // args.procs,
        1,
        args.iters,
    )

    # Case A: serial (1 proc, N pages)
    serial = _run_serial_case(
        pages_total=args.pages_total,
        iters=args.iters,
        device_mode=args.device_mode,
        block_size=args.block_size,
        num_layers=args.num_layers,
        base_page=args.base_page,
        verbose=args.verbose,
        contiguous_layout=contiguous_layout,
    )
    _print_table(
        "Serial case (1 proc maps N pages)", serial, args.pages_total, 1, args.iters
    )

    # Case B: parallel (P procs, N/P pages each)
    parallel = _run_parallel_case(
        pages_total=args.pages_total,
        procs=args.procs,
        iters=args.iters,
        device_mode=args.device_mode,
        block_size=args.block_size,
        num_layers=args.num_layers,
        base_page=args.base_page,
        verbose=args.verbose,
        contiguous_layout=contiguous_layout,
    )
    _print_table(
        f"Parallel case ({args.procs} procs map N/P each)",
        parallel,
        args.pages_total,
        args.procs,
        args.iters,
    )

    # Speedup vs serial (wall-clock)
    speedups = [
        s / p if p > 0 else float("inf")
        for s, p in zip(serial.per_iter_wall, parallel.per_iter_wall)
    ]
    mean_speedup = sum(speedups) / len(speedups)
    print(
        f"\nSpeedup (parallel vs serial): mean x{mean_speedup:.2f}   "
        f"(ideal being approximately {args.procs:.2f} if perfect overlap)\n"
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
