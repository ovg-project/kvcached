import argparse
import asyncio
import inspect
import multiprocessing as mp
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch

from kvcached.tp_ipc_util import broadcast_kv_tensors_created_to_workers
from kvcached.utils import PAGE_SIZE


def get_broadcast_impl(name: str):
    """
    Return the 'broadcast_map_to_kv_tensors_to_workers' callable for the
    implementation requested by `name`.

    Valid names: 'seq', 'thread', 'async'

    We use `importlib.util.spec_from_file_location` to load the
    module by absolute **file path** instead of a dotted import.
    This keeps the benchmark fully self-contained and avoids touching the
    global package namespace.  When the repo later promotes
    `benchmarks/` to a real package, we can replace this loader with a
    simple `import importlib; importlib.import_module(...)`.
    """
    name_map = {
        "seq": "sequential_sync",
        "thread": "threadpool",
        "async": "python_async_await",
    }
    if len(set(name_map.values())) != len(name_map.values()):
        raise RuntimeError("Duplicate module filenames detected in name_map")

    try:
        mod_name = name_map[name]
    except KeyError:
        raise ValueError(
            f"Unknown broadcast impl '{name}'. Choose from: " + ", ".join(name_map)
        ) from None

    here = Path(__file__).resolve().parent
    mod_path = here / "broadcast_map_impl" / f"{mod_name}.py"

    spec = spec_from_file_location(mod_name, mod_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    fn = module.broadcast_map_to_kv_tensors_to_workers

    if inspect.iscoroutinefunction(fn):

        def wrapper(*args, **kwargs):
            """Sync wrapper so caller doesn’t need to know this is async."""
            return asyncio.run(fn(*args, **kwargs))

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = f"[wrapped async] {fn.__doc__ or ''}"
        return wrapper

    return fn


# ---------------- Worker code ---------------- #
def _worker_entry(
    rank: int,
    tp_size: int,
    num_layers: int,
    kvcache_shape: Tuple[int, ...],
    block_size: int,
    async_sched: bool,
) -> None:
    """
    One worker process = one GPU = one TP rank.
    """
    try:
        torch.cuda.set_device(rank)
        from kvcached.integration.vllm.interfaces import init_kvcached

        init_kvcached(
            tp_rank=rank, tp_size=tp_size, is_worker=True, async_sched=async_sched
        )

        # Let the C++ layer allocates some virtual memory.
        from kvcached.integration.vllm.interfaces import alloc_kv_cache

        alloc_kv_cache(
            kvcache_shape=kvcache_shape,
            block_size=block_size,
            dtype=torch.float16,
            device=f"cuda:{rank}",
            num_layers=num_layers,
        )

        print(f"[worker {rank}] initialised - waiting for commands.", flush=True)

        # Keep the process alive; the listener thread does the real work.
        while True:
            time.sleep(3600)

    except KeyboardInterrupt:
        pass
    finally:
        print(f"[worker {rank}] shutting down.", flush=True)
        from kvcached.integration.vllm.interfaces import shutdown_kvcached

        shutdown_kvcached()


# ---------------- Main benchmark code ---------------- #
def run_benchmark(
    tp_size: int,
    iters: int,
    pages_per_iter: int,
    block_size: int,
    num_layers: int,
    async_sched: bool,
    verbose: bool,
    broadcast_fn: Callable,
    impl_key: str,
) -> None:
    mp.set_start_method("spawn", force=True)

    # Spawn workers
    procs: List[mp.Process] = []
    kvcache_shape = (
        2,  # stacked K/V
        30000,  # num_blocks will be overwritten by alloc_kv_cache()
        16,
        2,
        128,
    )  # small toy tensor – change if you like

    for rank in range(tp_size):
        p = mp.Process(
            target=_worker_entry,
            args=(rank, tp_size, num_layers, kvcache_shape, block_size, async_sched),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # Wait until every worker reports its KV tensors exist
    if verbose:
        print("Waiting for workers to become ready...", flush=True)
    while True:
        try:
            if broadcast_kv_tensors_created_to_workers(tp_size):
                break
        except ConnectionRefusedError:
            pass  # listener not up yet
        time.sleep(0.1)
    if verbose:
        print("All workers ready – starting benchmark.\n", flush=True)

    # -------- Benchmark loop --------
    map_times: List[float] = []
    unmap_times: List[float] = []
    from kvcached.tp_ipc_util import broadcast_unmap_from_kv_tensors_to_workers

    for it in range(iters):
        # Synthetic offsets: contiguous pages → byte-offset list
        # 50 is just arbitrary --- but it cannot exceed the number of virtual pages
        page_ids = [50 + i for i in range(pages_per_iter)]
        offsets = [pid * PAGE_SIZE for pid in page_ids]

        # ---------- MAP ----------
        t0 = time.time()
        broadcast_fn(tp_size, offsets)
        t1 = time.time()
        # ---------- UNMAP ----------
        t2 = time.time()
        broadcast_unmap_from_kv_tensors_to_workers(tp_size, offsets)
        t3 = time.time()

        map_t = t1 - t0
        map_times.append(map_t)
        unmap_t = t3 - t2
        unmap_times.append(unmap_t)

        if verbose:
            print(f"[iter {it:02d}] map={map_t:.6f}s  unmap={unmap_t:.6f}s", flush=True)

    # -------- Summary --------
    def _stats(ts: list[float]) -> tuple[float, float, float, float]:
        ts_arr = np.asarray(ts)
        mean = ts_arr.mean()
        mx = ts_arr.max()
        mn = ts_arr.min()
        p95 = np.percentile(ts_arr, 95)  # 95-th percentile
        return mean, mx, mn, p95

    mean_map, max_map, min_map, p95_map = _stats(map_times)
    mean_unm, max_unm, min_unm, p95_unm = _stats(unmap_times)
    # per-page ms (mean over iterations)
    per_page_map = mean_map / pages_per_iter
    per_page_unm = mean_unm / pages_per_iter

    print("\n=== IPC Benchmark Summary ===")
    print(
        f"Broadcast impl       : {impl_key} "
        f"({broadcast_fn.__module__.split('.')[-1]}.py)"
    )
    print(f"TP size                 : {tp_size}")
    print(f"Iterations              : {iters}")
    print(f"Pages Per Iteration     : {pages_per_iter}")
    print()

    header = "| Metric        |  Map (ms) | Unmap (ms) |"
    bound = "+---------------+-----------+------------+"
    sep = "|---------------|-----------|------------|"
    row = "| {:<13} | {:>9.2f} | {:>10.2f} |"

    print(bound)
    print(header)
    print(sep)
    print(row.format("mean", mean_map * 1e3, mean_unm * 1e3))
    print(row.format("p95", p95_map * 1e3, p95_unm * 1e3))
    print(row.format("max", max_map * 1e3, max_unm * 1e3))
    print(row.format("min", min_map * 1e3, min_unm * 1e3))
    print(sep)
    print(row.format("per-page mean", per_page_map * 1e3, per_page_unm * 1e3))
    print(bound)

    # Clean up worker processes
    for p in procs:
        p.terminate()
        p.join()


# ---------------- CLI ---------------- #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark kvcached TP IPC overhead.")
    parser.add_argument(
        "--tp-size",
        type=int,
        required=True,
        help="Number of tensor-parallel ranks / GPUs.",
    )
    parser.add_argument("--iters", type=int, default=20, help="Benchmark iterations.")
    parser.add_argument(
        "--pages-per-iter",
        type=int,
        default=1,
        help="How many blocks to map per iteration.",
    )
    parser.add_argument(
        "--block-size", type=int, default=16, help="Block size (num of tokens)."
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=32,
        help="Dummy layer count for alloc_kv_cache().",
    )
    parser.add_argument(
        "--async-sched",
        action="store_true",
        help="Enable kvcached async scheduler mode.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-iteration details and worker GPU-timing logs.",
    )
    parser.add_argument(
        "--map-impl",
        choices=["seq", "thread", "async"],
        default="seq",
        help="Which broadcast implementation to benchmark (default: seq).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    impl_key = args.map_impl
    broadcast_fn = get_broadcast_impl(impl_key)

    try:
        run_benchmark(
            tp_size=args.tp_size,
            iters=args.iters,
            pages_per_iter=args.pages_per_iter,
            block_size=args.block_size,
            num_layers=args.num_layers,
            async_sched=args.async_sched,
            verbose=args.verbose,
            broadcast_fn=broadcast_fn,
            impl_key=impl_key,
        )
    except KeyboardInterrupt:
        print("\nInterrupted – shutting down.", flush=True)
        sys.exit(0)
