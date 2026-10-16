# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Measure native growth, release, reserve reuse, rejection and calibration.

Run identical optimized candidate builds on an otherwise idle GPU. This measures
allocator operations, not model serving. Alternate candidate order externally.
"""

import argparse
import json
import multiprocessing as mp
import os
import resource
import tempfile
import threading
import time
import traceback
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def worker(index, config, directory, barrier, output):
    try:
        run_worker(index, config, directory, barrier, output)
    except BaseException:
        output.put({"worker": index, "error": traceback.format_exc()})
        raise


def run_worker(index, config, directory, barrier, output):
    identity = f"budget-ab-{os.getpid()}-{index}"
    os.environ["KVCACHED_IPC_NAME"] = identity
    os.environ["KVCACHED_PHYSICAL_GROWTH_LOCK_DIR"] = directory
    import torch

    import kvcached._C as native
    from kvcached import vmm_ops as vmm

    page = 2 * 1024 * 1024
    vmm.init_kvcached("cuda:0", page, False)
    tensors = vmm.create_kv_tensors(12 * page, 1, "cuda:0", 2, 2, page_size=page)
    cases = [("growth", 1), ("growth", 5), ("calibration", 5),
             ("rejection", 1), ("reuse", 1)]
    try:
        for case_index, (mode, count) in enumerate(cases):
            offsets = [i * page for i in range(count)]
            samples = {key: [] for key in ["admission_us", "reserve_us", "map_us",
                                          "release_us", "cycle_us"]}
            attempts = 0
            misses = 0
            calibration = {"attempts": 0, "applied": 0, "errors": []}
            stop = threading.Event()
            calibrator = None
            allocator = None
            if mode == "rejection":
                assert vmm.set_instance_budget_limit(0, 1, 0) == "applied"
            if mode == "reuse":
                allocator = native.PageAllocator(
                    2, 6 * page, page, contiguous_layout=False,
                    enable_page_prealloc=True, num_kv_buffers=2, ipc_name=identity)
                allocator.set_use_worker_ipc(False)
                first = allocator.alloc_page()
                allocator.free_page(first.page_id)
                assert allocator.get_num_reserved_pages() == 1
            # Reuse exercises the actual PageAllocator reserve list. The
            # background worker is intentionally stopped during this hot loop.
            iterations = max(1000, config["iterations"] * 10) if mode == "reuse" else config["iterations"]
            warmup = 20

            def calibrate():
                while not stop.wait(.01):
                    try:
                        calibration["attempts"] += 1
                        calibration["applied"] += int(vmm.recalibrate_physical_budget(0))
                    except Exception as error:
                        calibration["errors"].append(str(error))

            barrier.wait(timeout=120)
            if mode == "calibration":
                calibrator = threading.Thread(target=calibrate)
                calibrator.start()
            before = None
            cpu_start = None
            wall_start = None
            try:
                for iteration in range(iterations + warmup):
                    if iteration == warmup:
                        barrier.wait(timeout=120)
                        before = vmm.physical_budget_state(0)
                        cpu_start = resource.getrusage(resource.RUSAGE_SELF)
                        wall_start = time.perf_counter_ns()
                        attempts = misses = 0
                    started = time.perf_counter_ns()
                    admission = reserve = mapped = released = 0
                    if mode == "reuse":
                        allocated = allocator.alloc_page()
                        allocator.free_page(allocated.page_id)
                    else:
                        transaction = f"{case_index}-{iteration}"
                        while True:
                            attempts += 1
                            prepared = vmm.prepare_map_to_kv_tensors(transaction, offsets, 0)
                            if prepared["success"]:
                                assert mode != "rejection"
                                break
                            misses += 1
                            assert prepared["capacity_rejections"] == 1, prepared
                            if mode == "rejection":
                                break
                            assert prepared["shortfall_bytes"] == 0, prepared
                            assert time.perf_counter_ns() - started < 10_000_000_000, prepared
                            time.sleep(.001)
                        admission = prepared["admission_us"]
                        reserve = prepared["reserve_us"]
                        if mode != "rejection":
                            assert prepared["required_bytes"] == count * 4 * page
                            committed = vmm.commit_prepared_map(transaction, 0)
                            assert committed["success"]
                            mapped = committed["map_us"]
                            release_start = time.perf_counter_ns()
                            assert vmm.unmap_from_kv_tensors(offsets, 0)
                            released = (time.perf_counter_ns() - release_start) / 1000
                    elapsed = (time.perf_counter_ns() - started) / 1000
                    if iteration >= warmup:
                        for key, value in zip(samples, [admission, reserve, mapped, released, elapsed]):
                            samples[key].append(value)
                wall_end = time.perf_counter_ns()
                wall_seconds = (wall_end - wall_start) / 1e9
                cpu_end = resource.getrusage(resource.RUSAGE_SELF)
            finally:
                stop.set()
                if calibrator:
                    calibrator.join(timeout=20)
                    assert not calibrator.is_alive()
            after = vmm.physical_budget_state(0)
            sync = {key: after[key] - before[key] for key in before if key.startswith("sync_")}
            assert not calibration["errors"], calibration
            if mode == "reuse":
                assert after["instance_charged_bytes"] == before["instance_charged_bytes"]
                assert sync["sync_acquisitions"] == 1, sync  # final snapshot only
                allocator.trim()
                allocator.release_shared_segment()
                del allocator
            else:
                assert after["instance_charged_bytes"] == 0, after
            if mode == "rejection":
                assert misses == iterations and attempts == iterations
                assert vmm.set_instance_budget_limit(2**64 - 1, 2, 0) == "applied"
            barrier.wait(timeout=120)
            settled = vmm.physical_budget_state(0)
            assert settled["charged_bytes"] == 0, settled
            output.put({
                "worker": index, "workers": config["workers"], "mode": mode,
                "logical_pages": count, "batch_bytes": count * 4 * page,
                "iterations": iterations, "seconds": wall_seconds,
                "measurement_started_ns": wall_start, "measurement_finished_ns": wall_end,
                "operations_per_second": iterations / wall_seconds,
                "cpu_user_seconds": cpu_end.ru_utime - cpu_start.ru_utime,
                "cpu_system_seconds": cpu_end.ru_stime - cpu_start.ru_stime,
                "admission_attempts": attempts, "admission_misses": misses,
                "calibration": calibration, "sync": sync,
                "latency_us": {key: {"p50": percentile(values, .5),
                                     "p95": percentile(values, .95),
                                     "p99": percentile(values, .99),
                                     "max": max(values)}
                               for key, values in samples.items()},
                "samples_us": samples,
            })
            barrier.wait(timeout=120)
    finally:
        del tensors
        vmm.shutdown_kvcached()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--label", required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.iterations > 0
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="kvcached-budget-ab-") as directory:
        barrier = context.Barrier(args.workers)
        output = context.Queue()
        config = {"workers": args.workers, "iterations": args.iterations}
        processes = [context.Process(target=worker, args=(i, config, directory, barrier, output))
                     for i in range(args.workers)]
        for process in processes:
            process.start()
        records = []
        # Drain while workers run: raw samples may exceed the pipe buffer.
        try:
            for _ in range(args.workers * 5):
                record = output.get(timeout=240)
                if "error" in record:
                    raise RuntimeError(record["error"])
                records.append(record)
            for process in processes:
                process.join(30)
                assert process.exitcode == 0, process.exitcode
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(10)
    result = {"label": args.label, "round": args.round, "workers": args.workers,
              "iterations": args.iterations, "records": records}
    args.output.write_text(json.dumps(result, indent=2))
    for mode, pages in [("growth", 1), ("growth", 5), ("calibration", 5),
                        ("rejection", 1), ("reuse", 1)]:
        rows = [row for row in records if row["mode"] == mode and row["logical_pages"] == pages]
        rates = [row["operations_per_second"] for row in rows]
        print(json.dumps({"label": args.label, "round": args.round, "workers": args.workers,
                          "mode": mode, "pages": pages,
                          "aggregate_ops_per_second": sum(row["iterations"] for row in rows) * 1e9 / (max(row["measurement_finished_ns"] for row in rows) - min(row["measurement_started_ns"] for row in rows)),
                          "jain_completion_rate_index": sum(rates)**2 / (len(rates) * sum(rate**2 for rate in rates)),
                          "misses": sum(row["admission_misses"] for row in rows)}), flush=True)


if __name__ == "__main__":
    main()
