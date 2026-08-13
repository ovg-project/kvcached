#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Measure pinned-memory transfer cost for one logical kvcached page."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-size-mb", type=int, default=2)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--kv-buffers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def summarize_ms(samples: List[float]) -> Dict[str, float]:
    if not samples:
        raise ValueError("timing samples must not be empty")
    ordered = sorted(samples)
    p95_index = math.ceil(0.95 * len(ordered)) - 1
    p99_index = math.ceil(0.99 * len(ordered)) - 1
    return {
        "max": max(ordered),
        "mean": statistics.fmean(ordered),
        "min": min(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[p95_index],
        "p99": ordered[p99_index],
    }


def benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    for name in ("page_size_mb", "layers", "kv_buffers", "iterations"):
        validate_positive(name, getattr(args, name))
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CPU offload benchmark")

    device = torch.device(args.device)
    payload_count = args.layers * args.kv_buffers
    page_bytes = args.page_size_mb * 1024 * 1024
    logical_page_bytes = payload_count * page_bytes

    gpu_pages: List[torch.Tensor] = [
        torch.empty(page_bytes, dtype=torch.uint8, device=device)
        for _ in range(payload_count)
    ]
    cpu_pages: List[torch.Tensor] = [
        torch.empty(page_bytes, dtype=torch.uint8, pin_memory=True)
        for _ in range(payload_count)
    ]
    for index, gpu_page in enumerate(gpu_pages):
        gpu_page.fill_(index % 251)

    stream = torch.cuda.Stream(device=device)

    def copy_d2h() -> None:
        with torch.cuda.stream(stream):
            for cpu_page, gpu_page in zip(cpu_pages, gpu_pages):
                cpu_page.copy_(gpu_page, non_blocking=True)

    def copy_h2d() -> None:
        with torch.cuda.stream(stream):
            for gpu_page, cpu_page in zip(gpu_pages, cpu_pages):
                gpu_page.copy_(cpu_page, non_blocking=True)

    for _ in range(args.warmup):
        copy_d2h()
        copy_h2d()
    stream.synchronize()

    def timed_ms(operation) -> List[float]:
        events = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            operation()
            end.record(stream)
            events.append((start, end))
        events[-1][1].synchronize()
        return [start.elapsed_time(end) for start, end in events]

    d2h_samples = timed_ms(copy_d2h)
    d2h_ms = summarize_ms(d2h_samples)
    expected = [index % 251 for index in range(payload_count)]
    observed_cpu = [int(cpu_page[0]) for cpu_page in cpu_pages]
    if observed_cpu != expected:
        raise RuntimeError("D2H correctness check failed")

    for gpu_page in gpu_pages:
        gpu_page.zero_()
    h2d_samples = timed_ms(copy_h2d)
    h2d_ms = summarize_ms(h2d_samples)
    observed_gpu = [int(gpu_page[0]) for gpu_page in gpu_pages]
    if observed_gpu != expected:
        raise RuntimeError("H2D correctness check failed")

    def effective_gbps(duration_ms: float) -> float:
        return logical_page_bytes / (duration_ms / 1000) / 1_000_000_000

    return {
        "cuda_device": torch.cuda.get_device_name(device),
        "d2h_effective_gbps": effective_gbps(d2h_ms["mean"]),
        "d2h_ms": d2h_ms,
        "device": str(device),
        "h2d_effective_gbps": effective_gbps(h2d_ms["mean"]),
        "h2d_ms": h2d_ms,
        "iterations": args.iterations,
        "kv_buffers": args.kv_buffers,
        "layers": args.layers,
        "logical_page_bytes": logical_page_bytes,
        "page_size_bytes": page_bytes,
        "payload_count": payload_count,
        "torch_version": torch.__version__,
    }


def main() -> int:
    args = parse_args()
    result = benchmark(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
