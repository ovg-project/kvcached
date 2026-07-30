#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Exercise a real kvcached VMM page through GPU -> CPU -> GPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--page-size-mb", type=int, default=2)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def summarize(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "min": min(ordered),
        "p50": statistics.median(ordered),
        "max": max(ordered),
    }


def main() -> int:
    args = parse_args()
    if min(args.page_size_mb, args.layers, args.pages, args.cycles) <= 0:
        raise ValueError("page size, layers, pages, and cycles must be positive")

    import torch

    from kvcached.cpu_offload import (
        CPUOffloadManager,
        PageGeometry,
        PageTensorLayout,
        PinnedMemoryOffloadStore,
        TorchPageTransferBackend,
    )
    from kvcached.vmm_ops import (
        PageAllocator,
        create_kv_tensors,
        init_kvcached,
        shutdown_kvcached,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    page_size = args.page_size_mb * 1024 * 1024
    geometry = PageGeometry(
        page_size=page_size,
        num_layers=args.layers,
        num_kv_buffers=2,
    )
    mem_size_per_layer = args.pages * page_size
    raw_tensor_bytes = mem_size_per_layer * geometry.num_kv_buffers
    page_ids: List[int] = []
    allocator = None

    init_kvcached(args.device, page_size, False)
    try:
        raw_tensors = create_kv_tensors(
            raw_tensor_bytes,
            1,
            args.device,
            args.layers,
            geometry.num_kv_buffers,
        )
        allocator = PageAllocator(
            args.layers,
            mem_size_per_layer,
            page_size,
            1,
            0,
            False,
            False,
            False,
            geometry.num_kv_buffers,
        )
        for _ in range(args.pages):
            page_ids.append(allocator.alloc_page().page_id)

        layout = PageTensorLayout(
            geometry,
            [
                int(tensor.numel()) * int(tensor.element_size())
                for tensor in raw_tensors
            ],
            contiguous_layout=False,
        )

        def gpu_views(page_id: int):
            views = []
            for span in layout.spans(page_id):
                flat = raw_tensors[span.tensor_index].view(torch.uint8).reshape(-1)
                views.append(flat.narrow(0, span.byte_offset, span.size_bytes))
            return views

        expected: Dict[int, List[int]] = {}
        for page_id in page_ids:
            expected[page_id] = []
            for payload_index, view in enumerate(gpu_views(page_id)):
                value = (page_id * geometry.payload_count + payload_index + 1) % 251
                view.fill_(value)
                expected[page_id].append(value)
        torch.cuda.synchronize()

        backend = TorchPageTransferBackend(
            raw_tensors,
            geometry,
            contiguous_layout=False,
            release_gpu_page=allocator.offload_page,
            allocate_gpu_page=allocator.restore_page,
            commit_gpu_page=lambda _page_id: None,
            rollback_gpu_page=allocator.offload_page,
        )
        store = PinnedMemoryOffloadStore(
            geometry,
            max_bytes=args.pages * geometry.logical_page_bytes,
        )
        manager = CPUOffloadManager(store, backend)

        offload_ms: List[float] = []
        restore_ms: List[float] = []
        reclaimed_bytes: List[int] = []

        for _ in range(args.cycles):
            for page_id in page_ids:
                free_before, _ = torch.cuda.mem_get_info(args.device)
                start = time.perf_counter()
                result = manager.offload(page_id)
                offload_ms.append((time.perf_counter() - start) * 1000)
                free_after, _ = torch.cuda.mem_get_info(args.device)
                reclaimed_bytes.append(max(0, free_after - free_before))
                if not result.stored or not allocator.is_page_offloaded(page_id):
                    raise RuntimeError(f"page {page_id} did not enter offloaded state")

            if allocator.get_num_offloaded_pages() != len(page_ids):
                raise RuntimeError("offloaded page count does not match")

            for page_id in page_ids:
                start = time.perf_counter()
                if not manager.restore(page_id):
                    raise RuntimeError(f"page {page_id} was missing from CPU store")
                restore_ms.append((time.perf_counter() - start) * 1000)
                if allocator.is_page_offloaded(page_id):
                    raise RuntimeError(f"page {page_id} remained offloaded")
                observed = [int(view[0]) for view in gpu_views(page_id)]
                if observed != expected[page_id]:
                    raise RuntimeError(
                        f"round-trip mismatch for page {page_id}: "
                        f"expected={expected[page_id]}, observed={observed}"
                    )

        report: Dict[str, Any] = {
            "cycles": args.cycles,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
            "layers": args.layers,
            "logical_page_bytes": geometry.logical_page_bytes,
            "offload_ms": summarize(offload_ms),
            "page_size_bytes": page_size,
            "pages": args.pages,
            "reclaimed_bytes_per_page": summarize(
                [float(value) for value in reclaimed_bytes]
            ),
            "restore_ms": summarize(restore_ms),
            "round_trips": args.cycles * args.pages,
            "status": "passed",
            "torch": torch.__version__,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    finally:
        if allocator is not None:
            for page_id in page_ids:
                if allocator.is_page_offloaded(page_id):
                    allocator.restore_page(page_id)
            if page_ids:
                allocator.free_pages(page_ids)
            allocator.trim()
        shutdown_kvcached()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
