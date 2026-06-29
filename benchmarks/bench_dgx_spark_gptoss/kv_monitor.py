#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Sample peak PHYSICAL KV-cache footprint during a benchmark run.

For kvcached: reads the POSIX-shm MemInfoStruct (/dev/shm/<ipc>) and tracks the
high-water of used_size + prealloc_size = bytes actually mapped on the GPU.
For both modes: scrapes /metrics for the LOGICAL fill gauge and the
saturation guards (num_requests_waiting, num_preemptions_total) — these must
stay ~0 for the run to count as "same performance".

Run it alongside `vllm bench serve`; on SIGTERM/SIGINT it writes the peak JSON.

  python kv_monitor.py --name-filter gptoss \
      --metrics-url http://localhost:12346/metrics \
      --interval 0.2 --out cN_mem.json
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
import urllib.request

from kvcached.cli.utils import MemInfoStruct, RwLockedShm

SHM_DIR = "/dev/shm"
GUARD_METRICS = (
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",   # older alias
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
)


def detect_segments(name_filter: str) -> list[str]:
    out = []
    try:
        for f in os.listdir(SHM_DIR):
            if name_filter and name_filter not in f:
                continue
            try:
                if os.stat(os.path.join(SHM_DIR, f)).st_size != MemInfoStruct.SHM_SIZE:
                    continue
                with RwLockedShm(f, MemInfoStruct.SHM_SIZE, RwLockedShm.RLOCK) as mm:
                    if MemInfoStruct.from_buffer(mm).total_size > 0:
                        out.append(f)
            except Exception:
                continue
    except FileNotFoundError:
        pass
    return sorted(out)


def read_segment(name: str):
    try:
        with RwLockedShm(name, MemInfoStruct.SHM_SIZE, RwLockedShm.RLOCK) as mm:
            return MemInfoStruct.from_buffer(mm)
    except FileNotFoundError:
        return None


def scrape_metrics(url: str) -> dict[str, float]:
    res: dict[str, float] = {}
    if not url:
        return res
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            for line in r.read().decode("utf-8", "ignore").splitlines():
                if not line or line[0] == "#" or " " not in line:
                    continue
                key, _, val = line.rpartition(" ")
                base = key.split("{", 1)[0]
                if base in GUARD_METRICS:
                    try:
                        res[base] = max(res.get(base, float("-inf")), float(val))
                    except ValueError:
                        pass
    except Exception:
        pass
    return res


def gpu_used_bytes() -> int | None:
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return int(total - free)
    except Exception:
        pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name-filter", default="", help="substring of the kvcached shm name (e.g. gptoss); empty=baseline/no shm")
    ap.add_argument("--metrics-url", default="")
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = run until SIGTERM")
    ap.add_argument("--gpu", action="store_true", help="also sample torch.cuda.mem_get_info (heavy import)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))

    peak_used = peak_prealloc = peak_phys = 0
    peak_total = 0
    peak_metrics: dict[str, float] = {}
    peak_gpu = 0
    seg_names: set[str] = set()
    samples = 0
    t0 = time.time()

    while not stop["v"]:
        used = prealloc = total = 0
        for name in (detect_segments(args.name_filter) if args.name_filter else []):
            mi = read_segment(name)
            if mi is None:
                continue
            seg_names.add(name)
            used += mi.used_size
            prealloc += mi.prealloc_size
            total += mi.total_size
        peak_used = max(peak_used, used)
        peak_prealloc = max(peak_prealloc, prealloc)
        peak_phys = max(peak_phys, used + prealloc)
        peak_total = max(peak_total, total)

        m = scrape_metrics(args.metrics_url)
        for k, v in m.items():
            peak_metrics[k] = max(peak_metrics.get(k, float("-inf")), v)

        if args.gpu:
            g = gpu_used_bytes()
            if g is not None:
                peak_gpu = max(peak_gpu, g)

        samples += 1
        if args.duration and (time.time() - t0) >= args.duration:
            break
        time.sleep(args.interval)

    gib = 1024 ** 3
    out = {
        "samples": samples,
        "duration_s": round(time.time() - t0, 2),
        "segments": sorted(seg_names),
        "kvcached_limit_gib": round(peak_total / gib, 3),          # virtual pool cap (total_size)
        "kvcached_peak_used_gib": round(peak_used / gib, 3),       # live working set
        "kvcached_peak_prealloc_gib": round(peak_prealloc / gib, 3),
        "kvcached_peak_physical_gib": round(peak_phys / gib, 3),   # used+prealloc = true GPU footprint
        "peak_gpu_used_gib": round(peak_gpu / gib, 3) if peak_gpu else None,
        "peak_metrics": peak_metrics,                              # incl. num_preemptions_total, num_requests_waiting
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
