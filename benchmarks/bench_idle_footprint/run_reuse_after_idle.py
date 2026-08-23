#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""What an eviction policy's victim choice costs in prefix-cache hit rate.

Two eviction policies make identical choices while traffic is running: live
requests hold blocks all over the pool, so hardly any page is fully evictable
and a page-aware policy has nothing to choose from. They only diverge once the
pool goes quiet and every cached block becomes a candidate.

So the measurement needs three things in order: send traffic, let it go idle
long enough for the trim to run, then send more traffic. And that second round
has to ask for the *cold* prompts the first round cached -- hot prefixes survive
under any policy, and fresh prompts were never cached at all, so neither can
tell two policies apart. `--cold-pool` gives both rounds one fixed set of
prompts to draw from, so the second round re-requests what the first one cached
and its hit rate reads out directly what the trim kept.

    MODEL=/path/to/Qwen3-4B ./run_reuse_after_idle.py

Run it once per branch and compare `hit_rate_after_idle` alongside `idle_gb`.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from probe_mem import detect_segments  # noqa: E402
from run_idle_footprint import (  # noqa: E402
    geometry,
    http_text,
    launch,
    metric,
    our_segments,
    snapshot,
    wait_idle,
    wait_ready,
)

GB = 1024**3


def counters(port):
    text = http_text(f"http://127.0.0.1:{port}/metrics")
    return (metric(text, "vllm:prefix_cache_queries_total") or 0.0,
            metric(text, "vllm:prefix_cache_hits_total") or 0.0)


def send_traffic(port, workload, seed, hot_seed):
    cmd = [sys.executable, os.path.join(HERE, "workload.py"),
           "--port", str(port), "--model", "bench",
           "--seed", str(seed), "--hot-seed", str(hot_seed)]
    subprocess.run(cmd + workload.split(), check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--gap", type=float, default=45.0,
                    help="idle seconds between the two rounds of traffic")
    ap.add_argument("--idle-settle", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workload", default=(
        "--requests 1400 --concurrency 96 --hot-prefixes 8 --hot-tokens 128 "
        "--suffix-tokens 64 --cold-tokens-min 192 --cold-tokens-max 1024 "
        "--hot-ratio 0.2 --max-tokens-min 16 --max-tokens-max 192 "
        "--cold-pool 200"),
        help="args for workload.py; keep --cold-pool or the second round "
             "measures nothing")
    ap.add_argument("--serve-arg", action="append", default=[])
    args = ap.parse_args()
    if not args.model:
        ap.error("pass --model or set MODEL")
    if "--cold-pool" not in args.workload:
        print("warning: without --cold-pool the second round only revisits hot "
              "prefixes, which every policy keeps, and hit rate will not move",
              file=sys.stderr)

    log = os.path.join(HERE, "server.log")
    before = set(detect_segments())
    proc = launch(args.model, args.port, log, args.serve_arg)
    result = {"gap_s": args.gap, "workload": args.workload}
    try:
        if not wait_ready(args.port, log, proc):
            print(f"server failed to start; see {log}")
            return 1
        segs = our_segments(before)
        result["geometry"] = geometry(log)
        bpp = result["geometry"].get("bytes_per_page")

        # First round fills the cache from the pool.
        q0, h0 = counters(args.port)
        send_traffic(args.port, args.workload, seed=1234, hot_seed=99)
        wait_idle(args.port, args.idle_settle)
        q1, h1 = counters(args.port)
        result["hit_rate_before_idle"] = round((h1 - h0) / (q1 - q0), 4) \
            if q1 > q0 else None

        # Going quiet is when the trim finally has pages to choose from, and
        # when memory comes back.
        print(f"idle gap {args.gap}s ...", flush=True)
        time.sleep(args.gap)
        gap = snapshot(segs)
        result["idle_gb"] = round(gap["used_gb"], 2)
        result["idle_pages"] = round(gap["used_gb"] * GB / bpp) if bpp else None

        # Second round re-requests the same pool: whatever the trim kept, hits.
        send_traffic(args.port, args.workload, seed=5678, hot_seed=99)
        wait_idle(args.port, args.idle_settle)
        q2, h2 = counters(args.port)
        result["hit_rate_after_idle"] = round((h2 - h1) / (q2 - q1), 4) \
            if q2 > q1 else None
        result["hits_after_idle"] = int(h2 - h1)

        print(json.dumps(result, indent=1), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=1)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
