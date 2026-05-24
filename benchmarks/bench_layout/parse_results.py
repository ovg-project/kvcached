#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Aggregate sweep results into a markdown table (median across 3 seeds)."""
import glob
import json
import os
import statistics
import sys


def median(xs):
    return statistics.median(xs)


def load_one(path):
    with open(path) as f:
        return json.load(f)


def collect(label, dir_):
    files = sorted(glob.glob(os.path.join(dir_, f"{label}.seed*.json")))
    runs = []
    for fp in files:
        d = load_one(fp)
        if d.get("completed", 0) == 0 and d.get("num_prompts", 0) > 0:
            # benchmark crashed or all reqs failed
            runs.append({"failed": True, "file": os.path.basename(fp)})
            continue
        runs.append({
            "throughput":   d.get("request_throughput"),
            "mean_ttft":    d.get("mean_ttft_ms"),
            "p99_ttft":     d.get("p99_ttft_ms"),
            "mean_tpot":    d.get("mean_tpot_ms"),
            "p99_tpot":     d.get("p99_tpot_ms"),
            "completed":    d.get("completed"),
        })
    return runs


def median_of(runs, key):
    valid = [r[key] for r in runs if not r.get("failed") and r.get(key) is not None]
    if not valid:
        return None
    return median(valid)


def main(dir_):
    configs = [
        ("A_vanilla_inf",          "vanilla vLLM (no kvcached), rate=inf"),
        ("B_kvcached_default_inf", "kvcached default, rate=inf"),
        ("C_layout_false_inf",     "kvcached + LAYOUT=false, rate=inf"),
        ("D_reserved200_inf",      "kvcached + RESERVED=200, rate=inf"),
        ("E_both_inf",             "kvcached + LAYOUT=false + RESERVED=200, rate=inf"),
        ("F_vanilla_r16",          "vanilla vLLM, rate=16"),
        ("G_kvcached_default_r16", "kvcached default, rate=16"),
        ("H_best_r16",             "kvcached + LAYOUT=false + RESERVED=200, rate=16"),
    ]

    print(f"{'config':<55} {'tput':>8} {'mTTFT':>10} {'p99TTFT':>10} {'mTPOT':>8} {'p99TPOT':>9} seeds")
    rows = []
    for label, desc in configs:
        runs = collect(label, dir_)
        if not runs:
            continue
        ok = sum(1 for r in runs if not r.get("failed") and r.get("completed", 0) > 0)
        failed = sum(1 for r in runs if r.get("failed") or r.get("completed", 0) == 0)
        tput = median_of(runs, "throughput")
        mttft = median_of(runs, "mean_ttft")
        pttft = median_of(runs, "p99_ttft")
        mtpot = median_of(runs, "mean_tpot")
        ptpot = median_of(runs, "p99_tpot")
        def s(v, d=2):
            return f"{v:.{d}f}" if v is not None else "—"
        print(f"{desc:<55} {s(tput,2):>8} {s(mttft,1):>10} {s(pttft,1):>10} {s(mtpot,2):>8} {s(ptpot,2):>9}  ok={ok} fail={failed}")
        rows.append((label, desc, ok, failed, tput, mttft, pttft, mtpot, ptpot))

    # Markdown table
    print()
    print("| config | tput (req/s) | mean TTFT (ms) | P99 TTFT (ms) | mean TPOT (ms) | P99 TPOT (ms) |")
    print("|---|--:|--:|--:|--:|--:|")
    for label, desc, ok, failed, tput, mttft, pttft, mtpot, ptpot in rows:
        suffix = f" *(failed in {failed}/{ok+failed})*" if failed else ""
        def s(v, d=2):
            return f"{v:.{d}f}" if v is not None else "—"
        print(f"| {desc}{suffix} | {s(tput)} | {s(mttft,1)} | {s(pttft,1)} | {s(mtpot)} | {s(ptpot)} |")


if __name__ == "__main__":
    dir_ = sys.argv[1] if len(sys.argv) > 1 else "."
    main(dir_)
