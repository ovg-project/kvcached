# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Summarize TTFT results from one or more workload runs.

Usage: python analyze.py static=logs/static/results.jsonl offload=... kvcached=...
"""

import json
import sys

import numpy as np


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def pct(vals, q):
    return float(np.percentile(vals, q)) if vals else float("nan")


def summarize(recs):
    out = {}
    for r in recs:
        if r.get("error") or r.get("ttft") is None:
            key = (r["instance"], r["phase"], "errors")
            out.setdefault(key, []).append(1)
            continue
        key = (r["instance"], r["phase"])
        out.setdefault(key, []).append(r["ttft"])
    return out


def main():
    runs = {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        runs[name] = summarize(load(path))

    all_keys = sorted({k for s in runs.values() for k in s if len(k) == 2})
    header = f"{'instance/phase':<16}" + "".join(
        f"{name + ' mean':>14}{name + ' p50':>13}{name + ' p99':>13}{'n':>5}"
        for name in runs)
    print(header)
    print("-" * len(header))
    for key in all_keys:
        inst, phase = key
        row = f"{inst + '/' + phase:<16}"
        for name, s in runs.items():
            vals = s.get(key, [])
            nerr = len(s.get((inst, phase, "errors"), []))
            if vals:
                row += (f"{np.mean(vals) * 1000:>12.0f}ms"
                        f"{pct(vals, 50) * 1000:>11.0f}ms"
                        f"{pct(vals, 99) * 1000:>11.0f}ms"
                        f"{len(vals):>5}")
            else:
                row += f"{'-':>14}{'-':>13}{'-':>13}{'-':>5}"
            if nerr:
                row += f" ({nerr} err)"
        print(row)

    # Headline: burst-phase comparison across runs
    print()
    for phase in ("burst", "seed"):
        for name, s in runs.items():
            vals = [v for (i, p), vv in s.items() if p == phase for v in vv]
            if vals:
                print(f"{phase:>6} {name:<10} mean={np.mean(vals)*1000:7.0f}ms "
                      f"p50={pct(vals,50)*1000:7.0f}ms p99={pct(vals,99)*1000:7.0f}ms "
                      f"n={len(vals)}")
        print()


if __name__ == "__main__":
    main()
