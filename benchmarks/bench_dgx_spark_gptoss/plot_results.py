# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Plot Phase 1 results: performance parity + physical-KV footprint gap.

Reads results/<mode>/cN.json  (vllm bench serve output)
      results/<mode>/cN_mem.json (kv_monitor peak footprint + guards)
      logs/main_baseline_pool.txt (vanilla static KV pool, from serve log)

Produces under results/:
  perf_vs_concurrency.png    (TTFT/throughput overlaid, should coincide)
  kv_footprint_vs_concurrency.png (baseline reserved pool vs kvcached physical)
  phase1_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GIB = 1024 ** 3
# fp8 KV, hybrid-manager disabled => all 36 layers counted as full attention.
KV_BYTES_PER_TOKEN = 36 * 8 * 64 * 2 * 1  # layers*kv_heads*head_dim*(K+V)*fp8_bytes = 36864


def _get(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def load_mode(results_dir: str, mode: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, mode, "c*.json"))):
        base = os.path.basename(path)
        m = re.match(r"c(\d+)\.json$", base)
        if not m:
            continue
        c = int(m.group(1))
        perf = json.load(open(path))
        mem_path = os.path.join(results_dir, mode, f"c{c}_mem.json")
        mem = json.load(open(mem_path)) if os.path.exists(mem_path) else {}
        out[c] = {"perf": perf, "mem": mem}
    return out


def baseline_pool_gib(logs_dir: str, baseline_data: dict[int, dict]) -> float | None:
    """Static KV pool of vanilla vLLM (constant, regardless of load)."""
    txt = os.path.join(logs_dir, "main_baseline_pool.txt")
    if os.path.exists(txt):
        s = open(txt).read()
        m = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", s)
        if m:
            return float(m.group(1))
        m = re.search(r"reserved for KV Cache is ([\d.]+)\s*GiB", s)
        if m:
            return float(m.group(1))
        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", s)
        if m:
            toks = int(m.group(1).replace(",", ""))
            return toks * KV_BYTES_PER_TOKEN / GIB
    # fallback: peak device memory in baseline is ~flat (weights + full pool)
    peaks = [d["mem"].get("peak_gpu_used_gib") for d in baseline_data.values() if d["mem"].get("peak_gpu_used_gib")]
    return max(peaks) if peaks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--logs-dir", default=None)
    ap.add_argument("--phase", type=int, default=1)
    args = ap.parse_args()
    logs_dir = args.logs_dir or os.path.join(os.path.dirname(args.results_dir), "logs")

    base = load_mode(args.results_dir, "baseline")
    kvc = load_mode(args.results_dir, "kvcached")
    concs = sorted(set(base) | set(kvc))
    if not concs:
        print("no results found")
        return

    pool = baseline_pool_gib(logs_dir, base)

    rows = []
    for c in concs:
        b, k = base.get(c, {}), kvc.get(c, {})
        bp, kp = b.get("perf", {}), k.get("perf", {})
        bm, km = b.get("mem", {}), k.get("mem", {})
        rows.append({
            "C": c,
            "base_thpt": _get(bp, "request_throughput"),
            "kvc_thpt": _get(kp, "request_throughput"),
            "base_ttft_mean": _get(bp, "mean_ttft_ms"),
            "kvc_ttft_mean": _get(kp, "mean_ttft_ms"),
            "base_ttft_p99": _get(bp, "p99_ttft_ms"),
            "kvc_ttft_p99": _get(kp, "p99_ttft_ms"),
            "base_tpot_mean": _get(bp, "mean_tpot_ms"),
            "kvc_tpot_mean": _get(kp, "mean_tpot_ms"),
            "base_pool_gib": pool,
            "kvc_phys_gib": km.get("kvcached_peak_physical_gib"),
            "kvc_used_gib": km.get("kvcached_peak_used_gib"),
            "base_preempt": (bm.get("peak_metrics") or {}).get("vllm:num_preemptions_total"),
            "kvc_preempt": (km.get("peak_metrics") or {}).get("vllm:num_preemptions_total"),
        })

    # CSV
    csv_path = os.path.join(args.results_dir, "phase1_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")
    for r in rows:
        print(r)

    # --- plot 1: performance parity
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(concs, [r["base_ttft_mean"] for r in rows], "o-", label="baseline mean TTFT")
    ax[0].plot(concs, [r["kvc_ttft_mean"] for r in rows], "s--", label="kvcached mean TTFT")
    ax[0].plot(concs, [r["base_ttft_p99"] for r in rows], "o:", alpha=.5, label="baseline p99")
    ax[0].plot(concs, [r["kvc_ttft_p99"] for r in rows], "s:", alpha=.5, label="kvcached p99")
    ax[0].set_xlabel("max concurrency")
    ax[0].set_ylabel("TTFT (ms)")
    ax[0].set_title("TTFT (should coincide = same performance)")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[1].plot(concs, [r["base_thpt"] for r in rows], "o-", label="baseline")
    ax[1].plot(concs, [r["kvc_thpt"] for r in rows], "s--", label="kvcached")
    ax[1].set_xlabel("max concurrency")
    ax[1].set_ylabel("request throughput (req/s)")
    ax[1].set_title("Throughput (should coincide)")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    p1 = os.path.join(args.results_dir, "perf_vs_concurrency.png")
    fig.savefig(p1, dpi=130)
    print(f"wrote {p1}")

    # --- plot 2: KV footprint gap
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if pool:
        ax.axhline(pool, color="C0", ls="-", label=f"baseline static pool ({pool:.1f} GiB, constant)")
    ax.plot(concs, [r["kvc_phys_gib"] for r in rows], "s--", color="C1", label="kvcached peak physical (used+prealloc)")
    ax.plot(concs, [r["kvc_used_gib"] for r in rows], "^:", color="C2", alpha=.6, label="kvcached peak used")
    ax.set_xlabel("max concurrency")
    ax.set_ylabel("GPU KV memory (GiB)")
    ax.set_title("Physical KV footprint: kvcached only pays for what it uses")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    p2 = os.path.join(args.results_dir, "kv_footprint_vs_concurrency.png")
    fig.savefig(p2, dpi=130)
    print(f"wrote {p2}")

    # saturation sanity
    bad = [r["C"] for r in rows if (r["base_preempt"] or 0) > 0 or (r["kvc_preempt"] or 0) > 0]
    if bad:
        print(f"WARNING: preemptions seen at C={bad} -> NOT iso-performance there; lower load or raise the cap.")


if __name__ == "__main__":
    main()
