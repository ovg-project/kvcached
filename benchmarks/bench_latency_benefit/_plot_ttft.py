#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Plot mean and p99 TTFT averaged across all instances for baseline vs kvcached."""

import json
import re
from collections import defaultdict
from pathlib import Path

import argparse

import matplotlib.pyplot as plt
import numpy as np

_here = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=Path, default=_here / "results/sweep")
parser.add_argument("--kvcached", type=Path, default=_here / "results/sweep2048-kvcached")
parser.add_argument("--out-dir",  type=Path, default=_here / "results")
parser.add_argument("--suffix",   type=str,  default="",
                    help="Suffix appended to output filenames, e.g. '_1024kvcached'")
args_cli = parser.parse_args()

BASELINE_DIR = args_cli.baseline
KVCACHED_DIR = args_cli.kvcached
RESULTS_DIR  = args_cli.out_dir
SUFFIX       = args_cli.suffix


def load_dir(dirpath: Path, inst: int | None = None, allowed_peaks: set | None = None) -> dict:
    """Load JSON result files; return {peak_rps: [list of per-instance dicts]}.
    If inst is given, only load files matching that instance number.
    If allowed_peaks is given, skip peaks not in the set."""
    data = defaultdict(list)
    pattern = f"*inst{inst}.json" if inst is not None else "*.json"
    for f in sorted(dirpath.glob(pattern)):
        m = re.search(r"peak(\d+)", f.name)
        if not m:
            continue
        peak_rps = int(m.group(1))
        if allowed_peaks is not None and peak_rps not in allowed_peaks:
            continue
        with open(f) as fh:
            d = json.load(fh)
        data[peak_rps].append(d)
    return data


def common_peaks(dir_a: Path, dir_b: Path) -> set:
    """Return peak RPS values present in both directories."""
    def peaks(d): return {int(re.search(r"peak(\d+)", f.name).group(1))
                          for f in d.glob("*.json") if re.search(r"peak(\d+)", f.name)}
    a = peaks(dir_a) if dir_a.exists() else set()
    b = peaks(dir_b) if dir_b.exists() else set()
    return a & b if (a and b) else (a or b)


def aggregate(data: dict) -> tuple:
    """Weighted-average TTFT metrics across instances per peak_rps (weighted by completed requests).
    Returns sorted arrays."""
    rps_vals, mean_ttft, p99_ttft = [], [], []
    for rps in sorted(data.keys()):
        instances = data[rps]
        weights, means, p99s = [], [], []
        for d in instances:
            if d.get("mean_ttft_ms") is None or d.get("p99_ttft_ms") is None:
                continue
            w = d.get("completed", d.get("num_prompts", 1))
            weights.append(w)
            means.append(d["mean_ttft_ms"])
            p99s.append(d["p99_ttft_ms"])
        if not means:
            continue
        weights = np.array(weights, dtype=float)
        rps_vals.append(rps)
        mean_ttft.append(np.average(means, weights=weights))
        p99_ttft.append(np.average(p99s, weights=weights))
    return np.array(rps_vals), np.array(mean_ttft) / 1000, np.array(p99_ttft) / 1000  # ms → s


def make_2x2_plot(b_rps, b_mean, b_p99, k_rps, k_mean, k_p99, title, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"TTFT: Baseline vs kvcached — {title}\n"
                 "(6× Qwen2.5-7B on MI300X, prompt=256, completion=2048)",
                 fontsize=13)
    for col, (b_y, k_y, ylabel, metric) in enumerate([
        (b_mean, k_mean, "Mean TTFT (s)", "Mean TTFT"),
        (b_p99,  k_p99,  "P99 TTFT (s)",  "P99 TTFT"),
    ]):
        for row, (yscale, scale_label) in enumerate([("linear", ""), ("log", " (log scale)")]):
            ax = axes[row][col]
            if len(b_rps):
                ax.plot(b_rps, b_y, "o-", color="#e07b39", label="Baseline (static GPU mem)", linewidth=2)
            ax.plot(k_rps, k_y, "s-", color="#4c72b0", label="kvcached (elastic)",         linewidth=2)
            ax.set_xlabel("Peak RPS (per instance)")
            ax.set_ylabel(ylabel)
            ax.set_title(metric + scale_label)
            ax.set_yscale(yscale)
            ax.legend()
            ax.grid(True, alpha=0.3, which="both")
            all_rps = np.concatenate([b_rps, k_rps])
            if len(all_rps):
                ax.set_xlim(left=all_rps.min() - 1, right=all_rps.max() + 1)
            if yscale == "linear":
                ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


# --- Only plot peaks present in both dirs ---
_allowed = common_peaks(BASELINE_DIR, KVCACHED_DIR)

# --- Averaged across all instances ---
b_rps, b_mean, b_p99 = aggregate(load_dir(BASELINE_DIR, allowed_peaks=_allowed)) if BASELINE_DIR.exists() else (np.array([]), np.array([]), np.array([]))
k_rps, k_mean, k_p99 = aggregate(load_dir(KVCACHED_DIR, allowed_peaks=_allowed))
make_2x2_plot(b_rps, b_mean, b_p99, k_rps, k_mean, k_p99,
              "All instances (averaged)", RESULTS_DIR / f"ttft_comparison{SUFFIX}.png")

# --- Per-instance plots ---
for inst in range(1, 7):
    b_rps_i, b_mean_i, b_p99_i = aggregate(load_dir(BASELINE_DIR, inst=inst, allowed_peaks=_allowed)) if BASELINE_DIR.exists() else (np.array([]), np.array([]), np.array([]))
    k_rps_i, k_mean_i, k_p99_i = aggregate(load_dir(KVCACHED_DIR, inst=inst, allowed_peaks=_allowed))
    make_2x2_plot(b_rps_i, b_mean_i, b_p99_i, k_rps_i, k_mean_i, k_p99_i,
                  f"Instance {inst} only",
                  RESULTS_DIR / f"ttft_comparison{SUFFIX}_inst{inst}.png")

# --- Summary table ---
b_rps, b_mean, b_p99 = aggregate(load_dir(BASELINE_DIR, allowed_peaks=_allowed)) if BASELINE_DIR.exists() else (np.array([]), np.array([]), np.array([]))
k_rps, k_mean, k_p99 = aggregate(load_dir(KVCACHED_DIR, allowed_peaks=_allowed))
all_rps = sorted(set(b_rps) | set(k_rps))

print("\n--- Mean TTFT (s) ---")
print(f"{'RPS':>5}  {'Baseline':>10}  {'kvcached':>10}  {'Speedup':>8}")
for rps in all_rps:
    b = b_mean[b_rps == rps][0] if rps in b_rps else float("nan")
    k = k_mean[k_rps == rps][0] if rps in k_rps else float("nan")
    speedup = b / k if (b == b and k == k and k > 0) else float("nan")
    print(f"{rps:>5}  {b:>10.2f}  {k:>10.2f}  {speedup:>8.2f}x")

print("\n--- P99 TTFT (s) ---")
print(f"{'RPS':>5}  {'Baseline':>10}  {'kvcached':>10}  {'Speedup':>8}")
for rps in all_rps:
    b = b_p99[b_rps == rps][0] if rps in b_rps else float("nan")
    k = k_p99[k_rps == rps][0] if rps in k_rps else float("nan")
    speedup = b / k if (b == b and k == k and k > 0) else float("nan")
    print(f"{rps:>5}  {b:>10.2f}  {k:>10.2f}  {speedup:>8.2f}x")
