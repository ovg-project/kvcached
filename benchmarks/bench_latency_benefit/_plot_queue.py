#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Visualize in-flight request queue depth over time for a given peak_rps sweep."""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("peak_rps", type=int, nargs="?", default=16)
_parser.add_argument("--dir", type=Path, action="append", dest="dirs",
                     help="Results directory to plot (can be repeated for multiple dirs). "
                          "Defaults to both sweep2048-kvcached and sweep.")
_parser.add_argument("--label", type=str, action="append", dest="labels",
                     help="Label for each --dir (same order). Defaults to dir name.")
_args = _parser.parse_args()

PEAK_RPS   = _args.peak_rps
RESULTS_DIR = Path(__file__).parent / "results"

_default_dirs = [
    Path(__file__).parent / "results/sweep2048-kvcached",
    Path(__file__).parent / "results/sweep",
]
_default_labels = ["kvcached", "baseline"]

DIRS   = _args.dirs   if _args.dirs   else _default_dirs
LABELS = _args.labels if _args.labels else (_default_labels if not _args.dirs else
                                             [d.name for d in DIRS])


def load_inflight(path: Path):
    """Return (wall_times, in_flight_counts, data_dict) using wall_time for alignment."""
    with open(path) as f:
        d = json.load(f)

    ts = d["request_timestamps"]
    sends    = sorted([t["wall_time"] for t in ts if t["type"] == "send"])
    completes = sorted([t["wall_time"] for t in ts if t["type"] == "complete"])

    # Build event list: +1 at send, -1 at complete
    events = [(t, +1) for t in sends] + [(t, -1) for t in completes]
    events.sort()

    times, depths = [], []
    depth = 0
    for t, delta in events:
        times.append(t)
        depth += delta
        depths.append(depth)

    return np.array(times), np.array(depths), d


def plot_queue(dirpath: Path, peak_rps: int, label: str, axes, color_cycle, global_t0: float):
    files = sorted(dirpath.glob(f"*peak{peak_rps}*inst*.json"))
    if not files:
        return False
    for f in files:
        m = re.search(r"inst(\d+)", f.name)
        inst = int(m.group(1)) if m else 0
        times, depths, d = load_inflight(f)
        color = next(color_cycle)
        axes.step(times - global_t0, depths, where="post", color=color, linewidth=1.2,
                  label=f"{label} inst{inst}")
    return True


# --- Compute global t0 across all instances (earliest wall_time) ---
all_files = [f for d in DIRS for f in sorted(d.glob(f"*peak{PEAK_RPS}*inst*.json"))]
if not all_files:
    print(f"No data found for peak_rps={PEAK_RPS} in {[str(d) for d in DIRS]}")
    sys.exit(1)

global_t0 = min(
    min(t["wall_time"] for t in json.load(open(f))["request_timestamps"]
        if t["type"] == "send")
    for f in all_files
)

# --- Build figure ---
fig, axes = plt.subplots(figsize=(14, 5))
dir_names = ", ".join(d.name for d in DIRS)
fig.suptitle(f"In-flight requests over time — peak_rps={PEAK_RPS} (wall-clock aligned)\n"
             f"(6× Qwen2.5-7B on MI300X, prompt=256, completion=2048)  [{dir_names}]",
             fontsize=13)

# One color palette per directory
palettes = [
    ["#4c72b0", "#6a9fd8", "#88c0f0", "#2050a0", "#103880", "#082060"],  # blues
    ["#e07b39", "#e8a060", "#f0c090", "#c05020", "#a03010", "#803000"],  # oranges
    ["#2ca02c", "#5fd35f", "#98df8a", "#1a7a1a", "#0f520f", "#073507"],  # greens
    ["#9467bd", "#c5b0d5", "#7b4fa0", "#5c3080", "#3d1f60", "#1e0f30"],  # purples
]

for idx, (dirpath, label) in enumerate(zip(DIRS, LABELS)):
    color_cycle = iter(palettes[idx % len(palettes)])
    plot_queue(dirpath, PEAK_RPS, label, axes, color_cycle, global_t0)

# Annotate rps ramp-up from inst1 of first available dir
from datetime import datetime, timezone
sample_file = next((sorted(d.glob(f"*peak{PEAK_RPS}*inst1.json")) for d in DIRS
                    if sorted(d.glob(f"*peak{PEAK_RPS}*inst1.json"))), None)
sample_file = sample_file[0] if sample_file else None
if sample_file:
    with open(sample_file) as f:
        d = json.load(f)
    rps_events = d.get("rps_change_events", [])
    prev_rps = None
    for ev in rps_events:
        ev_wall = datetime.fromisoformat(ev["timestamp"]).replace(tzinfo=timezone.utc).timestamp()
        rel = ev_wall - global_t0
        rps = ev["rps"]
        if rps != prev_rps and rel >= 0:
            axes.axvline(rel, color="gray", linewidth=0.5, alpha=0.4)
            if rps % 4 == 0:
                axes.text(rel + 0.3, 5, f"{rps}rps", fontsize=7, color="gray",
                          ha="left", va="bottom", rotation=90)
        prev_rps = rps

axes.set_xlabel("Time (s)")
axes.set_ylabel("In-flight requests")
axes.set_title(f"Peak RPS = {PEAK_RPS}")
axes.legend(ncol=2, fontsize=8)
axes.grid(True, alpha=0.3)
axes.set_xlim(left=0)
axes.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
dir_suffix = "_".join(d.name for d in DIRS) if _args.dirs else "combined"
output = RESULTS_DIR / f"queue_peak{PEAK_RPS}_{dir_suffix}.png"
plt.savefig(output, dpi=150, bbox_inches="tight")
print(f"Saved: {output}")
