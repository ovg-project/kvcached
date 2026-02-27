# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Record GPU VRAM usage over time using rocm-smi.

Use this alongside (or instead of) record_kvmem.py to capture total GPU
memory consumption during a benchmark run. Particularly useful for the
baseline (no-kvcached) run where no IPC segments exist.

Run concurrently with the benchmark in a separate terminal:

    # kvcached run
    python record_rocm_mem.py --output results/rocm-kvcached.csv

    # baseline run
    python record_rocm_mem.py --output results/rocm-baseline.csv

Output columns:
    timestamp, elapsed_s, card,
    vram_total_gb, vram_used_gb, vram_used_pct,
    gpu_util_pct, vram_alloc_pct
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import time


def sample_rocm_smi() -> list[dict]:
    """
    Call rocm-smi and return a list of per-card dicts with memory stats.
    Returns empty list on error.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--showmemuse", "--showuse", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        data = json.loads(result.stdout)
    except Exception:
        return []

    cards = []
    for card_name, fields in data.items():
        if not card_name.startswith("card"):
            continue
        try:
            total_b = int(fields.get("VRAM Total Memory (B)", 0))
            used_b  = int(fields.get("VRAM Total Used Memory (B)", 0))
            total_gb   = total_b / 1e9
            used_gb    = used_b  / 1e9
            used_pct   = (used_b / total_b * 100) if total_b else 0.0
            gpu_util   = float(fields.get("GPU use (%)", 0) or 0)
            vram_alloc = float(fields.get("GPU Memory Allocated (VRAM%)", 0) or 0)
            cards.append({
                "card":          card_name,
                "vram_total_gb": total_gb,
                "vram_used_gb":  used_gb,
                "vram_used_pct": used_pct,
                "gpu_util_pct":  gpu_util,
                "vram_alloc_pct": vram_alloc,
            })
        except (ValueError, TypeError):
            continue
    return cards


def main():
    parser = argparse.ArgumentParser(
        description="Record GPU VRAM usage over time via rocm-smi."
    )
    parser.add_argument(
        "--output",
        default="results/rocm_mem.csv",
        help="Output CSV file (default: results/rocm_mem.csv).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds. Runs until Ctrl-C if omitted.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    stop = {"flag": False}
    def _handler(sig, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    print(f"Recording GPU VRAM → {args.output}  (interval={args.interval}s, Ctrl-C to stop)\n")

    # Print header sample to confirm rocm-smi works
    sample = sample_rocm_smi()
    if not sample:
        print("ERROR: rocm-smi returned no data. Is ROCm installed and GPU accessible?")
        return
    for c in sample:
        print(f"  {c['card']}: {c['vram_total_gb']:.1f} GB total")
    print()

    start_wall = time.time()
    row_count = 0

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "elapsed_s",
            "card",
            "vram_total_gb",
            "vram_used_gb",
            "vram_used_pct",
            "gpu_util_pct",
            "vram_alloc_pct",
        ])

        while not stop["flag"]:
            now     = time.time()
            elapsed = now - start_wall
            cards   = sample_rocm_smi()

            for c in cards:
                writer.writerow([
                    f"{now:.3f}",
                    f"{elapsed:.2f}",
                    c["card"],
                    f"{c['vram_total_gb']:.3f}",
                    f"{c['vram_used_gb']:.3f}",
                    f"{c['vram_used_pct']:.2f}",
                    f"{c['gpu_util_pct']:.1f}",
                    f"{c['vram_alloc_pct']:.1f}",
                ])
                row_count += 1
                print(
                    f"[{elapsed:6.1f}s] {c['card']}  "
                    f"used={c['vram_used_gb']:.2f}/{c['vram_total_gb']:.1f} GB  "
                    f"({c['vram_used_pct']:.1f}%)  "
                    f"gpu={c['gpu_util_pct']:.0f}%"
                )

            f.flush()

            if args.duration and elapsed >= args.duration:
                break

            time.sleep(args.interval)

    print(f"\nDone. Wrote {row_count} rows to {args.output}")


if __name__ == "__main__":
    main()
