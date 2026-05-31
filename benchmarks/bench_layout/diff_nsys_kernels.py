# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Diff per-kernel GPU time between two nsys traces.

Usage:
    python diff_nsys_kernels.py <baseline.nsys-rep> <variant.nsys-rep>

Prints kernels ordered by absolute delta (variant - baseline). Useful for
identifying which kernels regress between KVCACHED_CONTIGUOUS_LAYOUT=true and
false.
"""
import csv
import subprocess
import sys
from collections import defaultdict
from io import StringIO


def kernel_sums(nsys_rep_path: str) -> dict:
    """Returns {kernel_name: (total_ns, instances)} for the given trace."""
    result = subprocess.run(
        [
            "nsys",
            "stats",
            "--report",
            "cuda_gpu_kern_sum",
            "--format",
            "csv",
            "--force-overwrite=true",
            nsys_rep_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # nsys stats emits header text, then a CSV table. Find the CSV.
    text = result.stdout
    # Find a line that looks like a header row.
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("Time (%)") or line.startswith("\"Time (%)"):
            start = i
            break
    if start is None:
        raise RuntimeError(
            f"No CSV header found in nsys stats output for {nsys_rep_path}.\n"
            f"First 500 chars:\n{text[:500]}")

    csv_text = "\n".join(lines[start:])
    reader = csv.DictReader(StringIO(csv_text))
    sums: dict = defaultdict(lambda: [0, 0])
    for row in reader:
        name = row["Name"]
        total_ns = int(row["Total Time (ns)"])
        instances = int(row["Instances"])
        sums[name][0] += total_ns
        sums[name][1] += instances
    return {k: tuple(v) for k, v in sums.items()}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    baseline_path, variant_path = sys.argv[1], sys.argv[2]
    print(f"baseline = {baseline_path}", file=sys.stderr)
    print(f"variant  = {variant_path}", file=sys.stderr)

    base = kernel_sums(baseline_path)
    var = kernel_sums(variant_path)

    all_kernels = set(base) | set(var)
    rows = []
    for k in all_kernels:
        b_ns, b_n = base.get(k, (0, 0))
        v_ns, v_n = var.get(k, (0, 0))
        delta = v_ns - b_ns
        rows.append((k, b_ns, b_n, v_ns, v_n, delta))

    rows.sort(key=lambda r: r[5])  # by delta ascending (negative first = sped up)

    total_b = sum(r[1] for r in rows)
    total_v = sum(r[3] for r in rows)
    print(
        f"\nTotal kernel time: baseline={total_b/1e6:,.1f} ms  "
        f"variant={total_v/1e6:,.1f} ms  "
        f"delta={ (total_v-total_b)/1e6:+,.1f} ms ({(total_v-total_b)/total_b*100:+.1f}%)\n"
    )

    print(f"{'kernel':<80} {'base ms':>10} {'var ms':>10} {'delta ms':>10} {'delta %':>8}  base_n  var_n")
    for k, b_ns, b_n, v_ns, v_n, d_ns in rows:
        if abs(d_ns) < 1_000_000:  # < 1 ms delta, skip noise
            continue
        pct = (d_ns / b_ns * 100) if b_ns else float("inf")
        print(
            f"{k[:80]:<80} {b_ns/1e6:>10.2f} {v_ns/1e6:>10.2f} {d_ns/1e6:>+10.2f} {pct:>+7.1f}%  {b_n:>6}  {v_n:>6}")


if __name__ == "__main__":
    main()
