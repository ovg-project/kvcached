#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Plot DGX Spark workflow benchmark results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _concurrency_from_name(path: Path) -> int | None:
    match = re.fullmatch(r"c(\d+)\.json", path.name)
    return int(match.group(1)) if match else None


def _load_phase(results_dir: Path, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase_dir = results_dir / phase
    if not phase_dir.exists():
        return rows
    for path in sorted(phase_dir.glob("c*.json")):
        concurrency = _concurrency_from_name(path)
        if concurrency is None:
            continue
        data = json.loads(path.read_text())
        data["concurrency"] = concurrency
        rows.append(data)
    rows.sort(key=lambda item: item["concurrency"])
    return rows


def _valid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("endpoint_type") == "guard-main-guard"
        and "mean_e2e_ms" in row
        and "p99_e2e_ms" in row
        and int(row.get("completed", 0)) > 0
        and int(row.get("failed", 0)) == 0
    ]


def _annotate_delta(
    phase_rows: dict[str, list[dict[str, Any]]],
    metric: str,
    stat: str,
    yoffset: int,
) -> None:
    """Annotate kvcached percentage delta relative to baseline."""
    if "kvcached" not in phase_rows or "baseline" not in phase_rows:
        return

    value_key = f"{stat}_{metric}_ms"
    baseline_by_c = {
        int(row["concurrency"]): float(row[value_key])
        for row in phase_rows["baseline"]
        if value_key in row and float(row[value_key]) > 0
    }
    kvcached_by_c = {
        int(row["concurrency"]): float(row[value_key])
        for row in phase_rows["kvcached"]
        if value_key in row
    }

    for concurrency in sorted(set(baseline_by_c) & set(kvcached_by_c)):
        baseline = baseline_by_c[concurrency]
        kvcached = kvcached_by_c[concurrency]
        delta = (kvcached - baseline) / baseline * 100.0
        color = "#188038" if delta <= 0 else "#b3261e"
        plt.annotate(
            f"{delta:+.0f}%",
            xy=(concurrency, kvcached),
            xytext=(0, yoffset),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=color,
        )


def _plot_metric(results_dir: Path, phases: list[str], metric: str, ylabel: str, output: Path) -> None:
    plt.figure(figsize=(8, 5))
    plotted = False
    phase_rows: dict[str, list[dict[str, Any]]] = {}
    for phase in phases:
        rows = _valid(_load_phase(results_dir, phase))
        if not rows:
            continue
        phase_rows[phase] = rows
        xs = [row["concurrency"] for row in rows]
        mean_key = f"mean_{metric}_ms"
        p99_key = f"p99_{metric}_ms"
        mean = [float(row.get(mean_key, 0.0)) for row in rows]
        p99 = [float(row.get(p99_key, 0.0)) for row in rows]
        plt.plot(xs, mean, marker="o", linewidth=2, label=f"{phase} mean")
        plt.plot(xs, p99, marker="s", linestyle="--", linewidth=2, label=f"{phase} p99")
        plotted = True

    if not plotted:
        raise SystemExit(f"no valid rows found for {metric}")

    _annotate_delta(phase_rows, metric, "mean", -14)
    _annotate_delta(phase_rows, metric, "p99", 10)
    plt.xlabel("Concurrent workflow requests")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()


def _write_summary(results_dir: Path, phases: list[str]) -> None:
    lines = [
        "phase,concurrency,completed,failed,mean_ttft_ms,p99_ttft_ms,mean_e2e_ms,p99_e2e_ms"
    ]
    for phase in phases:
        for row in _load_phase(results_dir, phase):
            lines.append(
                "{phase},{concurrency},{completed},{failed},{mean_ttft_ms:.3f},{p99_ttft_ms:.3f},{mean_e2e_ms:.3f},{p99_e2e_ms:.3f}".format(
                    phase=phase,
                    concurrency=row["concurrency"],
                    completed=int(row.get("completed", 0)),
                    failed=int(row.get("failed", 0)),
                    mean_ttft_ms=float(row.get("mean_ttft_ms", 0.0)),
                    p99_ttft_ms=float(row.get("p99_ttft_ms", 0.0)),
                    mean_e2e_ms=float(row.get("mean_e2e_ms", 0.0)),
                    p99_e2e_ms=float(row.get("p99_e2e_ms", 0.0)),
                )
            )
    (results_dir / "summary.csv").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="demo/dgx-spark/results")
    parser.add_argument("--phases", nargs="+", default=["kvcached", "baseline"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    _plot_metric(
        results_dir,
        args.phases,
        "ttft",
        "Workflow TTFT (ms)",
        results_dir / "ttft_vs_concurrency.png",
    )
    _plot_metric(
        results_dir,
        args.phases,
        "e2e",
        "End-to-end workflow latency (ms)",
        results_dir / "e2e_vs_concurrency.png",
    )
    _write_summary(results_dir, args.phases)
    print(f"wrote {results_dir / 'ttft_vs_concurrency.png'}")
    print(f"wrote {results_dir / 'e2e_vs_concurrency.png'}")
    print(f"wrote {results_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
