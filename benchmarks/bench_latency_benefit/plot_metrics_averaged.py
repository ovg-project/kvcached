#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_filename(filename):
    """
    Parse filename to extract configuration parameters
    Example: vllm-meta-llama-Llama-3.1-8B-Instruct-ramp-up-down-0to12to1-inc1-prompt_256-completion_128-1-delay-27-model-num-3.json
    Returns: (reqrate, completion_len, model_id, delay)
    """
    pattern = r"ramp-up-down-0to(\d+)to1.*completion_(\d+)-(\d+)-delay-(\d+)"
    match = re.search(pattern, filename)
    if match:
        reqrate = int(match.group(1))
        completion_len = int(match.group(2))
        model_id = int(match.group(3))
        delay = int(match.group(4))
        return reqrate, completion_len, model_id, delay
    return None


def load_metrics_data(base_path):
    """Load all metrics data from true and false folders"""
    data = {"true": defaultdict(dict), "false": defaultdict(dict)}

    for folder in ["true", "false"]:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                parsed = parse_filename(filename)
                if parsed:
                    reqrate, completion_len, model_id, delay = parsed

                    filepath = os.path.join(folder_path, filename)
                    try:
                        with open(filepath, "r") as f:
                            json_data = json.load(f)

                        key = (reqrate, completion_len, model_id, delay)
                        data[folder][key] = {
                            "mean_ttft_ms": json_data.get("mean_ttft_ms", 0),
                            "mean_tpot_ms": json_data.get("mean_tpot_ms", 0),
                            "mean_e2el_ms": json_data.get("mean_e2el_ms", 0),
                            "p99_ttft_ms": json_data.get("p99_ttft_ms", 0),
                            "p99_tpot_ms": json_data.get("p99_tpot_ms", 0),
                            "p99_e2el_ms": json_data.get("p99_e2el_ms", 0),
                        }
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")

    return data


def create_charts_by_completion_length(data):
    """Create charts averaged across models, with separate figures for each completion length"""
    # Define metrics and their display names
    metric_groups = [
        (["mean_ttft_ms", "p99_ttft_ms"], "Time to First Token (TTFT)", "ttft"),
        (["mean_tpot_ms", "p99_tpot_ms"], "Time per Output Token (TPOT)", "tpot"),
        (["mean_e2el_ms", "p99_e2el_ms"], "End-to-End Latency (E2E)", "e2el"),
    ]

    completion_lens = [256, 400]
    reqrates = [i for i in range(12, 21)]

    # Create charts for each metric group and each completion length
    for metric_pair, metric_display_name, metric_short in metric_groups:
        for comp_len in completion_lens:
            # Create subplot with 2 rows (mean and p99)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            for subplot_idx, (ax, metric) in enumerate(zip([ax1, ax2], metric_pair)):
                true_values = []
                false_values = []

                for reqrate in reqrates:
                    # Aggregate across all models for this reqrate and completion_len
                    true_vals = []
                    false_vals = []

                    for key, metrics_data in data["true"].items():
                        if (
                            key[0] == reqrate and key[1] == comp_len
                        ):  # reqrate and completion_len match
                            true_vals.append(metrics_data.get(metric, 0))

                    for key, metrics_data in data["false"].items():
                        if (
                            key[0] == reqrate and key[1] == comp_len
                        ):  # reqrate and completion_len match
                            false_vals.append(metrics_data.get(metric, 0))

                    # Average across all models for this reqrate and completion_len
                    true_val = np.mean(true_vals) if true_vals else 0
                    false_val = np.mean(false_vals) if false_vals else 0

                    true_values.append(true_val)
                    false_values.append(false_val)

                # Bar positions
                x_positions = np.arange(len(reqrates))
                bar_width = 0.35

                # Plot bars
                ax.bar(
                    x_positions - bar_width / 2,
                    true_values,
                    bar_width,
                    label="KV Cache",
                    color="#1f77b4",
                    alpha=0.8,
                )
                ax.bar(
                    x_positions + bar_width / 2,
                    false_values,
                    bar_width,
                    label="No Cache",
                    color="#ff7f0e",
                    alpha=0.8,
                )

                # Add gain annotations
                for i, (true_val, false_val) in enumerate(
                    zip(true_values, false_values)
                ):
                    if true_val > 0 and false_val > 0:
                        gain = false_val / true_val
                        # Position text above the higher bar
                        y_pos = max(true_val, false_val) * 1.1
                        ax.text(
                            x_positions[i],
                            y_pos,
                            f"{gain:.1f}x",
                            ha="center",
                            va="bottom",
                            fontsize=10,
                            color="red" if gain > 1 else "green",
                            fontweight="bold",
                        )

                # Configure subplot
                metric_type = "Mean" if "mean" in metric else "P99"
                ax.set_xlabel("Request Rate (req/s)")
                ax.set_ylabel(f"{metric_type} {metric_display_name} (ms)")
                ax.set_title(
                    f"{metric_type} {metric_display_name} - Completion Length {comp_len} (Averaged across models)"
                )
                ax.set_xticks(x_positions)
                ax.set_xticklabels(reqrates)
                ax.set_yscale("log")  # Set y-axis to log scale
                if subplot_idx == 0:  # Only show legend on first subplot
                    ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save the plot with completion length information in filename
            output_filename = f"avg_models_comp{comp_len}_{metric_short}_comparison.png"
            plt.savefig(output_filename, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Saved chart: {output_filename}")


def main():
    base_path = "results/metrics"

    print("Loading metrics data...")
    raw_data = load_metrics_data(base_path)

    print("Creating charts by completion length (averaged across models)...")
    create_charts_by_completion_length(raw_data)

    print("Chart generation completed!")


if __name__ == "__main__":
    main()
