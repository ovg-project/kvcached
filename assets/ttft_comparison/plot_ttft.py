#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set paper-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'legend.frameon': True,
    'figure.dpi': 300
})

# Configuration
COMPLETION_LENS = [256]
REQRATES = list(range(12, 21))
KV_CACHE_COLOR = '#FF6B35'  # Bright Orange
NO_CACHE_COLOR = '#004E89'  # Deep Blue


def parse_filename(filename):
    """Parse filename to extract configuration parameters"""
    pattern = r'ramp-up-down-0to(\d+)to1.*completion_(\d+)-(\d+)-delay-(\d+)'
    match = re.search(pattern, filename)
    if match:
        return tuple(map(int, match.groups()))
    return None


def load_metrics_data(base_path):
    """Load all metrics data from true and false folders"""
    data = {'true': defaultdict(dict), 'false': defaultdict(dict)}

    for folder in ['true', 'false']:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                parsed = parse_filename(filename)
                if parsed:
                    try:
                        with open(os.path.join(folder_path, filename), 'r') as f:
                            json_data = json.load(f)
                        data[folder][parsed] = {
                            'mean_ttft_ms': json_data.get('mean_ttft_ms', 0),
                            'p99_ttft_ms': json_data.get('p99_ttft_ms', 0)
                        }
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
    return data


def get_metric_values(data, metric, comp_len):
    """Extract metric values for a specific completion length"""
    true_values, false_values = [], []

    for reqrate in REQRATES:
        true_vals = [metrics[metric] for key, metrics in data['true'].items()
                     if key[0] == reqrate and key[1] == comp_len]
        false_vals = [metrics[metric] for key, metrics in data['false'].items()
                      if key[0] == reqrate and key[1] == comp_len]

        true_values.append(np.mean(true_vals) if true_vals else 0)
        false_values.append(np.mean(false_vals) if false_vals else 0)

    return true_values, false_values


def create_chart(true_values, false_values, metric_name, comp_len):
    """Create a single chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(len(REQRATES))
    bar_width = 0.35

    ax.bar(x_positions - bar_width/2, true_values, bar_width,
           label='with kvcached', color=KV_CACHE_COLOR, alpha=0.85,
           edgecolor='black', linewidth=0.8)
    ax.bar(x_positions + bar_width/2, false_values, bar_width,
           label='w/o kvcached', color=NO_CACHE_COLOR, alpha=0.85,
           edgecolor='black', linewidth=0.8)

    # Add speedup annotations
    for i, (true_val, false_val) in enumerate(zip(true_values, false_values)):
        if true_val > 0 and false_val > 0:
            speedup = false_val / true_val
            y_pos = max(true_val, false_val) * 1.08
            ax.text(x_positions[i], y_pos, f'{speedup:.1f}×',
                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                   color='#FFFFFF', bbox=dict(boxstyle="round,pad=0.3",
                                            facecolor='#2E7D32', alpha=0.9))

    ax.set_xlabel('Request Rate (req/s)', fontweight='bold', fontsize=16)
    ax.set_ylabel(f'{metric_name} TTFT (ms)', fontweight='bold', fontsize=16)
    ax.set_title(f'{metric_name} Time to First Token', fontweight='bold', fontsize=18, pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(REQRATES)
    ax.set_yscale('log')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # Clean layout
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.2)

    plt.tight_layout()

    output_filename = f'ttft_{metric_name.lower()}_comp{comp_len}_paper.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_filename}")


def main():
    base_path = "results/metrics"

    print("Loading metrics data...")
    data = load_metrics_data(base_path)

    print("Creating publication-ready TTFT charts...")
    for comp_len in COMPLETION_LENS:
        # Mean TTFT
        true_values, false_values = get_metric_values(data, 'mean_ttft_ms', comp_len)
        create_chart(true_values, false_values, 'Mean', comp_len)

        # P99 TTFT
        true_values, false_values = get_metric_values(data, 'p99_ttft_ms', comp_len)
        create_chart(true_values, false_values, 'P99', comp_len)

    print("Chart generation completed!")


if __name__ == "__main__":
    main()