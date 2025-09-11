#!/usr/bin/env python3
"""
Plot request rate vs time for multiple models from benchmark results.

Usage:
    python plot_request_rate.py --result-files results/*.json --output plot.png
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_results(result_files: List[str]) -> Dict[str, dict]:
    """Load benchmark results from JSON files."""
    results = {}
    model_name_counts = {}
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract model name from result data or filename
            base_model_name = data.get('model_id', Path(file_path).stem.split('-')[0:-4])
            if isinstance(base_model_name, list):
                base_model_name = '-'.join(base_model_name)
            
            # Handle duplicate model names by adding a suffix
            if base_model_name in model_name_counts:
                model_name_counts[base_model_name] += 1
                model_name = f"{base_model_name}-{model_name_counts[base_model_name]}"
            else:
                model_name_counts[base_model_name] = 1
                model_name = base_model_name
            
            results[model_name] = data
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return results


def synchronize_timestamps(results: Dict[str, dict]) -> Dict[str, dict]:
    """
    Synchronize timestamps across different models using wall-clock time.
    
    Args:
        results: Dictionary of model_name -> benchmark_result
        
    Returns:
        Dictionary with synchronized timestamps
    """
    # Find the earliest global start time across all models
    earliest_start = None
    
    for model_name, result_data in results.items():
        if 'global_start_timestamp' in result_data:
            start_time = result_data['global_start_timestamp']
            if earliest_start is None or start_time < earliest_start:
                earliest_start = start_time
    
    if earliest_start is None:
        print("Warning: No global timestamps found, using original timestamps")
        return results
    
    print(f"Synchronizing timestamps relative to earliest start: {earliest_start}")
    
    # Synchronize all timestamps relative to the earliest start
    synchronized_results = {}
    
    for model_name, result_data in results.items():
        synchronized_data = result_data.copy()
        
        if 'request_timestamps' in result_data:
            timestamps = result_data['request_timestamps']
            synchronized_timestamps = []
            
            for ts in timestamps:
                # Convert wall-clock time to unified time
                if 'wall_time' in ts:
                    unified_time = ts['wall_time'] - earliest_start
                    synchronized_ts = ts.copy()
                    synchronized_ts['unified_time'] = unified_time
                    synchronized_timestamps.append(synchronized_ts)
                else:
                    # Fallback to original time if wall_time not available
                    synchronized_timestamps.append(ts)
            
            synchronized_data['request_timestamps'] = synchronized_timestamps
        
        synchronized_results[model_name] = synchronized_data
    
    return synchronized_results


def calculate_request_rate_timeseries(timestamps: List[dict], window_size: float = 1.0, use_unified_time: bool = True) -> Tuple[List[float], List[float]]:
    """
    Calculate request rate over time using a sliding window.
    
    Args:
        timestamps: List of timestamp dictionaries with 'time', 'model', 'type' fields
        window_size: Window size in seconds for rate calculation
        use_unified_time: Whether to use unified_time instead of time
        
    Returns:
        Tuple of (time_points, request_rates)
    """
    if not timestamps:
        return [], []
    
    # Filter for send events only
    send_events = [t for t in timestamps if t['type'] == 'send']
    
    if not send_events:
        return [], []
    
    # Choose time field
    time_field = 'unified_time' if use_unified_time and 'unified_time' in send_events[0] else 'time'
    
    # Sort by time
    send_events.sort(key=lambda x: x.get(time_field, x.get('time', 0)))
    
    # Get time range
    min_time = send_events[0].get(time_field, send_events[0].get('time', 0))
    max_time = send_events[-1].get(time_field, send_events[-1].get('time', 0))
    
    # Create time points
    time_points = np.arange(min_time, max_time + window_size, window_size / 2)
    request_rates = []
    
    for t in time_points:
        # Count requests in window [t - window_size/2, t + window_size/2]
        window_start = t - window_size / 2
        window_end = t + window_size / 2
        
        count = sum(1 for event in send_events 
                   if window_start <= event.get(time_field, event.get('time', 0)) <= window_end)
        
        # Convert to requests per second
        rate = count / window_size
        request_rates.append(rate)
    
    return time_points.tolist(), request_rates


def plot_request_rates(results: Dict[str, dict], output_path: str, window_size: float = 1.0):
    """
    Plot request rate vs time for multiple models.
    
    Args:
        results: Dictionary of model_name -> benchmark_result
        output_path: Output file path for the plot
        window_size: Window size for rate calculation in seconds
    """
    # Synchronize timestamps across models
    synchronized_results = synchronize_timestamps(results)
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, (model_name, result_data) in enumerate(synchronized_results.items()):
        if 'request_timestamps' not in result_data:
            print(f"Warning: No request_timestamps found in {model_name}")
            continue
            
        timestamps = result_data['request_timestamps']
        
        # Calculate request rate timeseries
        time_points, rates = calculate_request_rate_timeseries(timestamps, window_size)
        
        if not time_points:
            print(f"Warning: No valid timestamps for {model_name}")
            continue
        
        # Plot the line
        color = colors[i % len(colors)]
        plt.plot(time_points, rates, label=model_name, color=color, linewidth=2, alpha=0.8)
        
        # Add theoretical average line
        total_requests = len([t for t in timestamps if t['type'] == 'send'])
        duration = result_data.get('duration', max(time_points))
        theoretical_rate = total_requests / duration if duration > 0 else 0
        
        plt.axhline(y=theoretical_rate, color=color, linestyle='--', alpha=0.5, 
                   label=f'{model_name} (avg: {theoretical_rate:.1f} req/s)')
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Request Rate (requests/second)', fontsize=12)
    plt.title('Request Rate vs Time by Model', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Window size: {window_size}s"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show summary statistics
    print("\n=== Summary Statistics ===")
    for model_name, result_data in synchronized_results.items():
        if 'request_timestamps' in result_data:
            timestamps = result_data['request_timestamps']
            send_events = [t for t in timestamps if t['type'] == 'send']
            
            if send_events:
                total_requests = len(send_events)
                duration = result_data.get('duration', 0)
                avg_rate = total_requests / duration if duration > 0 else 0
                
                print(f"{model_name}:")
                print(f"  Total requests: {total_requests}")
                print(f"  Duration: {duration:.2f}s")
                print(f"  Average rate: {avg_rate:.2f} req/s")
                print()


def main():
    parser = argparse.ArgumentParser(description='Plot request rate vs time from benchmark results')
    
    parser.add_argument('--result-files', nargs='+', default=None,
                       help='Path(s) to benchmark result JSON files (default: all files in results/metrics/)')
    parser.add_argument('--output', default='results/figures/multi_model_request_rate.png',
                       help='Output plot file path (default: results/figures/multi_model_request_rate.png)')
    parser.add_argument('--window-size', type=float, default=1.0,
                       help='Window size in seconds for rate calculation (default: 1.0)')
    
    args = parser.parse_args()
    
    # Auto-discover result files if not specified
    if args.result_files is None:
        metrics_pattern = "results/metrics/*.json"
        args.result_files = glob.glob(metrics_pattern)
        if not args.result_files:
            print("Error: No JSON files found in results/metrics/")
            return 1
        print(f"Auto-discovered {len(args.result_files)} result files: {args.result_files}")
    
    # Load results
    results = load_benchmark_results(args.result_files)
    
    if not results:
        print("Error: No valid benchmark results found")
        return 1
    
    print(f"Loaded {len(results)} benchmark results")
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Synchronize timestamps across models first
    print("Synchronizing timestamps across models...")
    
    # Create plot
    plot_request_rates(results, args.output, args.window_size)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())