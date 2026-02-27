# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark sleep/wake latency for kvcached models.

Measures two things:
  1. Raw wake latency  — time for POST /wake_up to return (weights reloaded)
  2. Cold TTFT         — end-to-end time from firing a request at a sleeping
                         model to receiving the first token (wake + queue + prefill)

Usage:
    # Make sure the router and all 3 instances are running first, then:
    python bench_sleep_wake.py \
        --router-port 8080 \
        --instance-host localhost \
        --instance-port 30000 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --runs 5 \
        --output results/sleep_wake.json
"""

import argparse
import json
import os
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _router_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def _instance_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def wait_for_health(host: str, port: int, timeout: float = 120.0, label: str = "server"):
    """Poll /health until the server responds 200 or timeout."""
    url = _instance_url(host, port, "/health")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"{label} at {url} did not become healthy within {timeout}s")


def put_to_sleep(router_host: str, router_port: int, model: str) -> float:
    """PUT model to sleep via router. Returns time taken (s)."""
    encoded = model.replace("/", "%2F")
    url = _router_url(router_host, router_port, f"/action/sleep/{encoded}")
    t0 = time.perf_counter()
    r = requests.post(url, timeout=60)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return elapsed


def is_sleeping(router_host: str, router_port: int, model: str) -> bool:
    """Check sleep status via router."""
    url = _router_url(router_host, router_port, "/sleep/status")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    # Response is a dict of model_name -> {is_sleeping: bool, ...} or similar
    for key, val in data.items():
        if model in key or key in model:
            if isinstance(val, dict):
                return val.get("is_sleeping", False) or val.get("sleeping", False)
            return bool(val)
    return False


def measure_raw_wake(instance_host: str, instance_port: int) -> float:
    """
    Call POST /wake_up directly on the instance and time it.
    Returns wall-clock seconds until the call returns.
    """
    url = _instance_url(instance_host, instance_port, "/wake_up")
    t0 = time.perf_counter()
    r = requests.post(url, timeout=300)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return elapsed


def measure_cold_ttft(
    router_host: str,
    router_port: int,
    model: str,
    prompt: str = "Tell me a short joke.",
    max_tokens: int = 32,
) -> float:
    """
    Send a streaming completion request to a sleeping model via the router
    and measure time-to-first-token (which includes wake-up time).
    Returns seconds to first token.
    """
    url = _router_url(router_host, router_port, "/v1/completions")
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    first_token_time = None
    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line and first_token_time is None:
                first_token_time = time.perf_counter() - t0
                break  # We only need the first token

    if first_token_time is None:
        raise RuntimeError("No tokens received from model")
    return first_token_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sleep/wake latency for kvcached models."
    )
    parser.add_argument("--router-host",    default="localhost")
    parser.add_argument("--router-port",    type=int, default=8080)
    parser.add_argument("--instance-host",  default="localhost",
                        help="Direct host of the instance to sleep/wake.")
    parser.add_argument("--instance-port",  type=int, required=True,
                        help="Direct port of the instance to sleep/wake (e.g. 30000).")
    parser.add_argument("--model",          required=True,
                        help="Model name as registered with the router.")
    parser.add_argument("--runs",           type=int, default=5,
                        help="Number of sleep/wake cycles to measure (default: 5).")
    parser.add_argument("--min-sleep",      type=int, default=85,
                        help="Seconds to wait after sleep before waking (must exceed "
                             "min_sleep_duration in config, default: 85).")
    parser.add_argument("--output",         default="results/sleep_wake.json",
                        help="Path to save JSON results.")
    parser.add_argument("--prompt",         default="Tell me a short joke.",
                        help="Prompt to use for cold-TTFT measurement.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    print(f"Sleep/wake benchmark")
    print(f"  Model:         {args.model}")
    print(f"  Instance:      {args.instance_host}:{args.instance_port}")
    print(f"  Router:        {args.router_host}:{args.router_port}")
    print(f"  Runs:          {args.runs}")
    print(f"  Min sleep:     {args.min_sleep}s")
    print(f"  Output:        {args.output}\n")

    # Verify instance is healthy before starting
    print("Checking instance health...")
    wait_for_health(args.instance_host, args.instance_port, label="instance")
    print("Instance is healthy.\n")

    results = []

    for run in range(1, args.runs + 1):
        print(f"--- Run {run}/{args.runs} ---")

        # 1. Put the model to sleep
        print(f"  Putting {args.model} to sleep...")
        sleep_call_latency = put_to_sleep(args.router_host, args.router_port, args.model)
        print(f"  Sleep API call returned in {sleep_call_latency*1000:.0f}ms")

        # 2. Wait for min_sleep_duration to elapse (required by SleepManager)
        print(f"  Waiting {args.min_sleep}s (min_sleep_duration)...")
        time.sleep(args.min_sleep)

        # 3. Measure raw wake latency (direct instance call)
        print("  Measuring raw wake latency (POST /wake_up)...")
        t_wake_start = time.time()
        raw_wake_s = measure_raw_wake(args.instance_host, args.instance_port)
        print(f"  Raw wake latency: {raw_wake_s*1000:.0f}ms")

        # 4. Confirm instance healthy again
        wait_for_health(args.instance_host, args.instance_port, label="instance (post-wake)")

        # 5. Put back to sleep for cold-TTFT test
        print(f"  Putting {args.model} back to sleep for cold-TTFT test...")
        put_to_sleep(args.router_host, args.router_port, args.model)
        print(f"  Waiting {args.min_sleep}s...")
        time.sleep(args.min_sleep)

        # 6. Measure cold TTFT (request fires while model is sleeping; router wakes it)
        print("  Measuring cold TTFT (request → sleeping model via router)...")
        cold_ttft_s = measure_cold_ttft(
            args.router_host, args.router_port, args.model, prompt=args.prompt
        )
        print(f"  Cold TTFT: {cold_ttft_s*1000:.0f}ms")

        # 7. Measure warm TTFT for comparison (model is now awake)
        print("  Measuring warm TTFT (model already awake)...")
        warm_ttft_s = measure_cold_ttft(
            args.router_host, args.router_port, args.model, prompt=args.prompt
        )
        print(f"  Warm TTFT: {warm_ttft_s*1000:.0f}ms")
        print(f"  Wake overhead in TTFT: {(cold_ttft_s - warm_ttft_s)*1000:.0f}ms\n")

        results.append({
            "run": run,
            "sleep_call_latency_ms": round(sleep_call_latency * 1000, 1),
            "raw_wake_latency_ms":   round(raw_wake_s * 1000, 1),
            "cold_ttft_ms":          round(cold_ttft_s * 1000, 1),
            "warm_ttft_ms":          round(warm_ttft_s * 1000, 1),
            "wake_overhead_ms":      round((cold_ttft_s - warm_ttft_s) * 1000, 1),
        })

    # Summary
    raw_wake_vals  = [r["raw_wake_latency_ms"]  for r in results]
    cold_ttft_vals = [r["cold_ttft_ms"]          for r in results]
    warm_ttft_vals = [r["warm_ttft_ms"]          for r in results]
    overhead_vals  = [r["wake_overhead_ms"]       for r in results]

    def _stats(vals):
        vals = sorted(vals)
        n = len(vals)
        return {
            "mean_ms":   round(sum(vals) / n, 1),
            "min_ms":    round(vals[0], 1),
            "p50_ms":    round(vals[n // 2], 1),
            "p90_ms":    round(vals[int(n * 0.9)], 1),
            "max_ms":    round(vals[-1], 1),
        }

    summary = {
        "model":              args.model,
        "instance_port":      args.instance_port,
        "runs":               args.runs,
        "min_sleep_s":        args.min_sleep,
        "raw_wake_latency":   _stats(raw_wake_vals),
        "cold_ttft":          _stats(cold_ttft_vals),
        "warm_ttft":          _stats(warm_ttft_vals),
        "wake_overhead":      _stats(overhead_vals),
        "raw_results":        results,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Raw wake latency  (POST /wake_up): mean={summary['raw_wake_latency']['mean_ms']}ms  p90={summary['raw_wake_latency']['p90_ms']}ms")
    print(f"Cold TTFT (asleep → first token):  mean={summary['cold_ttft']['mean_ms']}ms  p90={summary['cold_ttft']['p90_ms']}ms")
    print(f"Warm TTFT (already awake):         mean={summary['warm_ttft']['mean_ms']}ms  p90={summary['warm_ttft']['p90_ms']}ms")
    print(f"Wake overhead in TTFT:             mean={summary['wake_overhead']['mean_ms']}ms  p90={summary['wake_overhead']['p90_ms']}ms")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
