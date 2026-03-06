#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Test script for validating vLLM prefix cache eviction with kvcached.

This test demonstrates cache eviction behavior by:
- Using a small cache size (configurable via KVCACHED_PREFIX_CACHE_MAX_SIZE)
- Creating multiple distinct prefixes (A, B, C, D, E)
- Sending requests in a pattern that fills cache and triggers LRU eviction
- Verifying eviction occurs by checking logs and latency patterns
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

import requests


def get_prefix_variant(variant: str) -> str:
    """
    Generate different prefixes for testing eviction.

    Each prefix is distinct to create separate cache entries.
    Prefixes are moderately long (~500-800 tokens) to fill cache meaningfully.

    Args:
        variant: One of "A", "B", "C", "D", "E"

    Returns:
        A unique prefix string for the specified variant
    """
    prefixes = {
        "A": """You are a mathematics tutor specializing in ALGEBRA. Solve algebra problems step by step.

Examples:
Question: Solve for x: 2x + 5 = 15
Answer: Subtract 5 from both sides: 2x = 10. Divide by 2: x = 5.

Question: Solve for y: 3y - 7 = 11
Answer: Add 7 to both sides: 3y = 18. Divide by 3: y = 6.

Question: Solve for z: 5z + 2 = 22
Answer: Subtract 2 from both sides: 5z = 20. Divide by 5: z = 4.

Question: Solve for a: 4a - 3 = 17
Answer: Add 3 to both sides: 4a = 20. Divide by 4: a = 5.

Question: Solve for b: 7b + 1 = 36
Answer: Subtract 1 from both sides: 7b = 35. Divide by 7: b = 5.

""",
        "B": """You are a mathematics tutor specializing in GEOMETRY. Solve geometry problems step by step.

Examples:
Question: A rectangle has length 12 cm and width 8 cm. What is its area?
Answer: Area = length × width = 12 × 8 = 96 square cm.

Question: A circle has radius 7 cm. What is its circumference? (Use π = 3.14)
Answer: Circumference = 2πr = 2 × 3.14 × 7 = 43.96 cm.

Question: A triangle has base 10 cm and height 6 cm. What is its area?
Answer: Area = (1/2) × base × height = (1/2) × 10 × 6 = 30 square cm.

Question: A square has side length 9 cm. What is its perimeter?
Answer: Perimeter = 4 × side = 4 × 9 = 36 cm.

Question: A rectangle has length 15 m and width 10 m. What is its perimeter?
Answer: Perimeter = 2(length + width) = 2(15 + 10) = 50 m.

""",
        "C": """You are a mathematics tutor specializing in CALCULUS. Solve calculus problems step by step.

Examples:
Question: Find the derivative of f(x) = 3x² + 5x - 2
Answer: Apply power rule: f'(x) = 6x + 5.

Question: Find the derivative of f(x) = x³ - 4x² + 7
Answer: Apply power rule term by term: f'(x) = 3x² - 8x.

Question: Find the derivative of f(x) = 2x⁴ + 3x³ - x
Answer: Apply power rule: f'(x) = 8x³ + 9x² - 1.

Question: Find the integral of f(x) = 4x
Answer: ∫4x dx = 2x² + C.

Question: Find the integral of f(x) = 6x² - 3
Answer: ∫(6x² - 3) dx = 2x³ - 3x + C.

""",
        "D": """You are a mathematics tutor specializing in STATISTICS. Solve statistics problems step by step.

Examples:
Question: Find the mean of: 5, 8, 12, 15, 20
Answer: Sum = 5 + 8 + 12 + 15 + 20 = 60. Count = 5. Mean = 60/5 = 12.

Question: Find the median of: 3, 7, 9, 15, 21
Answer: Already sorted. Middle value (position 3) is 9. Median = 9.

Question: Find the range of: 10, 25, 18, 32, 45
Answer: Maximum = 45, Minimum = 10. Range = 45 - 10 = 35.

Question: Find the mean of: 100, 150, 200, 250
Answer: Sum = 100 + 150 + 200 + 250 = 700. Count = 4. Mean = 700/4 = 175.

Question: Find the mode of: 2, 3, 3, 5, 7, 3, 9
Answer: 3 appears most frequently (3 times). Mode = 3.

""",
        "E": """You are a mathematics tutor specializing in TRIGONOMETRY. Solve trigonometry problems step by step.

Examples:
Question: If sin(θ) = 0.5, what is θ in degrees? (0° ≤ θ ≤ 90°)
Answer: sin(30°) = 0.5, so θ = 30°.

Question: If cos(θ) = 0.866, what is θ in degrees? (0° ≤ θ ≤ 90°)
Answer: cos(30°) ≈ 0.866, so θ ≈ 30°.

Question: If tan(θ) = 1, what is θ in degrees? (0° ≤ θ ≤ 90°)
Answer: tan(45°) = 1, so θ = 45°.

Question: In a right triangle, if opposite = 3 and hypotenuse = 5, find sin(θ)
Answer: sin(θ) = opposite/hypotenuse = 3/5 = 0.6.

Question: In a right triangle, if adjacent = 4 and hypotenuse = 5, find cos(θ)
Answer: cos(θ) = adjacent/hypotenuse = 4/5 = 0.8.

""",
    }

    if variant not in prefixes:
        raise ValueError(f"Invalid variant: {variant}. Must be one of A, B, C, D, E")

    return prefixes[variant]


def get_variant_question(variant: str) -> str:
    """Generate a question matching the prefix variant."""
    questions = {
        "A": "Question: Solve for x: 6x + 8 = 32",
        "B": "Question: A rectangle has length 20 cm and width 15 cm. What is its area?",
        "C": "Question: Find the derivative of f(x) = 5x² + 2x - 3",
        "D": "Question: Find the mean of: 12, 18, 24, 30, 36",
        "E": "Question: If sin(θ) = 0.707, what is θ in degrees? (0° ≤ θ ≤ 90°)",
    }
    return questions.get(variant, "Question: Solve this problem.")


def send_completion(base_url: str, model: str, prompt: str, max_tokens: int) -> Tuple[str, float]:
    """
    Send completion request and measure latency.

    Returns:
        Tuple of (response_text, latency_seconds)
    """
    url = f"{base_url}/v1/completions"
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    start = time.perf_counter()
    r = requests.post(url, json=body, timeout=120)
    latency = time.perf_counter() - start

    if r.status_code != 200:
        raise RuntimeError(f"Request failed with status {r.status_code}: {r.text}")

    response_data = r.json()
    response_text = response_data["choices"][0]["text"]

    return response_text, latency


def test_eviction(base_url: str, model: str, cache_size: int = 10):
    """
    Test cache eviction with multiple distinct prefixes.

    Strategy:
    1. Send requests with prefix A (fills cache partially)
    2. Send requests with prefix B (fills more)
    3. Send requests with prefix C (fills more)
    4. Send requests with prefix D (may trigger eviction)
    5. Send requests with prefix E (should trigger eviction)
    6. Return to prefix A - should be cache miss due to eviction

    Args:
        base_url: Base URL for the vLLM server
        model: Model name to use
        cache_size: Expected cache size (for display purposes)
    """
    variants = ["A", "B", "C", "D", "E"]

    print(f"\n{'='*80}")
    print("CACHE EVICTION TEST - LRU Eviction Behavior")
    print(f"{'='*80}")
    print("\nTest configuration:")
    print(f"  Cache size: {cache_size} blocks (set via KVCACHED_PREFIX_CACHE_MAX_SIZE)")
    print(f"  Number of prefixes: {len(variants)}")
    print(f"  Model: {model}")
    print("\nTest strategy:")
    print("  1. Send requests with prefixes A, B, C, D, E (fills cache)")
    print("  2. Cache should fill up and start evicting LRU entries")
    print("  3. Return to prefix A - should be cache miss (evicted)")
    print("\nExpected eviction pattern:")
    print("  - Prefix A cached first → evicted first (LRU)")
    print("  - Later prefixes remain in cache")
    print(f"\n{'='*80}\n")

    latencies: Dict[str, List[float]] = {v: [] for v in variants}

    # Phase 1: Fill cache with different prefixes
    print("Phase 1: Filling cache with distinct prefixes\n")

    for variant in variants:
        prefix = get_prefix_variant(variant)
        question = get_variant_question(variant)
        prompt = prefix + question

        try:
            response, latency = send_completion(base_url, model, prompt, max_tokens=50)
            latencies[variant].append(latency)

            prefix_words = len(prefix.split())
            print(f"Prefix {variant}: {latency:.3f}s - MISS (first use, ~{prefix_words * 1.3:.0f} tokens)")

        except Exception as e:
            print(f"ERROR with prefix {variant}: {e}")
            raise

    print(f"\n{'─'*80}\n")

    # Phase 2: Return to prefix A - should be evicted
    print("Phase 2: Returning to prefix A (testing if evicted)\n")

    time.sleep(0.5)

    prefix_a = get_prefix_variant("A")
    question_a = get_variant_question("A")
    prompt_a = prefix_a + question_a

    try:
        response, latency = send_completion(base_url, model, prompt_a, max_tokens=50)
        latencies["A"].append(latency)

        first_a_latency = latencies["A"][0]
        second_a_latency = latencies["A"][1]

        print(f"Prefix A (revisit): {latency:.3f}s")
        print(f"  First use:  {first_a_latency:.3f}s")
        print(f"  Second use: {second_a_latency:.3f}s")

        if second_a_latency >= first_a_latency * 0.8:
            print("  └─ Likely EVICTED (similar latency to first request)")
        else:
            print("  └─ Likely CACHED (faster than first request)")

    except Exception as e:
        print(f"ERROR revisiting prefix A: {e}")
        raise

    print(f"\n{'='*80}")

    # Analysis
    print("\n=== RESULTS ===\n")
    print("Latency summary:")
    for variant in variants:
        if len(latencies[variant]) > 0:
            print(f"  Prefix {variant}: {latencies[variant][0]:.3f}s", end="")
            if len(latencies[variant]) > 1:
                print(f" (revisit: {latencies[variant][1]:.3f}s)")
            else:
                print()

    # Check if prefix A was evicted
    if len(latencies["A"]) > 1:
        first_a = latencies["A"][0]
        second_a = latencies["A"][1]
        ratio = second_a / first_a

        print("\nPrefix A eviction check:")
        print(f"  First request:  {first_a:.3f}s (cache miss)")
        print(f"  Second request: {second_a:.3f}s (after filling cache with B, C, D, E)")
        print(f"  Latency ratio:  {ratio:.2f}x")

        if ratio >= 0.8:
            print("\n✓ EVICTION DETECTED: Prefix A was likely evicted from cache")
            print(f"  Second request is {ratio:.2f}x as slow as first (near 1.0 = evicted)")
        else:
            print("\n⚠ NO CLEAR EVICTION: Prefix A may still be cached")
            print(f"  Second request is only {ratio:.2f}x as slow as first")
            print("  Try reducing cache size or using longer prefixes")

    print("\nTo verify eviction in server logs, run:")
    print("  grep -i 'evict' server.log")
    print("  grep -i 'cache_size' server.log")
    print("\nExpected log patterns:")
    print(f"  [kvcached][DEBUG] Cached block X with hash ..., cache_size=Y/{cache_size}")
    print("  [kvcached][DEBUG] Evicting LRU entry with hash b'...' : block_id=X")

    print(f"\n{'='*80}\n")


def test_eviction_with_repeated_access(base_url: str, model: str, cache_size: int = 10):
    """
    Alternative eviction test: Send multiple requests per prefix to better fill cache.

    Strategy:
    1. Send 3 requests with prefix A (caches A)
    2. Send 3 requests with prefix B (caches B)
    3. Send 3 requests with prefix C (caches C)
    4. Send 3 requests with prefix D (may evict A)
    5. Return to prefix A - should be cache miss
    """
    variants = ["A", "B", "C", "D"]
    requests_per_prefix = 3

    print(f"\n{'='*80}")
    print("CACHE EVICTION TEST (REPEATED ACCESS) - LRU Eviction Behavior")
    print(f"{'='*80}")
    print("\nTest configuration:")
    print(f"  Cache size: {cache_size} blocks")
    print(f"  Prefixes: {len(variants)} (A, B, C, D)")
    print(f"  Requests per prefix: {requests_per_prefix}")
    print(f"  Total requests: {len(variants) * requests_per_prefix + requests_per_prefix}")
    print("\nStrategy:")
    print(f"  - Send {requests_per_prefix} requests with each prefix (A→B→C→D)")
    print("  - Each prefix should cache ~3-4 blocks")
    print(f"  - With cache_size={cache_size}, prefix A should be evicted")
    print("  - Return to A to verify eviction")
    print(f"\n{'='*80}\n")

    latencies: Dict[str, List[float]] = {v: [] for v in variants}

    # Phase 1: Send multiple requests per prefix
    print("Phase 1: Filling cache with multiple requests per prefix\n")

    for variant in variants:
        prefix = get_prefix_variant(variant)
        question = get_variant_question(variant)

        print(f"Prefix {variant}:")

        for req_num in range(requests_per_prefix):
            prompt = prefix + question + f" (request {req_num + 1})"

            try:
                response, latency = send_completion(base_url, model, prompt, max_tokens=50)
                latencies[variant].append(latency)

                status = "MISS" if req_num == 0 else "HIT"
                print(f"  Request {req_num + 1}: {latency:.3f}s - Cache {status}")

            except Exception as e:
                print(f"  ERROR: {e}")
                raise

        print()

    print(f"{'─'*80}\n")

    # Phase 2: Return to prefix A
    print("Phase 2: Returning to prefix A (checking for eviction)\n")

    time.sleep(0.5)

    prefix_a = get_prefix_variant("A")
    question_a = get_variant_question("A")
    prompt_a = prefix_a + question_a + " (eviction check)"

    try:
        response, latency = send_completion(base_url, model, prompt_a, max_tokens=50)
        latencies["A"].append(latency)

        avg_initial_a = sum(latencies["A"][:requests_per_prefix]) / requests_per_prefix
        final_a = latencies["A"][-1]

        print("Prefix A (revisit):")
        print(f"  Initial requests avg: {avg_initial_a:.3f}s")
        print(f"  Revisit latency:      {final_a:.3f}s")
        print(f"  Ratio:                {final_a / latencies["A"][0]:.2f}x")

        if final_a >= latencies["A"][0] * 0.75:
            print("  └─ Status: Likely EVICTED (high latency similar to first request)")
        else:
            print("  └─ Status: Likely CACHED (lower latency)")

    except Exception as e:
        print(f"ERROR revisiting prefix A: {e}")
        raise

    print(f"\n{'='*80}")

    # Final analysis
    print("\n=== FINAL ANALYSIS ===\n")

    first_latencies = {v: latencies[v][0] for v in variants if len(latencies[v]) > 0}
    cached_latencies = {v: sum(latencies[v][1:requests_per_prefix]) / (requests_per_prefix - 1)
                       for v in variants if len(latencies[v]) >= requests_per_prefix}

    print("First request latency (cache miss) per prefix:")
    for variant in variants:
        if variant in first_latencies:
            print(f"  Prefix {variant}: {first_latencies[variant]:.3f}s")

    print("\nAverage cached request latency per prefix:")
    for variant in variants:
        if variant in cached_latencies:
            speedup = first_latencies[variant] / cached_latencies[variant]
            print(f"  Prefix {variant}: {cached_latencies[variant]:.3f}s (speedup: {speedup:.2f}x)")

    if len(latencies["A"]) > requests_per_prefix:
        final_a = latencies["A"][-1]
        first_a = latencies["A"][0]
        ratio = final_a / first_a

        print("\nEviction verification for Prefix A:")
        print(f"  First access:      {first_a:.3f}s")
        print(f"  After eviction:    {final_a:.3f}s")
        print(f"  Latency ratio:     {ratio:.2f}x")

        if ratio >= 0.75:
            print("\n✓ SUCCESS: Eviction detected!")
            print("  Prefix A was evicted and had to be reprocessed")
            print(f"  Latency increased to {ratio:.2f}x of original (near 1.0 confirms eviction)")
        else:
            print("\n⚠ UNCLEAR: Eviction not clearly demonstrated")
            print(f"  Latency ratio {ratio:.2f}x suggests prefix may still be cached")
            print("  Try: smaller cache size or longer prefixes")

    print("\nVerify in server logs with:")
    print("  grep -E '(Evicting|cache_size)' server.log | tail -30")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test vLLM prefix cache eviction with multiple distinct prefixes"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model name (default: meta-llama/Llama-3.2-1B)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12346,
        help="vLLM server port (default: 12346)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="vLLM server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=10,
        help="Expected cache size (for display, actual size set via env var)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "repeated"],
        default="repeated",
        help="Test mode: 'simple' (one request per prefix) or 'repeated' (multiple requests per prefix)"
    )

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Check environment variable
    env_cache_size = os.environ.get("KVCACHED_PREFIX_CACHE_MAX_SIZE")
    if env_cache_size:
        print(f"\n✓ KVCACHED_PREFIX_CACHE_MAX_SIZE is set to: {env_cache_size}")
    else:
        print("\n⚠ WARNING: KVCACHED_PREFIX_CACHE_MAX_SIZE not set")
        print("  Set it before starting the server for this test to work:")
        print(f"  export KVCACHED_PREFIX_CACHE_MAX_SIZE={args.cache_size}")
        print("  Then restart the vLLM server")
        print("\n  Continuing anyway...\n")

    print("\nConfiguration:")
    print(f"  Server: {base_url}")
    print(f"  Model: {args.model}")
    print(f"  Test mode: {args.mode}")

    # Test server health
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"\nERROR: Server health check failed (status {health_response.status_code})")
            print("Make sure the vLLM server is running with prefix caching enabled.")
            return 1
    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Cannot connect to server at {base_url}")
        print(f"  {e}")
        print("\nMake sure to start the server first:")
        print("  bash benchmarks/test_prefix_cache/start_server.sh")
        return 1

    # Run the test
    try:
        if args.mode == "simple":
            test_eviction(base_url, args.model, args.cache_size)
        else:
            test_eviction_with_repeated_access(base_url, args.model, args.cache_size)
        return 0
    except Exception as e:
        print(f"\nERROR during test execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
