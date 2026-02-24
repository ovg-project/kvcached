#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Test script for validating vLLM prefix caching with kvcached.

This script sends multiple requests with a shared few-shot prefix to verify that:
1. The prefix is cached after the first request
2. Subsequent requests reuse cached blocks
3. Cached requests show measurable latency improvements
"""

import argparse
import time
from typing import List, Tuple

import requests


def get_few_shot_prefix() -> str:
    """
    Return a few-shot learning prefix with multiple examples.
    This will be shared across all requests to test prefix caching.
    """
    return """Question: Roger has 5 tennis balls. He buys 2 more. How many does he have?
Answer: 7

Question: The cafeteria had 23 apples. If they used 20, how many do they have?
Answer: 3

Question: Michael had 58 golf balls. On Tuesday, he lost 23. How many does he have?
Answer: 35

Question: There were nine computers in the server room. Five more were installed. How many are there?
Answer: 14

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: 8

"""


def generate_questions(num_questions: int) -> List[str]:
    """Generate a list of math questions to append to the prefix."""
    questions = [
        "Question: Betty has 3 books. She buys 5 more. How many does she have?",
        "Question: John has 8 marbles. He loses 2. How many does he have?",
        "Question: Sarah has 10 candies. She gives away 3. How many does she have?",
        "Question: Tom has 15 coins. He finds 7 more. How many does he have?",
        "Question: Alice has 20 stickers. She uses 4. How many does she have?",
        "Question: Bob has 12 pencils. He buys 6 more. How many does he have?",
        "Question: Carol has 25 stamps. She loses 8. How many does she have?",
        "Question: David has 30 cards. He gives away 12. How many does he have?",
        "Question: Emma has 18 toys. She receives 9 more. How many does she have?",
        "Question: Frank has 40 points. He loses 15. How many does he have?",
    ]
    return questions[:num_questions]


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


def test_prefix_caching(base_url: str, model: str, num_questions: int = 5):
    """
    Test prefix caching with shared few-shot examples.

    Args:
        base_url: Base URL for the vLLM server
        model: Model name to use
        num_questions: Number of questions to test (default: 5)
    """
    # Construct few-shot prefix
    few_shot_prefix = get_few_shot_prefix()

    # Generate questions
    questions = generate_questions(num_questions)

    latencies = []
    responses = []

    print(f"\nTesting prefix caching with {num_questions} requests...")
    print(f"Shared prefix length: {len(few_shot_prefix)} chars")
    print(f"Shared prefix tokens: ~{len(few_shot_prefix.split())} words\n")
    print("=" * 80)

    for i, question in enumerate(questions):
        prompt = few_shot_prefix + question

        try:
            response, latency = send_completion(base_url, model, prompt, max_tokens=50)
            latencies.append(latency)
            responses.append(response.strip())

            status = "MISS (populating cache)" if i == 0 else "HIT (reusing cached prefix)"
            print(f"Request {i+1}/{num_questions}: {latency:.3f}s - Expected cache {status}")
            print(f"  Question: {question[:60]}...")
            print(f"  Answer: {response.strip()[:60]}...")
            print()

        except Exception as e:
            print(f"ERROR in request {i+1}: {e}")
            raise

    print("=" * 80)

    # Analysis
    if len(latencies) > 1:
        first_latency = latencies[0]
        avg_cached_latency = sum(latencies[1:]) / len(latencies[1:])
        speedup = first_latency / avg_cached_latency if avg_cached_latency > 0 else 0

        print("\n=== Results ===")
        print(f"First request (cache miss):     {first_latency:.3f}s")
        print(f"Avg cached requests (hit):      {avg_cached_latency:.3f}s")
        print(f"Speedup from caching:           {speedup:.2f}x")

        if speedup > 1.2:
            print(f"\n✓ SUCCESS: Cache speedup detected ({speedup:.2f}x faster)")
        else:
            print(f"\n⚠ WARNING: Low or no speedup detected ({speedup:.2f}x)")
            print("  Check server logs for 'Cache hit' messages")

        print("\nNote: Check server logs (KVCACHED_LOG_LEVEL=DEBUG) for:")
        print("  - 'Cache hit for hash' messages (expected for requests 2+)")
        print("  - 'Cached block' messages (expected for request 1)")
    else:
        print("\n=== Results ===")
        print(f"Only {len(latencies)} request(s) sent. Need at least 2 to compare.")


def main():
    parser = argparse.ArgumentParser(
        description="Test vLLM prefix caching with kvcached integration"
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
        "--num-questions",
        type=int,
        default=5,
        help="Number of questions to test (default: 5)"
    )

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print("Configuration:")
    print(f"  Server: {base_url}")
    print(f"  Model: {args.model}")
    print(f"  Questions: {args.num_questions}")

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
        print("  bash benchmarks/simple_bench/start_server.sh vllm --enable-prefix-caching")
        return 1

    # Run the test
    try:
        test_prefix_caching(base_url, args.model, args.num_questions)
        return 0
    except Exception as e:
        print(f"\nERROR during test execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
