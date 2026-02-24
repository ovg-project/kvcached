#!/usr/bin/env python3
"""
Advanced test for vLLM prefix caching with long shared prefix.

This test demonstrates high latency improvement (2-3x speedup) by using:
- Very long shared prefix (2500-3000 tokens)
- Many few-shot examples (25-30 examples)
- Multiple requests with same prefix, different questions
"""

import argparse
import time
import requests
from typing import List, Tuple


def get_long_prefix() -> str:
    """
    Generate a long prefix with many examples for high cache benefit.
    
    Returns a prefix with ~2500-3000 tokens including:
    - Detailed instructions
    - 25 few-shot examples
    """
    instruction = """You are an expert math tutor specializing in elementary mathematics. Your task is to solve math problems step by step, showing all your work clearly and explaining your reasoning. Always follow these guidelines:

1. Read the problem carefully and identify what is being asked
2. List the given information
3. Show each step of your calculation
4. Verify your answer makes sense
5. State the final answer clearly

Here are examples to guide your responses:

"""
    
    # Generate 25 diverse few-shot examples
    examples = [
        """Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
Answer: Roger starts with 5 balls. He buys 2 cans with 3 balls each, so 2 × 3 = 6 new balls. Total: 5 + 6 = 11 tennis balls.

""",
        """Question: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
Answer: Start with 23 apples. Used 20, so 23 - 20 = 3 apples left. Then bought 6 more, so 3 + 6 = 9 apples.

""",
        """Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he found 15. How many golf balls does he have now?
Answer: Michael starts with 58 balls. Lost 23: 58 - 23 = 35 balls. Found 15: 35 + 15 = 50 golf balls.

""",
        """Question: There were 9 computers in the server room. 5 more computers were installed each day for 3 days. How many computers are in the server room?
Answer: Start with 9 computers. Added 5 per day for 3 days: 5 × 3 = 15 new computers. Total: 9 + 15 = 24 computers.

""",
        """Question: Olivia has $23. She bought 5 bagels for $3 each. How much money does she have left?
Answer: Olivia starts with $23. Each bagel costs $3, and she buys 5: 5 × $3 = $15. Money left: $23 - $15 = $8.

""",
        """Question: A library had 47 books. They received a donation of 18 books on Monday and 22 books on Friday. How many books does the library have?
Answer: Start with 47 books. Monday donation: 47 + 18 = 65 books. Friday donation: 65 + 22 = 87 books total.

""",
        """Question: Jack had 120 trading cards. He gave 35 to his friend and then bought 3 packs with 12 cards each. How many cards does he have?
Answer: Jack starts with 120 cards. Gave away 35: 120 - 35 = 85 cards. Bought 3 packs with 12 each: 3 × 12 = 36 new cards. Total: 85 + 36 = 121 cards.

""",
        """Question: A bakery made 144 cookies in the morning and 96 cookies in the afternoon. They sold 180 cookies. How many cookies are left?
Answer: Morning batch: 144 cookies. Afternoon batch: 96 cookies. Total made: 144 + 96 = 240 cookies. Sold 180: 240 - 180 = 60 cookies left.

""",
        """Question: Emily has 7 boxes of crayons. Each box contains 24 crayons. She gives 3 boxes to her sister. How many crayons does Emily have left?
Answer: Emily has 7 boxes with 24 crayons each. Gives away 3 boxes, keeps 4 boxes: 4 × 24 = 96 crayons.

""",
        """Question: A farmer planted 8 rows of corn with 15 plants in each row. Later, he planted 4 more rows with 12 plants each. How many corn plants did he plant in total?
Answer: First planting: 8 rows × 15 plants = 120 plants. Second planting: 4 rows × 12 plants = 48 plants. Total: 120 + 48 = 168 plants.

""",
        """Question: Sophie read 45 pages on Saturday and 38 pages on Sunday. If her book has 200 pages, how many pages does she have left to read?
Answer: Pages read: 45 + 38 = 83 pages. Total pages: 200. Pages left: 200 - 83 = 117 pages.

""",
        """Question: A store had 250 bottles of water. They sold 87 bottles on Monday, 93 bottles on Tuesday, and received a shipment of 120 bottles on Wednesday. How many bottles do they have?
Answer: Start: 250 bottles. Monday: 250 - 87 = 163. Tuesday: 163 - 93 = 70. Wednesday shipment: 70 + 120 = 190 bottles.

""",
        """Question: Marcus has $150. He buys a video game for $45 and a controller for $28. Then his grandma gives him $20. How much money does he have?
Answer: Marcus starts with $150. Game costs $45: 150 - 45 = $105. Controller costs $28: 105 - 28 = $77. Grandma gives $20: 77 + 20 = $97.

""",
        """Question: A parking lot has 6 rows with 18 parking spaces in each row. Currently, 45 spaces are occupied. How many spaces are empty?
Answer: Total spaces: 6 rows × 18 spaces = 108 spaces. Occupied: 45 spaces. Empty: 108 - 45 = 63 spaces.

""",
        """Question: Lisa collected 234 seashells over the summer. She gave 78 to her friend and sold 96 at a craft fair. How many seashells does she have left?
Answer: Lisa starts with 234 seashells. Gave away 78: 234 - 78 = 156. Sold 96: 156 - 96 = 60 seashells left.

""",
        """Question: A school ordered 15 boxes of markers. Each box contains 36 markers. If 4 boxes were damaged and returned, how many markers does the school have?
Answer: Total ordered: 15 boxes × 36 markers = 540 markers. Returned 4 boxes: 4 × 36 = 144 markers returned. Kept: 540 - 144 = 396 markers.

""",
        """Question: Noah runs 3 miles every weekday and 5 miles on each weekend day. How many miles does he run in one week?
Answer: Weekdays (5 days): 5 × 3 = 15 miles. Weekend (2 days): 2 × 5 = 10 miles. Total: 15 + 10 = 25 miles per week.

""",
        """Question: A movie theater has 8 screens. Each screen can seat 145 people. If 6 screens are full and 2 screens are empty, how many people are in the theater?
Answer: Each screen seats 145 people. 6 screens are full: 6 × 145 = 870 people in the theater.

""",
        """Question: Anna bought 4 notebooks for $2.50 each and 6 pens for $1.25 each. How much did she spend in total?
Answer: Notebooks: 4 × $2.50 = $10.00. Pens: 6 × $1.25 = $7.50. Total: $10.00 + $7.50 = $17.50.

""",
        """Question: A garden has 12 rows of tomato plants with 18 plants per row. If each plant produces 8 tomatoes, how many tomatoes are produced in total?
Answer: Total plants: 12 rows × 18 plants = 216 plants. Tomatoes: 216 plants × 8 tomatoes = 1,728 tomatoes.

""",
        """Question: Ben has saved $345. He wants to buy a bicycle that costs $289. How much money will he have left after buying the bicycle?
Answer: Ben has $345. Bicycle costs $289. Money left: $345 - $289 = $56.

""",
        """Question: A recipe calls for 3 cups of flour for every 2 cups of sugar. If you want to use 12 cups of flour, how many cups of sugar do you need?
Answer: The ratio is 3 cups flour : 2 cups sugar. For 12 cups flour: 12 ÷ 3 = 4 times the recipe. Sugar needed: 4 × 2 = 8 cups of sugar.

""",
        """Question: A train travels 65 miles per hour. How many miles will it travel in 4 hours and 30 minutes?
Answer: 4 hours at 65 mph: 4 × 65 = 260 miles. 30 minutes (0.5 hours): 0.5 × 65 = 32.5 miles. Total: 260 + 32.5 = 292.5 miles.

""",
        """Question: Katie has 480 stickers. She wants to divide them equally among herself and 5 friends. How many stickers will each person get?
Answer: Total people: Katie + 5 friends = 6 people. Stickers per person: 480 ÷ 6 = 80 stickers each.

""",
        """Question: A rectangular garden is 24 feet long and 15 feet wide. What is the area of the garden in square feet?
Answer: Area of rectangle = length × width. Area = 24 feet × 15 feet = 360 square feet.

""",
        """Question: Sam scores 85, 92, 78, 88, and 95 on his five math tests. What is his average score?
Answer: Total points: 85 + 92 + 78 + 88 + 95 = 438 points. Number of tests: 5. Average: 438 ÷ 5 = 87.6 points.

""",
        """Question: A store sells pencils in packs of 12. If a teacher needs 150 pencils for her class, how many packs should she buy?
Answer: Each pack has 12 pencils. 150 ÷ 12 = 12.5 packs. Since you can't buy half a pack, she needs to buy 13 packs.

""",
        """Question: A swimming pool is being filled with water at a rate of 8 gallons per minute. How many gallons will be in the pool after 2 hours?
Answer: 2 hours = 120 minutes. Rate is 8 gallons per minute. Total: 120 × 8 = 960 gallons.

""",
        """Question: Julia has 3 quarters, 5 dimes, 8 nickels, and 12 pennies. How much money does she have in total?
Answer: Quarters: 3 × $0.25 = $0.75. Dimes: 5 × $0.10 = $0.50. Nickels: 8 × $0.05 = $0.40. Pennies: 12 × $0.01 = $0.12. Total: $0.75 + $0.50 + $0.40 + $0.12 = $1.77.

""",
        """Question: A concert hall has 3 levels. The first level has 240 seats, the second level has 180 seats, and the third level has 120 seats. If 85% of seats are sold, how many tickets were sold?
Answer: Total seats: 240 + 180 + 120 = 540 seats. Tickets sold: 540 × 0.85 = 459 tickets.

""",
    ]
    
    return instruction + "".join(examples)


def generate_test_questions(num_questions: int) -> List[str]:
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
        "Question: Grace has 55 coins. She spends 28. How many does she have?",
        "Question: Henry has 100 stickers. He gives 45 to friends. How many does he have?",
        "Question: Iris has 75 marbles. She loses 18 and finds 10. How many does she have?",
        "Question: Jake has 64 cards. He trades 20 for 15 new ones. How many does he have?",
        "Question: Kate has 80 pencils. She donates 35. How many does she have?",
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


def test_high_speedup(base_url: str, model: str, num_questions: int = 15):
    """
    Test prefix caching with very long shared prefix for high speedup.
    
    Args:
        base_url: Base URL for the vLLM server
        model: Model name to use
        num_questions: Number of questions to test (default: 15)
    """
    prefix = get_long_prefix()
    
    # Count tokens (rough estimate)
    prefix_words = len(prefix.split())
    prefix_chars = len(prefix)
    
    questions = generate_test_questions(num_questions)
    
    latencies = []
    responses = []
    
    print(f"\n{'='*80}")
    print(f"HIGH SPEEDUP TEST - Long Prefix Caching")
    print(f"{'='*80}")
    print(f"\nTest configuration:")
    print(f"  Shared prefix length: {prefix_chars} chars (~{prefix_words} words, ~{prefix_words * 1.3:.0f} tokens)")
    print(f"  Number of requests: {num_questions}")
    print(f"  Model: {model}")
    print(f"\nExpected behavior:")
    print(f"  - First request: Processes full prefix (~{prefix_words * 1.3:.0f} tokens) - SLOW")
    print(f"  - Cached requests: Only process unique question (~50 tokens) - FAST")
    print(f"  - Expected speedup: 2-3x or higher\n")
    print(f"{'='*80}\n")
    
    for i, question in enumerate(questions):
        prompt = prefix + question
        
        try:
            response, latency = send_completion(base_url, model, prompt, max_tokens=50)
            latencies.append(latency)
            responses.append(response.strip())
            
            status = "MISS (populating cache)" if i == 0 else "HIT (reusing cached prefix)"
            print(f"Request {i+1}/{num_questions}: {latency:.3f}s - Cache {status}")
            
            if i == 0:
                print(f"  └─ Processing ~{prefix_words * 1.3:.0f} tokens (full prefix)")
            else:
                print(f"  └─ Processing ~50 tokens (question only)")
            
        except Exception as e:
            print(f"ERROR in request {i+1}: {e}")
            raise
    
    print(f"\n{'='*80}")
    
    # Analysis
    if len(latencies) > 1:
        first_latency = latencies[0]
        avg_cached_latency = sum(latencies[1:]) / len(latencies[1:])
        speedup = first_latency / avg_cached_latency if avg_cached_latency > 0 else 0
        
        min_cached = min(latencies[1:])
        max_cached = max(latencies[1:])
        
        print(f"\n=== RESULTS ===\n")
        print(f"First request (cache miss):     {first_latency:.3f}s")
        print(f"Cached requests (avg):          {avg_cached_latency:.3f}s")
        print(f"Cached requests (min/max):      {min_cached:.3f}s / {max_cached:.3f}s")
        print(f"\n🚀 Speedup from caching:         {speedup:.2f}x")
        print(f"   Latency reduction:            {(1 - 1/speedup) * 100:.1f}%")
        
        if speedup >= 2.0:
            print(f"\n✓ SUCCESS: High cache speedup achieved! ({speedup:.2f}x faster)")
            print(f"  The long prefix ({prefix_words * 1.3:.0f} tokens) provides significant benefit.")
        elif speedup >= 1.5:
            print(f"\n✓ GOOD: Moderate cache speedup detected ({speedup:.2f}x faster)")
            print(f"  Check server logs for cache hit confirmations.")
        else:
            print(f"\n⚠ WARNING: Lower than expected speedup ({speedup:.2f}x)")
            print(f"  Expected: 2-3x speedup with long prefix")
            print(f"  Possible causes:")
            print(f"    - Cache not enabled or not working")
            print(f"    - Network overhead dominating")
            print(f"    - Server load or throttling")
        
        print(f"\nTo verify caching behavior, check server logs for:")
        print(f"  - Request 1: 'Cached block' messages (populating cache)")
        print(f"  - Requests 2-{num_questions}: 'Cache hit for hash' messages (using cache)")
        print(f"\nCommand: grep -i 'cache' server.log | tail -20")
    else:
        print(f"\n=== RESULTS ===")
        print(f"Only {len(latencies)} request(s) sent. Need at least 2 to compare.")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test vLLM prefix caching with long prefix for high speedup"
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
        default=15,
        help="Number of questions to test (default: 15)"
    )
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print(f"\nConfiguration:")
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
        print(f"  bash benchmarks/test_prefix_cache/start_server.sh")
        return 1
    
    # Run the test
    try:
        test_high_speedup(base_url, args.model, args.num_questions)
        return 0
    except Exception as e:
        print(f"\nERROR during test execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
