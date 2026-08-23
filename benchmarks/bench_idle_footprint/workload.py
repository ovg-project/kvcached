# SPDX-License-Identifier: Apache-2.0
"""Issue #359 workload: make an idle instance's prefix cache scatter across pages.

Two kinds of traffic mixed together:

  * HOT prefixes revisited throughout the run. Every revisit is a cache hit,
    which moves those blocks back to the most-recently-used end, so an age-only
    eviction policy never drops them. They keep the block ids they were first
    allocated at, spread over whatever pages were filling at that moment.
  * a long tail of COLD prompts seen once. They churn through the cache and get
    evicted, but a page they shared with a surviving block cannot be unmapped.

Decode length matters as much as prompt length: prefill allocates a run of
blocks in one go and packs, while decode grows a request one block at a time
into whatever hole is free. Staggered decodes are what interleaves live
requests across pages in the first place -- without them the cache stays packed
and #359 does not reproduce.

Prompts are lists of token ids so prefix sharing is exact.
"""
import argparse
import asyncio
import json
import random
import time

import aiohttp

# Well inside any modern vocab, above the special-token range.
TOKEN_LO, TOKEN_HI = 1000, 100000


def build_requests(args):
    """Deterministically build the full request list for a run."""
    rng = random.Random(args.seed)

    def toks(n, r=None):
        r = r or rng
        return [r.randint(TOKEN_LO, TOKEN_HI) for _ in range(n)]

    # The hot prefixes get their own seed so a second burst can reuse exactly
    # the same ones while its cold tail is all new.
    hot_rng = random.Random(args.hot_seed if args.hot_seed is not None
                            else args.seed)
    hot = [toks(args.hot_tokens, hot_rng) for _ in range(args.hot_prefixes)]

    # Optional shared pool of cold prompts so two bursts can draw from the same
    # set: hot prefixes always survive eviction, so a second burst that only
    # revisits them cannot tell one eviction policy from another.
    cold_pool = None
    if args.cold_pool:
        pool_rng = random.Random(args.cold_pool_seed)
        cold_pool = [
            [pool_rng.randint(TOKEN_LO, TOKEN_HI)
             for _ in range(pool_rng.randint(args.cold_tokens_min,
                                             args.cold_tokens_max))]
            for _ in range(args.cold_pool)
        ]

    requests = []
    for i in range(args.requests):
        # Introduce hot prefixes gradually so they are first allocated at
        # widely separated times, i.e. on different pages.
        introduced = min(len(hot),
                         1 + int(len(hot) * (i / max(1, args.requests - 1))))
        if rng.random() < args.hot_ratio:
            prefix = hot[rng.randrange(introduced)]
            prompt = prefix + toks(args.suffix_tokens)
            kind = "hot"
        else:
            if cold_pool is not None:
                prompt = cold_pool[rng.randrange(len(cold_pool))]
            else:
                n = rng.randint(args.cold_tokens_min, args.cold_tokens_max)
                prompt = toks(n)
            kind = "cold"
        decode = rng.randint(args.max_tokens_min, args.max_tokens_max)
        requests.append((kind, prompt, decode))
    return requests


async def one_request(session, url, model, prompt, max_tokens, results):
    t0 = time.time()
    try:
        async with session.post(
                url,
                json={"model": model, "prompt": prompt,
                      "max_tokens": max_tokens, "temperature": 0.0},
                timeout=aiohttp.ClientTimeout(total=600)) as resp:
            body = await resp.json()
            ok = resp.status == 200
    except Exception as e:  # noqa: BLE001
        ok, body = False, {"error": str(e)}
    results.append({"ok": ok, "latency": time.time() - t0,
                    "err": None if ok else str(body)[:200]})


async def run(args):
    requests = build_requests(args)
    url = f"http://127.0.0.1:{args.port}/v1/completions"
    sem = asyncio.Semaphore(args.concurrency)
    results = []

    async def guarded(session, prompt, decode):
        async with sem:
            await one_request(session, url, args.model, prompt, decode, results)

    t0 = time.time()
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(guarded(session, p, d))
                 for _kind, p, d in requests]
        done = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            done += 1
            if done % max(1, len(tasks) // 20) == 0:
                print(f"  {done}/{len(tasks)} done "
                      f"({time.time() - t0:.0f}s)", flush=True)
    elapsed = time.time() - t0

    okc = sum(1 for r in results if r["ok"])
    lats = sorted(r["latency"] for r in results if r["ok"])
    summary = {
        "requests": len(results), "ok": okc, "failed": len(results) - okc,
        "elapsed_s": elapsed,
        "throughput_rps": okc / elapsed if elapsed else 0.0,
        "latency_p50": lats[len(lats) // 2] if lats else None,
        "latency_p99": lats[int(len(lats) * 0.99)] if lats else None,
        "sample_errors": [r["err"] for r in results if not r["ok"]][:3],
    }
    print(json.dumps(summary, indent=1), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--requests", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--hot-prefixes", type=int, default=24)
    ap.add_argument("--hot-tokens", type=int, default=256)
    ap.add_argument("--suffix-tokens", type=int, default=64)
    ap.add_argument("--cold-tokens-min", type=int, default=192)
    ap.add_argument("--cold-tokens-max", type=int, default=1024)
    ap.add_argument("--hot-ratio", type=float, default=0.5)
    ap.add_argument("--max-tokens-min", type=int, default=16)
    ap.add_argument("--max-tokens-max", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hot-seed", type=int, default=None)
    ap.add_argument("--cold-pool", type=int, default=0)
    ap.add_argument("--cold-pool-seed", type=int, default=4242)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    asyncio.run(run(args))
