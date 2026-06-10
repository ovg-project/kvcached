# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Staggered multi-turn burst workload for comparing KV-memory strategies.

Two vLLM instances share one GPU. Each instance receives:
  1. A "seed" phase: N conversations send turn-1 (long unique prefix).
  2. A "burst" phase: all N conversations return with turn-2 (same prefix +
     a short new suffix) at a high request rate.
Bursts are staggered so that while one instance peaks, the other is nearly
idle (only trickle traffic). Prefix reuse in the burst gives CPU-offloading
baselines their best case: evicted KV can be restored from CPU instead of
recomputed.

TTFT is measured per request via streaming completions.
"""

import argparse
import asyncio
import json
import random
import time

import aiohttp

PROMPT_LEN = 2048  # turn-1 prompt tokens (unique per conversation)
TURN2_SUFFIX = 192  # extra tokens appended for turn-2
OUTPUT_LEN = 512  # decode length for every request
TRICKLE_PROMPT_LEN = 128  # short probe requests for the idle instance

ID_LOW, ID_HIGH = 1000, 50000  # safe ordinary-token id range for Qwen3


def gen_conversations(num_convs: int, seed: int):
    rng = random.Random(seed)
    return [[rng.randrange(ID_LOW, ID_HIGH) for _ in range(PROMPT_LEN + TURN2_SUFFIX)]
            for _ in range(num_convs)]


async def send_one(session: aiohttp.ClientSession, port: int, model: str,
                   prompt_ids, output_len: int, tag: dict, results: list):
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": output_len,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    t_send = time.perf_counter()
    ttft = None
    ntokens = 0
    error = None
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error = f"http_{resp.status}: {(await resp.text())[:200]}"
            else:
                async for raw in resp.content:
                    line = raw.strip()
                    if not line or not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    if ttft is None:
                        ttft = time.perf_counter() - t_send
                    ntokens += 1
    except Exception as e:  # noqa: BLE001
        error = repr(e)[:200]
    latency = time.perf_counter() - t_send
    results.append({
        **tag,
        "t_send": t_send,
        "ttft": ttft,
        "latency": latency,
        "num_chunks": ntokens,
        "error": error,
    })
    if error:
        print(f"[warn] request failed: {tag} -> {error}")


async def run_phase(session, port, model, phase_name, instance, convs, turn,
                    rate, results, start_event):
    """Send one request per conversation at a fixed rate (req/s)."""
    await start_event.wait()
    tasks = []
    t0 = time.perf_counter()
    for i, conv in enumerate(convs):
        target = t0 + i / rate
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        prompt = conv[:PROMPT_LEN] if turn == 1 else conv
        tag = {
            "instance": instance,
            "phase": phase_name,
            "conv_id": i,
            "turn": turn,
            "prompt_len": len(prompt),
        }
        tasks.append(
            asyncio.create_task(
                send_one(session, port, model, prompt, OUTPUT_LEN, tag, results)))
    await asyncio.gather(*tasks)


async def run_trickle(session, port, model, instance, results, stop_event,
                      interval: float, seed: int):
    """Low-rate short probes against the (mostly) idle instance."""
    rng = random.Random(seed)
    i = 0
    while not stop_event.is_set():
        prompt = [rng.randrange(ID_LOW, ID_HIGH) for _ in range(TRICKLE_PROMPT_LEN)]
        tag = {
            "instance": instance,
            "phase": "trickle",
            "conv_id": i,
            "turn": 0,
            "prompt_len": len(prompt),
        }
        asyncio.create_task(
            send_one(session, port, model, prompt, 16, tag, results))
        i += 1
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def main(args):
    convs_a = gen_conversations(args.num_convs, seed=1234)
    convs_b = gen_conversations(args.num_convs, seed=5678)
    results: list = []

    seed_dur = args.num_convs / args.seed_rate
    burst_dur = args.num_convs / args.burst_rate
    # Timeline: A seed | gap | A burst (B trickles) | gap | B seed | gap | B burst (A trickles)
    t_a_seed = 0.0
    t_a_burst = t_a_seed + seed_dur + args.gap
    t_b_seed = t_a_burst + burst_dur + args.gap
    t_b_burst = t_b_seed + seed_dur + args.gap
    print(f"timeline: A_seed@{t_a_seed:.0f}s A_burst@{t_a_burst:.0f}s "
          f"B_seed@{t_b_seed:.0f}s B_burst@{t_b_burst:.0f}s")

    conn = aiohttp.TCPConnector(limit=512)
    timeout = aiohttp.ClientTimeout(total=20 * 60)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:

        async def at(t_offset, coro):
            await asyncio.sleep(t_offset)
            return await coro

        evt_now = asyncio.Event()
        evt_now.set()

        stop_trickle_b = asyncio.Event()
        stop_trickle_a = asyncio.Event()

        async def trickle_b():
            # B trickles during A's burst window
            await asyncio.sleep(t_a_burst - 5)
            stop = asyncio.Event()
            task = asyncio.create_task(
                run_trickle(session, args.port_b, args.model, "B", results,
                            stop, args.trickle_interval, 42))
            await asyncio.sleep(burst_dur + 15)
            stop.set()
            await task

        async def trickle_a():
            # A trickles during B's burst window
            await asyncio.sleep(t_b_burst - 5)
            stop = asyncio.Event()
            task = asyncio.create_task(
                run_trickle(session, args.port_a, args.model, "A", results,
                            stop, args.trickle_interval, 43))
            await asyncio.sleep(burst_dur + 15)
            stop.set()
            await task

        await asyncio.gather(
            at(t_a_seed,
               run_phase(session, args.port_a, args.model, "seed", "A",
                         convs_a, 1, args.seed_rate, results, evt_now)),
            at(t_a_burst,
               run_phase(session, args.port_a, args.model, "burst", "A",
                         convs_a, 2, args.burst_rate, results, evt_now)),
            at(t_b_seed,
               run_phase(session, args.port_b, args.model, "seed", "B",
                         convs_b, 1, args.seed_rate, results, evt_now)),
            at(t_b_burst,
               run_phase(session, args.port_b, args.model, "burst", "B",
                         convs_b, 2, args.burst_rate, results, evt_now)),
            trickle_b(),
            trickle_a(),
        )

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--port-a", type=int, default=8100)
    p.add_argument("--port-b", type=int, default=8200)
    p.add_argument("--num-convs", type=int, default=48)
    p.add_argument("--seed-rate", type=float, default=2.0)
    p.add_argument("--burst-rate", type=float, default=8.0)
    p.add_argument("--gap", type=float, default=20.0)
    p.add_argument("--trickle-interval", type=float, default=5.0)
    p.add_argument("--output", default="results.jsonl")
    args = p.parse_args()
    asyncio.run(main(args))
