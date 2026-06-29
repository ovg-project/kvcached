# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Guardrail -> LLM workflow load generator (Phase 2 co-location).

Each request: input-guard chat (small model) -> stream main LLM (gpt-oss-120b).
Measures per-request workflow TTFT (first main token), main TTFT, and E2E, plus
request throughput, under a fixed --max-concurrency. ShareGPT or random prompts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _mean(v: list[float]) -> float:
    return statistics.fmean(v) if v else 0.0


def _pct(v: list[float], p: float) -> float:
    if not v:
        return 0.0
    o = sorted(v)
    return o[max(0, min(len(o) - 1, round((p / 100.0) * (len(o) - 1))))]


def _msg(d: dict[str, Any]) -> str:
    c = d["choices"][0]["message"].get("content")
    return c if isinstance(c, str) else ""


def _delta(d: dict[str, Any]) -> str:
    try:
        c = (d["choices"][0].get("delta") or {}).get("content")
    except (KeyError, IndexError, TypeError):
        return ""
    return c if isinstance(c, str) else ""


def _load_sharegpt(path: Path, n: int, seed: int, max_chars: int) -> list[str]:
    data = json.loads(path.read_text())
    prompts: list[str] = []
    for row in data:
        convs = row.get("conversations") if isinstance(row, dict) else None
        if not isinstance(convs, list):
            continue
        for turn in convs:
            if not isinstance(turn, dict):
                continue
            if str(turn.get("from") or turn.get("role") or "").lower() not in {"human", "user"}:
                continue
            val = turn.get("value") or turn.get("content")
            if isinstance(val, str) and 16 <= len(val.strip()) <= max_chars:
                prompts.append(val.strip())
                break
    if not prompts:
        raise ValueError(f"no usable ShareGPT prompts in {path}")
    rng = random.Random(seed)
    rng.shuffle(prompts)
    while len(prompts) < n:
        prompts.extend(prompts[: n - len(prompts)])
    return prompts[:n]


def _make_random(n: int, words: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    filler = "Summarize the situation, list tradeoffs, and give a recommendation. "
    prompts = [f"Request {i}: " + (filler * ((max(16, words) // 8) + 2)).strip() for i in range(n)]
    rng.shuffle(prompts)
    return prompts


async def _post_chat(session, base, model, messages, max_tokens, disable_thinking, timeout):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t = time.perf_counter()
    async with session.post(f"{base.rstrip('/')}/v1/chat/completions", json=body,
                            timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        txt = await r.text()
        if r.status >= 400:
            raise RuntimeError(f"{model} HTTP {r.status}: {txt[:300]}")
        return _msg(json.loads(txt)), time.perf_counter() - t


def _completion_delta(d: dict[str, Any]) -> str:
    try:
        t = d["choices"][0].get("text")
    except (KeyError, IndexError, TypeError):
        return ""
    return t if isinstance(t, str) else ""


async def _stream_main(session, base, model, prompt, max_tokens, disable_thinking, timeout):
    # gpt-oss Harmony chat returns null on this build; raw completions are
    # coherent, so the main model is driven via /v1/completions.
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "ignore_eos": True}
    t = time.perf_counter()
    first = None
    chunks: list[str] = []
    async with session.post(f"{base.rstrip('/')}/v1/completions", json=body,
                            timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        if r.status >= 400:
            raise RuntimeError(f"{model} HTTP {r.status}: {(await r.text())[:300]}")
        async for raw in r.content:
            for line in raw.decode("utf-8", "ignore").splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    d = _completion_delta(json.loads(payload))
                except json.JSONDecodeError:
                    continue
                if d:
                    if first is None:
                        first = time.perf_counter()
                    chunks.append(d)
    end = time.perf_counter()
    if first is None:
        first = end
    return "".join(chunks), first - t, end - t


async def _run_one(idx, prompt, args, session, sem, counters, lock):
    async with sem:
        async with lock:
            counters["inflight"] += 1
            counters["max_inflight"] = max(counters["max_inflight"], counters["inflight"])
        start = time.perf_counter()
        try:
            # Guard checks a bounded window of the input (its context is small and
            # a guardrail screens, not regenerates). Main gets the full prompt.
            guard_prompt = prompt[:args.guard_max_chars]
            _g, g_s = await _post_chat(session, args.guard_base_url, args.guard_model,
                                       [{"role": "user", "content": guard_prompt}],
                                       args.guard_max_tokens, False, args.request_timeout)
            m_start = time.perf_counter()
            _out, m_ttft, m_e2e = await _stream_main(session, args.main_base_url, args.main_model,
                                                     prompt, args.main_output_len,
                                                     args.disable_thinking, args.request_timeout)
            end = time.perf_counter()
            return {"index": idx, "ok": True,
                    "workflow_ttft_ms": (m_start + m_ttft - start) * 1000,
                    "main_ttft_ms": m_ttft * 1000, "e2e_ms": (end - start) * 1000,
                    "input_guard_ms": g_s * 1000, "main_e2e_ms": m_e2e * 1000}
        except Exception as exc:  # noqa: BLE001
            return {"index": idx, "ok": False, "error": str(exc),
                    "e2e_ms": (time.perf_counter() - start) * 1000}
        finally:
            async with lock:
                counters["inflight"] -= 1


def _build_result(args, dur, rows, max_inflight):
    ok = [r for r in rows if r.get("ok")]
    fail = [r for r in rows if not r.get("ok")]
    wf = [r["workflow_ttft_ms"] for r in ok]
    e2e = [r["e2e_ms"] for r in ok]
    mt = [r["main_ttft_ms"] for r in ok]
    return {
        "date": _now_tag(), "phase": args.phase, "model_id": args.main_model,
        "guard_model_id": args.guard_model, "dataset_name": args.dataset_name,
        "num_prompts": args.num_prompts, "max_concurrency": args.max_concurrency,
        "max_concurrent_requests": max_inflight, "duration": dur,
        "completed": len(ok), "failed": len(fail),
        "request_throughput": len(ok) / dur if dur > 0 else 0.0,
        "mean_ttft_ms": _mean(wf), "p99_ttft_ms": _pct(wf, 99),
        "mean_main_ttft_ms": _mean(mt), "p99_main_ttft_ms": _pct(mt, 99),
        "mean_e2e_ms": _mean(e2e), "p99_e2e_ms": _pct(e2e, 99),
        "errors": [r.get("error", "") for r in fail[:10]],
    }


async def _async_main(args):
    if args.dataset_name == "sharegpt":
        prompts = _load_sharegpt(Path(args.dataset_path), args.num_prompts, args.seed, args.max_prompt_chars)
    else:
        prompts = _make_random(args.num_prompts, args.random_input_len, args.seed)
    sem = asyncio.Semaphore(args.max_concurrency)
    counters = {"inflight": 0, "max_inflight": 0}
    lock = asyncio.Lock()
    conn = aiohttp.TCPConnector(limit=max(args.max_concurrency * 3, 16))
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=conn) as session:
        rows = await asyncio.gather(*[
            _run_one(i, p, args, session, sem, counters, lock) for i, p in enumerate(prompts)])
    res = _build_result(args, time.perf_counter() - t0, rows, counters["max_inflight"])
    Path(args.result_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result_file).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: res[k] for k in ("completed", "failed", "mean_ttft_ms", "p99_ttft_ms",
                                          "mean_e2e_ms", "request_throughput")}, indent=2))
    return 0 if res["completed"] > 0 else 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--main-base-url", required=True)
    p.add_argument("--guard-base-url", required=True)
    p.add_argument("--main-model", required=True)
    p.add_argument("--guard-model", required=True)
    p.add_argument("--dataset-name", choices=["sharegpt", "random"], default="sharegpt")
    p.add_argument("--dataset-path", default="")
    p.add_argument("--random-input-len", type=int, default=256)
    p.add_argument("--main-output-len", type=int, default=512)
    p.add_argument("--guard-max-tokens", type=int, default=16)
    p.add_argument("--guard-max-chars", type=int, default=3000)
    p.add_argument("--max-prompt-chars", type=int, default=12000)
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--max-concurrency", type=int, required=True)
    p.add_argument("--request-timeout", type=float, default=1800)
    p.add_argument("--phase", default="kvcached")
    p.add_argument("--result-file", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable-thinking", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_async_main(parse_args())))
