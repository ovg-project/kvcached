#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Load generator for the DGX Spark Guardrail -> LLM -> Guardrail workflow."""

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


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round((percentile / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _extract_message(data: dict[str, Any]) -> str:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"bad chat response: {data}") from exc
    content = message.get("content")
    if isinstance(content, str):
        return content
    return ""


def _extract_stream_delta(data: dict[str, Any]) -> str:
    try:
        delta = data["choices"][0].get("delta") or {}
    except (KeyError, IndexError, TypeError):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    return ""


def _make_random_prompts(num_prompts: int, input_len: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    topics = [
        "capacity planning",
        "incident response",
        "security review",
        "quarterly reporting",
        "developer onboarding",
        "customer support",
    ]
    filler = (
        "Summarize the situation, list concrete tradeoffs, and give a concise recommendation. "
    )
    prompts: list[str] = []
    for i in range(num_prompts):
        topic = rng.choice(topics)
        target_words = max(16, input_len)
        body = (filler * ((target_words // 12) + 2)).strip()
        prompts.append(
            f"Request {i}: You are assisting an enterprise workstation user with {topic}. "
            f"{body}"
        )
    return prompts


def _load_sharegpt(path: Path, num_prompts: int, seed: int, max_prompt_chars: int) -> list[str]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"ShareGPT file must contain a list: {path}")

    prompts: list[str] = []
    for row in data:
        conversations = row.get("conversations") if isinstance(row, dict) else None
        if not isinstance(conversations, list):
            continue
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from") or turn.get("role") or "").lower()
            if role not in {"human", "user"}:
                continue
            value = turn.get("value") or turn.get("content")
            if not isinstance(value, str):
                continue
            text = value.strip()
            if 16 <= len(text) <= max_prompt_chars:
                prompts.append(text)
                break

    if not prompts:
        raise ValueError(f"no usable ShareGPT prompts found in {path}")

    rng = random.Random(seed)
    rng.shuffle(prompts)
    while len(prompts) < num_prompts:
        prompts.extend(prompts[: num_prompts - len(prompts)])
    return prompts[:num_prompts]


async def _post_chat(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    disable_thinking: bool,
    request_timeout: float,
) -> tuple[str, float]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    start = time.perf_counter()
    async with session.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=body,
        timeout=aiohttp.ClientTimeout(total=request_timeout),
    ) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"{model} HTTP {resp.status}: {text[:500]}")
        data = json.loads(text)
    return _extract_message(data), time.perf_counter() - start


async def _stream_main(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    disable_thinking: bool,
    request_timeout: float,
) -> tuple[str, float, float]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    start = time.perf_counter()
    first_token_at: float | None = None
    chunks: list[str] = []
    async with session.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=body,
        timeout=aiohttp.ClientTimeout(total=request_timeout),
    ) as resp:
        if resp.status >= 400:
            text = await resp.text()
            raise RuntimeError(f"{model} HTTP {resp.status}: {text[:500]}")
        async for raw in resp.content:
            for line in raw.decode("utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    delta = _extract_stream_delta(json.loads(payload))
                except json.JSONDecodeError:
                    continue
                if delta:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunks.append(delta)

    end = time.perf_counter()
    if first_token_at is None:
        first_token_at = end
    return "".join(chunks), first_token_at - start, end - start


async def _run_one(
    idx: int,
    prompt: str,
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    counters: dict[str, int],
    counter_lock: asyncio.Lock,
) -> dict[str, Any]:
    async with semaphore:
        async with counter_lock:
            counters["inflight"] += 1
            counters["max_inflight"] = max(counters["max_inflight"], counters["inflight"])
        start = time.perf_counter()
        try:
            input_guard, input_guard_s = await _post_chat(
                session,
                args.guard_base_url,
                args.guard_model,
                [{"role": "user", "content": prompt}],
                args.guard_max_tokens,
                False,
                args.request_timeout,
            )
            blocked = input_guard.strip().lower().startswith("unsafe")
            if blocked and args.respect_guard_decision:
                end = time.perf_counter()
                return {
                    "index": idx,
                    "ok": True,
                    "blocked": True,
                    "workflow_ttft_ms": 0.0,
                    "main_ttft_ms": 0.0,
                    "e2e_ms": (end - start) * 1000,
                    "input_guard_ms": input_guard_s * 1000,
                    "main_e2e_ms": 0.0,
                    "output_guard_ms": 0.0,
                }

            main_start = time.perf_counter()
            output, main_ttft_s, main_e2e_s = await _stream_main(
                session,
                args.main_base_url,
                args.main_model,
                prompt,
                args.main_output_len,
                args.disable_thinking,
                args.request_timeout,
            )
            first_token_at = main_start + main_ttft_s
            _, output_guard_s = await _post_chat(
                session,
                args.guard_base_url,
                args.guard_model,
                [{"role": "user", "content": output[: args.output_guard_max_chars]}],
                args.guard_max_tokens,
                False,
                args.request_timeout,
            )
            end = time.perf_counter()
            return {
                "index": idx,
                "ok": True,
                "blocked": False,
                "workflow_ttft_ms": (first_token_at - start) * 1000,
                "main_ttft_ms": main_ttft_s * 1000,
                "e2e_ms": (end - start) * 1000,
                "input_guard_ms": input_guard_s * 1000,
                "main_e2e_ms": main_e2e_s * 1000,
                "output_guard_ms": output_guard_s * 1000,
                "output_chars": len(output),
            }
        except Exception as exc:  # noqa: BLE001 - benchmark records request-level failures
            return {
                "index": idx,
                "ok": False,
                "blocked": False,
                "error": str(exc),
                "e2e_ms": (time.perf_counter() - start) * 1000,
            }
        finally:
            async with counter_lock:
                counters["inflight"] -= 1


def _build_result(args: argparse.Namespace, duration_s: float, rows: list[dict[str, Any]], max_inflight: int) -> dict[str, Any]:
    completed_rows = [row for row in rows if row.get("ok") and not row.get("blocked")]
    failed_rows = [row for row in rows if not row.get("ok")]
    blocked_rows = [row for row in rows if row.get("blocked")]

    workflow_ttft = [float(row["workflow_ttft_ms"]) for row in completed_rows]
    main_ttft = [float(row["main_ttft_ms"]) for row in completed_rows]
    e2e = [float(row["e2e_ms"]) for row in completed_rows]
    input_guard = [float(row["input_guard_ms"]) for row in completed_rows]
    main_e2e = [float(row["main_e2e_ms"]) for row in completed_rows]
    output_guard = [float(row["output_guard_ms"]) for row in completed_rows]

    result: dict[str, Any] = {
        "date": _now_tag(),
        "phase": args.phase,
        "backend": "openai-chat-workflow",
        "endpoint_type": "guard-main-guard",
        "model_id": args.main_model,
        "guard_model_id": args.guard_model,
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "num_prompts": args.num_prompts,
        "max_concurrency": args.max_concurrency,
        "max_concurrent_requests": max_inflight,
        "duration": duration_s,
        "completed": len(completed_rows),
        "failed": len(failed_rows),
        "blocked": len(blocked_rows),
        "request_throughput": len(completed_rows) / duration_s if duration_s > 0 else 0.0,
        "mean_ttft_ms": _mean(workflow_ttft),
        "p99_ttft_ms": _percentile(workflow_ttft, 99),
        "mean_e2e_ms": _mean(e2e),
        "p99_e2e_ms": _percentile(e2e, 99),
        "mean_main_ttft_ms": _mean(main_ttft),
        "p99_main_ttft_ms": _percentile(main_ttft, 99),
        "mean_input_guard_ms": _mean(input_guard),
        "p99_input_guard_ms": _percentile(input_guard, 99),
        "mean_main_e2e_ms": _mean(main_e2e),
        "p99_main_e2e_ms": _percentile(main_e2e, 99),
        "mean_output_guard_ms": _mean(output_guard),
        "p99_output_guard_ms": _percentile(output_guard, 99),
    }

    if args.save_detailed:
        result["requests"] = rows
    else:
        result["errors"] = [row.get("error", "") for row in failed_rows[:10]]
    return result


async def _async_main(args: argparse.Namespace) -> int:
    if args.dataset_name == "sharegpt":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for ShareGPT")
        prompts = _load_sharegpt(
            Path(args.dataset_path),
            args.num_prompts,
            args.seed,
            args.max_prompt_chars,
        )
    elif args.dataset_name == "random":
        prompts = _make_random_prompts(args.num_prompts, args.random_input_len, args.seed)
    else:
        raise ValueError(f"unsupported dataset: {args.dataset_name}")

    semaphore = asyncio.Semaphore(args.max_concurrency)
    counters = {"inflight": 0, "max_inflight": 0}
    counter_lock = asyncio.Lock()
    connector = aiohttp.TCPConnector(limit=max(args.max_concurrency * 3, 16), force_close=False)
    started = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(_run_one(i, prompt, args, session, semaphore, counters, counter_lock))
            for i, prompt in enumerate(prompts)
        ]
        rows = await asyncio.gather(*tasks)
    duration_s = time.perf_counter() - started

    result = _build_result(args, duration_s, rows, counters["max_inflight"])
    result_file = Path(args.result_file)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "completed",
        "failed",
        "blocked",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_e2e_ms",
        "p99_e2e_ms",
    )}, indent=2))
    return 0 if result["completed"] > 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-base-url", required=True)
    parser.add_argument("--guard-base-url", required=True)
    parser.add_argument("--main-model", required=True)
    parser.add_argument("--guard-model", required=True)
    parser.add_argument("--dataset-name", choices=["sharegpt", "random"], default="sharegpt")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--random-input-len", type=int, default=256)
    parser.add_argument("--main-output-len", type=int, default=2048)
    parser.add_argument("--guard-max-tokens", type=int, default=32)
    parser.add_argument("--output-guard-max-chars", type=int, default=12000)
    parser.add_argument("--max-prompt-chars", type=int, default=12000)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--max-concurrency", type=int, required=True)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--phase", default="kvcached")
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--respect-guard-decision", action="store_true")
    parser.add_argument("--save-detailed", action="store_true")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
