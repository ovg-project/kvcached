# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Stdlib HTTP client for the vLLM and SGLang servers under test.

Every request is greedy, so a kvcached server and a native server that see the
same inputs must return the same token ids for the same sequential requests.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

BASE_PROMPTS = [
    "The capital of France is",
    "Write a short Python function that returns the n-th Fibonacci number.\n",
    "Explain in three sentences why the sky is blue.",
    "List the first ten prime numbers separated by commas:",
    "Translate to German: 'The weather is nice today and we are going to the park.'",
    "Once upon a time in a small village by the sea,",
]

_WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi "
          "rho sigma tau upsilon phi chi psi omega river mountain cloud engine memory page block "
          "cache tensor kernel graph scheduler request token stream buffer vector matrix").split()

APC_QUESTIONS = [
    "Question: what is the first word of the document? Answer:",
    "Question: how many times does the word 'cache' appear? Answer:",
    "Question: write a title for this document. Answer:",
    "Question: is the word 'omega' present? Answer:",
]


def long_document(n_words: int, seed: int) -> str:
    rnd = random.Random(seed)
    body = " ".join(rnd.choice(_WORDS) for _ in range(n_words))
    return f"Document #{seed}: {body}"


def long_prompt(n_words: int, seed: int) -> str:
    return long_document(n_words, seed) + "\n\nSummarize the document above in one sentence:"


def gen_prompts() -> list[str]:
    """Short prompts plus long ones that span many KV blocks and pages."""
    return list(BASE_PROMPTS) + [long_prompt(n, 1000 + n) for n in (300, 900, 1800)]


def apc_prompts() -> list[str]:
    prefix = long_document(1300, 4242) + "\n\n"
    return [prefix + q for q in APC_QUESTIONS]


@dataclass
class Result:
    status: int
    token_ids: Optional[list[int]]
    text: Optional[str]
    prompt_tokens: Optional[int]
    cached_tokens: Optional[int]
    latency_s: float
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.token_ids is not None


@dataclass
class Phase:
    """One pass over a prompt list."""
    name: str
    concurrency: int
    results: list[Result] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return bool(self.results) and all(r.ok for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "concurrency": self.concurrency,
                "results": [asdict(r) for r in self.results]}


def _post(url: str, body: dict[str, Any], timeout: float) -> tuple[int, Any, float]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read().decode(errors="replace")
    except Exception as e:  # connection refused, timeout, reset
        return 0, {"error": f"{type(e).__name__}: {e}"}, time.time() - t0
    try:
        return status, json.loads(raw), time.time() - t0
    except ValueError:
        return status, {"error": raw[:2000]}, time.time() - t0


class EngineClient:
    """Common interface; subclasses speak each server's API."""

    def __init__(self, host: str, port: int, timeout: float = 1800.0) -> None:
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=10) as r:
                return r.status == 200
        except Exception:
            return False

    def wait_ready(self, timeout: float, alive: Callable[[], bool]) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.health():
                return True
            if not alive():
                return False
            time.sleep(3)
        return False

    def generate(self, prompt: str, max_tokens: int, ignore_eos: bool = False) -> Result:
        raise NotImplementedError

    def reset_prefix_cache(self) -> bool:
        raise NotImplementedError

    def run_phase(self, name: str, prompts: list[str], max_tokens: int,
                  concurrency: int, ignore_eos: bool = False) -> Phase:
        phase = Phase(name, concurrency)
        if concurrency <= 1:
            phase.results = [self.generate(p, max_tokens, ignore_eos) for p in prompts]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                phase.results = list(ex.map(lambda p: self.generate(p, max_tokens, ignore_eos),
                                            prompts))
        return phase


class VLLMClient(EngineClient):
    def generate(self, prompt: str, max_tokens: int, ignore_eos: bool = False) -> Result:
        body: dict[str, Any] = {"model": "e2e", "prompt": prompt, "temperature": 0,
                                "max_tokens": max_tokens, "return_token_ids": True, "seed": 0}
        if ignore_eos:
            body["ignore_eos"] = True
        status, j, dt = _post(self.base + "/v1/completions", body, self.timeout)
        if status != 200:
            return Result(status, None, None, None, None, dt, str(j.get("error", j))[:2000])
        ch = (j.get("choices") or [{}])[0]
        usage = j.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        return Result(status, ch.get("token_ids"), ch.get("text"), usage.get("prompt_tokens"),
                      cached, dt)

    def reset_prefix_cache(self) -> bool:
        # Needs VLLM_SERVER_DEV_MODE=1 on the server.
        status, _, _ = _post(self.base + "/reset_prefix_cache", {}, 120)
        return status == 200


class SGLangClient(EngineClient):
    def generate(self, prompt: str, max_tokens: int, ignore_eos: bool = False) -> Result:
        sp: dict[str, Any] = {"temperature": 0, "max_new_tokens": max_tokens}
        if ignore_eos:
            sp["ignore_eos"] = True
        status, j, dt = _post(self.base + "/generate", {"text": prompt, "sampling_params": sp},
                              self.timeout)
        if isinstance(j, list):
            j = j[0] if j else {}
        if status != 200:
            return Result(status, None, None, None, None, dt, str(j.get("error", j))[:2000])
        mi = j.get("meta_info") or {}
        return Result(status, j.get("output_ids"), j.get("text"), mi.get("prompt_tokens"),
                      mi.get("cached_tokens"), dt)

    def reset_prefix_cache(self) -> bool:
        status, _, _ = _post(self.base + "/flush_cache", {}, 120)
        return status == 200


def make_client(engine: str, host: str, port: int) -> EngineClient:
    if engine == "vllm":
        return VLLMClient(host, port)
    if engine == "sglang":
        return SGLangClient(host, port)
    raise ValueError(f"unknown engine {engine!r}")


def first_divergence(a: Optional[list[int]], b: Optional[list[int]]) -> Optional[dict[str, Any]]:
    """Index and tokens of the first difference, or None when identical."""
    a = a or []
    b = b or []
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"index": i, "a": x, "b": y}
    if len(a) != len(b):
        return {"index": min(len(a), len(b)), "len_a": len(a), "len_b": len(b)}
    return None
