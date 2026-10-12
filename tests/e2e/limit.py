# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""A kvcached server's memory limit is lowered while its prefix cache holds
most of its KV memory, then raised again.

The limit is set with `kvctl limit` next to the server:
  1. the prefix cache is filled with distinct long prompts;
  2. the limit is lowered to a quarter of what the cache grew, and the idle
     server must evict its prefix cache down to it;
  3. short and long requests must succeed without exceeding it;
  4. the limit is restored, and the server must grow past the lower one.
Greedy probes before and after must give the same tokens.
"""

from __future__ import annotations

import fcntl
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from e2e import client as cl
from e2e.elastic import MIB, SEGMENT, Usage
from e2e.matrix import Case, LimitTest
from e2e.run import CaseResult, Container, Server, check_gpu_released, launch, stop_and_check

FILL_REQUESTS = 24
FILL_WORDS = 1500
SHORT_REQUESTS = 100
SHORT_WORDS = 150
LONG_REQUESTS = 4
LONG_WORDS = 1200
# How long the idle server may take to evict down to a lowered limit.
IDLE_TIMEOUT = 30

# For setup.json (the CI results issue shows it); keep in step with run_limit.
WORKLOAD = (
    "one kvcached server, contiguous KV layout",
    f"fill: {FILL_REQUESTS} distinct {FILL_WORDS}-word filler prompts with 1 output token, "
    "8 at a time",
    "`kvctl limit` lowers the server's limit to its idle usage plus a quarter of what the "
    "fill added",
    f"load under the lower limit: {SHORT_REQUESTS} distinct {SHORT_WORDS}-word prompts with "
    f"16 output tokens, 8 at a time, then {LONG_REQUESTS} distinct {LONG_WORDS}-word prompts "
    "with 16 output tokens, one at a time",
    "the limit is restored and the fill is repeated with new prompts",
    "3 greedy probe prompts of 32 tokens before and after",
)
CHECKS = {
    "<server>_ready": "the server answers /health within the ready timeout",
    "limit_set": "`kvctl limit` succeeds, when lowering and when restoring the limit",
    "limit_cache_filled": "every fill request succeeds and the KV usage grows",
    "limit_released_idle": f"within {IDLE_TIMEOUT} s of lowering the limit, with no "
                           "requests, the KV usage is within it",
    "limit_requests_ok": "every request under the lower limit succeeds",
    "limit_held_under_load": "the KV usage stays within the lower limit under that load",
    "limit_raised_regrows": "after the limit is restored, the KV usage grows past the "
                            "lower one",
    "<server>_probe_unchanged": "the probes give the same tokens before and after",
    "<server>_alive": "the server is still running at the end",
    "<server>_<shutdown check>": "the shutdown and cleanup checks of the server",
}


def read_limit(ipc: str) -> int:
    with open(f"/dev/shm/{ipc}", "rb") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        return SEGMENT.unpack(f.read(SEGMENT.size))[0]


def set_limit(ct: Container, ipc: str, limit: int) -> str:
    """Run `kvctl limit` where the server runs: its segment belongs to the
    server's user. Return the error, if any."""
    proc = ct.exec(f"kvctl limit {shlex.quote(ipc)} {int(limit)}")
    return "" if proc.returncode == 0 else proc.stderr.decode(errors="replace")[-300:]


def load(srv: Server, pool: ThreadPoolExecutor, n: int, words: int, max_tokens: int,
         seed: int) -> list[cl.Result]:
    futures = [pool.submit(srv.client.generate, cl.long_prompt(words, seed + i), max_tokens,
                           True) for i in range(n)]
    return [f.result() for f in futures]


def failures(results: list[cl.Result]) -> str:
    return f"{sum(not r.ok for r in results)}/{len(results)} failed"


def run_limit(ct: Container, test: LimitTest, run_id: str, port: int,
              timeout: float) -> CaseResult:
    res = CaseResult(test.name)
    case = Case(test.engine, test.model, "c1", tag="limit")
    srv: Optional[Server] = None
    usage: Optional[Usage] = None
    try:
        srv = launch(ct, case, run_id, port)
        res.log["setup"] = {"kind": "limit", **srv.setup}
        ready = srv.client.wait_ready(timeout, lambda: ct.alive(srv.pid))
        res.add(f"{case.name}_ready", ready)
        if not ready:
            return res
        probes = cl.BASE_PROMPTS[:3]
        base = srv.client.run_phase("probe", probes, 32, 1)
        usage = Usage(ct.gpu)
        mem: dict[str, float] = {}
        res.log["memory_mib"] = mem

        with ThreadPoolExecutor(max_workers=8) as pool:
            idle = usage.used(srv)
            results = load(srv, pool, FILL_REQUESTS, FILL_WORDS, 1, 50_000)
            filled = usage.used(srv)
            res.add("limit_cache_filled", all(r.ok for r in results) and filled > idle,
                    f"{failures(results)}; KV grew from {idle / MIB:.0f} to "
                    f"{filled / MIB:.0f} MiB")

            limit = idle + (filled - idle) // 4
            original = read_limit(srv.ipc)
            error = set_limit(ct, srv.ipc, limit)
            res.add("limit_set", not error, error)
            if error:
                return res
            start = time.time()
            released = usage.used(srv)
            while released > limit and time.time() - start < IDLE_TIMEOUT:
                time.sleep(1)
                released = usage.used(srv)
            res.add("limit_released_idle", released <= limit,
                    f"{released / MIB:.0f} MiB after {time.time() - start:.0f} s, "
                    f"limit {limit / MIB:.0f} MiB")

            start = time.time()
            results = load(srv, pool, SHORT_REQUESTS, SHORT_WORDS, 16, 60_000)
            with ThreadPoolExecutor(max_workers=1) as one:
                results += load(srv, one, LONG_REQUESTS, LONG_WORDS, 16, 70_000)
            # Samples are a second apart; a short load can end between two.
            peak = max(usage.peak(srv, start), usage.used(srv))
            res.add("limit_requests_ok", all(r.ok for r in results), failures(results))
            res.add("limit_held_under_load", peak <= limit,
                    f"peak {peak / MIB:.0f} MiB, limit {limit / MIB:.0f} MiB")

            error = set_limit(ct, srv.ipc, original)
            res.add("limit_set", not error, error)
            start = time.time()
            results = load(srv, pool, FILL_REQUESTS, FILL_WORDS, 1, 80_000)
            regrown = max(usage.peak(srv, start), usage.used(srv))
            res.add("limit_raised_regrows", all(r.ok for r in results) and regrown > limit,
                    f"{failures(results)}; peak {regrown / MIB:.0f} MiB")
            mem.update(idle=idle / MIB, filled=filled / MIB, limit=limit / MIB,
                       released=released / MIB, peak_under_limit=peak / MIB,
                       regrown=regrown / MIB)

        after = srv.client.run_phase("probe", probes, 32, 1)
        same = all(x.token_ids == y.token_ids for x, y in zip(base.results, after.results))
        res.add(f"{case.name}_probe_unchanged", after.all_ok and same)
        res.add(f"{case.name}_alive", ct.alive(srv.pid))
        return res
    finally:
        if usage is not None:
            usage.stop()
        if srv is not None:
            stop_and_check(ct, srv, res, prefix=f"{case.name}_")
        check_gpu_released(res, ct.gpus)
