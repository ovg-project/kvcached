# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Two kvcached servers on one GPU: memory one releases, the other can use.

Each server's KV usage is read from its kvcached segments in /dev/shm, and the
load is sized from a calibration burst, so the checks do not depend on how
fast requests finish:
  1. A gets enough concurrent long requests to need most of the free memory;
  2. A finishes, its prefix cache is reset, and it must give at least half
     of that memory back;
  3. B gets its own such load, and must grow past what was free while A was
     at its peak, which it can only do with the memory A gave back;
  4. both get that load at once, more than fits: requests may queue, but all
     must finish and neither server may crash.
Greedy probes before and after must give the same tokens on each server.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import struct
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from e2e import client as cl
from e2e.matrix import BATCH_FLAG, ELASTIC_BATCH, Case, ElasticPair
from e2e.run import (
    CaseResult,
    Container,
    Server,
    case_setup,
    check_gpu_released,
    launch,
    stop_and_check,
)

# Distinct prompts, so the prefix cache shares nothing between requests, and
# ignore_eos, so every request holds its KV until it reaches MAX_TOKENS.
PROMPT_WORDS = 3000
MAX_TOKENS = 1024
# Each engine's load is sized to need this fraction of the free memory.
TARGET = 0.7
MIB = 1 << 20
SEGMENT = struct.Struct("=3q")

# For setup.json (the CI results issue shows it); keep in step with run_elastic.
WORKLOAD = (
    f"both servers use the contiguous KV layout and a batch limit of {ELASTIC_BATCH}; "
    "A starts first, then B, on the same GPU",
    f"every load request is a distinct {PROMPT_WORDS}-word filler prompt with "
    f"{MAX_TOKENS} output tokens and ignore_eos",
    "calibration: 4 such requests per server measure the KV memory one request holds",
    f"each server's load is sized to need {TARGET:.0%} of the GPU memory free after "
    f"calibration, at most {ELASTIC_BATCH} requests",
    "phases: (1) A gets its load; (2) A's prefix cache is reset; (3) B gets its load and "
    "its prefix cache is reset; (4) both get their loads at the same time",
    "3 greedy probe prompts of 32 tokens on each server before and after",
)
CHECKS = {
    "<server>_ready": "the server answers /health within the ready timeout",
    "a_requests_ok, b_requests_ok": "every request of the server's own load succeeds",
    "a_filled_memory": "A's KV usage grows by more than half of the free memory",
    "a_released_memory": "after its prefix cache is reset, A keeps less than half of "
                         "what it grew",
    "b_reused_a_memory": "B's KV usage grows past what was free at A's peak",
    "contended_requests_ok": "every request succeeds while both loads run",
    "<server>_probe_unchanged": "the probes give the same tokens before and after",
    "<server>_alive": "the server is still running at the end",
    "<server>_<shutdown check>": "the shutdown and cleanup checks of each server",
}


def gpu_memory_mib(gpu: str) -> tuple[int, int]:
    out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True)
    used, total = (int(x) for x in out.stdout.strip().split(","))
    return used, total


def read_segments() -> dict[str, int]:
    """Used KV bytes of every kvcached segment in /dev/shm, read as kvctl does:
    three int64s (limit, used, preallocated) under a shared flock. /dev/shm
    is shared with the containers (--ipc=host), and this is much faster than
    running kvctl once a second."""
    used = {}
    for name in os.listdir("/dev/shm"):
        path = os.path.join("/dev/shm", name)
        try:
            if os.path.getsize(path) != SEGMENT.size:
                continue
            with open(path, "rb") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                limit, in_use, _ = SEGMENT.unpack(f.read(SEGMENT.size))
        except (OSError, struct.error):
            continue
        if limit > 0:
            used[name] = in_use
    return used


class Usage:
    """Sample the kvcached segments and the GPU's memory once a second."""

    def __init__(self, gpu: str) -> None:
        self.gpu = gpu
        self.samples: list[tuple[float, int, dict[str, int]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def of(srv: Server, segments: dict[str, int]) -> int:
        # A server can have one segment per KV cache group: <ipc>, <ipc>_g1, ...
        return sum(v for k, v in segments.items()
                   if k == srv.ipc or k.startswith(srv.ipc + "_g"))

    def used(self, srv: Server) -> int:
        return self.of(srv, read_segments())

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((time.time(), gpu_memory_mib(self.gpu)[0], read_segments()))
            time.sleep(1)

    def peak(self, srv: Server, since: float) -> int:
        return max((self.of(srv, u) for t, _, u in self.samples if t >= since), default=0)

    def gpu_peak_mib(self, since: float) -> int:
        return max((g for t, g, _ in self.samples if t >= since), default=0)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def burst(srv: Server, pool: ThreadPoolExecutor, n: int, seed: int) -> list[cl.Result]:
    """Send n distinct long requests at once and wait for all of them."""
    futures = [pool.submit(srv.client.generate, cl.long_prompt(PROMPT_WORDS, seed + i),
                           MAX_TOKENS, True) for i in range(n)]
    return [f.result() for f in futures]


def drain(usage: Usage, srv: Server, timeout: float = 60) -> int:
    """Reset the server's prefix cache, so that finished requests hold no KV,
    and wait until its usage stops falling; return it."""
    srv.client.reset_prefix_cache()
    last = usage.used(srv)
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(5)
        now = usage.used(srv)
        if now >= last:
            return now
        last = now
    return last


def start_ballast(ct: Container, mib: int) -> str:
    """Hold `mib` of GPU memory, to emulate a smaller GPU when testing locally."""
    (ct.out_dir / "elastic-ballast").mkdir(parents=True, exist_ok=True)
    code = ("import time, torch; "
            f"x = torch.empty({mib} << 20, dtype=torch.uint8, device='cuda'); "
            "print('ballast ready', flush=True); time.sleep(1e9)")
    pid = ct.start_server("elastic-ballast", {}, ["python3", "-c", code])
    deadline = time.time() + 120
    while time.time() < deadline:
        log = ct.out_dir / "elastic-ballast" / "serve.log"
        if log.exists() and "ballast ready" in log.read_text(errors="replace"):
            return pid
        time.sleep(1)
    raise RuntimeError("GPU ballast did not start")


def run_elastic(ct_a: Container, ct_b: Container, pair: ElasticPair, run_id: str,
                base_port: int, timeout: float, emulate_gpu_mib: int = 0) -> CaseResult:
    """Server A runs in ct_a and server B in ct_b; both use GPU ct_a.gpu."""
    res = CaseResult(pair.name)
    gpu = ct_a.gpu
    ballast: Optional[str] = None
    if emulate_gpu_mib:
        _, total = gpu_memory_mib(gpu)
        if total > emulate_gpu_mib:
            ballast = start_ballast(ct_a, total - emulate_gpu_mib)
            res.log["ballast_mib"] = total - emulate_gpu_mib
    case_a = Case(pair.engine_a, pair.model_a, "c1", tag="elastic")
    case_b = Case(pair.engine_b, pair.model_b, "c1", tag="elastic")
    setup = {"kind": "elastic", "gpu": gpu, "batch": ELASTIC_BATCH,
             "a": case_setup(case_a), "b": case_setup(case_b)}
    res.log["setup"] = setup
    started: list[tuple[Container, Server]] = []
    usage: Optional[Usage] = None
    try:
        # Start one after the other so the second sees the first's weights.
        for host, case, port, extra in ((ct_a, case_a, base_port + 1, pair.extra_args_a),
                                        (ct_b, case_b, base_port + 2, pair.extra_args_b)):
            batch = (BATCH_FLAG[case.engine], str(ELASTIC_BATCH))
            srv = launch(host, case, run_id, port, extra_args=(*batch, *extra))
            setup["a" if case is case_a else "b"] = srv.setup
            started.append((host, srv))
            ready = srv.client.wait_ready(timeout, lambda: host.alive(srv.pid))
            res.add(f"{case.name}_ready", ready)
            if not ready:
                return res
        (_, a), (_, b) = started
        probes = cl.BASE_PROMPTS[:3]
        base = {s.ipc: s.client.run_phase("probe", probes, 32, 1) for _, s in started}
        usage = Usage(gpu)
        mem: dict[str, float] = {}
        res.log["memory_mib"] = mem
        res.log["segments"] = {k: v for k, v in read_segments().items()
                               if k.startswith((a.ipc, b.ipc))}

        with ThreadPoolExecutor(max_workers=4 * ELASTIC_BATCH) as pool:
            # Calibrate: the KV memory one request of this shape holds.
            per_request = {}
            for srv in (a, b):
                before = usage.used(srv)
                t = time.time()
                burst(srv, pool, 4, 1_000)
                per_request[srv.ipc] = max(usage.peak(srv, t) - before, MIB) / 4
                drain(usage, srv)
            idle_gpu, total = gpu_memory_mib(gpu)
            free = (total - idle_gpu) * MIB
            # More than the batch limit would only queue.
            n = {s.ipc: min(math.ceil(TARGET * free / per_request[s.ipc]), ELASTIC_BATCH)
                 for s in (a, b)}
            mem.update(free=free / MIB, per_request_a=per_request[a.ipc] / MIB,
                       per_request_b=per_request[b.ipc] / MIB, n_a=n[a.ipc], n_b=n[b.ipc])

            # 1. A takes most of the free memory.
            a_idle = usage.used(a)
            t = time.time()
            results = burst(a, pool, n[a.ipc], 10_000)
            a_growth = usage.peak(a, t) - a_idle
            free_at_a_peak = (total - usage.gpu_peak_mib(t)) * MIB
            res.add("a_requests_ok", all(r.ok for r in results),
                    f"{sum(not r.ok for r in results)}/{len(results)} failed")
            res.add("a_filled_memory", a_growth > 0.5 * free,
                    f"A grew {a_growth / MIB:.0f} MiB of {free / MIB:.0f} MiB free")

            # 2. A gives its KV memory back.
            a_after = drain(usage, a)
            mem["gpu_free_after_a"] = total - gpu_memory_mib(gpu)[0]
            res.add("a_released_memory", a_after - a_idle < 0.5 * a_growth,
                    f"A still holds {(a_after - a_idle) / MIB:.0f} of the "
                    f"{a_growth / MIB:.0f} MiB it grew")

            # 3. B grows past what was free while A was at its peak.
            b_idle = usage.used(b)
            t = time.time()
            results = burst(b, pool, n[b.ipc], 20_000)
            b_growth = usage.peak(b, t) - b_idle
            res.add("b_requests_ok", all(r.ok for r in results),
                    f"{sum(not r.ok for r in results)}/{len(results)} failed")
            res.add("b_reused_a_memory", b_growth > free_at_a_peak,
                    f"B grew {b_growth / MIB:.0f} MiB; {free_at_a_peak / MIB:.0f} MiB "
                    "was free at A's peak")
            drain(usage, b)

            # 4. Both at once.
            fa = pool.submit(burst, a, pool, n[a.ipc], 30_000)
            fb = pool.submit(burst, b, pool, n[b.ipc], 40_000)
            results = fa.result() + fb.result()
            res.add("contended_requests_ok", all(r.ok for r in results),
                    f"{sum(not r.ok for r in results)}/{len(results)} failed")
            mem.update(a_growth=a_growth / MIB, a_after=(a_after - a_idle) / MIB,
                       free_at_a_peak=free_at_a_peak / MIB, b_growth=b_growth / MIB)

        for host, srv in started:
            after = srv.client.run_phase("probe", probes, 32, 1)
            same = all(x.token_ids == y.token_ids
                       for x, y in zip(base[srv.ipc].results, after.results))
            res.add(f"{srv.case.name}_probe_unchanged", after.all_ok and same)
            res.add(f"{srv.case.name}_alive", host.alive(srv.pid))
        return res
    finally:
        if usage is not None:
            usage.stop()
            # Per-second timeline, MiB: GPU memory used and each server's KV.
            timeline = [{"t": round(t, 1), "gpu": g,
                         **{srv.case.name: usage.of(srv, u) >> 20 for _, srv in started}}
                        for t, g, u in usage.samples]
            (ct_a.out_dir / f"{pair.name}-usage.json").write_text(json.dumps(timeline))
        for host, srv in started:
            stop_and_check(host, srv, res, prefix=f"{srv.case.name}_")
        if ballast is not None:
            ct_a.stop_server(ballast)
        check_gpu_released(res)
