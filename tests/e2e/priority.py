# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Two kvcached servers with different priorities on one GPU: the lower one
gives its memory back once the higher one posts its demand (#585).

L, the lower priority, fills the GPU with prefix cache and goes idle; H, the
higher priority, then gets a burst it does not have the memory for. The
watermark and reclaim must move the memory with no outside control, where
the limit case needs an operator's `kvctl limit`:
  1. `kvctl priority` sets L below H, before any load;
  2. L's load fills most of the free memory with prefix cache, which it
     keeps while idle (its retention cap is lifted at startup);
  3. H gets enough concurrent long requests to need most of what L holds on
     top of what is free: the ledger must show H's shortfall and L's
     watermark, L must release most of what it grew, and H must grow past
     what was free when its burst landed, which it can only do with the
     memory L gave back;
  4. every request of both servers must finish, and greedy probes before
     and after must give the same tokens on each server.
The ledger evidence comes from `kvctl list --json` sampled during the burst,
the reader kvcached provides; the file layout stays private to kvcached.
"""

from __future__ import annotations

import json
import math
import shlex
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Optional, Sequence

from e2e import client as cl
from e2e.elastic import (
    MAX_TOKENS,
    MIB,
    PROMPT_WORDS,
    TARGET,
    Usage,
    burst,
    drain,
    gpu_memory_mib,
    start_ballast,
)
from e2e.limit import failures, load
from e2e.matrix import BATCH_FLAG, ELASTIC_BATCH, Case, PriorityPair
from e2e.run import (
    CaseResult,
    Check,
    Container,
    Server,
    case_setup,
    check_gpu_released,
    gpu_info,
    launch,
    stop_and_check,
)

# Larger is more important; the values go to `kvctl priority`.
PRIORITY_L = 0
PRIORITY_H = 1
# The fill runs 8 requests at a time, like the limit case's, so L's cache,
# not its running requests, holds the memory. It can need more requests than
# the batch limit, since every finished prompt stays cached; the cap only
# bounds a miscalibrated run, and the ballast emulates a smaller GPU.
FILL_CONCURRENCY = 8
FILL_CAP = 4 * ELASTIC_BATCH
# Seconds between ledger samples; one sample is one kvctl run.
LEDGER_POLL = 2.0

# For setup.json (the CI results issue shows it); keep in step with run_priority.
WORKLOAD = (
    "two kvcached servers on one GPU, contiguous KV layout and a batch limit of "
    f"{ELASTIC_BATCH}; L starts first, then H",
    "L keeps every finished prompt: it starts with KVCACHED_MAX_CACHED_TOKENS=-1",
    f"`kvctl priority` sets L to {PRIORITY_L} and H to {PRIORITY_H} before any load",
    "calibration: 4 fill-shaped requests on L and 4 burst-shaped requests on H "
    "measure the KV memory one request holds",
    f"fill: distinct {PROMPT_WORDS}-word filler prompts with 1 output token, "
    f"{FILL_CONCURRENCY} at a time, sized to {TARGET:.0%} of the free memory; "
    "L then idles",
    f"burst: distinct {PROMPT_WORDS}-word filler prompts with {MAX_TOKENS} output "
    f"tokens and ignore_eos, all at once, sized to {TARGET:.0%} of the free memory "
    "plus what L holds",
    f"`kvctl list --json` is sampled every {LEDGER_POLL:.0f} s during the burst for "
    "H's shortfall and L's watermark",
    "3 greedy probe prompts of 32 tokens on each server before and after",
)
CHECKS = {
    "<server>_ready": "the server answers /health within the ready timeout",
    "priority_set": "`kvctl priority` succeeds, for L and for H",
    "l_requests_ok, h_requests_ok": "every request of the server's own load succeeds",
    "l_filled_memory": "L's KV usage grows by more than half of the free memory",
    "l_held_at_burst": "when H's burst lands, L still holds at least half of what it "
                       "grew, the pressure the next checks need",
    "h_shortfall_posted": "a ledger sample during the burst shows a shortfall posted "
                          "by H",
    "l_watermark_raised": "a ledger sample during the burst shows a watermark on L",
    "l_released_for_h": "after the burst, L holds less than half of what it grew",
    "h_reused_l_memory": "H's KV usage grows past what was free when its burst "
                         "landed",
    "<server>_probe_unchanged": "the probes give the same tokens before and after",
    "<server>_alive": "the server is still running at the end",
    "<server>_<shutdown check>": "the shutdown and cleanup checks of each server",
}


def set_priority(ct: Container, ipc: str, priority: int) -> str:
    """Run `kvctl priority` where the server runs: its segment and ledger
    belong to the server's user. kvctl prints the publish result and exits 0
    even when the publish was rejected, so only an accepted publish passes.
    Return the error, if any."""
    proc = ct.exec(f"kvctl priority {shlex.quote(ipc)} {int(priority)}")
    if proc.returncode != 0:
        return proc.stderr.decode(errors="replace")[-300:] or "kvctl failed"
    lines = proc.stdout.decode(errors="replace").strip().splitlines()
    try:
        result = json.loads(lines[-1]) if lines else {}
    except ValueError:
        result = {}
    if result.get("accepted") is True:
        return ""
    return lines[-1][-300:] if lines else "no kvctl output"


def ledger_stats(text: str, ipcs: Sequence[str]) -> dict[str, dict[str, int]]:
    """The case's ledger fields per instance, summed over its GPUs, from the
    JSON `kvctl list --json` prints. An instance the output does not carry (a
    dead server, a kvctl without these fields) is absent, and a missing field
    counts as 0, so the ledger checks fail on missing evidence, not an error."""
    try:
        entries = json.loads(text)
    except ValueError:
        return {}
    if not isinstance(entries, list):
        return {}
    stats: dict[str, dict[str, int]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("ipc") not in ipcs:
            continue
        gpus = [g for g in entry.get("gpus") or () if isinstance(g, dict)]
        stats[entry["ipc"]] = {
            field: sum(int(g.get(field, 0)) for g in gpus)
            for field in ("charged_bytes", "watermark_bytes", "shortfall_bytes")}
    return stats


class Ledger:
    """Sample both servers' ledger state during the burst via `kvctl list
    --json`, run where the servers run, as `kvctl priority` is."""

    def __init__(self, ct: Container, ipcs: Sequence[str]) -> None:
        self.ct = ct
        self.ipcs = tuple(ipcs)
        self.samples: list[tuple[float, dict[str, dict[str, int]]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        cmd = f"kvctl list --json {shlex.join(self.ipcs)}"
        while not self._stop.is_set():
            proc = self.ct.exec(cmd, timeout=60)
            text = proc.stdout.decode(errors="replace") if proc.returncode == 0 else ""
            self.samples.append((time.time(), ledger_stats(text, self.ipcs)))
            self._stop.wait(LEDGER_POLL)

    def peak(self, ipc: str, field: str, since: float = 0.0) -> int:
        return max((s.get(ipc, {}).get(field, 0) for t, s in self.samples if t >= since),
                   default=0)

    def timeline(self) -> list[dict[str, Any]]:
        """Per-sample ledger stats in MiB, for the case log."""
        return [{"t": round(t, 1),
                 **{ipc: {k: v >> 20 for k, v in s.get(ipc, {}).items()}
                    for ipc in self.ipcs}} for t, s in self.samples]

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def priority_checks(mem: Mapping[str, float], shortfall_h: int,
                    watermark_l: int) -> list[Check]:
    """Whether the memory and the ledger moved as #585's watermark semantics
    say. `mem` is in bytes: L's usage idle (l_idle), what the fill grew it
    (l_growth), at the burst (l_at_burst) and after it (l_after), H's growth
    during the burst (h_growth), and the GPU memory free when the burst
    landed (free_at_burst). The ledger peaks are H's posted shortfall and
    L's watermark during the burst, 0 when no sample carried one."""
    held = mem["l_at_burst"] - mem["l_idle"]
    kept = mem["l_after"] - mem["l_idle"]
    grew = f"of the {mem['l_growth'] / MIB:.0f} MiB it grew"
    return [
        Check("l_held_at_burst", held >= 0.5 * mem["l_growth"],
              f"L held {held / MIB:.0f} MiB {grew} when H's burst landed"),
        Check("h_shortfall_posted", shortfall_h > 0,
              f"H's largest sampled shortfall is {shortfall_h / MIB:.0f} MiB"),
        Check("l_watermark_raised", watermark_l > 0,
              f"L's largest sampled watermark is {watermark_l / MIB:.0f} MiB"),
        Check("l_released_for_h", kept < 0.5 * mem["l_growth"],
              f"L still holds {kept / MIB:.0f} MiB {grew}"),
        Check("h_reused_l_memory", mem["h_growth"] > mem["free_at_burst"],
              f"H grew {mem['h_growth'] / MIB:.0f} MiB; {mem['free_at_burst'] / MIB:.0f} "
              "MiB was free when its burst landed"),
    ]


def latency_ms(results: Sequence[cl.Result]) -> dict[str, float]:
    """p50 and max request latency, for the per-server service record."""
    seconds = [r.latency_s for r in results] or [0.0]
    return {"p50": round(statistics.median(seconds) * 1000, 1),
            "max": round(max(seconds) * 1000, 1)}


def run_priority(ct_l: Container, ct_h: Container, pair: PriorityPair, run_id: str,
                 base_port: int, timeout: float, emulate_gpu_mib: int = 0) -> CaseResult:
    """Server L runs in ct_l and server H in ct_h, both on the run's GPU
    pair.gpu, where memory is sampled."""
    res = CaseResult(pair.name)
    if pair.gpu >= len(gpu_info(ct_l.gpus)):
        res.add("gpus_available", False, f"the case needs GPU {pair.gpu} of the run")
        return res
    gpu = ct_l.host_gpu(pair.gpu)
    ballast: Optional[str] = None
    if emulate_gpu_mib:
        _, total = gpu_memory_mib(gpu)
        if total > emulate_gpu_mib:
            ballast = start_ballast(ct_l, total - emulate_gpu_mib, pair.gpu)
            res.log["ballast_mib"] = total - emulate_gpu_mib
    case_l = Case(pair.engine_l, pair.model_l, "c1", tag="priority-lo")
    case_h = Case(pair.engine_h, pair.model_h, "c1", tag="priority-hi")
    setup: dict[str, Any] = {"kind": "priority", "gpu": gpu, "batch": ELASTIC_BATCH,
                             "priorities": {"l": PRIORITY_L, "h": PRIORITY_H},
                             "l": case_setup(case_l), "h": case_setup(case_h)}
    res.log["setup"] = setup
    started: list[tuple[Container, Server]] = []
    usage: Optional[Usage] = None
    ledger: Optional[Ledger] = None
    try:
        # Start one after the other so the second sees the first's weights.
        # L keeps every finished prompt: the fill, not a retention cap, must
        # decide how much its idle cache holds.
        for host, case, port, extra, env in (
                (ct_l, case_l, base_port + 1, pair.extra_args_l,
                 {**ct_l.gpu_env(pair.gpu), "KVCACHED_MAX_CACHED_TOKENS": "-1"}),
                (ct_h, case_h, base_port + 2, pair.extra_args_h,
                 ct_h.gpu_env(pair.gpu))):
            batch = (BATCH_FLAG[case.engine], str(ELASTIC_BATCH))
            srv = launch(host, case, run_id, port, extra_args=(*batch, *extra),
                         extra_env=env)
            setup["l" if case is case_l else "h"] = srv.setup
            started.append((host, srv))
            ready = srv.client.wait_ready(timeout, lambda: host.alive(srv.pid))
            res.add(f"{case.name}_ready", ready)
            if not ready:
                return res
        (_, low), (_, high) = started
        for (host, srv), prio in zip(started, (PRIORITY_L, PRIORITY_H)):
            error = set_priority(host, srv.ipc, prio)
            res.add("priority_set", not error, error)
            if error:
                return res
        probes = cl.BASE_PROMPTS[:3]
        base = {s.ipc: s.client.run_phase("probe", probes, 32, 1) for _, s in started}
        usage = Usage(gpu, (low.ipc, high.ipc))
        mem: dict[str, float] = {}
        res.log["memory_mib"] = mem
        lat: dict[str, dict[str, float]] = {}
        res.log["latency_ms"] = lat

        with ThreadPoolExecutor(max_workers=4 * ELASTIC_BATCH) as pool, \
                ThreadPoolExecutor(max_workers=FILL_CONCURRENCY) as fill_pool:
            # Calibrate L with the fill shape: a finished prompt stays in its
            # cache, so 4 requests' growth sizes one. H gets the burst shape,
            # requests that hold their KV until they finish.
            before = usage.used(low)
            load(low, fill_pool, 4, PROMPT_WORDS, 1, 1_000)
            per_fill = max(usage.used(low) - before, MIB) / 4
            drain(usage, low)
            before = usage.used(high)
            t = time.time()
            burst(high, pool, 4, 2_000)
            per_burst = max(usage.peak(high, t) - before, MIB) / 4
            drain(usage, high)

            # 1. L fills most of the free memory with prefix cache and idles.
            idle_gpu, total = gpu_memory_mib(gpu)
            free = (total - idle_gpu) * MIB
            l_idle = usage.used(low)
            n_fill = min(math.ceil(TARGET * free / per_fill), FILL_CAP)
            results = load(low, fill_pool, n_fill, PROMPT_WORDS, 1, 10_000)
            l_growth = usage.used(low) - l_idle
            lat["l_fill"] = latency_ms(results)
            res.add("l_requests_ok", all(r.ok for r in results), failures(results))
            res.add("l_filled_memory", l_growth > 0.5 * free,
                    f"L grew {l_growth / MIB:.0f} MiB of {free / MIB:.0f} MiB free")

            # 2. H's burst needs most of what L holds on top of what is
            # free. Its unfit requests post H's demand, the ledger shows it
            # as H's shortfall and L's watermark, and L, idle, must give its
            # cache back, which H maps. The burst-phase offsets locate this
            # phase in the servers' logs, as the elastic case's offset does.
            for name, case, ct in (("l", case_l, ct_l), ("h", case_h, ct_h)):
                log_path = ct.out_dir / case.name / "serve.log"
                res.log[f"{name}_log_burst_offset"] = (log_path.stat().st_size
                                                       if log_path.exists() else 0)
            burst_gpu, _ = gpu_memory_mib(gpu)
            free_at_burst = (total - burst_gpu) * MIB
            l_at_burst = usage.used(low)
            h_idle = usage.used(high)
            n_burst = min(math.ceil(
                TARGET * (free_at_burst + (l_at_burst - l_idle)) / per_burst),
                ELASTIC_BATCH)
            ledger = Ledger(ct_l, (low.ipc, high.ipc))
            t_burst = time.time()
            results = burst(high, pool, n_burst, 20_000)
            ledger.stop()
            h_growth = usage.peak(high, t_burst) - h_idle
            l_after = usage.used(low)
            lat["h_burst"] = latency_ms(results)
            res.add("h_requests_ok", all(r.ok for r in results), failures(results))
            mem.update(free=free / MIB, per_fill=per_fill / MIB,
                       per_burst=per_burst / MIB, n_fill=n_fill, n_burst=n_burst,
                       l_growth=l_growth / MIB, l_held=(l_at_burst - l_idle) / MIB,
                       l_kept=(l_after - l_idle) / MIB, h_growth=h_growth / MIB,
                       free_at_burst=free_at_burst / MIB)
            res.checks += priority_checks(
                {"l_idle": l_idle, "l_growth": l_growth, "l_at_burst": l_at_burst,
                 "l_after": l_after, "h_growth": h_growth,
                 "free_at_burst": free_at_burst},
                ledger.peak(high.ipc, "shortfall_bytes", t_burst),
                ledger.peak(low.ipc, "watermark_bytes", t_burst))
            res.log["ledger_mib"] = ledger.timeline()
            drain(usage, high)

        for host, srv in started:
            after = srv.client.run_phase("probe", probes, 32, 1)
            same = all(x.token_ids == y.token_ids
                       for x, y in zip(base[srv.ipc].results, after.results))
            res.add(f"{srv.case.name}_probe_unchanged", after.all_ok and same)
            res.add(f"{srv.case.name}_alive", host.alive(srv.pid))
        return res
    finally:
        if ledger is not None:
            ledger.stop()
        if usage is not None:
            usage.stop()
            # Per-second timeline, MiB: GPU memory used and each server's KV.
            timeline = [{"t": round(t, 1), "gpu": g,
                         **{srv.case.name: usage.of(srv, u) >> 20 for _, srv in started}}
                        for t, g, u in usage.samples]
            (ct_l.out_dir / f"{pair.name}-usage.json").write_text(json.dumps(timeline))
        for host, srv in started:
            stop_and_check(host, srv, res, prefix=f"{srv.case.name}_")
        if ballast is not None:
            ct_l.stop_server(ballast)
        check_gpu_released(res, ct_l.gpus)
