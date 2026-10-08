#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Serving throughput and latency of vLLM with kvcached against native vLLM.

Runs on a GPU host with vLLM and kvcached installed (the weekly CI uses an
H100 sandbox built from the vLLM image). For each model the native and the
kvcached server alternate in ABBA order; each start runs `vllm bench serve`
on random 1024-token prompts with 512 output tokens at concurrency 1, 32 and
128. The default vLLM configuration is used, Inductor included, as users run
it. Results are the median of the two runs of each arm.

Usage:
    python3 benchmarks/ci/perf.py --out /tmp/perf [--models gemma3_270m qwen38_27b]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

# Revisions are pinned so that every week serves the same files.
MODELS: dict[str, dict[str, Any]] = {
    # Sliding-window layers free KV blocks on every decode step, and each step
    # is short: the configuration most sensitive to kvcached's CPU overhead.
    # An ungated copy of google/gemma-3-270m-it, so the job needs no HF token.
    "gemma3_270m": dict(hf_id="unsloth/gemma-3-270m-it",
                        revision="23cf460f6bb16954176b3ddcc8d4f250501458a9",
                        args=["--max-model-len", "2048"]),
    "qwen38_27b": dict(hf_id="Qwen/Qwen3.8-27B",
                       revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                       # The default 1024 sequences exceed the Mamba state slots
                       # that fit next to the weights; 128 is the top concurrency.
                       args=["--max-model-len", "4096", "--max-num-seqs", "128"]),
    # Small, for trying the script; not part of the weekly run.
    "qwen05b": dict(hf_id="Qwen/Qwen2.5-0.5B-Instruct",
                    revision="7ae557604adf67be50417f59c2c2f167def9a775",
                    args=["--max-model-len", "2048"]),
}
WEEKLY = ("gemma3_270m", "qwen38_27b")
ARMS = ("native", "kvcached", "kvcached", "native")
CONCURRENCY = {1: 16, 32: 320, 128: 640}  # concurrency -> number of prompts
WARMUP = (32, 64)  # concurrency, prompts
SERVER_ARGS = ["--served-model-name", "perf", "--host", "127.0.0.1", "--seed", "0",
               "--gpu-memory-utilization", "0.9"]
BENCH_ARGS = ["--backend", "vllm", "--served-model-name", "perf", "--dataset-name", "random",
              "--random-input-len", "1024", "--random-output-len", "512", "--ignore-eos",
              "--request-rate", "inf", "--seed", "0", "--percentile-metrics", "ttft,tpot,itl,e2el",
              "--metric-percentiles", "50,99"]
METRICS = ("request_throughput", "output_throughput", "mean_ttft_ms", "median_ttft_ms",
           "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms")
PORT = 18100


def model_path(model: str) -> str:
    """Local snapshot of the model's pinned revision, downloaded if needed."""
    from huggingface_hub import snapshot_download

    spec = MODELS[model]
    return snapshot_download(spec["hf_id"], revision=spec["revision"])


def arm_env(arm: str, ipc: str) -> dict[str, str]:
    env = dict(os.environ)
    if arm == "kvcached":
        env.update(ENABLE_KVCACHED="true", KVCACHED_AUTOPATCH="1", KVCACHED_IPC_NAME=ipc)
    else:
        env.update(ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0")
    return env


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                          "-i", "0"], capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


class Sampler:
    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, gpu_used_mib())
            time.sleep(0.5)

    def stop(self) -> int:
        self._stop.set()
        self._t.join()
        return self.peak


def bench(path: str, concurrency: int, prompts: int, out: Path) -> dict[str, Any]:
    cmd = ["vllm", "bench", "serve", "--model", path, "--base-url", f"http://127.0.0.1:{PORT}",
           *BENCH_ARGS, "--num-prompts", str(prompts), "--max-concurrency", str(concurrency),
           "--save-result", "--result-dir", str(out.parent), "--result-filename", out.name]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out.with_suffix(".log")).write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"vllm bench serve failed; see {out.with_suffix('.log')}")
    return json.loads(out.read_text())


def run_server(model: str, arm: str, run: int, out: Path, ready_timeout: float) -> dict[str, Any]:
    spec = MODELS[model]
    path = model_path(model)
    tag = f"{model}-{run}-{arm}"
    log = out / f"{tag}-serve.log"
    cmd = ["vllm", "serve", path, *SERVER_ARGS, "--port", str(PORT), *spec["args"]]
    (out / f"{tag}-cmd.txt").write_text(shlex.join(cmd) + "\n")
    with open(log, "w") as f:
        server = subprocess.Popen(cmd, env=arm_env(arm, f"perf_{model}_{run}"), stdout=f,
                                  stderr=subprocess.STDOUT, start_new_session=True)
    result: dict[str, Any] = {"model": model, "arm": arm, "run": run}
    try:
        deadline = time.time() + ready_timeout
        while not healthy():
            if server.poll() is not None or time.time() > deadline:
                raise RuntimeError(f"{tag} server did not become ready; see {log}")
            time.sleep(5)
        idle = gpu_used_mib()
        # Warm up so compilation and graph capture stay out of the numbers.
        bench(path, *WARMUP, out / f"{tag}-warmup.json")
        for concurrency, prompts in CONCURRENCY.items():
            sampler = Sampler()
            data = bench(path, concurrency, prompts, out / f"{tag}-c{concurrency}.json")
            peak = sampler.stop()
            result[f"c{concurrency}"] = {m: data.get(m) for m in METRICS}
            result[f"c{concurrency}"]["peak_gpu_mib"] = peak
            print(f"[{time.strftime('%H:%M:%S')}] {tag} c={concurrency}: "
                  f"{data.get('output_throughput', 0):.0f} tok/s, "
                  f"TPOT {data.get('mean_tpot_ms', 0):.2f} ms", flush=True)
        time.sleep(10)
        result["idle_gpu_mib"] = idle
        result["after_load_gpu_mib"] = gpu_used_mib()
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGINT)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
    return result


def summarize(runs: list[dict[str, Any]], errors: list[str], out: Path) -> str:
    lines = ["| model | concurrency | metric | native | kvcached | kvcached / native |",
             "|---|---|---|---|---|---|"]
    table = []
    for model in dict.fromkeys(r["model"] for r in runs):
        for c in CONCURRENCY:
            for metric in ("output_throughput", "mean_tpot_ms", "p99_tpot_ms", "mean_ttft_ms"):
                vals = {}
                for arm in ("native", "kvcached"):
                    xs = [r[f"c{c}"][metric] for r in runs
                          if r["model"] == model and r["arm"] == arm and f"c{c}" in r]
                    vals[arm] = statistics.median(xs) if xs else None
                n, k = vals["native"], vals["kvcached"]
                ratio = k / n if n and k else None
                table.append(dict(model=model, concurrency=c, metric=metric, native=n,
                                  kvcached=k, ratio=ratio))
                fmt = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
                lines.append(f"| {model} | {c} | {metric} | {fmt(n)} | {fmt(k)} | {fmt(ratio)} |")
    (out / "summary.json").write_text(json.dumps({"runs": runs, "table": table,
                                                  "errors": errors}, indent=1))
    md = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(md)
    return md


def write_setup(out: Path, models: list[str]) -> None:
    """Record what this run measured, for the CI results issue (tools/ci/report.py)."""
    import importlib.metadata as md

    import torch

    def version(package: str) -> Optional[str]:
        try:
            return md.version(package)
        except md.PackageNotFoundError:
            return None

    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout
    rows = [[x.strip() for x in line.split(",")] for line in gpus.splitlines()]
    setup = {
        "kvcached_sha": os.environ.get("KVCACHED_SHA"),
        "gpus": [dict(zip(("name", "memory", "driver"), r)) for r in rows if len(r) == 3],
        "versions": {"vllm": version("vllm"), "kvcached": version("kvcached"),
                     "torch": torch.__version__, "cuda": torch.version.cuda},
        "models": {m: MODELS[m] for m in models},
        "server": shlex.join(["vllm", "serve", "MODEL", *SERVER_ARGS, "--port", str(PORT)]),
        "kvcached_env": {k: v for k, v in arm_env("kvcached", "PER_START").items()
                         if os.environ.get(k) != v},
        "bench": shlex.join(["vllm", "bench", "serve", "--model", "MODEL", *BENCH_ARGS,
                             "--num-prompts", "N", "--max-concurrency", "C"]),
        "arms": list(ARMS),
        "concurrency": {str(c): n for c, n in CONCURRENCY.items()},
        "warmup": {"concurrency": WARMUP[0], "prompts": WARMUP[1]},
    }
    (out / "setup.json").write_text(json.dumps(setup, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", nargs="*", default=list(WEEKLY), choices=list(MODELS))
    ap.add_argument("--ready-timeout", type=float, default=1800)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    write_setup(out, a.models)
    runs: list[dict[str, Any]] = []
    errors: list[str] = []
    for model in a.models:
        for run, arm in enumerate(ARMS):
            try:
                runs.append(run_server(model, arm, run, out, a.ready_timeout))
            except Exception as e:
                errors.append(f"{model} {arm} run {run}: {e}")
                print(f"FAILED {errors[-1]}", flush=True)
    md = summarize(runs, errors, out)
    print(md)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(md)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
