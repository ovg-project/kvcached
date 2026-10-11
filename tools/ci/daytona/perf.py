#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Run the weekly H100 job in a Daytona GPU sandbox and fetch its results.

Creates an ephemeral H100 sandbox from the vLLM image, uploads this checkout,
runs tools/ci/daytona/sandbox_job.sh in the background while streaming its
log, downloads the results and stops the sandbox. Stopping an ephemeral
sandbox deletes it; a TTL ends it if this script never gets that far.

Needs DAYTONA_API_KEY with sandbox scopes. All models are ungated; HF_TOKEN is
passed on when set.

Usage:
    python3 tools/ci/daytona/perf.py --out results [--models qwen05b --skip-correctness]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from daytona import CreateSandboxFromImageParams, Daytona, GpuType, Image, Resources

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests"))
from e2e.matrix import ENGINES  # noqa: E402

SRC = "/tmp/kvcached-src"
OUT = "/tmp/kvcached-ci-out"
LOG = "/tmp/kvcached-ci.log"
RC = "/tmp/kvcached-ci.rc"


def create_sandbox(d: Daytona, ttl_minutes: int, wait_minutes: int):
    params = CreateSandboxFromImageParams(
        name=f"kvcached-ci-perf-{int(time.time())}",
        image=Image.base(ENGINES["vllm"].image).entrypoint(["sleep", "infinity"]),
        resources=Resources(cpu=16, memory=128, disk=300, gpu=1, gpu_type=[GpuType.H100]),
        ephemeral=True,
        auto_stop_interval=0,
        ttl_minutes=ttl_minutes,
        env_vars={k: v for k, v in os.environ.items() if k == "HF_TOKEN" and v},
        labels={"kvcached-ci": "perf"},
    )
    # The organization has no server-side queueing, so wait for a free GPU here.
    deadline = time.time() + wait_minutes * 60
    while True:
        try:
            return d.create(params, timeout=3600,
                            on_snapshot_create_logs=lambda line: print("[image]", line))
        except Exception as e:
            msg = str(e)
            permanent = any(s in msg for s in ("credits", "Forbidden", "Unauthorized",
                                               "Access denied", "not enabled"))
            if permanent or time.time() > deadline:
                raise
            print(f"Sandbox not created, retrying in 5 minutes: {msg[:300]}", flush=True)
            time.sleep(300)


def run(sb, cmd: str, timeout: int = 600) -> str:
    r = sb.process.exec(cmd, timeout=timeout)
    if r.exit_code != 0:
        raise RuntimeError(f"`{cmd}` failed ({r.exit_code}): {r.result[-2000:]}")
    return r.result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", nargs="*", default=["gemma3_270m", "qwen38_27b"])
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--ttl-minutes", type=int, default=300)
    ap.add_argument("--wait-minutes", type=int, default=180,
                    help="how long to keep retrying while no H100 is free")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    d = Daytona()
    t0 = time.time()
    try:
        sb = create_sandbox(d, a.ttl_minutes, a.wait_minutes)
    except Exception as e:
        # For the CI results issue: the job did not run, and why.
        (a.out / "error.txt").write_text(f"No H100 sandbox: {e}"[:2000] + "\n")
        raise
    print(f"Sandbox {sb.id} ready after {time.time() - t0:.0f}s", flush=True)
    (a.out / "sandbox.json").write_text(json.dumps({
        "image": ENGINES["vllm"].image, "gpu": "H100", "cpu": 16, "memory_gb": 128,
        "wait_s": round(time.time() - t0)}, indent=1))
    rc = 1
    try:
        archive = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar.gz", "HEAD"],
                                 capture_output=True, check=True).stdout
        sb.fs.upload_file(archive, f"{SRC}.tar.gz")
        run(sb, f"rm -rf {SRC} && mkdir -p {SRC} && tar -xzf {SRC}.tar.gz -C {SRC}")

        # The sandbox gets an exported tree without .git, so pass the commit along.
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        job = " ".join([f"KVCACHED_SHA={sha}", f"{SRC}/tools/ci/daytona/sandbox_job.sh", SRC,
                        OUT, *a.models] + (["--skip-correctness"] if a.skip_correctness else []))
        run(sb, f"setsid nohup bash -c {shlex.quote(job + f'; echo $? > {RC}')} "
                f"> {LOG} 2>&1 < /dev/null &")

        offset = 0
        while True:
            time.sleep(30)
            try:
                chunk = sb.process.exec(f"tail -c +{offset + 1} {LOG}", timeout=60).result
            except Exception as e:  # a transient API error must not end the run
                print(f"(log poll failed: {e})", flush=True)
                continue
            if chunk:
                print(chunk, end="", flush=True)
                offset += len(chunk.encode())
            if sb.process.exec(f"test -f {RC}", timeout=60).exit_code == 0:
                break
        rc = int(run(sb, f"cat {RC}").strip() or 1)

        run(sb, f"mkdir -p {OUT} && cp {LOG} {OUT}/run.log; tar -czf {OUT}.tar.gz -C {OUT} . "
                "|| true", timeout=600)
        data = sb.fs.download_file(f"{OUT}.tar.gz")
        tarfile.open(fileobj=io.BytesIO(data)).extractall(a.out, filter="data")
        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        for name in ("perf/summary.md", "e2e/summary.md"):
            path = a.out / name
            if path.exists():
                print(path.read_text())
                if step_summary:
                    with open(step_summary, "a") as f:
                        f.write(f"\n### {name.split('/')[0]}\n\n" + path.read_text())
    finally:
        try:
            sb.stop()  # deletes an ephemeral sandbox
        except Exception as e:
            print(f"Stopping the sandbox failed ({e}); trying delete", flush=True)
            sb.delete()
        print(f"Sandbox ended after {time.time() - t0:.0f}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
