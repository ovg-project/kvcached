#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Measure the physical memory a kvcached instance still holds once it is idle.

Issue #359: with prefix caching on, an idle instance keeps far more physical
memory than KVCACHED_MAX_CACHED_TOKENS implies. The cap counts blocks, but
memory comes back a page at a time, and only once every block on that page is
free -- so what an idle instance holds is decided by how many pages its
surviving blocks are spread over, not by how many blocks it kept.

This drives a real vLLM server, then reads the number kvcached itself
publishes: `used_size` in its MemInfo shared-memory segment, which is
`num_inuse_pages * num_layers * page_size * num_kv_buffers`. (nvidia-smi would
be no good here -- it also counts weights and activations.)

Usage:
    MODEL=/path/to/Qwen3-4B ./run_idle_footprint.py
    ./run_idle_footprint.py --workload "--requests 1000 --concurrency 32"

Run it once per branch and compare the "idle" line.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
GB = 1024**3

sys.path.insert(0, HERE)
from probe_mem import detect_segments, read_all  # noqa: E402


def http_text(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def metric(text, name):
    m = re.search(rf'^{re.escape(name)}\{{[^}}]*\}}\s+([0-9.e+-]+)$', text,
                  re.MULTILINE)
    return float(m.group(1)) if m else None


def _vllm_bin():
    """Find the `vllm` console script, preferring the one next to this
    interpreter so `venv/bin/python run_idle_footprint.py` works without
    activating the venv first."""
    local = os.path.join(os.path.dirname(sys.executable), "vllm")
    if os.path.exists(local):
        return local
    found = shutil.which("vllm")
    if not found:
        raise RuntimeError("cannot find the `vllm` executable; activate the "
                           "environment vLLM is installed in")
    return found


def launch(model, port, log, extra):
    env = dict(os.environ)
    env.setdefault("ENABLE_KVCACHED", "true")
    env.setdefault("KVCACHED_AUTOPATCH", "1")
    env.setdefault("VLLM_USE_V1", "1")
    # kvcached patches the V1 GPUModelRunner. vLLM picks the V2 runner by
    # default for some models, and then kvcached's worker-side init hook never
    # fires and KV init dies; keep it on the path kvcached supports.
    env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    cmd = [_vllm_bin(), "serve", model, "--port", str(port),
           "--served-model-name", "bench", "--max-model-len",
           os.environ.get("MAX_MODEL_LEN", "8192")] + extra
    with open(log, "w") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                env=env)


def wait_ready(port, log, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return False


def geometry(log):
    """Recover the page geometry the server chose, from its own startup log."""
    with open(log, errors="ignore") as f:
        text = f.read()
    m = re.search(r"num_layers=(\d+).*?page_size=(\d+)MB.*?num_kv_buffers=(\d+)",
                  text)
    if not m:
        return {}
    layers, page_mb, buffers = (int(m.group(1)), int(m.group(2)),
                                int(m.group(3)))
    return {"num_layers": layers, "page_size_mb": page_mb,
            "num_kv_buffers": buffers,
            "bytes_per_page": page_mb * 1024**2 * layers * buffers}


def our_segments(before):
    """The MemInfo segments this run's server created.

    Reading every segment in /dev/shm would fold in any co-located kvcached
    instance -- which is the whole point of kvcached, so it is not hypothetical
    -- and any segment a previously killed server left behind. Deleting them is
    not an option for the same reason, so diff against what existed at launch.
    """
    ours = [n for n in detect_segments() if n not in before]
    if not ours:
        print("warning: no new MemInfo segment appeared; is kvcached enabled?",
              file=sys.stderr)
    return ours


def snapshot(names):
    live = [s for s in read_all(names) if "error" not in s]
    return {"used_gb": sum(s["used_gb"] for s in live),
            "prealloc_gb": sum(s["prealloc_gb"] for s in live)}


def wait_idle(port, settle, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            text = http_text(f"http://127.0.0.1:{port}/metrics")
            if (metric(text, "vllm:num_requests_running") or 0) == 0 and \
               (metric(text, "vllm:num_requests_waiting") or 0) == 0:
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    time.sleep(settle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--idle-settle", type=float, default=25.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="",
                    help="suffix for the timeline file, to keep runs apart")
    ap.add_argument("--workload", default="",
                    help="extra args forwarded to workload.py")
    ap.add_argument("--serve-arg", action="append", default=[],
                    help="extra args forwarded to `vllm serve`")
    args = ap.parse_args()
    if not args.model:
        ap.error("pass --model or set MODEL")

    log = os.path.join(HERE, "server.log")
    before = set(detect_segments())
    proc = launch(args.model, args.port, log, args.serve_arg)
    try:
        if not wait_ready(args.port, log, proc):
            print(f"server failed to start; see {log}")
            return 1
        segs = our_segments(before)
        geo = geometry(log)
        print(f"geometry: {geo}", flush=True)

        # Sample throughout, not just at the end: the shape matters. Page-aware
        # eviction cannot free anything while requests are in flight (nearly
        # every page holds a live block), so its whole effect appears as a step
        # at the moment the last request drains.
        timeline = os.path.join(HERE, f"timeline{args.tag}.jsonl")
        sampler = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "probe_mem.py"),
             "--watch", "1", "--jsonl", timeline]
            + (["--seg", segs[0]] if segs else []),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        subprocess.run([sys.executable, os.path.join(HERE, "workload.py"),
                        "--port", str(args.port), "--model", "bench"]
                       + args.workload.split(), check=False)
        after = snapshot(segs)
        wait_idle(args.port, args.idle_settle)
        idle = snapshot(segs)

        sampler.terminate()
        text = http_text(f"http://127.0.0.1:{args.port}/metrics")
        q = metric(text, "vllm:prefix_cache_queries_total") or 0
        h = metric(text, "vllm:prefix_cache_hits_total") or 0

        bpp = geo.get("bytes_per_page")
        result = {
            "after_workload_gb": round(after["used_gb"], 2),
            "idle_gb": round(idle["used_gb"], 2),
            "idle_prealloc_gb": round(idle["prealloc_gb"], 2),
            "idle_pages": round(idle["used_gb"] * GB / bpp) if bpp else None,
            "prefix_cache_hits": int(h),
            "prefix_cache_hit_rate": round(h / q, 4) if q else None,
            "geometry": geo,
            "timeline": timeline,
        }
        print(json.dumps(result, indent=1), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=1)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
