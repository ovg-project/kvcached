#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Serve real models with and without kvcached and compare the results.

For each engine, one long-lived container from the official image gets
kvcached installed from this checkout. Each case is one server start inside
it: kvcached with the contiguous and the non-contiguous KV layout, and the
native engine. A case passes when kvcached is
active, the server stays healthy, sequential greedy outputs and prefix-cache
hits match the native run, and shutdown leaves no shm segment, socket
directory or GPU process behind.

Usage:
    python3 tests/e2e/run.py --profile nightly --out /tmp/e2e
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# Import the sibling modules as the `e2e` package, as mypy and pytest see them.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e import client as cl  # noqa: E402
from e2e.matrix import ENGINES, FP8_KV, MODELS, PROFILES, Case, EngineSpec  # noqa: E402

DOCKER = shlex.split(os.environ.get("E2E_DOCKER", "docker"))
OUT_IN_CONTAINER = "/e2e-out"
SRC_IN_CONTAINER = "/opt/kvcached-src"
ERROR_SIGNATURES = (r"Traceback \(most recent call last\)", r"CUDA error",
                    r"illegal memory access", r"CUDA out of memory", r"Failed to patch",
                    r"Error applying")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class CaseResult:
    name: str
    checks: list[Check] = field(default_factory=list)
    log: dict[str, Any] = field(default_factory=dict)
    phases: dict[str, cl.Phase] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(name, bool(ok), detail))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "checks": [asdict(c) for c in self.checks],
                "log": self.log, "phases": {k: p.to_dict() for k, p in self.phases.items()}}


def sh(cmd: list[str], timeout: Optional[float] = None, check: bool = False,
       input_bytes: Optional[bytes] = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}): "
                           f"{proc.stderr.decode(errors='replace')[-2000:]}")
    return proc


def gpu_select(gpus: str) -> list[str]:
    """nvidia-smi's selection of the GPUs a run uses: host indices, or all."""
    return [] if gpus == "all" else ["-i", gpus]


def gpu_processes(gpus: str = "all") -> list[str]:
    """Compute processes on the GPUs a run uses."""
    out = sh(["nvidia-smi", *gpu_select(gpus), "--query-compute-apps=pid",
              "--format=csv,noheader"])
    return [x.strip() for x in out.stdout.decode().splitlines() if x.strip()]


# True while a process group has a member that is not a zombie: orphaned
# workers can stay zombies when PID 1 does not reap them.
GROUP_ALIVE = """python3 -c 'import os, sys
for p in filter(str.isdigit, os.listdir("/proc")):
    try:
        f = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()
    except OSError:
        continue
    if f[0] != "Z" and f[2] == sys.argv[1]:
        sys.exit(0)
sys.exit(1)' """


class Container:
    """A long-lived engine container; servers run inside it via docker exec."""

    # Where out_dir is visible to the processes the servers run in.
    out_path = OUT_IN_CONTAINER

    def __init__(self, engine: EngineSpec, name: str, gpus: str, out_dir: Path,
                 hf_cache: Path) -> None:
        self.engine = engine
        self.name = name
        # The host GPUs of the run, comma-separated, or "all"; the container
        # sees only these. TP cases span them; the others use the first, on
        # which memory is sampled.
        self.gpus = gpus
        self.gpu = "0" if gpus == "all" else gpus.split(",")[0]
        self.out_dir = out_dir

        sh(DOCKER + ["rm", "-f", name])
        # --init reaps the engine's orphaned workers, so an exited server's
        # process group really disappears instead of lingering as zombies.
        device = "all" if gpus == "all" else f'"device={gpus}"'
        sh(DOCKER + ["run", "-d", "--init", "--name", name, "--gpus", device,
                     "--ipc=host", "--network=host", "-v", f"{hf_cache}:/root/.cache/huggingface",
                     "-v", f"{out_dir}:{OUT_IN_CONTAINER}", "-e", "HF_TOKEN",
                     "--entrypoint", "sleep", engine.image, "infinity"], check=True)

    def exec(self, script: str, timeout: Optional[float] = None,
             input_bytes: Optional[bytes] = None) -> subprocess.CompletedProcess:
        flags = ["-i"] if input_bytes is not None else []
        return sh(DOCKER + ["exec", *flags, self.name, "bash", "-c", script],
                  timeout=timeout, input_bytes=input_bytes)

    def spawn(self, script: str) -> None:
        sh(DOCKER + ["exec", "-d", self.name, "bash", "-c", script], check=True)

    def install_kvcached(self, repo: Path, log: Path) -> None:
        """Install kvcached non-editable from the current checkout, like a user would."""
        if (repo / ".git").exists():
            # Committed and untracked files, without build output or ignored files.
            src = sh(["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others",
                      "--exclude-standard"], check=True).stdout
            tar = sh(["tar", "-C", str(repo), "--null", "-T", "-", "-cf", "-"],
                     input_bytes=src, check=True).stdout
        else:  # an exported tree, as the CI uploads to the VM
            tar = sh(["tar", "-C", str(repo), "--exclude=__pycache__", "-cf", "-", "."],
                     check=True).stdout
        script = f"""set -e
rm -rf {SRC_IN_CONTAINER} && mkdir -p {SRC_IN_CONTAINER} && tar -x -C {SRC_IN_CONTAINER}
cd {SRC_IN_CONTAINER}
python3 -m pip install -q -r requirements.txt ninja
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs${{LIBRARY_PATH:+:$LIBRARY_PATH}}
python3 -m pip install . --no-build-isolation
cd /tmp
python3 {SRC_IN_CONTAINER}/tools/dev_copy_pth.py --check
python3 -c 'import kvcached, kvcached.vmm_ops; print("kvcached", kvcached.__file__)'
"""
        proc = self.exec(script, timeout=3600, input_bytes=tar)
        log.write_bytes(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"installing kvcached in {self.name} failed; see {log}")

    def start_server(self, case_dir: str, env: dict[str, str], argv: list[str]) -> str:
        """Start a server in its own process group; return its pid (= the group id)."""
        log = f"{self.out_path}/{case_dir}/serve.log"
        pidfile = f"{self.out_path}/{case_dir}/server.pid"
        envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        cmd = " ".join(shlex.quote(a) for a in argv)
        # With job control on, the background job gets its own process group
        # (id = $!), so stopping the group also reaches the engine's workers.
        # `set -m` must run in this shell, not in a subshell of the job.
        script = f"cd /tmp; set -m; env {envs} {cmd} > {log} 2>&1 & echo $! > {pidfile}; wait"
        self.spawn(script)
        for _ in range(50):
            proc = self.exec(f"cat {pidfile} 2>/dev/null")
            pid = proc.stdout.decode().strip()
            if pid:
                return pid
            time.sleep(0.2)
        raise RuntimeError(f"server for {case_dir} did not start")

    def alive(self, pid: str) -> bool:
        return self.exec(GROUP_ALIVE + pid).returncode == 0

    def stop_server(self, pid: str, timeout: float = 90) -> bool:
        """Signal the process group (SIGINT, as Ctrl-C does, unless
        E2E_STOP_SIGNAL says otherwise), then SIGKILL; True if it exited on
        the first signal."""
        sig = os.environ.get("E2E_STOP_SIGNAL", "INT")
        self.exec(f"kill -{shlex.quote(sig)} -- -{pid} 2>/dev/null")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.alive(pid):
                return True
            time.sleep(1)
        self.exec(f"kill -KILL -- -{pid} 2>/dev/null")
        time.sleep(3)
        return False

    def tmp_entries(self, prefix: str) -> list[str]:
        out = self.exec(f"ls -1 /tmp | grep -F -- {shlex.quote(prefix)} || true")
        return [x for x in out.stdout.decode().split() if x]

    def versions(self) -> dict[str, Any]:
        """Installed versions of the engine, kvcached and torch."""
        code = ("import json, importlib.metadata as md, torch\n"
                "def v(p):\n"
                "    try:\n"
                "        return md.version(p)\n"
                "    except md.PackageNotFoundError:\n"
                "        return None\n"
                f"print(json.dumps({{'engine': v({self.engine.name!r}), 'kvcached': v('kvcached'), "
                "'torch': torch.__version__, 'cuda': torch.version.cuda}))")
        proc = self.exec(f"cd /tmp && python3 -c {shlex.quote(code)}", timeout=300)
        try:
            return json.loads(proc.stdout.decode().strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {}

    def image_digest(self) -> Optional[str]:
        proc = sh(DOCKER + ["image", "inspect", "--format", "{{json .RepoDigests}}",
                            self.engine.image])
        try:
            digests = json.loads(proc.stdout.decode() or "[]")
        except ValueError:
            return None
        return digests[0] if digests else None

    def remove(self) -> None:
        sh(DOCKER + ["rm", "-f", self.name])


class LocalHost(Container):
    """Run the servers directly on this machine, e.g. inside a sandbox built
    from the engine image, where there is no Docker."""

    def __init__(self, engine: EngineSpec, name: str, gpus: str, out_dir: Path) -> None:
        self.engine = engine
        self.name = name
        self.gpus = gpus
        self.gpu = "0" if gpus == "all" else gpus.split(",")[0]
        self.out_dir = out_dir
        self.out_path = str(out_dir)

    def exec(self, script: str, timeout: Optional[float] = None,
             input_bytes: Optional[bytes] = None) -> subprocess.CompletedProcess:
        return sh(["bash", "-c", script], timeout=timeout, input_bytes=input_bytes)

    def spawn(self, script: str) -> None:
        subprocess.Popen(["bash", "-c", script], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)

    def image_digest(self) -> Optional[str]:
        return None

    def remove(self) -> None:
        pass


def server_argv(engine: str, model_id: str, args: tuple[str, ...], port: int) -> list[str]:
    if engine == "vllm":
        return ["vllm", "serve", model_id, "--served-model-name", "e2e", "--host", "127.0.0.1",
                "--port", str(port), "--seed", "42", "--enable-prompt-tokens-details",
                "--enable-prefix-caching", "--async-scheduling",
                # CUDA graphs stay on; skipping Inductor keeps outputs bit-exact
                # between runs, so kvcached and native can be compared token by token.
                "--compilation-config.backend=eager", *args]
    return ["python3", "-m", "sglang.launch_server", "--model-path", model_id,
            "--served-model-name", "e2e", "--host", "127.0.0.1", "--port", str(port),
            "--random-seed", "42", "--enable-cache-report", *args]


def parse_log(text: str, spec: EngineSpec) -> dict[str, Any]:
    info: dict[str, Any] = {}
    m = re.search(spec.patched_re, text)
    info["patched"] = [p.strip() for p in m.group(1).split(",")] if m else []
    m = re.search(spec.capacity_re, text)
    info["capacity"] = int(m.group(1)) if m else None
    info["tokens"] = [int(x.replace(",", "")) for x in re.findall(spec.tokens_re, text)]
    info["kvcached_lines"] = len(re.findall(r"\[kvcached\]", text))
    # kvcached can enlarge a hybrid model's attention block so that it tiles
    # its physical page (vLLM); the native run must use the same block size.
    m = re.search(r"Setting attention block size to (\d+) tokens \(was (\d+)\) so the KV unit", text)
    info["block_size"] = (int(m.group(1)), int(m.group(2))) if m else None
    info["errors"] = sorted({s for s in ERROR_SIGNATURES if re.search(s, text)})
    return info


def ipc_name(case_name: str, run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", f"e2e_{run_id}_{case_name}")


@dataclass
class Server:
    case: Case
    ipc: str
    pid: str
    client: cl.EngineClient
    # What the server was started with, for setup.json.
    setup: dict[str, Any] = field(default_factory=dict)


def server_env(case: Case, ipc: str) -> dict[str, str]:
    env = dict(ENGINES[case.engine].env)
    if case.kvcached:
        env.update(ENABLE_KVCACHED="true", KVCACHED_AUTOPATCH="1", KVCACHED_IPC_NAME=ipc,
                   KVCACHED_CONTIGUOUS_LAYOUT="true" if case.layout == "c1" else "false")
    else:
        env.update(ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0")
    return env


def case_setup(case: Case) -> dict[str, Any]:
    model = MODELS[case.model]
    return {"engine": case.engine, "model": case.model, "hf_id": model.hf_id,
            "layout": case.layout, "tp": case.tp,
            "kv_cache_dtype": FP8_KV[case.engine][1] if case.fp8_kv else None,
            "model_args": list(model.args[case.engine])}


def launch(ct: Container, case: Case, run_id: str, port: int,
           extra_args: tuple[str, ...] = ()) -> Server:
    model = MODELS[case.model]
    case_dir = ct.out_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)

    ipc = ipc_name(case.name, run_id)
    env = server_env(case, ipc)
    argv = server_argv(case.engine, model.hf_id, case.args() + extra_args, port)
    cmd = " ".join(f"{k}={v}" for k, v in env.items()) + " " + shlex.join(argv)
    (case_dir / "cmd.txt").write_text(cmd + "\n")
    setup = case_setup(case)
    setup.update(env={k: v for k, v in env.items() if k != "KVCACHED_IPC_NAME"},
                 extra_args=list(extra_args), cmd=cmd)
    pid = ct.start_server(case.name, env, argv)
    return Server(case, ipc, pid, cl.make_client(case.engine, "127.0.0.1", port), setup)


def stop_and_check(ct: Container, srv: Server, res: CaseResult, prefix: str = "") -> None:
    """Stop a server and record the shutdown, log and cleanup checks."""
    spec = ENGINES[srv.case.engine]
    model = MODELS[srv.case.model]
    log_path = ct.out_dir / srv.case.name / "serve.log"
    before_stop = log_path.stat().st_size if log_path.exists() else 0
    res.add(f"{prefix}clean_exit_on_signal", ct.stop_server(srv.pid))
    raw = log_path.read_bytes() if log_path.exists() else b""
    log = parse_log(raw.decode(errors="replace"), spec)
    # Engines print KeyboardInterrupt tracebacks while they shut down; only
    # errors raised while serving count.
    log["errors_after_stop"] = log["errors"]
    log["errors"] = parse_log(raw[:before_stop].decode(errors="replace"), spec)["errors"]
    res.log[srv.case.name] = log
    res.add(f"{prefix}no_error_signatures", not log["errors"],
            ", ".join(re.sub(r"\\(.)", r"\1", e) for e in log["errors"]))
    if srv.case.kvcached:
        required = set(spec.required_patches) | set(model.extra_patches.get(srv.case.engine, ()))
        missing = sorted(required - set(log["patched"]))
        res.add(f"{prefix}kvcached_active", not missing and log["capacity"] is not None,
                f"missing patches {missing}" if missing else "")
        leftover_shm = sorted(x for x in os.listdir("/dev/shm") if x.startswith(srv.ipc))
        res.add(f"{prefix}shm_removed", not leftover_shm, ", ".join(leftover_shm))
        leftover_tmp = ct.tmp_entries(srv.ipc)
        res.add(f"{prefix}socket_dirs_removed", not leftover_tmp, ", ".join(leftover_tmp))
    else:
        res.add(f"{prefix}kvcached_inactive", log["kvcached_lines"] == 0,
                f"{log['kvcached_lines']} [kvcached] lines")


def check_gpu_released(res: CaseResult, gpus: str) -> None:
    deadline = time.time() + 60
    while gpu_processes(gpus) and time.time() < deadline:
        time.sleep(2)
    procs = gpu_processes(gpus)
    res.add("no_gpu_process_left", not procs, ", ".join(procs))


# What every server case does and checks, for setup.json (the CI results
# issue shows it); keep these in step with run_server_case and stop_and_check.
WORKLOAD = (
    "gen_seq: 9 prompts (6 short, and 300, 900 and 1800 words of filler text), greedy, "
    "64 output tokens, one request at a time",
    "gen_conc: the same 9 prompts, all at once",
    "the prefix cache is reset",
    "apc_0, apc_1, apc_2: 4 questions about one shared 1300-word document, greedy, "
    "32 output tokens, one request at a time",
    "apc_conc: the same 4 questions, all at once",
    "the server is stopped with SIGINT to its process group (E2E_STOP_SIGNAL), "
    "SIGKILL after 90 s",
)
CHECKS = {
    "server_ready": "the server answers /health within the ready timeout",
    "<phase>_ok": "every request of the phase returns HTTP 200 with output tokens",
    "prefix_cache_reset": "the prefix-cache reset endpoint succeeds",
    "server_alive_after_load": "the server process group is still running after the load",
    "clean_exit_on_signal": "the server exits within 90 s of the stop signal",
    "no_error_signatures": "no traceback, CUDA error or patch failure in the log "
                           "before the stop signal",
    "kvcached_active": "kvcached applied every required patch and sized the KV cache",
    "kvcached_inactive": "the native server's log has no [kvcached] line",
    "shm_removed": "no /dev/shm segment of the server is left",
    "socket_dirs_removed": "no /tmp/kvcached-tp-* directory of the server is left",
    "no_gpu_process_left": "no process is left on the GPUs of the run",
    "<phase>_tokens_equal": "gen_seq and apc_0..2 give the same greedy tokens as native",
    "apc_<n>_cached_tokens_equal": "every request reports the same cached_tokens as native",
}


def run_server_case(ct: Container, case: Case, run_id: str, port: int, timeout: float,
                    extra_args: tuple[str, ...] = ()) -> CaseResult:
    res = CaseResult(case.name)
    srv = launch(ct, case, run_id, port, extra_args)
    res.log["setup"] = srv.setup
    client = srv.client
    pid = srv.pid
    ready = client.wait_ready(timeout, lambda: ct.alive(pid))
    res.add("server_ready", ready)
    if ready:
        res.phases["gen_seq"] = client.run_phase("gen_seq", cl.gen_prompts(), 64, 1)
        res.phases["gen_conc"] = client.run_phase("gen_conc", cl.gen_prompts(), 64, 9)
        res.add("prefix_cache_reset", client.reset_prefix_cache())
        for rnd in range(3):
            res.phases[f"apc_{rnd}"] = client.run_phase(f"apc_{rnd}", cl.apc_prompts(), 32, 1)
        res.phases["apc_conc"] = client.run_phase("apc_conc", cl.apc_prompts(), 32, 4)
        for name, phase in res.phases.items():
            res.add(f"{name}_ok", phase.all_ok,
                    "; ".join(f"{r.status} {r.error}" for r in phase.results if not r.ok)[:500])
        # Warm rounds are compared with the native engine's warm rounds, not
        # with the cold round: SGLang on L4 already differs there natively.
        res.add("server_alive_after_load", ct.alive(pid))

    stop_and_check(ct, srv, res)
    check_gpu_released(res, ct.gpus)
    return res


def compare_with_native(kv: CaseResult, native: CaseResult) -> list[Check]:
    checks = []
    for phase in ("gen_seq", "apc_0", "apc_1", "apc_2"):
        a, b = kv.phases.get(phase), native.phases.get(phase)
        if a is None or b is None:
            checks.append(Check(f"{phase}_tokens_equal", False, "phase missing"))
            continue
        divs = [(i, cl.first_divergence(x.token_ids, y.token_ids))
                for i, (x, y) in enumerate(zip(a.results, b.results))]
        bad = [f"prompt {i}: {d}" for i, d in divs if d is not None]
        checks.append(Check(f"{phase}_tokens_equal", not bad, "; ".join(bad)[:500]))
        if phase.startswith("apc"):
            ca = [r.cached_tokens for r in a.results]
            cb = [r.cached_tokens for r in b.results]
            checks.append(Check(f"{phase}_cached_tokens_equal", ca == cb,
                                f"kvcached {ca} vs native {cb}"))
    return checks


def report(r: CaseResult) -> None:
    """One line per finished case, so that a CI log shows progress."""
    failed = [c.name for c in r.checks if not c.ok]
    print(f"{time.strftime('%H:%M:%S')} {r.name}: {'PASS' if r.ok else 'FAIL'}"
          + (f" ({', '.join(failed)})" if failed else ""), flush=True)


def write_summary(out: Path, results: list[CaseResult]) -> bool:
    ok = all(r.ok for r in results)
    (out / "summary.json").write_text(json.dumps(
        {"ok": ok, "cases": [r.to_dict() for r in results]}, indent=1))
    lines = ["| case | result | failed checks |", "|---|---|---|"]
    for r in results:
        failed = [f"{c.name} ({c.detail})" if c.detail else c.name for c in r.checks if not c.ok]
        lines.append(f"| {r.name} | {'PASS' if r.ok else 'FAIL'} | {'; '.join(failed)[:300]} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write("\n".join(lines) + "\n")
    return ok


def gpu_info(gpus: str = "all") -> list[dict[str, str]]:
    out = sh(["nvidia-smi", *gpu_select(gpus), "--query-gpu=name,memory.total,driver_version",
              "--format=csv,noheader"])
    rows = [[x.strip() for x in line.split(",")] for line in out.stdout.decode().splitlines()]
    return [dict(zip(("name", "memory", "driver"), row)) for row in rows if len(row) == 3]


def model_revision(hf_cache: Path, hf_id: str) -> Optional[str]:
    """The commit of the model the servers loaded: the cache's main ref."""
    ref = hf_cache / "hub" / f"models--{hf_id.replace('/', '--')}" / "refs" / "main"
    try:
        return ref.read_text().strip() or None
    except OSError:
        return None


def kvcached_sha(repo: Path) -> Optional[str]:
    # The CI ships an exported tree without .git and passes the commit along.
    if os.environ.get("KVCACHED_SHA"):
        return os.environ["KVCACHED_SHA"]
    if (repo / ".git").exists():
        return sh(["git", "-C", str(repo), "rev-parse", "HEAD"]).stdout.decode().strip() or None
    return None


def write_setup(out: Path, profile: str, sha: Optional[str], hf_cache: Path, port: int,
                gpus: str, engines: dict[str, dict[str, Any]],
                results: list[CaseResult]) -> None:
    """Record what this run tested, for the CI results issue (tools/ci/report.py)."""
    from e2e import elastic

    setups = [{"name": r.name, **r.log["setup"]} for r in results if "setup" in r.log]
    hf_ids = {s["hf_id"] for s in setups if "hf_id" in s}
    hf_ids |= {s[k]["hf_id"] for s in setups for k in ("a", "b") if k in s}
    for engine, info in engines.items():
        info.update(env=dict(ENGINES[engine].env),
                    server=shlex.join(server_argv(engine, "MODEL", (), port)))
    setup = {
        "profile": profile, "kvcached_sha": sha, "gpus": gpu_info(gpus), "engines": engines,
        "stop_signal": os.environ.get("E2E_STOP_SIGNAL", "INT"),
        "model_revisions": {h: model_revision(hf_cache, h) for h in sorted(hf_ids)},
        "workload": WORKLOAD, "checks": CHECKS,
        "cases": [s for s in setups if s.get("kind") != "elastic"],
        "elastic": [s for s in setups if s.get("kind") == "elastic"],
        "elastic_workload": elastic.WORKLOAD, "elastic_checks": elastic.CHECKS,
    }
    (out / "setup.json").write_text(json.dumps(setup, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="nightly", choices=sorted(PROFILES))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--hf-cache", type=Path,
                    default=Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser())
    ap.add_argument("--gpus", default="all",
                    help="host GPUs the run uses, comma-separated indices (default: all); "
                         "the containers see only these, and single-GPU servers use the first")
    ap.add_argument("--engines", nargs="*", help="subset of the profile's engines")
    ap.add_argument("--models", nargs="*", help="subset of the profile's models")
    ap.add_argument("--skip-elastic", action="store_true")
    ap.add_argument("--skip-cases", action="store_true", help="run only the elastic tests")
    ap.add_argument("--local", action="store_true",
                    help="run the servers on this machine instead of in engine containers")
    ap.add_argument("--no-install", action="store_true",
                    help="use the kvcached already installed instead of this checkout")
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--ready-timeout", type=float, default=1800)
    ap.add_argument("--elastic-gpu-mib", type=int, default=0,
                    help="hold GPU memory during the elastic test so that only this much "
                         "is usable, to run it on a larger GPU than the CI one")
    a = ap.parse_args()

    profile = PROFILES[a.profile]
    run_id = time.strftime("%m%d%H%M")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results: list[CaseResult] = []

    engines = [e for e in profile.engines if not a.engines or e in a.engines]
    containers: dict[str, Container] = {}
    engine_info: dict[str, dict[str, Any]] = {}
    try:
        for engine in engines:
            spec = ENGINES[engine]
            name = f"kvcached-e2e-{engine}-{run_id}"
            ct = (LocalHost(spec, name, a.gpus, out) if a.local
                  else Container(spec, name, a.gpus, out, a.hf_cache))
            setup = CaseResult(f"{engine}-setup")
            try:
                if not a.no_install:
                    ct.install_kvcached(a.repo, out / f"{engine}-install.log")
                setup.add("install_kvcached", True)
                containers[engine] = ct
                engine_info[engine] = {"image": spec.image, "digest": ct.image_digest(),
                                       **ct.versions()}
            except Exception as e:
                setup.add("install_kvcached", False, str(e))
                ct.remove()
            results.append(setup)

        for group in [] if a.skip_cases else profile.groups:
            host = containers.get(group.engine)
            if host is None or (a.models and group.model not in a.models):
                continue
            block_size: Optional[tuple[int, int]] = None
            done: dict[str, CaseResult] = {}
            for case in group.cases():
                extra: tuple[str, ...] = ()
                note = ""
                if not case.kvcached and block_size:
                    extra = ("--block-size", str(block_size[0]))
                    note = (f"native ran with kvcached's --block-size {block_size[0]}; "
                            f"its default is {block_size[1]}")
                r = run_server_case(host, case, run_id, a.port, a.ready_timeout, extra)
                if note:
                    r.add("native_block_size_matched", True, note)
                if case.kvcached and block_size is None:
                    block_size = r.log.get(case.name, {}).get("block_size")
                done[case.name] = r
                results.append(r)
            native = done[[c for c in group.cases() if not c.kvcached][0].name]
            for case in group.cases():
                if case.kvcached:
                    done[case.name].checks += compare_with_native(done[case.name], native)
                report(done[case.name])

        if not a.skip_elastic:
            from e2e.elastic import run_elastic
            for pair in profile.elastic:
                ct_a, ct_b = containers.get(pair.engine_a), containers.get(pair.engine_b)
                if ct_a and ct_b:
                    results.append(run_elastic(ct_a, ct_b, pair, run_id, a.port,
                                               a.ready_timeout, a.elastic_gpu_mib))
                    report(results[-1])
    finally:
        for ct in containers.values():
            ct.remove()

    write_setup(out, a.profile, kvcached_sha(a.repo), a.hf_cache, a.port, a.gpus, engine_info,
                results)
    ok = write_summary(out, results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
