#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Compare kvcached-only and kvcached+TriAttention vLLM serving."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CSV_FIELDS = [
    "label",
    "workload",
    "model",
    "prompt_repeat",
    "kv_budget",
    "divide_length",
    "max_tokens",
    "concurrency",
    "requests",
    "successes",
    "failures",
    "success_rate",
    "elapsed_secs",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tpot_mean_ms",
    "e2e_mean_ms",
    "e2e_p50_ms",
    "output_chunks",
    "output_chunks_per_sec",
    "gpu_mib_before",
    "gpu_mib_peak",
    "gpu_mib_avg",
    "gpu_mib_p95",
    "gpu_mib_after",
    "gpu_mib_delta_peak",
    "compression_events",
    "compression_skipped_events",
    "compression_skip_reasons",
    "free_blocks_events",
    "freed_blocks",
    "triattention_activation_seen",
    "triattention_activation_failures",
    "error_count",
    "log_path",
]


PROMPT_UNIT = (
    "Segment {idx}: kvcached maps KV pages; TriAttention compresses long KV "
    "cache using sparse stats, budget, window, and block reclaim."
)


DEFAULT_STATS_RELATIVE_PATH = (
    "triattention/calibration/for_aime25_experiment/qwen3_8b.pt"
)


def default_triattention_root() -> str:
    script_root = Path(__file__).resolve().parents[1]
    candidates = [
        script_root.parent / "triattention-main",
        Path.cwd().parent / "triattention-main",
    ]
    for local_root in candidates:
        if local_root.exists():
            return str(local_root)
    return "/root/triattention-main"


@dataclass
class RequestMetrics:
    ok: bool
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    output_chunks: int = 0
    error: str = ""


@dataclass
class ServerHandle:
    proc: subprocess.Popen[str]
    log_path: Path
    log_thread: threading.Thread


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def count_regex(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))


def sum_first_group(pattern: str, text: str) -> int:
    return sum(int(match.group(1)) for match in re.finditer(pattern, text))


def read_gpu_mib(gpu_index: int) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return None
    first = out.strip().splitlines()[0] if out.strip() else ""
    try:
        return int(first.strip())
    except ValueError:
        return None


class GpuSampler:
    def __init__(self, gpu_index: int, interval_secs: float) -> None:
        self.gpu_index = gpu_index
        self.interval_secs = interval_secs
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "GpuSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = read_gpu_mib(self.gpu_index)
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval_secs)


def make_prompt(prompt_repeat: int) -> str:
    body = "\n\n".join(PROMPT_UNIT.format(idx=i) for i in range(1, prompt_repeat + 1))
    return (
        body
        + "\n\nFinal task: summarize the roles of kvcached and TriAttention, "
        "then explain which server-side logs prove that compression or block "
        "reclaim happened. Answer in three concise numbered points."
    )


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_server(base_url: str, timeout_secs: float) -> None:
    deadline = time.monotonic() + timeout_secs
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy within {timeout_secs}s ({last_error})")


def tokenize_count(base_url: str, model_name: str, prompt: str) -> int | None:
    try:
        out = http_json(
            "POST",
            f"{base_url}/tokenize",
            {"model": model_name, "prompt": prompt, "add_special_tokens": False},
            timeout=120,
        )
    except Exception:
        return None
    for key in ("tokens", "token_ids", "prompt_token_ids"):
        value = out.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def fit_prompt_to_context(
    *,
    base_url: str,
    model_name: str,
    requested_prompt_repeat: int,
    max_model_len: int,
    max_tokens: int,
    margin_tokens: int,
) -> tuple[str, int, int | None]:
    """Return a prompt that should pass vLLM's context-length validation."""

    target_prompt_tokens = max_model_len - max_tokens - margin_tokens
    if target_prompt_tokens <= 0:
        raise ValueError(
            "max_model_len must be larger than max_tokens + margin_tokens "
            f"(got {max_model_len}, {max_tokens}, {margin_tokens})"
        )

    low = 1
    high = max(1, requested_prompt_repeat)
    best_repeat = 1
    best_tokens: int | None = None

    while low <= high:
        mid = (low + high) // 2
        candidate = make_prompt(mid)
        token_count_value = tokenize_count(base_url, model_name, candidate)
        if token_count_value is None:
            # If /tokenize is unavailable, fall back to the requested prompt.
            return make_prompt(requested_prompt_repeat), requested_prompt_repeat, None
        if token_count_value <= target_prompt_tokens:
            best_repeat = mid
            best_tokens = token_count_value
            low = mid + 1
        else:
            high = mid - 1

    return make_prompt(best_repeat), best_repeat, best_tokens


def stream_chat_request(
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    timeout_secs: float,
) -> RequestMetrics:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_chunk: float | None = None
    chunks = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout_secs) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_part = line[5:].strip()
                if data_part == "[DONE]":
                    break
                try:
                    event = json.loads(data_part)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content") if isinstance(delta, dict) else None
                if content:
                    chunks += 1
                    if first_chunk is None:
                        first_chunk = time.perf_counter()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            detail = str(exc)
        return RequestMetrics(ok=False, error=f"HTTPError {exc.code}: {detail}")
    except Exception as exc:
        return RequestMetrics(ok=False, error=f"{type(exc).__name__}: {exc}")

    end = time.perf_counter()
    ttft = (first_chunk or end) - start
    # A stream that opens and closes but yields no content tokens is a failure,
    # not a success: e.g. the engine crashed mid-decode. Counting it as a
    # success would report a (misleadingly low) GPU peak for a dead run.
    return RequestMetrics(
        ok=chunks > 0,
        ttft_ms=ttft * 1000.0,
        e2e_ms=(end - start) * 1000.0,
        output_chunks=chunks,
        error="" if chunks > 0 else "stream completed but produced 0 output tokens",
    )


def make_env(
    args: argparse.Namespace,
    enable_triattention: bool,
    budget: int | None,
    event_sink_path: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ENABLE_KVCACHED": "true",
            "ENABLE_TRIATTENTION": "1" if enable_triattention else "0",
        }
    )
    env.setdefault("KVCACHED_LOG_LEVEL", args.kvcached_log_level)
    env.setdefault("VLLM_USE_DEEP_GEMM", "0")
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    pythonpath_parts = [str(Path.cwd())]
    if args.triattention_root:
        pythonpath_parts.append(args.triattention_root)
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    if enable_triattention:
        env.update(
            {
                "TRIATTN_RUNTIME_MODEL_PATH": args.model,
                "TRIATTN_RUNTIME_DIVIDE_LENGTH": str(args.divide_length),
                "TRIATTN_RUNTIME_WINDOW_SIZE": str(args.window_size),
                "TRIATTN_RUNTIME_LOG_DECISIONS": "true",
                "TRIATTN_RUNTIME_REQUIRE_TRITON_SCORING": "true",
                "TRIATTN_RUNTIME_REQUIRE_PHYSICAL_RECLAIM": "true",
                "TRIATTN_RUNTIME_DISABLE_COMPRESSION": "false",
                "TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION": "true",
                "TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM": "true",
            }
        )
        if args.stats_path:
            env["TRIATTN_RUNTIME_SPARSE_STATS_PATH"] = args.stats_path
        if budget is not None:
            env["TRIATTN_RUNTIME_KV_BUDGET"] = str(budget)
        if event_sink_path is not None:
            env["TRIATTN_RUNTIME_EVENT_SINK_PATH"] = str(event_sink_path)
    return env


def start_server(
    args: argparse.Namespace,
    *,
    label: str,
    budget: int | None,
    log_path: Path,
) -> ServerHandle:
    enable_triattention = label == "triattention"
    event_sink_path = (
        log_path.with_suffix(".triattn-events.jsonl")
        if enable_triattention
        else None
    )
    if event_sink_path is not None:
        try:
            event_sink_path.unlink()
        except FileNotFoundError:
            pass
    env = make_env(
        args,
        enable_triattention=enable_triattention,
        budget=budget,
        event_sink_path=event_sink_path,
    )
    cmd = [
        args.vllm_command,
        "serve",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--port",
        str(args.port),
        "--no-enable-prefix-caching",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
    ]
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    for extra_arg in args.extra_vllm_arg:
        cmd.append(extra_arg)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    log_file.write("$ " + " ".join(cmd) + "\n")
    log_file.flush()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
    except Exception:
        log_file.close()
        raise

    def pump_log() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
        finally:
            log_file.close()

    log_thread = threading.Thread(target=pump_log, daemon=True)
    log_thread.start()
    return ServerHandle(proc=proc, log_path=log_path, log_thread=log_thread)


def stop_server(handle: ServerHandle, timeout_secs: float = 30) -> None:
    proc = handle.proc
    if proc.poll() is not None:
        handle.log_thread.join(timeout=2)
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=timeout_secs)
    except Exception:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
    handle.log_thread.join(timeout=5)


def parse_event_sink(event_sink_path: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "applied": 0,
        "skipped": 0,
        "errors": 0,
        "freed_blocks": 0,
        "skip_reasons": Counter(),
    }
    if event_sink_path is None or not event_sink_path.exists():
        return out
    try:
        lines = event_sink_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        status = str(event.get("status", ""))
        if status == "applied":
            out["applied"] += 1
            details = event.get("details")
            if isinstance(details, dict):
                try:
                    out["freed_blocks"] += int(details.get("reclaimed_block_count") or 0)
                except (TypeError, ValueError):
                    pass
        elif status == "skipped":
            out["skipped"] += 1
            reason = event.get("reason")
            if isinstance(reason, str):
                out["skip_reasons"][reason] += 1
        elif status == "error":
            out["errors"] += 1
    return out


def parse_server_log(log_path: Path, event_sink_path: Path | None = None) -> dict[str, Any]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        text = ""
    reclaimed = sum_first_group(r"reclaimed_blocks=([0-9]+)", text)
    scheduler_freed = sum_first_group(
        r"TriAttention scheduler FREE_BLOCKS:[^\n]*freed=([0-9]+)", text
    )
    applied_from_runner_log = count_regex(r"TriAttention compression applied", text)
    applied_from_scheduler_events = sum_first_group(
        r"TriAttention update_from_output: received [0-9]+ events \(([0-9]+) applied\)",
        text,
    )
    event_sink = parse_event_sink(event_sink_path)
    skip_reasons: Counter[str] = Counter(
        reason.replace(" ", "_")
        for pattern in (
            r"TriAttention compression skipped[^\n]*reason=([^\s,]+)",
            r"TriAttention compression skipped \(([^)]+)\)",
        )
        for reason in re.findall(pattern, text)
    )
    skip_reasons.update(event_sink["skip_reasons"])
    activation_failures = count_regex(
        r"\[TriAttention\].*Activation failed|"
        r"\[TriAttention\].*Runtime plugin activation failed",
        text,
    )
    return {
        "compression_events": max(
            applied_from_runner_log,
            applied_from_scheduler_events,
            int(event_sink["applied"]),
        ),
        "compression_skipped_events": max(
            count_regex(r"TriAttention compression skipped", text),
            int(event_sink["skipped"]),
        ),
        "compression_skip_reasons": ";".join(
            f"{reason}:{count}" for reason, count in skip_reasons.most_common()
        ),
        "free_blocks_events": (
            count_regex(r"TriAttention block reclaim", text)
            + count_regex(r"TriAttention scheduler FREE_BLOCKS", text)
        ),
        "freed_blocks": max(reclaimed, scheduler_freed, int(event_sink["freed_blocks"])),
        "triattention_activation_seen": bool(
            re.search(
                r"TriAttention].*activated|"
                r"TriAttentionWorker lazily injected|"
                r"Installed TriAttention runtime monkeypatch integration|"
                r"TriAttention monkeypatched Scheduler initialized",
                text,
            )
        ),
        "triattention_activation_failures": activation_failures,
        "error_count": count_regex(r"\bERROR\b|\[ERROR\]", text),
    }


def run_workload(
    args: argparse.Namespace,
    *,
    prompt: str,
    concurrency: int,
) -> tuple[list[RequestMetrics], float]:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                stream_chat_request,
                f"http://127.0.0.1:{args.port}",
                args.served_model_name,
                prompt,
                args.max_tokens,
                args.request_timeout,
            )
            for _ in range(concurrency)
        ]
        results = [future.result() for future in as_completed(futures)]
    elapsed = time.perf_counter() - start
    return results, elapsed


def build_row(
    args: argparse.Namespace,
    *,
    label: str,
    budget: int | None,
    concurrency: int,
    actual_prompt_repeat: int,
    elapsed_secs: float,
    request_metrics: list[RequestMetrics],
    gpu_before: int | None,
    gpu_after: int | None,
    gpu_samples: list[int],
    log_path: Path,
) -> dict[str, Any]:
    successes = [m for m in request_metrics if m.ok]
    failures = len(request_metrics) - len(successes)
    ttfts = [m.ttft_ms for m in successes]
    e2es = [m.e2e_ms for m in successes]
    tpots = [
        max(0.0, (m.e2e_ms - m.ttft_ms) / max(1, m.output_chunks - 1))
        for m in successes
        if m.output_chunks > 1
    ]
    chunks = sum(m.output_chunks for m in successes)
    gpu_peak = max(gpu_samples) if gpu_samples else (gpu_before or 0)
    gpu_avg = mean([float(x) for x in gpu_samples])
    gpu_p95 = percentile([float(x) for x in gpu_samples], 0.95)
    parsed = parse_server_log(
        log_path,
        event_sink_path=log_path.with_suffix(".triattn-events.jsonl"),
    )
    return {
        "label": label,
        "workload": "concurrency",
        "model": args.model,
        "prompt_repeat": actual_prompt_repeat,
        "kv_budget": "" if budget is None else budget,
        "divide_length": "" if budget is None else args.divide_length,
        "max_tokens": args.max_tokens,
        "concurrency": concurrency,
        "requests": len(request_metrics),
        "successes": len(successes),
        "failures": failures,
        "success_rate": f"{(len(successes) / max(1, len(request_metrics))):.4f}",
        "elapsed_secs": f"{elapsed_secs:.3f}",
        "ttft_mean_ms": f"{mean(ttfts):.3f}",
        "ttft_p50_ms": f"{percentile(ttfts, 0.50):.3f}",
        "ttft_p99_ms": f"{percentile(ttfts, 0.99):.3f}",
        "tpot_mean_ms": f"{mean(tpots):.3f}",
        "e2e_mean_ms": f"{mean(e2es):.3f}",
        "e2e_p50_ms": f"{percentile(e2es, 0.50):.3f}",
        "output_chunks": chunks,
        "output_chunks_per_sec": f"{(chunks / elapsed_secs if elapsed_secs > 0 else 0):.3f}",
        "gpu_mib_before": "" if gpu_before is None else gpu_before,
        "gpu_mib_peak": gpu_peak,
        "gpu_mib_avg": f"{gpu_avg:.3f}",
        "gpu_mib_p95": f"{gpu_p95:.3f}",
        "gpu_mib_after": "" if gpu_after is None else gpu_after,
        "gpu_mib_delta_peak": "" if gpu_before is None else max(0, gpu_peak - gpu_before),
        **parsed,
        "triattention_activation_seen": str(parsed["triattention_activation_seen"]).lower(),
        "log_path": str(log_path),
    }


def write_row(output_csv: Path, row: dict[str, Any]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = output_csv.exists()
    with output_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare kvcached-only and kvcached+TriAttention on a long-context vLLM workload."
    )
    parser.add_argument("--model", default="/root/data/models/Qwen/Qwen3-8B")
    parser.add_argument("--served-model-name", default="qwen3-8b")
    parser.add_argument(
        "--stats-path",
        default=None,
        help=(
            "TriAttention sparse stats .pt file. Defaults to "
            f"<triattention-root>/{DEFAULT_STATS_RELATIVE_PATH}."
        ),
    )
    parser.add_argument("--triattention-root", default=default_triattention_root())
    parser.add_argument("--vllm-command", default="vllm")
    parser.add_argument("--port", type=int, default=12346)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=Path("results/results_vllm_tri_compare.csv"))
    parser.add_argument("--log-dir", type=Path, default=Path(tempfile.gettempdir()) / "triattn-compare")
    parser.add_argument("--prompt-repeat", type=int, default=1200)
    parser.add_argument(
        "--prompt-token-margin",
        type=int,
        default=512,
        help=(
            "Safety margin kept below max_model_len - max_tokens when auto-fitting "
            "the long prompt."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrencies", type=parse_int_list, default=[4, 8, 16])
    parser.add_argument("--budgets", type=parse_int_list, default=[1024, 2048])
    parser.add_argument("--divide-length", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=2400)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--kvcached-log-level", default="DEBUG")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--online", dest="offline", action="store_false", help="Allow HF/Transformers network lookups.")
    parser.set_defaults(offline=True)
    parser.add_argument(
        "--extra-vllm-arg",
        action="append",
        default=[],
        help="Append one raw argument to vllm serve. Repeat for multiple args.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Run only TriAttention rows.",
    )
    parser.add_argument(
        "--skip-triattention",
        action="store_true",
        help="Run only kvcached baseline rows.",
    )
    return parser.parse_args()


def resolve_triattention_paths(args: argparse.Namespace) -> None:
    root = Path(args.triattention_root).expanduser()
    args.triattention_root = str(root)
    if args.stats_path is None:
        args.stats_path = str(root / DEFAULT_STATS_RELATIVE_PATH)
    else:
        args.stats_path = str(Path(args.stats_path).expanduser())


def validate_triattention_inputs(args: argparse.Namespace, plan: list[tuple[str, int | None, int, int]]) -> None:
    if not any(label == "triattention" for label, _budget, _pair_budget, _concurrency in plan):
        return
    root = Path(args.triattention_root)
    if not root.exists():
        raise FileNotFoundError(
            f"TriAttention root does not exist: {root}. Pass --triattention-root "
            "pointing at the uploaded triattention-main directory."
        )
    marker = root / "triattention" / "vllm" / "runtime" / "integration_monkeypatch.py"
    if not marker.exists():
        raise FileNotFoundError(
            f"TriAttention runtime integration was not found under: {root}. "
            "Expected triattention/vllm/runtime/integration_monkeypatch.py."
        )
    stats = Path(args.stats_path)
    if not stats.exists():
        raise FileNotFoundError(
            f"TriAttention sparse stats file does not exist: {stats}. "
            "Pass --stats-path pointing at qwen3_8b.pt or the stats for your model."
        )


def main() -> int:
    args = parse_args()
    resolve_triattention_paths(args)
    base_url = f"http://127.0.0.1:{args.port}"

    plan: list[tuple[str, int | None, int, int]] = []
    for concurrency in args.concurrencies:
        for budget in args.budgets:
            if not args.skip_baseline:
                plan.append(("kvcached", None, budget, concurrency))
            if not args.skip_triattention:
                plan.append(("triattention", budget, budget, concurrency))

    validate_triattention_inputs(args, plan)

    print(f"Writing CSV to {args.output_csv}")
    print(f"Server logs in {args.log_dir}")
    print(f"TriAttention root: {args.triattention_root}")
    print(f"TriAttention stats: {args.stats_path}")
    print(
        f"Workload: prompt_repeat={args.prompt_repeat}, max_tokens={args.max_tokens}, "
        f"concurrencies={args.concurrencies}, budgets={args.budgets}"
    )

    for idx, (label, budget, pair_budget, concurrency) in enumerate(plan, start=1):
        budget_tag = f"budget{pair_budget}"
        log_name = (
            f"concurrency-{label}-{budget_tag}-tok{args.max_tokens}-"
            f"conc{concurrency}-{int(time.time())}.log"
        )
        log_path = args.log_dir / log_name
        print(f"\n[{idx}/{len(plan)}] starting {label} {budget_tag} concurrency={concurrency}")
        handle = start_server(args, label=label, budget=budget, log_path=log_path)
        server_stopped = False
        try:
            wait_for_server(base_url, args.startup_timeout)
            prompt, actual_prompt_repeat, prompt_tokens = fit_prompt_to_context(
                base_url=base_url,
                model_name=args.served_model_name,
                requested_prompt_repeat=args.prompt_repeat,
                max_model_len=args.max_model_len,
                max_tokens=args.max_tokens,
                margin_tokens=args.prompt_token_margin,
            )
            if prompt_tokens is not None:
                total = prompt_tokens + args.max_tokens
                print(
                    f"prompt_repeat={actual_prompt_repeat}/{args.prompt_repeat}, "
                    f"prompt_tokens={prompt_tokens}, requested_total={total}, "
                    f"max_model_len={args.max_model_len}, margin={args.prompt_token_margin}"
                )
            else:
                print(
                    "prompt token count unavailable; using requested "
                    f"prompt_repeat={actual_prompt_repeat}"
                )
            gpu_before = read_gpu_mib(args.gpu_index)
            with GpuSampler(args.gpu_index, args.gpu_sample_interval) as sampler:
                request_metrics, elapsed_secs = run_workload(
                    args,
                    prompt=prompt,
                    concurrency=concurrency,
                )
            gpu_after = read_gpu_mib(args.gpu_index)
            stop_server(handle)
            server_stopped = True
            row = build_row(
                args,
                label=label,
                budget=budget,
                concurrency=concurrency,
                actual_prompt_repeat=actual_prompt_repeat,
                elapsed_secs=elapsed_secs,
                request_metrics=request_metrics,
                gpu_before=gpu_before,
                gpu_after=gpu_after,
                gpu_samples=sampler.samples,
                log_path=log_path,
            )
            write_row(args.output_csv, row)
            print(
                "done: "
                f"success={row['successes']}/{row['requests']} "
                f"ttft_mean_ms={row['ttft_mean_ms']} "
                f"e2e_mean_ms={row['e2e_mean_ms']} "
                f"gpu_peak={row['gpu_mib_peak']}MiB "
                f"compression_events={row['compression_events']} "
                f"skipped={row['compression_skipped_events']} "
                f"freed_blocks={row['freed_blocks']} "
                f"activation={row['triattention_activation_seen']} "
                f"activation_failures={row['triattention_activation_failures']} "
                f"errors={row['error_count']} "
                f"log={row['log_path']}"
            )
            if row.get("compression_skip_reasons"):
                print(f"skip_reasons={row['compression_skip_reasons']}")
            failures = [m.error for m in request_metrics if not m.ok]
            if failures:
                print("first failure:", failures[0])
        finally:
            if not server_stopped:
                stop_server(handle)
    print(f"\nFinished. Results: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
