#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Offline tiny hybrid/partial-prefix acceptance, not full-model qualification.

Only 0.28 V1 and 0.29 MRV2, non-contiguous storage, eager FP16, TP=PP=1.
Both schedules require native, elastic and actual CoW allocation-miss cases.
No downloads, agent repair, unsafe RPC serialization, TP/PP or MPS claims.
The trusted host must independently pin candidate and controller revisions.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from engine_compat_gpu_probe import (
    EXIT,
    Blocked,
    check,
    cleanup,
    environment,
    fingerprint,
    interrupted,
    save,
    version,
)

FIXTURE = Path(__file__).with_name("engine_compat_tiny_hybrid.json")
RUNNERS = {"v1": "vllm.v1.worker.gpu_model_runner", "v2": "vllm.v1.worker.gpu.model_runner"}
STAGES = tuple(
    f"{schedule}-{mode}"
    for schedule in ("sync", "async")
    for mode in ("native", "elastic", "fault")
)
FATAL = re.compile(
    r"illegal memory access|illegal instruction|invalid resource handle|EngineDeadError|"
    r"unspecified launch failure|unknown CUDA driver error|"
    r"EngineCore failed|Page \d+ is not mapped|Cannot get \d+ free blocks|CUDA out of memory",
    re.I,
)


def worker_evidence(worker):
    runner = worker.model_runner
    config = runner.kv_cache_config
    return dict(
        runner_module=type(runner).__module__,
        async_scheduling=runner.vllm_config.scheduler_config.async_scheduling,
        prefix_match_unit=runner.vllm_config.cache_config.prefix_match_unit,
        groups=[
            dict(
                type=type(group.kv_cache_spec).__name__,
                block_size=group.kv_cache_spec.block_size,
                page_size_bytes=group.kv_cache_spec.page_size_bytes,
                layers=list(group.layer_names),
            )
            for group in config.kv_cache_groups
        ],
    )


def worker(args, result):
    # Hooks are loaded through our generated sitecustomize in every spawned
    # process. The registered string-method RPC keeps safe serialization on.
    import torch
    import vllm
    from packaging.version import Version
    from vllm.v1.worker.gpu_worker import Worker

    version(args.version, result)
    check(Version(vllm.__version__).public == args.version, "Imported vLLM version differs")
    if not torch.version.cuda or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Blocked("Exactly one visible CUDA GPU is required")
    torch.cuda.set_device(0)
    mode = args._stage.split("-")[1]
    elastic = mode != "native"
    if elastic:
        import kvcached.integration.vllm.autopatch  # noqa: F401
        from kvcached.integration.vllm import patches

        check(
            Path(patches.__file__).resolve().is_relative_to(args.source),
            "Loaded patches outside candidate",
        )
    check(
        bool(getattr(Worker.determine_available_memory, "determine_available_memory", False))
        == elastic,
        "Worker patch activation differs from requested mode",
    )
    options = dict(
        model=str(args.output / "model"),
        load_format="dummy",
        skip_tokenizer_init=True,
        trust_remote_code=False,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dtype="half",
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
        enable_prefix_caching=True,
        prefix_match_unit=16,
        async_scheduling=args._stage.startswith("async-"),
        limit_mm_per_prompt={"image": 0, "video": 0},
        attention_config={"backend": "TRITON_ATTN"},
        mamba_cache_dtype="float16",
        mamba_ssm_cache_dtype="float32",
        seed=20260921,
    )
    result.update(
        options=options,
        rows=[],
        torch_version=str(torch.__version__),
        model_config_sha256=hashlib.sha256(
            (args.output / "model/config.json").read_bytes()
        ).hexdigest(),
    )
    llm = None
    try:
        llm = vllm.LLM(**options)
        evidence = llm.collective_rpc("hybrid_probe_evidence", timeout=60)
        result["worker_evidence"] = evidence
        check(len(evidence) == 1, "Expected one worker")
        check(evidence[0]["runner_module"] == RUNNERS[args.runner], "Wrong loaded runner")
        check(
            bool(evidence[0]["async_scheduling"]) == options["async_scheduling"], "Wrong schedule"
        )
        check(evidence[0]["prefix_match_unit"] == 16, "Wrong prefix hash unit")
        groups = evidence[0]["groups"]
        check(any(group["type"] == "MambaSpec" for group in groups), "No recurrent state")
        unit = max(group["block_size"] for group in groups)
        check(64 < unit < 1800, "Fixture does not exercise partial blocks")
        if elastic:
            from kvcached.integration.vllm import interfaces

            check(not interfaces._contiguous_layout, "Hybrid storage must be non-contiguous")
        lengths = sorted({48, 64, unit - 1, unit, unit + 1, min(2 * unit - 1, 1984)})
        sampling = vllm.SamplingParams(
            temperature=0, max_tokens=32, ignore_eos=True, detokenize=False
        )
        for length in lengths:
            prompts = [{"prompt_token_ids": [token] * length} for token in (17, 23)]
            for repeat in range(2):
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                check(len(outputs) == 2, "Missing requests")
                for prompt, output in zip(prompts, outputs):
                    check(output.finished and len(output.outputs) == 1, "Incomplete request")
                    completion = output.outputs[0]
                    check(
                        list(output.prompt_token_ids) == prompt["prompt_token_ids"],
                        "Prompt changed",
                    )
                    check(
                        len(completion.token_ids) == 32 and completion.finish_reason == "length",
                        "Truncated generation",
                    )
                    result["rows"].append(
                        dict(
                            input=prompt["prompt_token_ids"][0],
                            length=length,
                            repeat=repeat,
                            tokens=list(completion.token_ids),
                        )
                    )
        result["requests"] = len(result["rows"])
        result["output_tokens"] = sum(len(row["tokens"]) for row in result["rows"])
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
            if elastic:
                from kvcached.integration.vllm.interfaces import shutdown_kvcached

                shutdown_kvcached()
            result["explicit_shutdown_completed"] = True


def validate_case(data, markers, log, stage, runner):
    check(
        data.get("status") == "passed" and data.get("explicit_shutdown_completed"),
        "Incomplete shutdown",
    )
    check(
        data.get("requests") == 24 and data.get("output_tokens") == 768, "Incomplete request totals"
    )
    rows = data.get("rows", [])
    check(len(rows) == 24 and all(len(row["tokens"]) == 32 for row in rows), "Truncated output")
    vectors = {(row["input"], row["length"], row["repeat"]): row["tokens"] for row in rows}
    check(len(vectors) == 24, "Duplicate request evidence")
    evidence = data["worker_evidence"]
    check(
        len(evidence) == 1 and evidence[0]["runner_module"] == RUNNERS[runner],
        "Wrong actual runner",
    )
    check(evidence[0]["prefix_match_unit"] == 16, "Wrong hash unit")
    check(
        bool(evidence[0]["async_scheduling"]) == stage.startswith("async-"), "Wrong actual schedule"
    )
    groups = evidence[0]["groups"]
    check(any(group["type"] == "MambaSpec" for group in groups), "No recurrent state")
    unit = max(group["block_size"] for group in groups)
    check(64 < unit < 1800, "Invalid hybrid allocation geometry")
    expected = {
        (token, length, repeat)
        for token in (17, 23)
        for length in (48, 64, unit - 1, unit, unit + 1, min(2 * unit - 1, 1984))
        for repeat in (0, 1)
    }
    check(set(vectors) == expected, "Missing boundary request coverage")
    check(not FATAL.search(log), "Fatal GPU/lifetime error in case log")
    traces = [line for line in log.splitlines() if "Traceback" in line]
    if traces:
        check(
            all(re.match(r"WARNING .*\[base.py:\d+\] Traceback", line) for line in traces)
            and "Chat template warmup failed" in log
            and "Tokenizer not available when `skip_tokenizer_init=True`" in log,
            "Unexpected traceback in case log",
        )
    counts = collections.Counter(row["event"] for row in markers)
    check(counts["partial_hit"] > 0, "No partial-prefix hit exercised")
    for row in markers:
        if row["event"] == "partial_hit":
            check(
                row["hash_block_size"] == 16 < row["allocation_block_size"], "Invalid partial hit"
            )
        if row["event"] == "copy_submitted":
            check(row["runner"] == runner and row["count"] > 0, "Invalid worker copy")
    if not stage.endswith("native"):
        check(counts["copy_submitted"] > 0, "No worker copy exercised")
    hits = [row["hit"] for row in markers if row["event"] == "cow_admission_miss"]
    check(
        hits == ([1, 2] if stage.endswith("fault") else []),
        "Missing or unexpected fault injections",
    )
    return vectors, dict(counts)


def run_case(args, stage):
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--source",
        str(args.source),
        "--output",
        str(args.output),
        "--version",
        args.version,
        "--runner",
        args.runner,
        "--candidate-sha",
        args.candidate_sha,
        "--_stage",
        stage,
    ]
    env = environment(args, stage)
    elastic = not stage.endswith("native")
    env.update(
        ENABLE_KVCACHED="true" if elastic else "false",
        KVCACHED_AUTOPATCH="1" if elastic else "0",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        KVCACHED_PAGE_SIZE_MB="16",
        KVCACHED_MAX_CACHED_TOKENS="4096",
        HYBRID_MARKER=str(args.output / f"{stage}.jsonl"),
        HYBRID_FAULT="cow-admission" if stage.endswith("fault") else "",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        (str(args.output / "hooks"), str(Path(__file__).parent), str(args.source))
    )
    # Reuse only this probe's own compilation caches across its six children.
    for key in (
        "HF_HOME",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "VLLM_CACHE_ROOT",
        "FLASHINFER_WORKSPACE_BASE",
    ):
        env[key] = str(args.output / "cache" / key.lower())
    # Unix-domain sockets must fit sockaddr_un even when evidence lives in a
    # deeply nested workspace. Remove only this case's private directory, after
    # its entire child process group has been stopped.
    with tempfile.TemporaryDirectory(prefix="kch-", dir="/tmp" if os.name == "posix" else None) as ipc_dir, (
        args.output / f"{stage}.log"
    ).open("w", encoding="utf-8") as log:
        env["TMPDIR"] = ipc_dir
        process = subprocess.Popen(
            command,
            cwd=args.output,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=args.timeout)
        finally:
            cleanup(process)
    if process.returncode not in (0, 1):
        raise Blocked(f"Child could not complete: {stage} ({process.returncode})")
    check(process.returncode == 0, f"Child failed: {stage} ({process.returncode})")
    data = json.loads((args.output / f"{stage}.json").read_text())
    check(
        data.get("candidate_sha") == args.candidate_sha
        and data.get("expected_vllm_version") == args.version,
        "Child candidate/version identity differs",
    )
    check(
        data.get("requested_runner") == args.runner and data.get("layout") == args.layout,
        "Child runner/layout identity differs",
    )
    check(
        data.get("model_config_sha256") == hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "Child fixture differs from trusted configuration",
    )
    marker_path = args.output / f"{stage}.jsonl"
    markers = (
        [json.loads(line) for line in marker_path.read_text().splitlines()]
        if marker_path.exists()
        else []
    )
    vectors, counts = validate_case(
        data, markers, (args.output / f"{stage}.log").read_text(), stage, args.runner
    )
    return data, vectors, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--runner", choices=("v1", "v2"), required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--layout", choices=("non-contiguous",), default="non-contiguous")
    parser.add_argument("--mode", choices=("compare",), default="compare")
    parser.add_argument("--timeout", type=int, default=600, help="Per-child timeout, including cold kernel compilation")
    parser.add_argument("--_stage", choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        not re.fullmatch(r"[0-9a-f]{40}", args.candidate_sha)
        or not 1 <= args.timeout <= 3600
        or not re.fullmatch(r"0\.(28|29)\.\d+", args.version)
        or args.runner != ("v1" if args.version.startswith("0.28.") else "v2")
    ):
        parser.error("Only pinned 0.28 V1 or 0.29 MRV2 configurations are qualified")
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if not args._stage:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Refusing to overwrite existing evidence", file=sys.stderr)
            return 2
    result = dict(
        status="failed",
        comparison="not_run",
        candidate_sha=args.candidate_sha,
        expected_vllm_version=args.version,
        requested_runner=args.runner,
        layout=args.layout,
    )
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        if args._stage:
            worker(args, result)
        else:
            if os.name != "posix" or not args.source.is_dir():
                raise Blocked("A POSIX runtime and existing candidate checkout are required")
            version(args.version, result)
            result["source_fingerprint_before"] = fingerprint(args.source, args.output)
            (args.output / "model").mkdir()
            (args.output / "model/config.json").write_bytes(FIXTURE.read_bytes())
            (args.output / "hooks").mkdir()
            (args.output / "hooks/sitecustomize.py").write_text(
                "import engine_compat_hybrid_hooks\n"
            )
            result["cases"] = []
            baseline = None
            for stage in STAGES:
                data, vectors, counts = run_case(args, stage)
                if stage.endswith("native"):
                    baseline = data, vectors
                else:
                    check(baseline is not None, "Missing native control")
                    check(data["options"] == baseline[0]["options"], "Native options differ")
                    check(
                        data["worker_evidence"] == baseline[0]["worker_evidence"],
                        "Native geometry differs",
                    )
                    check(
                        data["model_config_sha256"] == baseline[0]["model_config_sha256"],
                        "Native model differs",
                    )
                    check(vectors == baseline[1], "Output tokens differ from native control")
                result["cases"].append(
                    dict(stage=stage, requests=24, output_tokens=768, markers=counts)
                )
                save(args.output / "result.json", result)
            result["comparison"] = "passed"
        result["status"] = "passed"
    except (Blocked, KeyboardInterrupt, subprocess.TimeoutExpired) as exc:
        result.update(status="blocked", error=str(exc))
        traceback.print_exc()
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM, previous)
        if "source_fingerprint_before" in result:
            result["source_fingerprint_after"] = fingerprint(args.source, args.output)
            if result["source_fingerprint_before"] != result["source_fingerprint_after"]:
                result.update(status="failed", error="Candidate source changed during validation")
        save(args.output / f"{args._stage or 'result'}.json", result)
    print(json.dumps(result), flush=True)
    return EXIT[result["status"]]


if __name__ == "__main__":
    sys.exit(main())
