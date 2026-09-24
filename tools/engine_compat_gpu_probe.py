#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Offline single-GPU vLLM smoke/fault probe; exit 0=passed, 1=failed, 2=blocked.

Run a trusted copy in a disposable, credential-free POSIX vLLM/CUDA runtime with
candidate kvcached already built/installed. No installs or downloads are performed.
Use --source CHECKOUT --output NEW_DIRECTORY --version EXACT_VERSION --mode compare.
Pass --candidate-sha HOST_SHA to avoid requiring Git in the runtime; it is supplied
provenance, not a Git-verified identity. Source fingerprints must remain unchanged.
An omitted --model generates a tiny random Llama and local WordLevel tokenizer.
CUDA contexts belong only to isolated children; the supervisor is standard-library-only.
Logs/results are runtime evidence, not attestations: record the runtime exit and
candidate identity independently on the trusted host. Environment filtering is
not a sandbox; never mount credentials. HEAD does not identify uncommitted changes.
No distributed KV transaction chaos, rollback/restart, profiling snapshots,
graphs, MPS, or multi-instance coverage. Random weights do not test model quality.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

EXIT = {"passed": 0, "failed": 1, "blocked": 2}
IGNORED = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
OPTIONS = dict(
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    dtype="float16",
    max_model_len=256,
    max_num_seqs=4,
    max_num_batched_tokens=256,
    gpu_memory_utilization=0.4,
    enforce_eager=True,
    seed=0,
    distributed_executor_backend="uni",
    trust_remote_code=False,
)


class Blocked(RuntimeError):
    pass


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def save(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def version(expected, result):
    try:
        from packaging.version import Version
    except ImportError as exc:
        raise Blocked("The runtime requires packaging") from exc

    try:
        actual = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise Blocked("vLLM is not installed") from exc
    result["actual_vllm_version"] = actual
    if Version(actual).public != Version(expected).public:
        raise Blocked(f"Expected exact vLLM {expected!r}, installed {actual!r}")


def fingerprint(source, output):
    digest = hashlib.sha256()
    count = 0

    def fail(error):
        raise error

    for root, directories, files in os.walk(source, onerror=fail):
        directories[:] = sorted(
            name
            for name in directories
            if name not in IGNORED and Path(root, name).resolve() != output
        )
        for name in sorted(files):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = Path(root, name)
            content = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    content.update(chunk)
            digest.update(path.relative_to(source).as_posix().encode() + b"\0" + content.digest())
            count += 1
    return dict(sha256=digest.hexdigest(), files=count)


def make_model(path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
    vocab.update({f"word{i}": i + 4 for i in range(124)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.save_pretrained(str(path))
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(device="cpu", dtype=torch.float16)
    model.save_pretrained(str(path), safe_serialization=True)
    check(not torch.cuda.is_initialized(), "CPU model generation initialized CUDA")


def runner_identity(llm, expected):
    runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    module = type(runner).__module__
    actual = {
        "vllm.v1.worker.gpu_model_runner": "v1",
        "vllm.v1.worker.gpu.model_runner": "v2",
    }.get(module)
    check(actual is not None, f"Unrecognized runner module: {module}")
    check(expected == "auto" or actual == expected, f"Expected {expected}, loaded {actual}")
    return dict(runner=actual, module=module)


def worker(args, result):
    version(args.version, result)
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise Blocked("The runtime requires a loadable CUDA PyTorch installation") from exc

    if args._stage == "patched":
        from kvcached.integration.vllm import patches

        origin = Path(patches.__file__).resolve()
        result["patch_module"] = str(origin)
        check(origin.is_relative_to(args.source), "Patch module is outside candidate checkout")
        try:
            extension = importlib.import_module("kvcached.vmm_ops")
        except (ImportError, OSError) as exc:
            raise Blocked("Prebuilt kvcached CUDA extension cannot be loaded") from exc
        result["extension_module"] = str(extension.__file__)
    if args._stage == "prepare":
        if args.model is None:
            try:
                make_model(args.output / "model")
            except ModuleNotFoundError as exc:
                raise Blocked("Model generation requires transformers and tokenizers") from exc
        elif not (args.model / "config.json").is_file():
            raise Blocked("--model must be a local model/tokenizer directory with config.json")
        return
    result.update(torch_version=str(torch.__version__), cuda_version=torch.version.cuda)
    if not torch.version.cuda or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Blocked("Exactly one visible CUDA GPU and CUDA-enabled PyTorch are required")
    torch.cuda.set_device(0)
    result["gpu"] = torch.cuda.get_device_name(0)
    if args._stage == "oom":
        _, total = torch.cuda.mem_get_info()
        result.update(total_gpu_bytes=int(total), requested_allocation_bytes=2 * int(total))
        try:
            torch.empty(2 * int(total), dtype=torch.uint8, device="cuda")
        except torch.cuda.OutOfMemoryError:
            result["allocation_oom_caught"] = True
            traceback.print_exc()
        else:
            raise AssertionError("Oversized CUDA allocation unexpectedly succeeded")
        torch.cuda.empty_cache()
        healthy = torch.ones(4096, device="cuda")
        check((healthy + 1).sum().item() == 8192, "GPU operation did not recover after OOM")
        torch.cuda.synchronize()
        result["gpu_operation_recovered"] = True
        return
    patched = args._stage == "patched"
    if patched:
        importlib.import_module("kvcached.integration.vllm.autopatch")
    import vllm
    from packaging.version import Version
    from vllm.v1.worker.gpu_worker import Worker

    if Version(vllm.__version__).public != Version(args.version).public:
        raise Blocked("Imported vLLM differs from the exact expected version")
    active = bool(getattr(Worker.determine_available_memory, "determine_available_memory", False))
    check(active == patched, f"Worker patch active={active}, expected {patched}")
    model = str(args.model or args.output / "model")
    result.update(worker_patch_active=active, llm_options=OPTIONS, model=model)
    llm = vllm.LLM(model=model, tokenizer=model, **OPTIONS)
    result["loaded_runner"] = runner_identity(llm, args.runner)
    if patched:
        from kvcached.integration.vllm import interfaces

        check(interfaces._contiguous_layout == (args.layout == "contiguous"),
              "Requested elastic storage layout was not activated")
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.encode(f"word{i} word{i + 1} word{i + 2}", add_special_tokens=False)
        for i in range(16)
    ]
    check(all(0 < len(ids) <= 240 for ids in prompts), "Prompt exceeds the context limit")
    sampling = dict(temperature=0.0, ignore_eos=True, max_tokens=16)
    result.update(prompt_token_ids=prompts, sampling_options=sampling)
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompts],
        vllm.SamplingParams(**sampling),
        use_tqdm=False,
    )
    check(len(outputs) == 16, "Expected 16 completed requests")
    result["token_ids"] = []
    for index, output in enumerate(outputs):
        check(list(output.prompt_token_ids) == prompts[index], "Prompt ordering/IDs changed")
        check(output.finished and len(output.outputs) == 1, "Request did not finish once")
        completion = output.outputs[0]
        ids = list(completion.token_ids)
        result["token_ids"].append(ids)
        check(len(ids) == 16 and completion.finish_reason == "length", "Incomplete completion")
    result["output_tokens"] = sum(map(len, result["token_ids"]))
    check(result["output_tokens"] == 256, "Expected 256 output tokens")
    if patched:
        from kvcached.observability import get_registered_kv_cache_pool_snapshot_dicts

        pools = get_registered_kv_cache_pool_snapshot_dicts(integration="vllm")
        result["kv_pools"] = pools
        check(
            any(pool["virtual_total_bytes"] > pool["mapped_bytes"] > 0 for pool in pools),
            "No partially mapped KVCached virtual KV pool was exercised",
        )
    # Embedded LLM users own shutdown; finish it while CUDA is still available.
    llm.llm_engine.engine_core.shutdown()
    if patched:
        from kvcached.integration.vllm.interfaces import shutdown_kvcached

        shutdown_kvcached()
    result["explicit_shutdown_completed"] = True


def environment(args, stage):
    allowed = (
        "PATH",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "VIRTUAL_ENV",
    )
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    root = args.output / "runtime" / stage
    for directory in ("home", "tmp", "cache"):
        (root / directory).mkdir(parents=True)
    env.update(
        HOME=str(root / "home"),
        TMPDIR=str(root / "tmp"),
        XDG_CONFIG_HOME=str(root / "home" / ".config"),
        XDG_CACHE_HOME=str(root / "cache"),
        PYTHONPATH=str(args.source),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
        VLLM_PLUGINS="",
        VLLM_USE_V1="1",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONHASHSEED="0",
        ENABLE_KVCACHED="true" if stage == "patched" else "false",
        KVCACHED_AUTOPATCH="1" if stage == "patched" else "0",
        KVCACHED_IPC_NAME="gpu_probe_" + uuid.uuid4().hex,
        KVCACHED_PAGE_SIZE_MB="2",
        KVCACHED_PAGE_PREALLOC_ENABLED="false",
        KVCACHED_MIN_RESERVED_PAGES="0",
        KVCACHED_MAX_RESERVED_PAGES="0",
        KVCACHED_MAX_CACHED_TOKENS="1024",
        KVCACHED_CONTIGUOUS_LAYOUT="true" if args.layout == "contiguous" else "false",
    )
    if args.runner != "auto":
        env["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "v2" else "0"
    env["CUDA_VISIBLE_DEVICES"] = (
        "" if stage == "prepare" else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    )
    for key in (
        "HF_HOME",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "VLLM_CACHE_ROOT",
        "FLASHINFER_WORKSPACE_BASE",
    ):
        env[key] = str(root / "cache" / key.lower())
    return env


def cleanup(process):
    # Kill only our session, including descendants after the leader has exited.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if sig == signal.SIGKILL:
                raise


def run(args, stage):
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
        "--_stage",
        stage,
        "--candidate-sha",
        args.sha,
        "--runner",
        args.runner,
        "--layout",
        args.layout,
    ]
    if args.model:
        command += ["--model", str(args.model)]
    env = environment(args, stage)
    observed = dict(status="failed", stage=stage, timed_out=False, log=f"{stage}.log")
    with (args.output / f"{stage}.log").open("w", encoding="utf-8") as log:
        log.write(json.dumps(dict(command=command, environment=env)) + "\n")
        log.flush()
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
        except subprocess.TimeoutExpired:
            observed["timed_out"] = True
            observed["status"] = "blocked"
            traceback.print_exc(file=log)
        finally:
            cleanup(process)
        observed["returncode"] = process.returncode
    report = args.output / f"{stage}.json"
    if report.is_file():
        detail = json.loads(report.read_text(encoding="utf-8"))
        observed["result"] = detail
        check(detail["expected_vllm_version"] == args.version, "Child version pin differs")
        check(detail["candidate_sha"] == args.sha, "Child candidate SHA differs")
        if not observed["timed_out"] and process.returncode == EXIT.get(detail["status"]):
            observed["status"] = detail["status"]
    print(stage, observed["status"], flush=True)
    return observed


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "version"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--runner", choices=("auto", "v1", "v2"), default="auto")
    parser.add_argument("--layout", choices=("contiguous", "non-contiguous"), default="contiguous")
    parser.add_argument("--mode", choices=("native", "patched", "compare"), default="compare")
    parser.add_argument(
        "--timeout", type=int, default=600, help="Per-child timeout, 1..3600 seconds"
    )
    parser.add_argument(
        "--_stage", choices=("prepare", "native", "patched", "oom"), help=argparse.SUPPRESS
    )
    parser.add_argument("--candidate-sha", help="Trusted-host 40-hex SHA; skips runtime Git lookup")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 3600:
        parser.error("--timeout must be between 1 and 3600")
    if args.candidate_sha is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", args.candidate_sha):
        parser.error("--candidate-sha must contain exactly 40 hexadecimal characters")
    args.source, args.output = Path(args.source).resolve(), Path(args.output).resolve()
    args.model = args.model.resolve() if args.model else None
    if not args._stage:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("BLOCKED: --output exists; refusing to overwrite evidence", file=sys.stderr)
            return 2
    result = dict(
        status="failed", expected_vllm_version=args.version, candidate_sha=args.candidate_sha,
        requested_runner=args.runner, layout=args.layout,
    )
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        if args._stage:
            worker(args, result)
            result["status"] = "passed"
        else:
            result.update(mode=args.mode, stages={}, comparison="not_run", limitations=__doc__)
            if not args.source.is_dir():
                raise Blocked("--source must be an existing candidate directory")
            if args.candidate_sha is not None:
                args.sha = args.candidate_sha.lower()
                result["candidate_sha_provenance"] = "supplied_by_trusted_host"
            else:
                try:
                    args.sha = (
                        subprocess.check_output(
                            ["git", "-C", str(args.source), "rev-parse", "--verify", "HEAD"],
                            stderr=subprocess.STDOUT,
                            timeout=15,
                        )
                        .decode()
                        .strip()
                    )
                except (OSError, subprocess.CalledProcessError) as exc:
                    raise Blocked("Supply --candidate-sha when runtime Git is unavailable") from exc
                result["candidate_sha_provenance"] = "runtime_git_head"
            result["candidate_sha"] = args.sha
            result["source_fingerprint_before"] = fingerprint(args.source, args.output)
            result["fingerprint_ignored"] = sorted(IGNORED) + ["*.pyc", "*.pyo", "--output"]
            version(args.version, result)
            if os.name != "posix":
                raise Blocked("POSIX runtime required for owned process-group cleanup")
            result["stages"]["prepare"] = run(args, "prepare")
            if result["stages"]["prepare"]["status"] == "passed":
                modes = ["native", "patched"] if args.mode == "compare" else [args.mode]
                for stage in modes + ["oom"]:
                    result["stages"][stage] = run(args, stage)
                    save(args.output / "result.json", result)
                if args.mode == "compare":
                    left, right = [result["stages"][mode] for mode in modes]
                    if left["status"] == right["status"] == "passed":
                        result["comparison"] = "failed"
                        for field in ("prompt_token_ids", "token_ids"):
                            check(
                                left["result"][field] == right["result"][field],
                                f"Native/patched {field} differ",
                            )
                        result["comparison"] = "passed"
            statuses = [item["status"] for item in result["stages"].values()]
            result["status"] = (
                "failed"
                if "failed" in statuses
                else "blocked" if "blocked" in statuses else "passed"
            )
    except (Blocked, KeyboardInterrupt) as exc:
        result.update(status="blocked", error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    except BaseException as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM, previous)
        if "source_fingerprint_before" in result:
            try:
                result["source_fingerprint_after"] = fingerprint(args.source, args.output)
                check(
                    result["source_fingerprint_before"] == result["source_fingerprint_after"],
                    "Candidate source files changed during the probe",
                )
            except Exception as exc:
                result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
                traceback.print_exc()
        save(args.output / f"{args._stage or 'result'}.json", result)
    print(json.dumps(result), flush=True)
    return EXIT[result["status"]]


if __name__ == "__main__":
    sys.exit(main())
