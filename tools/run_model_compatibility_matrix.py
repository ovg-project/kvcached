# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).parents[1]
DEFAULT_MANIFEST = ROOT / "tools" / "model_compatibility_matrix.json"
DEFAULT_SMOKE_SCRIPT = ROOT / "tools" / "run_engine_smoke.sh"
ENGINES = ("vllm", "sglang")
LAYOUTS = ("contiguous", "non-contiguous")
STATUSES = ("pass", "crash-at-startup", "garbled-output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the manually triggered kvcached model compatibility matrix."
    )
    parser.add_argument("--engine", choices=("all", *ENGINES), default="all")
    parser.add_argument("--layout", choices=("all", *LAYOUTS), default="all")
    parser.add_argument(
        "--model",
        default="all",
        help="Model key or Hugging Face model ID; use 'all' for the full matrix.",
    )
    parser.add_argument(
        "--model-override",
        default="",
        help="Override the model ID when exactly one manifest model is selected.",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--smoke-script", type=Path, default=DEFAULT_SMOKE_SCRIPT)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--fail-on-non-pass", action="store_true")
    return parser.parse_args()


def load_models(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("model matrix manifest must use schema_version 1")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("model matrix manifest must contain a non-empty models list")

    seen = set()
    for entry in models:
        required = ("id", "model", "architecture")
        if not isinstance(entry, dict) or any(not entry.get(key) for key in required):
            raise ValueError(f"invalid model matrix entry: {entry!r}")
        if entry["id"] in seen:
            raise ValueError(f"duplicate model matrix id: {entry['id']}")
        seen.add(entry["id"])
        if "page_size_mb" in entry and (
            not isinstance(entry["page_size_mb"], int)
            or entry["page_size_mb"] <= 0
        ):
            raise ValueError(f"invalid page_size_mb for {entry['id']}")
    return models


def select_values(requested: str, values: Iterable[str]) -> List[str]:
    return list(values) if requested == "all" else [requested]


def select_models(
    models: List[Dict[str, Any]], requested: str, override: str
) -> List[Dict[str, Any]]:
    if requested == "all":
        selected = [dict(model) for model in models]
    else:
        selected = [
            dict(model)
            for model in models
            if requested in (model["id"], model["model"])
        ]
        if not selected:
            known = ", ".join(model["id"] for model in models)
            raise ValueError(f"unknown model selector {requested!r}; choose one of: {known}")
    if override:
        if len(selected) != 1:
            raise ValueError("--model-override requires exactly one selected model")
        selected[0]["model"] = override
    return selected


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def synthesize_result(
    *, engine: str, model: str, layout: str, returncode: int
) -> Dict[str, Any]:
    return {
        "engine": engine,
        "exit_code": returncode,
        "layout": layout,
        "model": model,
        "phase": "unknown",
        "status": "crash-at-startup",
    }


def run_cell(
    *,
    entry: Dict[str, Any],
    engine: str,
    layout: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model = entry["model"]
    cell_dir = args.artifact_dir / engine / entry["id"] / layout
    cell_dir.mkdir(parents=True, exist_ok=True)
    result_file = cell_dir / "result.json"
    log_file = cell_dir / "runner.log"
    python = (
        os.environ.get("VLLM_PYTHON", "python")
        if engine == "vllm"
        else os.environ.get("SGLANG_PYTHON", "python")
    )
    env = os.environ.copy()
    env.update(
        {
            "ENGINE": engine,
            "LAYOUT": layout,
            "LOG_DIR": str(cell_dir),
            "MAX_MODEL_LEN": str(args.max_model_len),
            "MODEL": model,
            "PYTHON": python,
            "RESULT_FILE": str(result_file),
            "STARTUP_TIMEOUT": str(args.startup_timeout),
        }
    )
    if entry.get("page_size_mb"):
        env["KVCACHED_PAGE_SIZE_MB"] = str(entry["page_size_mb"])
    else:
        env.pop("KVCACHED_PAGE_SIZE_MB", None)

    started = time.monotonic()
    with log_file.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            ["bash", str(args.smoke_script)],
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
    duration = round(time.monotonic() - started, 3)
    if result_file.exists():
        result = json.loads(result_file.read_text(encoding="utf-8"))
    else:
        result = synthesize_result(
            engine=engine,
            model=model,
            layout=layout,
            returncode=completed.returncode,
        )
    if result.get("status") not in STATUSES:
        raise ValueError(f"smoke test returned unsupported status: {result!r}")
    result.update(
        {
            "architecture": entry["architecture"],
            "duration_seconds": duration,
            "log": str(log_file.relative_to(args.artifact_dir)),
            "model_key": entry["id"],
        }
    )
    return result


def markdown_report(results: List[Dict[str, Any]]) -> str:
    symbols = {
        "pass": "PASS",
        "crash-at-startup": "CRASH_AT_STARTUP",
        "garbled-output": "GARBLED_OUTPUT",
    }
    lines = [
        "# kvcached model compatibility matrix",
        "",
        "| Engine | Model | Layout | Result | Duration |",
        "|---|---|---|---|---:|",
    ]
    for result in results:
        lines.append(
            "| {engine} | `{model}` | {layout} | {status} | {duration:.1f}s |".format(
                engine=result["engine"],
                model=result["model"],
                layout=result["layout"],
                status=symbols[result["status"]],
                duration=result["duration_seconds"],
            )
        )
    counts = {status: 0 for status in STATUSES}
    for result in results:
        counts[result["status"]] += 1
    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- pass: {counts['pass']}",
            f"- crash-at-startup: {counts['crash-at-startup']}",
            f"- garbled-output: {counts['garbled-output']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.startup_timeout <= 0 or args.max_model_len <= 0:
        raise ValueError("timeouts and model length must be positive")
    models = select_models(load_models(args.manifest), args.model, args.model_override)
    engines = select_values(args.engine, ENGINES)
    layouts = select_values(args.layout, LAYOUTS)
    cells = [
        (entry, engine, layout)
        for engine in engines
        for entry in models
        for layout in layouts
    ]
    if args.check_only:
        print(
            f"Model compatibility preflight passed: cells={len(cells)}, "
            f"engines={','.join(engines)}, layouts={','.join(layouts)}"
        )
        return 0

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, (entry, engine, layout) in enumerate(cells, start=1):
        print(
            f"[{index}/{len(cells)}] engine={engine} model={entry['model']} "
            f"layout={layout}",
            flush=True,
        )
        result = run_cell(
            entry=entry,
            engine=engine,
            layout=layout,
            args=args,
        )
        results.append(result)
        print(f"  result={result['status']}", flush=True)

    summary_json = args.artifact_dir / "matrix-summary.json"
    summary_json.write_text(
        json.dumps({"results": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = markdown_report(results)
    (args.artifact_dir / "matrix-summary.md").write_text(report, encoding="utf-8")
    print(report)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as output:
            output.write(report)
    if args.fail_on_non_pass and any(result["status"] != "pass" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"model compatibility matrix error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
