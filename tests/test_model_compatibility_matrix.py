# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "run_model_compatibility_matrix.py"
MANIFEST = ROOT / "tools" / "model_compatibility_matrix.json"


def run_matrix(tmp_path: Path, *args: str, smoke_script: Optional[Path] = None):
    command = [
        sys.executable,
        str(SCRIPT),
        "--artifact-dir",
        str(tmp_path / "artifacts"),
    ]
    if smoke_script:
        command.extend(["--smoke-script", str(smoke_script)])
    command.extend(args)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_manifest_covers_issue_425_models():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    models = payload["models"]

    assert payload["schema_version"] == 1
    assert len(models) == 5
    assert {model["id"] for model in models} == {
        "qwen3",
        "gpt-oss",
        "qwen3.5",
        "gemma-4-e2b",
        "gemma-4-12b",
    }
    qwen35 = next(model for model in models if model["id"] == "qwen3.5")
    assert qwen35["page_size_mb"] == 4


def test_full_matrix_preflight_has_twenty_cells(tmp_path):
    completed = run_matrix(tmp_path, "--check-only")

    assert completed.returncode == 0
    assert "cells=20" in completed.stdout
    assert "engines=vllm,sglang" in completed.stdout


def test_filters_and_model_override_are_validated(tmp_path):
    completed = run_matrix(
        tmp_path,
        "--check-only",
        "--engine",
        "vllm",
        "--layout",
        "contiguous",
        "--model",
        "qwen3",
        "--model-override",
        "Qwen/Qwen3-0.6B",
    )
    assert completed.returncode == 0
    assert "cells=1" in completed.stdout

    completed = run_matrix(
        tmp_path,
        "--check-only",
        "--model-override",
        "Qwen/Qwen3-0.6B",
    )
    assert completed.returncode == 2
    assert "requires exactly one selected model" in completed.stdout


def test_cells_continue_and_write_json_and_markdown_reports(tmp_path):
    fake_smoke = tmp_path / "fake-smoke.sh"
    fake_smoke.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
status=pass
exit_code=0
if [[ "${LAYOUT}" == "non-contiguous" ]]; then
  status=garbled-output
  exit_code=1
fi
STATUS="${status}" EXIT_CODE="${exit_code}" python - <<'PY'
import json
import os
with open(os.environ["RESULT_FILE"], "w", encoding="utf-8") as output:
    json.dump({
        "engine": os.environ["ENGINE"],
        "exit_code": int(os.environ["EXIT_CODE"]),
        "layout": os.environ["LAYOUT"],
        "model": os.environ["MODEL"],
        "phase": "ready",
        "status": os.environ["STATUS"],
    }, output)
PY
exit "${exit_code}"
""",
        encoding="utf-8",
    )
    fake_smoke.chmod(0o755)

    completed = run_matrix(
        tmp_path,
        "--engine",
        "vllm",
        "--model",
        "qwen3",
        smoke_script=fake_smoke,
    )

    assert completed.returncode == 0
    summary_path = tmp_path / "artifacts" / "matrix-summary.json"
    results = json.loads(summary_path.read_text(encoding="utf-8"))["results"]
    assert [result["status"] for result in results] == [
        "pass",
        "garbled-output",
    ]
    report = (tmp_path / "artifacts" / "matrix-summary.md").read_text(
        encoding="utf-8"
    )
    assert "PASS" in report
    assert "GARBLED_OUTPUT" in report


def test_strict_mode_fails_after_writing_complete_report(tmp_path):
    fake_smoke = tmp_path / "fake-smoke.sh"
    fake_smoke.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
STATUS=crash-at-startup EXIT_CODE=1 python - <<'PY'
import json
import os
with open(os.environ["RESULT_FILE"], "w", encoding="utf-8") as output:
    json.dump({
        "engine": os.environ["ENGINE"],
        "exit_code": 1,
        "layout": os.environ["LAYOUT"],
        "model": os.environ["MODEL"],
        "phase": "starting",
        "status": "crash-at-startup",
    }, output)
PY
exit 1
""",
        encoding="utf-8",
    )
    fake_smoke.chmod(0o755)

    completed = run_matrix(
        tmp_path,
        "--engine",
        "sglang",
        "--layout",
        "contiguous",
        "--model",
        "qwen3",
        "--fail-on-non-pass",
        smoke_script=fake_smoke,
    )

    assert completed.returncode == 1
    assert (tmp_path / "artifacts" / "matrix-summary.json").exists()
