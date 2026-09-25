#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Operator-owned repair scopes and independent, non-skippable checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict

from engine_compat_artifact import allowed_paths, canonical

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / ".github/engine-compat"
PROBES = {
    "attention": ["tools/engine_compat_gpu_probe.py"],
    "hybrid": ["tools/engine_compat_hybrid_probe.py", "tools/engine_compat_hybrid_hooks.py",
               "tools/engine_compat_tiny_hybrid.json", "tools/engine_compat_gpu_probe.py"],
    "sharing": ["tools/engine_compat_sharing_probe.py", "tools/engine_compat_gpu_probe.py"],
}


def load_profile(name: str) -> Dict[str, Any]:
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", name):
        raise ValueError("Invalid compatibility profile name")
    path = PROFILES / f"{name}.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"No reviewed compatibility profile: {name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"task", "allow", "cpu_tests", "gpu_tests", "qualification", "releases"}
    if not required <= set(value) or set(value) - required - {"repair", "runner", "layouts", "probe"}:
        raise ValueError("Invalid compatibility profile schema")
    value.setdefault("repair", True)
    value.setdefault("runner", "auto")
    value.setdefault("layouts", ["contiguous"])
    value.setdefault("probe", "attention")
    if not isinstance(value["probe"], str) or value["probe"] not in PROBES:
        raise ValueError("Unknown trusted probe")
    if (type(value["repair"]) is not bool or value["runner"] not in ("auto", "v1", "v2")
            or not isinstance(value["layouts"], list) or not value["layouts"]
            or any(item not in ("contiguous", "non-contiguous") for item in value["layouts"])
            or len(set(value["layouts"])) != len(value["layouts"])):
        raise ValueError("Invalid reviewed runtime matrix")
    if value["probe"] == "hybrid" and (
        value["repair"] or value["layouts"] != ["non-contiguous"]
        or (value["runner"], value["releases"]) not in (("v1", ["0.28"]), ("v2", ["0.29"]))
    ):
        raise ValueError("Hybrid acceptance is validation-only for 0.28 V1 or 0.29 MRV2")
    if value["probe"] == "sharing" and (
        value["repair"] or value["runner"] != "v2" or value["releases"] != ["0.29"]
        or value["layouts"] != ["non-contiguous", "contiguous"]
    ):
        raise ValueError("Sharing acceptance is validation-only for 0.29 MRV2 in both layouts")
    if value["qualification"] != "single-gpu":
        raise ValueError("This controller has no validator for the requested qualification")
    if not isinstance(value["task"], str) or not value["task"].strip():
        raise ValueError("A reviewed task is required")
    # Keep artifact validation and repair write permissions on the same policy.
    allow_path = PROFILES / f"{name}-allow.json"
    allow = allowed_paths(allow_path)
    if value["allow"] != allow:
        raise ValueError("Profile and artifact allowlists differ")
    hashes = {}
    for field in ("cpu_tests", "gpu_tests"):
        tests = value[field]
        if not isinstance(tests, list) or not tests or len(set(tests)) != len(tests):
            raise ValueError("Each stage requires nonempty unique trusted checks")
        for test in tests:
            if not isinstance(test, str) or not re.fullmatch(r"tests/test_[a-z0-9_]+\.py", test):
                raise ValueError("Invalid trusted check path")
            check = ROOT / test
            if check.is_symlink() or not check.is_file():
                raise ValueError(f"Missing trusted check: {test}")
            hashes[test] = hashlib.sha256(check.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    for probe in PROBES[value["probe"]]:
        target = ROOT / probe
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"Missing trusted probe: {probe}")
        hashes[probe] = hashlib.sha256(target.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    value["name"] = name
    value["policy_digest"] = hashlib.sha256(canonical(dict(profile=value, checks=hashes))).hexdigest()
    return value


def validate_release(profile: Dict[str, Any], tag: str) -> None:
    match = re.fullmatch(r"v(\d+\.\d+)\.\d+", tag)
    if not match or not isinstance(profile["releases"], list):
        raise ValueError("A stable release and reviewed release range are required")
    if "*" not in profile["releases"] and match[1] not in profile["releases"]:
        raise ValueError(f"Profile {profile['name']} has no trusted contracts for {tag}")


def identity(profile: Dict[str, Any]) -> Dict[str, str]:
    return {key: profile[key] for key in ("name", "policy_digest")}


def validate_mode(profile: Dict[str, Any], mode: str) -> None:
    if mode not in ("validate", "repair") or (mode == "repair" and not profile["repair"]):
        raise ValueError(f"Profile {profile['name']} is validation-only; no agent repair is permitted")


def run_checks(profile: Dict[str, Any], source: Path, output: Path, *, gpu=False, trusted=True) -> int:
    """One pytest process per contract avoids module-mock contamination."""
    output.mkdir(parents=True, exist_ok=True)
    results = []
    status = 0
    for index, test in enumerate(profile["gpu_tests" if gpu else "cpu_tests"]):
        xml = output / f"{index}.xml"
        target = (ROOT if trusted else source) / test
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   "--confcutdir", str(target.parent), str(target), f"--junitxml={xml}"]
        env = dict(os.environ, PYTHONPATH=str(source), ENGINE_COMPAT_SOURCE=str(source),
                   PYTHONDONTWRITEBYTECODE="1",
                   PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0")
        result = subprocess.run(command, cwd=source, env=env, check=False)
        counts = dict(tests=0, failures=0, errors=0, skipped=0)
        if xml.exists():
            for suite in ET.parse(xml).getroot().iter("testsuite"):
                for key in counts:
                    counts[key] += int(suite.get(key, "0"))
        # A missing backend, empty collection, or skipped acceptance is not success.
        code = result.returncode
        if code == 0 and counts["tests"] > 0 and not any(counts[k] for k in ("skipped", "failures", "errors")):
            disposition = 0
        elif code == 1 and counts["tests"] > 0 and counts["failures"] > 0:
            disposition = 1
        else:
            disposition = 2
        status = max(status, disposition)
        results.append(dict(test=test, command=command, exit_code=code, counts=counts,
                            disposition=disposition))
    (output / "checks.json").write_text(json.dumps(dict(
        profile=identity(profile), trusted=trusted, checks=results, exit_code=status,
    ), indent=2) + "\n", encoding="utf-8")
    return status


def run_probe_matrix(profile, source, output, version, candidate_sha):
    """Execute every reviewed runner/layout cell; missing evidence cannot pass."""
    if not re.fullmatch(r"[0-9a-f]{40}", candidate_sha):
        raise ValueError("A pinned candidate SHA is required")
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    status = 0
    for layout in profile["layouts"]:
        destination = output / layout
        command = [sys.executable, str(ROOT / PROBES[profile["probe"]][0]),
                   "--source", str(source), "--output", str(destination),
                   "--version", version, "--candidate-sha", candidate_sha,
                   "--runner", profile["runner"], "--layout", layout, "--mode", "compare"]
        code = subprocess.run(command, check=False).returncode
        report = destination / "result.json"
        value = json.loads(report.read_text()) if report.is_file() else {}
        if (code == 0 and value.get("status") == "passed" and value.get("comparison") == "passed"
                and value.get("candidate_sha") == candidate_sha
                and value.get("expected_vllm_version") == version
                and value.get("requested_runner") == profile["runner"]
                and value.get("layout") == layout):
            disposition = 0
        else:
            disposition = 1 if code == 1 else 2
        cases.append(dict(layout=layout, runner=profile["runner"], exit_code=disposition))
        status = max(status, disposition)
    (output / "matrix.json").write_text(json.dumps(dict(
        profile=identity(profile), candidate_head=candidate_sha, version=version,
        cases=cases, exit_code=status,
    ), indent=2) + "\n", encoding="utf-8")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--mode", choices=("validate", "repair"), default="validate")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--candidate-tests", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--candidate-sha")
    args = parser.parse_args()
    try:
        profile = load_profile(args.profile)
        validate_mode(profile, args.mode)
        if args.tag:
            validate_release(profile, args.tag)
        expected = os.environ.get("ENGINE_COMPAT_POLICY_DIGEST")
        if expected and profile["policy_digest"] != expected:
            raise ValueError("Trusted profile changed between validation stages")
        if args.source:
            if args.output is None:
                parser.error("--output is required with --source")
            if args.probe:
                if not args.tag or not args.candidate_sha:
                    parser.error("--probe requires --tag and --candidate-sha")
                return run_probe_matrix(profile, args.source.resolve(), args.output.resolve(),
                                        args.tag[1:], args.candidate_sha)
            return run_checks(profile, args.source.resolve(), args.output.resolve(),
                              gpu=args.gpu, trusted=not args.candidate_tests)
        print(json.dumps(identity(profile)))
        return 0
    except (ValueError, OSError, ET.ParseError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
