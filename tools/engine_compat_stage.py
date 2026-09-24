#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Trusted job supervisor for release compatibility checks and report handoff."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from engine_compat_artifact import load_envelope, materialize, pack
from engine_compat_profile import (
    identity,
    load_profile,
    run_checks,
    validate_mode,
    validate_release,
)
from repair_engine_compat import run_command, tail

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_TIMEOUT = 2700
PUBLICATION_POLL_INTERVAL = 10
EXPECTED_TARGET_CHECKS = {"pre-commit"} | {
    f"{job} ({version})"
    for job in ("CPU pytest", "MyPy")
    for version in ("3.9", "3.10", "3.11", "3.12", "3.13")
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def record(output: Path, value: Dict[str, Any]) -> None:
    write_json(output / "stage.json", value)
    destination = os.environ.get("GITHUB_OUTPUT")
    if destination:
        with open(destination, "a", encoding="utf-8") as stream:
            for key in ("status", "candidate_head", "digest"):
                if key in value:
                    stream.write(f"{key}={value[key]}\n")
    print(json.dumps(value))


def checks(args) -> int:
    source = Path(os.environ["ENGINE_COMPAT_SOURCE"])
    return run_checks(load_profile(args.profile), source,
                      args.output / ("probes" if args.probes else "validation"),
                      trusted=args.probes)


def cpu(args) -> None:
    profile = load_profile(args.profile)
    validate_release(profile, args.tag)
    mode = getattr(args, "mode", "repair")
    validate_mode(profile, mode)
    allow = profile["allow"]
    if mode == "validate":
        if args.prior:
            raise ValueError("Validation-only mode does not retry a repair candidate")
        payload = pack(args.source, args.base, args.tag, allow)
        if payload["files"] or payload["candidate_head"] != args.base:
            raise ValueError("Validation-only mode requires the exact clean baseline")
        code = run_checks(profile, args.source, args.output / "contracts", trusted=True)
        if pack(args.source, args.base, args.tag, allow) != payload:
            raise ValueError("Candidate changed during validation-only checks")
        if code == 0:
            write_json(args.output / "candidate.json", payload)
        record(args.output, dict(
            status="candidate" if code == 0 else "failed" if code == 1 else "blocked",
            candidate_head=payload["candidate_head"], digest=payload["digest"],
            engine_sha=args.engine_sha, profile=identity(profile), mode=mode,
        ))
        return
    prior_report = None
    if args.prior:
        payload = load_envelope(args.prior / "candidate.json")
        materialize(args.source, payload, args.base, args.tag, allow)
        previous = json.loads((args.prior / "stage.json").read_text())
        if (
            previous.get("status") != "failed"
            or previous.get("digest") != payload["digest"]
            or previous.get("candidate_head") != payload["candidate_head"]
            or previous.get("engine_sha") != args.engine_sha
            or previous.get("profile") != identity(profile)
        ):
            raise ValueError("Previous GPU evidence does not match this repair attempt")
        prior_report = args.prior / "failure.json"
    checks_root = args.output / "checks"
    checks_root.mkdir()
    command = [sys.executable, str(Path(__file__).resolve()), "checks", "--profile", args.profile,
               "--output", str(args.output / "check-results")]
    write_json(
        checks_root / "checks.json",
        {
            "probes": [{"name": "vllm-contract", "argv": command + ["--probes"]}],
            "validation": [{"name": "vllm-regression", "argv": command}],
        },
    )
    task = args.output / "task.md"
    task.write_text(
        f"Adapt KVCached to upstream vLLM {args.tag} ({args.engine_sha}). "
        + profile["task"] + " "
        "Use the supplied engine source. Preserve supported older versions, inference mode, "
        "physical allocation failure handling and asynchronous worker ordering. "
        "Do not disable tests or bypass GPU failures. This round only runs CPU checks; "
        "an independent GPU job must still pass before any publication.\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(ROOT / "tools/repair_engine_compat.py"),
        "--source",
        str(args.source),
        "--engine-source",
        str(args.engine),
        "--checks",
        str(checks_root),
        "--task",
        str(task),
        "--output",
        str(args.output / "repair"),
        "--attempts",
        "1",
        "--codex",
        args.codex,
    ]
    for name in allow:
        command.extend(("--allow", name))
    if prior_report:
        command.extend(("--failure-report", str(prior_report)))
    result = run_command(command, ROOT, args.output / "repair.log", 1500, dict(os.environ))
    if result["exit_code"]:
        record(args.output, dict(status="blocked", reason="Bounded repair did not validate"))
        return
    payload = pack(args.source, args.base, args.tag, allow)
    write_json(args.output / "candidate.json", payload)
    record(
        args.output,
        dict(
            status="candidate",
            candidate_head=payload["candidate_head"],
            digest=payload["digest"],
            engine_sha=args.engine_sha,
            profile=identity(profile),
            mode=mode,
        ),
    )


def gpu(args) -> None:
    profile = load_profile(args.profile)
    validate_release(profile, args.tag)
    payload = load_envelope(args.prior / "candidate.json")
    previous = json.loads((args.prior / "stage.json").read_text())
    if (
        previous.get("status") != "candidate"
        or previous.get("digest") != payload["digest"]
        or previous.get("engine_sha") != args.engine_sha
        or previous.get("candidate_head") != payload["candidate_head"]
        or previous.get("profile") != identity(profile)
    ):
        raise ValueError("Candidate envelope is not from the expected CPU stage")
    head = materialize(args.source, payload, args.base, args.tag, profile["allow"])
    write_json(args.output / "candidate.json", payload)
    container_name = "kvcached-compat-" + uuid.uuid4().hex
    env = dict(os.environ, COMPAT_CONTAINER_NAME=container_name, ENGINE_COMPAT_PROFILE=args.profile,
               ENGINE_COMPAT_POLICY_DIGEST=profile["policy_digest"])
    command = [
        "bash",
        str(ROOT / "tools/engine_compat_gpu.sh"),
        str(args.source),
        str(args.output),
        args.tag,
        args.image,
    ]
    if args.gpu_command:
        command = json.loads(args.gpu_command.read_text(encoding="utf-8"))
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
        ):
            raise ValueError("Local GPU command must be an operator-owned argv array")
        env.update(
            ENGINE_COMPAT_SOURCE=str(args.source),
            ENGINE_COMPAT_OUTPUT=str(args.output),
            ENGINE_COMPAT_VERSION=args.tag[1:],
        )
    try:
        result = run_command(command, ROOT, args.output / "gpu.log", 2400, env)
    finally:
        if not args.gpu_command:
            subprocess.run(
                ["docker", "rm", "-f", container_name], capture_output=True, timeout=60, check=False
            )
    code = result["exit_code"]
    if code == 0:
        receipt = json.loads((args.output / "runtime/contracts/checks.json").read_text())
        if (receipt.get("profile") != identity(profile) or receipt.get("exit_code") != 0
                or receipt.get("trusted") is not True
                or [check.get("test") for check in receipt.get("checks", [])] != profile["gpu_tests"]):
            raise ValueError("GPU job did not run this profile's independent contracts")
        matrix = json.loads((args.output / "runtime/probe/matrix.json").read_text())
        expected_cases = [dict(layout=layout, runner=profile["runner"], exit_code=0)
                          for layout in profile["layouts"]]
        if (matrix.get("profile") != identity(profile) or matrix.get("candidate_head") != head
                or matrix.get("version") != args.tag[1:] or matrix.get("exit_code") != 0
                or matrix.get("cases") != expected_cases):
            raise ValueError("GPU job did not pass the complete runner/layout matrix")
    status = "passed" if code == 0 else "failed" if code == 1 else "blocked"
    value = dict(
        status=status,
        candidate_head=head,
        digest=payload["digest"],
        engine_sha=args.engine_sha,
        profile=identity(profile),
        mode=previous.get("mode", "repair"),
        exit_code=code,
    )
    if status == "failed":
        write_json(
            args.output / "failure.json",
            dict(status="failed", source_head=head, summary=tail(args.output / "gpu.log")),
        )
    record(args.output, value)


def publish(args) -> None:
    """Called only after independent full CI and GPU jobs pass; never executes candidate code."""
    profile = load_profile(args.profile)
    validate_release(profile, args.tag)
    payload = load_envelope(args.prior / "candidate.json")
    previous = json.loads((args.prior / "stage.json").read_text())
    if previous.get("mode", "repair") != "repair" or not profile["repair"]:
        raise ValueError("Validation-only evidence cannot authorize publication")
    if (
        previous.get("status") != "passed"
        or previous.get("digest") != payload["digest"]
        or previous.get("engine_sha") != args.engine_sha
        or previous.get("candidate_head") != payload["candidate_head"]
        or previous.get("profile") != identity(profile)
    ):
        raise ValueError("Publication requires matching successful GPU evidence")
    head = materialize(args.source, payload, args.base, args.tag, profile["allow"])
    if not payload["files"]:
        record(args.output, dict(status="passed", candidate_head=head, digest=payload["digest"]))
        return
    branch = f"automation/vllm-{args.tag}"
    if args.profile != "vllm":
        branch += f"-{args.profile}"
    remote = f"https://github.com/{args.publish_repository}.git"
    owner = args.publish_repository.split("/")[0]
    selector = f"{owner}:{branch}"
    deadline = time.monotonic() + PUBLICATION_TIMEOUT

    def command(argv):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("Publication verification timed out; checks are not confirmed")
        return subprocess.check_output(
            argv, text=True, encoding="utf-8", timeout=min(60, remaining)
        )

    def api(endpoint, *fields, method="GET", pages=False):
        argv = ["gh", "api", "--hostname", "github.com", "--method", method, endpoint]
        if pages:
            argv += ["--paginate", "--slurp", "-f", "per_page=100"]
        for field in fields:
            argv += ["-f", field]
        return json.loads(command(argv))

    def remote_head(repository, ref):
        output = command(
            [
                "git",
                "-C",
                str(args.source),
                "ls-remote",
                f"https://github.com/{repository}.git",
                ref,
            ]
        )
        matches = [line.split()[0] for line in output.splitlines() if line.split()[1:] == [ref]]
        if len(matches) > 1:
            raise ValueError("Ambiguous remote ref")
        return matches[0] if matches else None

    def verify_base():
        if remote_head(args.pr_repository, f"refs/heads/{args.base_branch}") != args.base:
            raise ValueError(
                "Target base moved or is missing; rebuild and revalidate the candidate"
            )

    def verify_pr(pr):
        if pr.get("state") != "open" or pr.get("merged") or pr.get("merged_at"):
            raise ValueError("Closed or merged PR requires human reconciliation")
        if pr.get("mergeable") is False or pr.get("mergeable_state") == "dirty":
            raise ValueError("Published PR has merge conflicts")
        for side, repository, ref, sha in (
            ("base", args.pr_repository, args.base_branch, args.base),
            ("head", args.publish_repository, branch, head),
        ):
            value = pr.get(side) or {}
            if (
                (value.get("repo") or {}).get("full_name", "").lower() != repository.lower()
                or value.get("ref") != ref
                or value.get("sha") != sha
            ):
                raise ValueError(f"PR {side} repository, ref or SHA mismatch")
        if type(pr.get("number")) is not int or pr["number"] <= 0:
            raise ValueError("Invalid PR number")

    verify_base()
    pulls = [
        pr
        for page in api(
            f"repos/{args.pr_repository}/pulls", "state=all", f"head={selector}", pages=True
        )
        for pr in page
    ]
    if len(pulls) > 1:
        raise ValueError("Multiple matching PRs require human reconciliation")
    if pulls:
        verify_pr(pulls[0])
    old = remote_head(args.publish_repository, f"refs/heads/{branch}")
    if old and old != head:
        raise ValueError("Existing branch changed; preserve it for human reconciliation")
    if not old:
        command(
            [
                "git",
                "-C",
                str(args.source),
                "push",
                f"--force-with-lease=refs/heads/{branch}:",
                remote,
                f"{head}:refs/heads/{branch}",
            ]
        )
    if remote_head(args.publish_repository, f"refs/heads/{branch}") != head:
        raise ValueError("Published branch SHA mismatch")
    if not pulls:
        body = (
            f"## Summary\n\nAdapt to vLLM {args.tag} ({args.profile}).\n\n"
            f"{profile['task']}\n\n"
            f"## Validation\n\nCandidate `{head}` passed independent GPU checks and the full "
            f"CPU CI matrix before publication. [Run and evidence]({args.run_url}).\n\n"
            "Single-GPU validation is not TP/PP, MPS or model-family certification. "
            "Human review is required; this workflow does not merge.\n"
        )
        pulls = [
            api(
                f"repos/{args.pr_repository}/pulls",
                f"head={selector}",
                f"head_repo={args.publish_repository.split('/')[1]}",
                f"base={args.base_branch}",
                f"title=fix: adapt vLLM {args.tag}",
                f"body={body}",
                method="POST",
            )
        ]
        verify_pr(pulls[0])
    number = pulls[0]["number"]
    while True:
        pr = api(f"repos/{args.pr_repository}/pulls/{number}")
        verify_pr(pr)
        if pr["number"] != number:
            raise ValueError("PR number changed")
        results = []
        target_checks = set()
        # Inspect both push and PR checks, addressed by the exact tested commit, not a branch.
        for repository in dict.fromkeys((args.publish_repository, args.pr_repository)):
            endpoint = f"repos/{repository}/commits/{head}"
            for page in api(endpoint + "/check-runs", "filter=latest", pages=True):
                for check in page["check_runs"]:
                    if check.get("head_sha") != head:
                        raise ValueError("Remote check SHA mismatch")
                    results.append(
                        check.get("conclusion") if check.get("status") == "completed" else "pending"
                    )
                    if repository == args.pr_repository:
                        target_checks.add(check.get("name"))
            for page in api(endpoint + "/status", pages=True):
                if page.get("sha") != head:
                    raise ValueError("Remote status SHA mismatch")
                results.extend(status.get("state") for status in page["statuses"])
        if any(result not in ("success", "pending", None) for result in results):
            raise ValueError("Remote checks did not succeed")
        verify_base()
        if remote_head(args.publish_repository, f"refs/heads/{branch}") != head:
            raise ValueError("Published branch SHA mismatch")
        if (
            pr.get("mergeable") is True
            and EXPECTED_TARGET_CHECKS <= target_checks
            and results
            and all(item == "success" for item in results)
        ):
            # Re-read identity after collecting checks; a concurrent PR retarget must fail closed.
            latest = api(f"repos/{args.pr_repository}/pulls/{number}")
            verify_pr(latest)
            if latest["number"] != number:
                raise ValueError("PR number changed")
            if latest.get("mergeable") is True:
                break
        time.sleep(min(PUBLICATION_POLL_INTERVAL, max(0, deadline - time.monotonic())))
    record(
        args.output,
        dict(
            status="passed",
            candidate_head=head,
            digest=payload["digest"],
            pr_url=f"https://github.com/{args.pr_repository}/pull/{number}",
            profile=identity(profile),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("checks", "cpu", "gpu", "publish"))
    for name in ("source", "engine", "prior", "output"):
        parser.add_argument("--" + name, type=Path)
    for name in ("base", "tag", "engine-sha", "publish-repository", "pr-repository", "run-url"):
        parser.add_argument("--" + name)
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--image", default="")
    parser.add_argument("--probes", action="store_true")
    parser.add_argument("--profile", default="vllm")
    parser.add_argument("--mode", choices=("validate", "repair"), default="repair")
    parser.add_argument(
        "--gpu-command",
        type=Path,
        help="Local replay only: trusted external argv JSON; not workflow input",
    )
    args = parser.parse_args()
    if args.stage == "checks":
        if args.output is None:
            parser.error("checks requires --output outside the immutable check directory")
        return checks(args)
    for name in ("source", "base", "tag", "engine_sha", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    for name in ("source", "engine", "prior", "output"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if not re.fullmatch(r"[0-9a-f]{40}", args.engine_sha):
        parser.error("--engine-sha must be the exact upstream commit")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        {"cpu": cpu, "gpu": gpu, "publish": publish}[args.stage](args)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        record(args.output, dict(status="blocked", reason=str(exc)))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
