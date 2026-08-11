#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Merge an engine's upstream branch into an OVG-maintained fork.

The command is intentionally independent of GitHub Actions.  It can be tested
with local repositories, run by a maintainer, or wrapped by an agent.  Every
run writes JSON and Markdown reports so conflicts and failed checks are useful
inputs for a later automated repair agent.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

SYNC_MANAGED_TRAILER = "OVG-Sync-Managed: true"


@dataclass
class CheckResult:
    command: str
    returncode: int
    output_tail: str


@dataclass
class SyncResult:
    engine: str
    status: str
    target_repository: str
    upstream_repository: str
    base_branch: str
    sync_branch: str
    strategy: str = "merge"
    upstream_commit: str = ""
    result_commit: str = ""
    worktree: str = ""
    conflict_files: List[str] = field(default_factory=list)
    checks: List[CheckResult] = field(default_factory=list)
    message: str = ""


class CommandError(RuntimeError):
    def __init__(self, command: Sequence[str], returncode: int, output: str):
        super().__init__(
            f"command failed ({returncode}): {shlex.join(command)}\n{output}"
        )
        self.command = list(command)
        self.returncode = returncode
        self.output = output


def run(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and completed.returncode != 0:
        raise CommandError(command, completed.returncode, completed.stdout)
    return completed


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(["git", *args], cwd=cwd, check=check)


def output_tail(output: str, max_lines: int = 80) -> str:
    lines = output.rstrip().splitlines()
    return "\n".join(lines[-max_lines:])


def write_reports(result: SyncResult, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# {result.engine} upstream sync",
        "",
        f"- Status: `{result.status}`",
        f"- Target: `{result.target_repository}`",
        f"- Upstream: `{result.upstream_repository}`",
        f"- Base branch: `{result.base_branch}`",
        f"- Sync branch: `{result.sync_branch}`",
        f"- Strategy: `{result.strategy}`",
    ]
    if result.upstream_commit:
        lines.append(f"- Upstream commit: `{result.upstream_commit}`")
    if result.result_commit:
        lines.append(f"- Result commit: `{result.result_commit}`")
    if result.worktree:
        lines.append(f"- Preserved worktree: `{result.worktree}`")
    if result.message:
        lines.extend(["", "## Summary", "", result.message])
    if result.conflict_files:
        lines.extend(["", "## Conflicts", ""])
        lines.extend(f"- `{path}`" for path in result.conflict_files)
    if result.checks:
        lines.extend(["", "## Checks", ""])
        for check_result in result.checks:
            lines.extend(
                [
                    f"### `{check_result.command}`",
                    "",
                    f"Exit code: `{check_result.returncode}`",
                    "",
                    "```text",
                    check_result.output_tail,
                    "```",
                ]
            )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sync_repository(args: argparse.Namespace, workdir: Path) -> SyncResult:
    result = SyncResult(
        engine=args.engine,
        status="initializing",
        target_repository=args.target_repository,
        upstream_repository=args.upstream_repository,
        base_branch=args.base_branch,
        sync_branch=args.sync_branch,
        strategy=args.strategy,
    )

    repository = workdir / "engine"
    result.worktree = str(repository) if args.keep_conflicts else ""
    run(
        [
            "git",
            "clone",
            "--no-tags",
            "--branch",
            args.base_branch,
            args.target_repository,
            str(repository),
        ]
    )
    git(repository, "config", "user.name", args.git_user_name)
    git(repository, "config", "user.email", args.git_user_email)
    git(repository, "remote", "add", "upstream", args.upstream_repository)
    git(repository, "fetch", "--no-tags", "upstream", args.upstream_branch)

    upstream_ref = f"upstream/{args.upstream_branch}"
    result.upstream_commit = git(
        repository, "rev-parse", upstream_ref
    ).stdout.strip()

    ancestor = git(
        repository,
        "merge-base",
        "--is-ancestor",
        upstream_ref,
        "HEAD",
        check=False,
    )
    if ancestor.returncode == 0:
        result.status = "up-to-date"
        result.result_commit = git(repository, "rev-parse", "HEAD").stdout.strip()
        result.message = "The OVG fork already contains the selected upstream commit."
        return result
    if ancestor.returncode not in (0, 1):
        raise CommandError(
            ["git", "merge-base", "--is-ancestor", upstream_ref, "HEAD"],
            ancestor.returncode,
            ancestor.stdout,
        )

    expected_remote_tip = ""
    if args.update_existing_branch:
        remote_sync_ref = f"refs/remotes/origin/{args.sync_branch}"
        remote_branch = git(
            repository,
            "show-ref",
            "--verify",
            "--quiet",
            remote_sync_ref,
            check=False,
        )
        if remote_branch.returncode == 0:
            expected_remote_tip = git(
                repository, "rev-parse", remote_sync_ref
            ).stdout.strip()
            remote_message = git(
                repository,
                "show",
                "-s",
                "--format=%B",
                remote_sync_ref,
            ).stdout.splitlines()
            if SYNC_MANAGED_TRAILER not in remote_message:
                result.status = "remote-diverged"
                result.result_commit = expected_remote_tip
                result.message = (
                    "The remote sync branch contains commits that were not "
                    "produced by the synchronization job. The update stopped "
                    "without pushing so maintainer or repair-agent work is not "
                    "overwritten."
                )
                return result
        elif remote_branch.returncode != 1:
            raise CommandError(
                ["git", "show-ref", "--verify", "--quiet", remote_sync_ref],
                remote_branch.returncode,
                remote_branch.stdout,
            )

    git(repository, "checkout", "-b", args.sync_branch)
    result.result_commit = git(repository, "rev-parse", "HEAD").stdout.strip()
    if args.strategy == "rebase":
        integrate_command = ["rebase", upstream_ref]
        abort_command = ["rebase", "--abort"]
    else:
        integrate_command = ["merge", "--no-ff", "--no-edit", upstream_ref]
        abort_command = ["merge", "--abort"]

    integration = git(repository, *integrate_command, check=False)
    if integration.returncode != 0:
        conflicts = git(
            repository,
            "diff",
            "--name-only",
            "--diff-filter=U",
            check=False,
        ).stdout.splitlines()
        result.status = "conflict"
        result.conflict_files = sorted(path for path in conflicts if path)
        result.message = (
            f"Git could not {args.strategy} upstream automatically. The conflict "
            "list and repository versions in the JSON report are intended as "
            "input for a maintainer or repair agent."
        )
        if not args.keep_conflicts:
            git(repository, *abort_command, check=False)
            result.worktree = ""
        return result

    for command_text in args.check:
        command = shlex.split(command_text)
        if not command:
            continue
        completed = run(command, cwd=repository, check=False)
        result.checks.append(
            CheckResult(
                command=command_text,
                returncode=completed.returncode,
                output_tail=output_tail(completed.stdout),
            )
        )
        if completed.returncode != 0:
            result.status = "check-failed"
            result.result_commit = git(
                repository, "rev-parse", "HEAD"
            ).stdout.strip()
            result.message = (
                f"The upstream {args.strategy} completed, but a compatibility "
                "check failed. No branch was pushed."
            )
            return result

    if args.push:
        if args.update_existing_branch:
            git(
                repository,
                "commit",
                "--allow-empty",
                "-m",
                "chore: record automated upstream sync",
                "-m",
                SYNC_MANAGED_TRAILER,
                "-m",
                f"OVG-Upstream-Commit: {result.upstream_commit}",
            )
        result.result_commit = git(repository, "rev-parse", "HEAD").stdout.strip()
        push_args = ["push"]
        if args.update_existing_branch:
            lease = f"refs/heads/{args.sync_branch}:{expected_remote_tip}"
            push_args.append(f"--force-with-lease={lease}")
        push_args.extend(["origin", f"HEAD:refs/heads/{args.sync_branch}"])
        git(repository, *push_args)
        result.message = (
            f"Upstream {args.strategy} completed, checks passed, and the sync "
            "branch was pushed."
        )
    else:
        result.result_commit = git(repository, "rev-parse", "HEAD").stdout.strip()
        result.message = (
            f"Upstream {args.strategy} completed and checks passed. "
            "Dry run: no branch was pushed."
        )
    result.status = "synced"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--target-repository", required=True)
    parser.add_argument("--upstream-repository", required=True)
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--upstream-branch", default="main")
    parser.add_argument("--sync-branch", required=True)
    parser.add_argument(
        "--strategy",
        choices=("merge", "rebase"),
        default="merge",
        help="How to integrate upstream into the OVG-maintained branch.",
    )
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        help="Compatibility check command. May be specified more than once.",
    )
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--update-existing-branch",
        action="store_true",
        help=(
            "Update a stable remote sync branch with force-with-lease. "
            "This is safe for automation-owned branches and avoids duplicate PRs."
        ),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Persistent parent directory for the cloned engine worktree.",
    )
    parser.add_argument(
        "--keep-conflicts",
        action="store_true",
        help="Leave integration conflict markers in --workdir for an agent to repair.",
    )
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--git-user-name", default="ovg-upstream-sync")
    parser.add_argument("--git-user-email", default="actions@users.noreply.github.com")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.keep_conflicts and args.workdir is None:
        raise SystemExit("--keep-conflicts requires --workdir")

    result: SyncResult
    try:
        if args.workdir is not None:
            args.workdir.mkdir(parents=True, exist_ok=True)
            result = sync_repository(args, args.workdir)
        else:
            with tempfile.TemporaryDirectory(prefix="ovg-engine-sync-") as temp_dir:
                result = sync_repository(args, Path(temp_dir))
    except Exception as exc:
        result = SyncResult(
            engine=args.engine,
            status="error",
            target_repository=args.target_repository,
            upstream_repository=args.upstream_repository,
            base_branch=args.base_branch,
            sync_branch=args.sync_branch,
            strategy=args.strategy,
            message=str(exc),
        )

    write_reports(result, args.result_json, args.report)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return {
        "up-to-date": 0,
        "synced": 0,
        "conflict": 2,
        "check-failed": 3,
        "remote-diverged": 4,
    }.get(result.status, 1)


if __name__ == "__main__":
    raise SystemExit(main())
