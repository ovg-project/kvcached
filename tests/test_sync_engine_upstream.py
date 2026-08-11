# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "tools" / "sync_engine_upstream.py"


def git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return completed.stdout.strip()


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def commit(repo: Path, message: str) -> None:
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        message,
    )


def initialize_repositories(tmp_path: Path):
    upstream = tmp_path / "upstream"
    target = tmp_path / "target.git"
    git(tmp_path, "init", "-b", "main", str(upstream))
    write(upstream / "engine.txt", "base\n")
    commit(upstream, "base")
    git(tmp_path, "clone", "--bare", str(upstream), str(target))
    return upstream, target


def run_sync(
    tmp_path: Path,
    upstream: Path,
    target: Path,
    *extra: str,
    sync_branch: str = "automation/test-sync",
):
    result_json = tmp_path / "result.json"
    report = tmp_path / "report.md"
    command = [
        sys.executable,
        str(SCRIPT),
        "--engine",
        "vllm",
        "--target-repository",
        str(target),
        "--upstream-repository",
        str(upstream),
        "--sync-branch",
        sync_branch,
        "--result-json",
        str(result_json),
        "--report",
        str(report),
        *extra,
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed, json.loads(result_json.read_text()), report.read_text()


def test_sync_pushes_checked_upstream_merge(tmp_path):
    upstream, target = initialize_repositories(tmp_path)
    write(upstream / "upstream.txt", "new upstream feature\n")
    commit(upstream, "upstream feature")

    completed, result, report = run_sync(
        tmp_path,
        upstream,
        target,
        "--check",
        f"{sys.executable} -c pass",
        "--push",
    )

    assert completed.returncode == 0
    assert result["status"] == "synced"
    assert result["checks"][0]["returncode"] == 0
    assert git(target, "rev-parse", "refs/heads/automation/test-sync")
    assert "Status: `synced`" in report


def test_rebase_strategy_replays_ovg_patch_on_new_upstream(tmp_path):
    upstream, target = initialize_repositories(tmp_path)

    target_worktree = tmp_path / "target-worktree"
    git(tmp_path, "clone", str(target), str(target_worktree))
    write(target_worktree / "ovg.txt", "kvcached integration\n")
    commit(target_worktree, "ovg patch")
    git(target_worktree, "push", "origin", "main")

    write(upstream / "upstream.txt", "new upstream feature\n")
    commit(upstream, "upstream feature")

    completed, result, report = run_sync(
        tmp_path,
        upstream,
        target,
        "--strategy",
        "rebase",
        "--push",
    )

    assert completed.returncode == 0
    assert result["status"] == "synced"
    assert result["strategy"] == "rebase"
    sync_tip = git(target, "rev-parse", "refs/heads/automation/test-sync")
    upstream_tip = git(upstream, "rev-parse", "main")
    assert git(target, "merge-base", sync_tip, upstream_tip) == upstream_tip
    assert "Strategy: `rebase`" in report


def test_sync_reports_up_to_date_without_creating_branch(tmp_path):
    upstream, target = initialize_repositories(tmp_path)

    completed, result, _ = run_sync(tmp_path, upstream, target)

    assert completed.returncode == 0
    assert result["status"] == "up-to-date"
    branches = git(target, "for-each-ref", "--format=%(refname)", "refs/heads")
    assert "automation/test-sync" not in branches


def test_failed_check_does_not_push_sync_branch(tmp_path):
    upstream, target = initialize_repositories(tmp_path)
    write(upstream / "upstream.txt", "new upstream feature\n")
    commit(upstream, "upstream feature")

    completed, result, report = run_sync(
        tmp_path,
        upstream,
        target,
        "--check",
        f"{sys.executable} -c 'raise SystemExit(7)'",
        "--push",
    )

    assert completed.returncode == 3
    assert result["status"] == "check-failed"
    assert result["checks"][0]["returncode"] == 7
    branches = git(target, "for-each-ref", "--format=%(refname)", "refs/heads")
    assert "automation/test-sync" not in branches
    assert "No branch was pushed." in report


def test_sync_reports_conflicting_files_for_agent_followup(tmp_path):
    upstream, target = initialize_repositories(tmp_path)

    target_worktree = tmp_path / "target-worktree"
    git(tmp_path, "clone", str(target), str(target_worktree))
    write(target_worktree / "engine.txt", "ovg patch\n")
    commit(target_worktree, "ovg patch")
    git(target_worktree, "push", "origin", "main")

    write(upstream / "engine.txt", "upstream rewrite\n")
    commit(upstream, "upstream rewrite")

    completed, result, report = run_sync(tmp_path, upstream, target)

    assert completed.returncode == 2
    assert result["status"] == "conflict"
    assert result["conflict_files"] == ["engine.txt"]
    assert "`engine.txt`" in report


def test_sync_can_leave_conflict_worktree_for_repair_agent(tmp_path):
    upstream, target = initialize_repositories(tmp_path)

    target_worktree = tmp_path / "target-worktree"
    git(tmp_path, "clone", str(target), str(target_worktree))
    write(target_worktree / "engine.txt", "ovg patch\n")
    commit(target_worktree, "ovg patch")
    git(target_worktree, "push", "origin", "main")

    write(upstream / "engine.txt", "upstream rewrite\n")
    commit(upstream, "upstream rewrite")

    repair_dir = tmp_path / "repair"
    completed, result, report = run_sync(
        tmp_path,
        upstream,
        target,
        "--strategy",
        "rebase",
        "--workdir",
        str(repair_dir),
        "--keep-conflicts",
        sync_branch="automation/repair-sync",
    )

    assert completed.returncode == 2
    assert result["status"] == "conflict"
    assert Path(result["worktree"]) == repair_dir / "engine"
    assert f"`{repair_dir / 'engine'}`" in report
    conflict_text = (repair_dir / "engine" / "engine.txt").read_text()
    assert "<<<<<<< HEAD" in conflict_text


def test_stable_sync_branch_updates_without_duplicate_branch(tmp_path):
    upstream, target = initialize_repositories(tmp_path)
    write(upstream / "first.txt", "first upstream feature\n")
    commit(upstream, "first upstream feature")
    completed, first, _ = run_sync(
        tmp_path,
        upstream,
        target,
        "--push",
        "--update-existing-branch",
    )
    assert completed.returncode == 0
    first_tip = git(target, "rev-parse", "refs/heads/automation/test-sync")

    write(upstream / "second.txt", "second upstream feature\n")
    commit(upstream, "second upstream feature")
    completed, second, _ = run_sync(
        tmp_path,
        upstream,
        target,
        "--push",
        "--update-existing-branch",
    )

    assert completed.returncode == 0
    assert first["sync_branch"] == second["sync_branch"]
    assert git(target, "rev-parse", "refs/heads/automation/test-sync") != first_tip


def test_sync_refuses_to_overwrite_repair_commit_on_remote_branch(tmp_path):
    upstream, target = initialize_repositories(tmp_path)
    write(upstream / "first.txt", "first upstream feature\n")
    commit(upstream, "first upstream feature")
    completed, _, _ = run_sync(
        tmp_path,
        upstream,
        target,
        "--push",
        "--update-existing-branch",
    )
    assert completed.returncode == 0

    repair_worktree = tmp_path / "repair-worktree"
    git(tmp_path, "clone", str(target), str(repair_worktree))
    git(repair_worktree, "checkout", "automation/test-sync")
    write(repair_worktree / "repair.txt", "human compatibility fix\n")
    commit(repair_worktree, "preserve this repair")
    git(repair_worktree, "push", "origin", "automation/test-sync")
    repair_tip = git(target, "rev-parse", "refs/heads/automation/test-sync")

    write(upstream / "second.txt", "second upstream feature\n")
    commit(upstream, "second upstream feature")
    completed, result, report = run_sync(
        tmp_path,
        upstream,
        target,
        "--push",
        "--update-existing-branch",
    )

    assert completed.returncode == 4
    assert result["status"] == "remote-diverged"
    assert git(target, "rev-parse", "refs/heads/automation/test-sync") == repair_tip
    assert "not overwritten" in report
    assert git(target, "show", "automation/test-sync:repair.txt") == (
        "human compatibility fix"
    )
