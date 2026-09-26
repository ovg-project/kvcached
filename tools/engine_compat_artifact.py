#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Transfer an allowlisted repair between isolated jobs without executing it."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List

MAX_BYTES = 16 * 1024 * 1024
IDENTITY = "KVCached compatibility automation"
EMAIL = "compatibility-bot@users.noreply.github.com"


def git(root: Path, *args: str, data=None, env=None) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], input=data, env=env)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _unique_object(pairs: List[Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate candidate envelope field")
        result[key] = value
    return result


def load_envelope(path: Path) -> Dict[str, Any]:
    """Read bounded JSON; callers must still validate against trusted work-item inputs."""
    limit = MAX_BYTES * 2
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Candidate envelope too large")
    try:
        payload = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as exc:
        raise ValueError("Invalid candidate envelope JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Invalid candidate envelope")
    return payload


def allowed_paths(path: Path) -> List[str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError("An explicit nonempty list of unique repair paths is required")
    for value in values:
        name = PurePosixPath(value)
        if (
            len(name.parts) < 2
            or name.is_absolute()
            or name.as_posix() != value
            or ".." in name.parts
            or "\\" in value
            or ":" in value
            or name.parts[0] not in ("kvcached", "tests")
        ):
            raise ValueError("Invalid repair path")
    return values


def validate(payload: Any, base: str, tag: str, allow: List[str]) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "base",
        "tag",
        "files",
        "candidate_head",
        "digest",
    }:
        raise ValueError("Invalid candidate envelope")
    if (
        not isinstance(base, str)
        or not isinstance(tag, str)
        or not re.fullmatch(r"[0-9a-f]{40}", base)
        or not re.fullmatch(r"v\d+\.\d+\.\d+(?:\.post\d+)?", tag)
    ):
        raise ValueError("Pin a full base SHA and a stable release tag")
    if (
        type(payload["schema"]) is not int
        or payload["schema"] != 1
        or not isinstance(payload["base"], str)
        or not isinstance(payload["tag"], str)
        or payload["base"] != base
        or payload["tag"] != tag
    ):
        raise ValueError("Candidate does not match the selected work item")
    if not isinstance(payload["candidate_head"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", payload["candidate_head"]
    ):
        raise ValueError("Invalid candidate SHA")
    if not isinstance(payload["digest"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", payload["digest"]
    ):
        raise ValueError("Candidate digest mismatch")
    if not isinstance(allow, list) or any(not isinstance(name, str) for name in allow):
        raise ValueError("Invalid repair allowlist")
    if not isinstance(payload["files"], dict) or set(payload["files"]) - set(allow):
        raise ValueError("Candidate contains out-of-scope paths")
    total = 0
    for name, record in payload["files"].items():
        if (
            not isinstance(record, dict)
            or set(record) != {"data", "sha256", "mode"}
            or any(not isinstance(value, str) for value in record.values())
            or record["mode"] not in ("100644", "100755")
        ):
            raise ValueError(f"Invalid file record: {name}")
        if len(record["data"]) > 4 * ((MAX_BYTES - total + 2) // 3):
            raise ValueError(f"Invalid file content: {name}")
        content = base64.b64decode(record["data"], validate=True)
        total += len(content)
        if total > MAX_BYTES or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"Invalid file content: {name}")
    unsigned = {k: v for k, v in payload.items() if k != "digest"}
    if payload["digest"] != digest(unsigned):
        raise ValueError("Candidate digest mismatch")


def commit_object(root: Path, base: str, tag: str, files: Dict[str, Any]) -> str:
    if not files:
        return base
    # A private index keeps reconstruction out of the user's/workflow's index.
    with tempfile.TemporaryDirectory(prefix="compat-index-", dir=root.parent) as temp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / "index"))
        env.update(
            GIT_AUTHOR_NAME=IDENTITY,
            GIT_AUTHOR_EMAIL=EMAIL,
            GIT_COMMITTER_NAME=IDENTITY,
            GIT_COMMITTER_EMAIL=EMAIL,
            GIT_AUTHOR_DATE="2000-01-01T00:00:00+0000",
            GIT_COMMITTER_DATE="2000-01-01T00:00:00+0000",
        )
        git(root, "read-tree", base, env=env)
        for name, record in sorted(files.items()):
            blob = git(root, "hash-object", "-w", "--stdin", data=base64.b64decode(record["data"]))
            git(
                root,
                "update-index",
                "--add",
                "--cacheinfo",
                f"{record['mode']},{blob.decode().strip()},{name}",
                env=env,
            )
        tree = git(root, "write-tree", env=env).decode().strip()
        message = f"fix: adapt engine release {tag}\n\nAutomatically prepared; requires human review.\n"
        return (
            git(
                root,
                "-c",
                "commit.gpgsign=false",
                "commit-tree",
                tree,
                "-p",
                base,
                data=message.encode(),
                env=env,
            )
            .decode()
            .strip()
        )


def pack(root: Path, base: str, tag: str, allow: List[str]) -> Dict[str, Any]:
    # Compare against the original base, including changes from previous rounds.
    changed = set(
        filter(None, git(root, "diff", "--name-only", "--no-renames", base).decode().splitlines())
    )
    changed.update(
        filter(None, git(root, "ls-files", "--others", "--exclude-standard").decode().splitlines())
    )
    if changed - set(allow):
        raise ValueError(
            "Out-of-scope candidate changes: " + ", ".join(sorted(changed - set(allow)))
        )
    files = {}
    for name in sorted(changed):
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("Deletion, symlinks and escaped paths are not allowed")
        content = path.read_bytes()
        tracked = git(root, "ls-tree", base, "--", name).decode()
        mode = tracked.split()[0] if tracked else "100644"
        if os.name != "nt":
            mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        files[name] = dict(
            data=base64.b64encode(content).decode("ascii"),
            mode=mode,
            sha256=hashlib.sha256(content).hexdigest(),
        )
    payload = dict(
        schema=1,
        base=base,
        tag=tag,
        files=files,
        candidate_head=commit_object(root, base, tag, files),
    )
    payload["digest"] = digest(payload)
    validate(payload, base, tag, allow)
    return payload


def materialize(root: Path, payload: Any, base: str, tag: str, allow: List[str]) -> str:
    validate(payload, base, tag, allow)
    if git(root, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("Materialization requires a clean disposable checkout")
    if git(root, "rev-parse", "HEAD").decode().strip() != base:
        raise ValueError("Materialization checkout is not at the recorded base")
    # Tree/blob operations cannot execute hooks, candidate scripts or Git filters.
    for name in payload["files"]:
        parent = (root / name).parent
        while parent != root:
            if parent.is_symlink():
                raise ValueError("Symlink parent in candidate path")
            parent = parent.parent
        existing = git(root, "ls-tree", base, "--", name).decode()
        if existing and existing.split()[0] not in ("100644", "100755"):
            raise ValueError("Cannot replace a symlink or submodule")
    head = commit_object(root, base, tag, payload["files"])
    if head != payload["candidate_head"]:
        raise ValueError("Reconstructed commit differs from the tested candidate")
    git(root, "-c", "core.hooksPath=/dev/null", "checkout", "--detach", head)
    return head


def verify_checkout(root: Path, payload: Any, base: str, tag: str, allow: List[str]) -> str:
    """Verify a quiescent checkout without trusting Git's worktree stat cache or flags."""
    validate(payload, base, tag, allow)
    root = root.resolve()
    head = payload["candidate_head"]
    if git(root, "rev-parse", "HEAD").decode().strip() != head:
        raise ValueError("Checkout HEAD differs from the expected candidate")
    if commit_object(root, base, tag, payload["files"]) != head:
        raise ValueError("Reconstructed commit differs from the expected candidate")
    tree = git(root, "--no-replace-objects", "rev-parse", f"{head}^{{tree}}").strip()
    try:
        index_tree = git(root, "--no-replace-objects", "write-tree").strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError("Cannot verify the checkout index tree") from exc
    if index_tree != tree:
        raise ValueError("Checkout index tree differs from the expected candidate")
    if git(root, "ls-files", "--others", "--exclude-standard", "-z"):
        raise ValueError("Checkout contains nonignored untracked files")
    emulated_symlinks = (
        os.name == "nt"
        and git(root, "config", "--type=bool", "--default=true", "--get", "core.symlinks").strip()
        == b"false"
    )

    # Read the expected tree, not ls-files flags or a status/diff cleanliness result.
    entries = git(root, "--no-replace-objects", "ls-tree", "-r", "-z", "--full-tree", head)
    for entry in filter(None, entries.split(b"\0")):
        header, raw_name = entry.split(b"\t", 1)
        mode, kind, expected_blob = header.split()
        name = os.fsdecode(raw_name)
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Invalid tracked file path")
        if kind != b"blob" or mode not in (b"100644", b"100755", b"120000"):
            raise ValueError(f"Unsupported tracked file mode: {name}")
        path = root / name
        parent = path.parent
        while parent != root:
            if parent.is_symlink() or not parent.resolve().is_relative_to(root):
                raise ValueError(f"Symlink or escaped tracked parent: {name}")
            parent = parent.parent
        try:
            info = path.lstat()
            if mode == b"120000" and stat.S_ISLNK(info.st_mode):
                content = os.fsencode(os.readlink(path))
                actual_blob = hashlib.sha1(
                    b"blob " + str(len(content)).encode("ascii") + b"\0" + content
                ).hexdigest()
            else:
                if not stat.S_ISREG(info.st_mode) or (mode == b"120000" and not emulated_symlinks):
                    raise ValueError(f"Tracked file type differs: {name}")
                if os.name != "nt" and bool(info.st_mode & 0o111) != (mode == b"100755"):
                    raise ValueError(f"Tracked file mode differs: {name}")
                hasher = hashlib.sha1(b"blob " + str(info.st_size).encode("ascii") + b"\0")
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(block)
                actual_blob = hasher.hexdigest()
        except OSError as exc:
            raise ValueError(f"Cannot read tracked file: {name}") from exc
        if actual_blob != expected_blob.decode("ascii"):
            raise ValueError(f"Tracked file content differs: {name}")
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "materialize", "verify"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--allow-list", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    allow = allowed_paths(args.allow_list)
    if args.command == "pack":
        payload = pack(args.source.resolve(), args.base, args.tag, allow)
        args.artifact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        payload = load_envelope(args.artifact)
        if args.command == "verify":
            verify_checkout(args.source.resolve(), payload, args.base, args.tag, allow)
        else:
            materialize(args.source.resolve(), payload, args.base, args.tag, allow)
    print(payload["candidate_head"])


if __name__ == "__main__":
    main()
