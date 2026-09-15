# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""CPU-only artifact integrity tests using disposable local Git repositories."""

import base64
import errno
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

spec = importlib.util.spec_from_file_location(
    "engine_artifact", Path(__file__).parents[1] / "tools/engine_compat_artifact.py"
)
assert spec is not None and spec.loader is not None
artifact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifact)

TAG = "v1.2.3"
ALLOW = ["kvcached/compat.py", "kvcached/run.py", "tests/test_added.py"]
FIX = b"value = 'fixed'\n"


def git(root, *args):
    return artifact.git(root, *args).decode("utf-8").strip()


def seal(payload):
    payload["digest"] = artifact.digest({k: v for k, v in payload.items() if k != "digest"})
    return payload


def record(content, mode="100644"):
    return {
        "data": base64.b64encode(content).decode("ascii"),
        "sha256": hashlib.sha256(content).hexdigest(),
        "mode": mode,
    }


def state(root):
    return (
        git(root, "rev-parse", "HEAD"),
        git(root, "status", "--porcelain", "--untracked-files=all"),
        git(root, "ls-files", "--stage"),
    )


def symlink(path, target):
    try:
        path.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314 or exc.errno in (errno.ENOSYS, errno.ENOTSUP):
            pytest.skip("This host cannot create filesystem symlinks")
        raise


@pytest.fixture
def source(tmp_path, monkeypatch):
    # Keep developer Git settings, hooks, and newline conversion out of the fixtures.
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "source with spaces"
    root.mkdir()
    git(root, "init", "-q")
    for key, value in (
        ("user.name", "Artifact test"),
        ("user.email", "artifact@example.invalid"),
        ("commit.gpgsign", "false"),
        ("core.autocrlf", "false"),
        ("core.filemode", "false" if os.name == "nt" else "true"),
    ):
        git(root, "config", key, value)
    (root / "kvcached").mkdir()
    (root / "tests").mkdir()
    (root / "kvcached/compat.py").write_bytes(b"value = 'original'\n")
    executable = root / "kvcached/run.py"
    executable.write_bytes(b"#!/usr/bin/env python3\nprint('original')\n")
    if os.name != "nt":
        executable.chmod(0o755)
    (root / "README.md").write_bytes(b"Unrelated documentation.\n")
    git(root, "add", ".")
    git(root, "update-index", "--chmod=+x", "kvcached/run.py")
    git(root, "commit", "-qm", "baseline")
    assert not git(root, "status", "--porcelain")
    return root


@pytest.fixture
def base(source):
    return git(source, "rev-parse", "HEAD")


@pytest.fixture
def clone(source, base, tmp_path):
    def create(name="target"):
        target = tmp_path / name
        git(tmp_path, "clone", "-q", "--no-local", str(source), str(target))
        git(target, "config", "core.autocrlf", "false")
        git(target, "checkout", "-q", "--detach", base)
        return target

    return create


@pytest.fixture
def payload(source, base):
    (source / "kvcached/compat.py").write_bytes(FIX)
    return artifact.pack(source, base, TAG, ALLOW)


def test_roundtrip_preserves_bytes_and_leaves_source_index_and_head_alone(source, base, clone):
    content = b"binary\x00\xff\r\ntrailing newline\n"
    (source / "kvcached/compat.py").write_bytes(content)
    before = state(source)
    payload = artifact.pack(source, base, TAG, ALLOW)
    assert state(source) == before
    assert set(payload["files"]) == {"kvcached/compat.py"}
    assert payload["files"]["kvcached/compat.py"] == record(content)

    target = clone()
    head = artifact.materialize(target, payload, base, TAG, ALLOW)
    assert head == payload["candidate_head"] == git(target, "rev-parse", "HEAD")
    assert git(target, "rev-parse", "HEAD^") == base
    assert (target / "kvcached/compat.py").read_bytes() == content
    assert (target / "README.md").read_bytes() == b"Unrelated documentation.\n"
    assert not git(target, "status", "--porcelain")


def test_no_change_roundtrip_uses_original_commit(source, base, clone):
    payload = artifact.pack(source, base, TAG, ALLOW)
    assert payload["files"] == {}
    assert payload["candidate_head"] == base
    target = clone()
    assert artifact.materialize(target, payload, base, TAG, ALLOW) == base
    assert state(target) == state(source)


def test_new_untracked_test_is_included(source, base, clone):
    content = b"def test_contract():\n    assert True\n"
    (source / "tests/test_added.py").write_bytes(content)
    payload = artifact.pack(source, base, TAG, ALLOW)
    assert payload["files"] == {"tests/test_added.py": record(content)}
    target = clone()
    artifact.materialize(target, payload, base, TAG, ALLOW)
    assert (target / "tests/test_added.py").read_bytes() == content
    assert git(target, "ls-files", "tests/test_added.py") == "tests/test_added.py"
    assert not git(target, "status", "--porcelain")


@pytest.mark.parametrize(
    "name,mode", [("kvcached/compat.py", "100644"), ("kvcached/run.py", "100755")]
)
def test_tracked_file_mode_survives_content_repair(source, base, clone, name, mode):
    (source / name).write_bytes(FIX)
    payload = artifact.pack(source, base, TAG, ALLOW)
    assert payload["files"][name]["mode"] == mode
    target = clone()
    artifact.materialize(target, payload, base, TAG, ALLOW)
    assert git(target, "ls-tree", "HEAD", "--", name).split()[0] == mode
    if os.name != "nt":
        assert bool((target / name).stat().st_mode & 0o111) == (mode == "100755")


@pytest.mark.parametrize("mode", ["100644", "100755"])
def test_materialize_preserves_new_file_git_modes(source, base, clone, mode):
    payload = artifact.pack(source, base, TAG, ALLOW)
    payload["files"]["tests/test_added.py"] = record(FIX, mode)
    payload["candidate_head"] = artifact.commit_object(source, base, TAG, payload["files"])
    seal(payload)
    target = clone()
    artifact.materialize(target, payload, base, TAG, ALLOW)
    assert git(target, "ls-tree", "HEAD", "--", "tests/test_added.py").split()[0] == mode
    assert (target / "tests/test_added.py").read_bytes() == FIX


@pytest.mark.skipif(os.name == "nt", reason="NTFS does not expose POSIX executable-bit edits")
def test_pack_includes_mode_only_change(source, base, clone):
    original = (source / "kvcached/compat.py").read_bytes()
    (source / "kvcached/compat.py").chmod(0o755)
    payload = artifact.pack(source, base, TAG, ALLOW)
    assert payload["files"] == {"kvcached/compat.py": record(original, "100755")}
    target = clone()
    artifact.materialize(target, payload, base, TAG, ALLOW)
    assert (target / "kvcached/compat.py").stat().st_mode & 0o111


def test_second_round_is_cumulative_against_original_base(source, base, clone):
    (source / "kvcached/compat.py").write_bytes(FIX)
    added = source / "tests/test_added.py"
    added.write_bytes(b"def test_first():\n    assert True\n")
    first = artifact.pack(source, base, TAG, ALLOW)
    second_source = clone("second round")
    artifact.materialize(second_source, first, base, TAG, ALLOW)
    (second_source / "tests/test_added.py").write_bytes(b"def test_second():\n    assert True\n")
    (second_source / "kvcached/run.py").write_bytes(b"print('second repair')\n")
    second = artifact.pack(second_source, base, TAG, ALLOW)
    assert second["base"] == base
    assert second["candidate_head"] != first["candidate_head"]
    assert set(second["files"]) == set(ALLOW)
    assert second["files"]["kvcached/compat.py"] == first["files"]["kvcached/compat.py"]
    target = clone("final target")
    artifact.materialize(target, second, base, TAG, ALLOW)
    for name in ALLOW:
        assert (target / name).read_bytes() == (second_source / name).read_bytes()
    assert git(target, "rev-list", "--count", f"{base}..HEAD") == "1"
    assert not git(target, "status", "--porcelain")


def test_candidate_has_deterministic_exact_commit(source, base, clone, monkeypatch):
    (source / "kvcached/compat.py").write_bytes(FIX)
    first = artifact.pack(source, base, TAG, ALLOW)
    other = clone("independent producer")
    (other / "kvcached/compat.py").write_bytes(FIX)
    for name, value in (
        ("GIT_AUTHOR_NAME", "Different author"),
        ("GIT_COMMITTER_NAME", "Different committer"),
        ("GIT_AUTHOR_EMAIL", "other@example.invalid"),
        ("GIT_COMMITTER_EMAIL", "other@example.invalid"),
        ("GIT_AUTHOR_DATE", "2025-01-01T12:00:00+0800"),
        ("GIT_COMMITTER_DATE", "2026-01-01T12:00:00-0700"),
    ):
        monkeypatch.setenv(name, value)
    git(other, "config", "commit.gpgsign", "true")
    second = artifact.pack(other, base, TAG, list(reversed(ALLOW)))
    assert second == first
    assert artifact.pack(source, base, TAG, ALLOW) == first

    tree = git(source, "rev-parse", first["candidate_head"] + "^{tree}")
    identity = (
        "KVCached compatibility automation "
        "<compatibility-bot@users.noreply.github.com> 946684800 +0000"
    )
    commit = (
        f"tree {tree}\nparent {base}\nauthor {identity}\ncommitter {identity}\n\n"
        f"fix: adapt vLLM {TAG}\n\nAutomatically prepared; requires human review.\n"
    ).encode("utf-8")
    expected = hashlib.sha1(
        b"commit " + str(len(commit)).encode("ascii") + b"\0" + commit
    ).hexdigest()
    assert first["candidate_head"] == expected
    target = clone("exact head target")
    assert artifact.materialize(target, first, base, TAG, ALLOW) == expected


@pytest.mark.parametrize(
    "corruption",
    ["digest", "content", "file-hash", "base64", "mode", "head", "schema", "extra", "missing"],
)
def test_corruption_is_rejected_before_checkout(payload, base, clone, corruption):
    if corruption == "digest":
        payload["digest"] = "0" * 64
    elif corruption == "content":
        payload["files"]["kvcached/compat.py"]["data"] = base64.b64encode(b"tampered").decode()
    elif corruption == "file-hash":
        payload["files"]["kvcached/compat.py"]["sha256"] = "0" * 64
    elif corruption == "base64":
        payload["files"]["kvcached/compat.py"]["data"] = "!not-base64!"
    elif corruption == "mode":
        payload["files"]["kvcached/compat.py"]["mode"] = "120000"
    elif corruption == "head":
        payload["candidate_head"] = "not-a-sha"
    elif corruption == "schema":
        payload["schema"] = 2
    elif corruption == "extra":
        payload["unexpected"] = True
    else:
        del payload["candidate_head"]
    if corruption != "digest":
        seal(payload)
    target = clone()
    before = state(target)
    with pytest.raises(ValueError):
        artifact.materialize(target, payload, base, TAG, ALLOW)
    assert state(target) == before
    assert (target / "kvcached/compat.py").read_bytes() == b"value = 'original'\n"


def test_rehashed_envelope_cannot_substitute_a_different_exact_head(payload, base, clone):
    payload["candidate_head"] = base
    seal(payload)
    target = clone()
    before = state(target)
    with pytest.raises(ValueError, match="Reconstructed commit"):
        artifact.materialize(target, payload, base, TAG, ALLOW)
    assert state(target) == before


def test_combined_file_size_limit(source, base, monkeypatch):
    payload = artifact.pack(source, base, TAG, ALLOW)
    payload["files"] = {"kvcached/compat.py": record(b"abc"), "tests/test_added.py": record(b"def")}
    seal(payload)
    monkeypatch.setattr(artifact, "MAX_BYTES", 5)
    with pytest.raises(ValueError, match="Invalid file content"):
        artifact.validate(payload, base, TAG, ALLOW)


@pytest.mark.parametrize("kind", ["tracked", "untracked", "staged"])
def test_pack_rejects_out_of_scope_changes(source, base, kind):
    path = source / ("README.md" if kind == "tracked" else "unexpected.txt")
    path.write_bytes(b"outside repair scope\n")
    if kind == "staged":
        git(source, "add", path.name)
    before = state(source)
    with pytest.raises(ValueError, match="Out-of-scope"):
        artifact.pack(source, base, TAG, ALLOW)
    assert state(source) == before


def test_materialize_rejects_rehashed_out_of_scope_file(payload, base, clone):
    payload["files"]["README.md"] = record(b"not allowed\n")
    seal(payload)
    target = clone()
    before = state(target)
    with pytest.raises(ValueError, match="out-of-scope"):
        artifact.materialize(target, payload, base, TAG, ALLOW)
    assert state(target) == before


def test_pack_rejects_deletion(source, base):
    (source / "kvcached/compat.py").unlink()
    with pytest.raises(ValueError, match="Deletion"):
        artifact.pack(source, base, TAG, ALLOW)


def test_allow_config_accepts_exact_repository_paths(tmp_path):
    config = tmp_path / "allow.json"
    config.write_text(json.dumps(ALLOW), encoding="utf-8")
    assert artifact.allowed_paths(config) == ALLOW


@pytest.mark.parametrize(
    "values",
    [
        [],
        {},
        ["tests/test_added.py", "tests/test_added.py"],
        ["../outside.py"],
        ["tests/../../outside.py"],
        ["kvcached/../README.md"],
        ["/tests/absolute.py"],
        ["C:/tests/absolute.py"],
        ["tests/C:stream"],
        ["tests\\..\\outside.py"],
        ["tests//test_added.py"],
        ["tests/./test_added.py"],
        ["tests/test_added.py/"],
        [".git/config"],
        ["tools/repair.py"],
        [""],
        ["."],
        ["tests"],
        ["kvcached"],
        [None],
    ],
)
def test_allow_config_rejects_traversal_and_invalid_paths(tmp_path, values):
    config = tmp_path / "allow.json"
    config.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError):
        artifact.allowed_paths(config)


def test_pack_rejects_symlink_to_outside_file(source, base, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"must not be imported\n")
    symlink(source / "tests/test_added.py", outside)
    with pytest.raises(ValueError, match="symlinks"):
        artifact.pack(source, base, TAG, ALLOW)
    assert outside.read_bytes() == b"must not be imported\n"


def test_materialize_rejects_symlink_parent_escape(source, base, clone, tmp_path):
    name = "tests/linked/test_added.py"
    payload = artifact.pack(source, base, TAG, ALLOW)
    payload["files"][name] = record(FIX)
    payload["candidate_head"] = artifact.commit_object(source, base, TAG, payload["files"])
    seal(payload)
    target = clone()
    outside = tmp_path / "outside directory"
    outside.mkdir()
    sentinel = outside / "test_added.py"
    sentinel.write_bytes(b"preserve me\n")
    (target / "tests").mkdir(exist_ok=True)
    symlink(target / "tests/linked", outside)
    (target / ".git/info/exclude").write_text("tests/linked\n", encoding="utf-8")
    before = state(target)
    assert not before[1]
    with pytest.raises(ValueError, match="Symlink parent"):
        artifact.materialize(target, payload, base, TAG, ALLOW + [name])
    assert state(target) == before
    assert sentinel.read_bytes() == b"preserve me\n"


def test_materialize_cannot_replace_tracked_symlink_even_without_os_symlink_support(source):
    # Build the symlink in Git's tree, so this guard is exercised on Windows too.
    link = (
        artifact.git(source, "hash-object", "-w", "--stdin", data=b"../README.md").decode().strip()
    )
    git(source, "update-index", "--add", "--cacheinfo", f"120000,{link},tests/test_added.py")
    git(source, "commit", "-qm", "tracked symlink")
    git(source, "-c", "core.symlinks=false", "checkout-index", "-f", "--", "tests/test_added.py")
    git(source, "config", "core.symlinks", "false")
    base = git(source, "rev-parse", "HEAD")
    payload = artifact.pack(source, base, TAG, ALLOW)
    payload["files"]["tests/test_added.py"] = record(FIX)
    payload["candidate_head"] = artifact.commit_object(source, base, TAG, payload["files"])
    seal(payload)
    before = state(source)
    assert not before[1]
    with pytest.raises(ValueError, match="Cannot replace a symlink"):
        artifact.materialize(source, payload, base, TAG, ALLOW)
    assert state(source) == before


@pytest.mark.parametrize("field", ["base", "tag"])
def test_stale_work_item_is_rejected(payload, base, clone, field):
    expected_base = "f" * 40 if field == "base" else base
    expected_tag = "v1.2.4" if field == "tag" else TAG
    target = clone()
    before = state(target)
    with pytest.raises(ValueError, match="selected work item"):
        artifact.materialize(target, payload, expected_base, expected_tag, ALLOW)
    assert state(target) == before


@pytest.mark.parametrize("tag", ["main", "1.2.3", "v1.2.3rc1", "v1.2.3-rc.1"])
def test_non_release_tag_is_rejected(payload, base, tag):
    payload["tag"] = tag
    seal(payload)
    with pytest.raises(ValueError, match="stable release tag"):
        artifact.validate(payload, base, tag, ALLOW)


def test_short_base_sha_is_rejected(payload, base):
    payload["base"] = base[:12]
    seal(payload)
    with pytest.raises(ValueError, match="full base SHA"):
        artifact.validate(payload, base[:12], TAG, ALLOW)


def test_clean_target_at_wrong_head_is_rejected(payload, base, clone):
    target = clone()
    git(
        target,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "different base",
    )
    before = state(target)
    assert not before[1]
    with pytest.raises(ValueError, match="not at the recorded base"):
        artifact.materialize(target, payload, base, TAG, ALLOW)
    assert state(target) == before


@pytest.mark.parametrize("kind", ["unstaged", "staged", "untracked"])
def test_dirty_target_is_rejected_without_losing_user_changes(payload, base, clone, kind):
    target = clone()
    path = target / ("notes.txt" if kind == "untracked" else "kvcached/compat.py")
    path.write_bytes(b"unfinished user work\n")
    if kind == "staged":
        git(target, "add", "kvcached/compat.py")
    before = state(target)
    with pytest.raises(ValueError, match="clean disposable checkout"):
        artifact.materialize(target, payload, base, TAG, ALLOW)
    assert state(target) == before
    assert path.read_bytes() == b"unfinished user work\n"


def test_cli_roundtrip_uses_allow_config(source, base, clone, tmp_path):
    (source / "kvcached/compat.py").write_bytes(FIX)
    config = tmp_path / "allow.json"
    config.write_text(json.dumps(ALLOW), encoding="utf-8")
    output = tmp_path / "candidate.json"

    def run(command, root):
        artifact_file = artifact.__file__
        assert artifact_file is not None
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(artifact_file)),
                command,
                "--source",
                str(root),
                "--base",
                base,
                "--tag",
                TAG,
                "--allow-list",
                str(config),
                "--artifact",
                str(output),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return completed.stdout.strip()

    expected = run("pack", source)
    target = clone()
    assert run("materialize", target) == expected == git(target, "rev-parse", "HEAD")
    assert run("verify", target) == expected
    assert (target / "kvcached/compat.py").read_bytes() == FIX
    assert not git(target, "status", "--porcelain")


@pytest.fixture
def json_envelope():
    return seal(
        {
            "schema": 1,
            "base": "a" * 40,
            "tag": TAG,
            "files": {"kvcached/compat.py": record(FIX)},
            "candidate_head": "b" * 40,
        }
    )


def test_load_envelope_exact_byte_limit(tmp_path, monkeypatch, json_envelope):
    content = artifact.canonical(json_envelope)
    content += b" " * (len(content) % 2)
    monkeypatch.setattr(artifact, "MAX_BYTES", len(content) // 2)
    path = tmp_path / "candidate.json"
    path.write_bytes(content)
    loaded = artifact.load_envelope(path)
    assert loaded == json_envelope
    artifact.validate(loaded, json_envelope["base"], TAG, ALLOW)
    path.write_bytes(content + b" ")
    with pytest.raises(ValueError, match="too large"):
        artifact.load_envelope(path)


def test_load_envelope_bounds_read_before_parsing(tmp_path, monkeypatch):
    sizes = []

    class ReadSpy(io.BytesIO):
        def read(self, size=-1):
            sizes.append(size)
            return super().read(size)

    stream = ReadSpy(b"x" * 100)
    monkeypatch.setattr(artifact, "MAX_BYTES", 8)
    monkeypatch.setattr(Path, "open", lambda self, mode: stream)
    monkeypatch.setattr(
        artifact.json, "loads", lambda *a, **kw: pytest.fail("Parsed oversized JSON")
    )
    with pytest.raises(ValueError, match="too large"):
        artifact.load_envelope(tmp_path / "candidate.json")
    assert sizes == [17]


@pytest.mark.parametrize(
    "content",
    [b"[]", b"null", b'{"schema":', b'{"data":"\xff"}', b'{"schema":1,"schema":1}'],
)
def test_load_envelope_rejects_malformed_json(tmp_path, content):
    path = tmp_path / "candidate.json"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        artifact.load_envelope(path)


def test_load_envelope_normalizes_excessive_nesting(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_bytes(b"[" * 2000 + b"0" + b"]" * 2000)
    with pytest.raises(ValueError):
        artifact.load_envelope(path)


@pytest.mark.parametrize(
    "field",
    [
        "root",
        "schema",
        "base",
        "tag",
        "candidate_head",
        "digest",
        "files",
        "record",
        "data",
        "sha256",
        "mode",
    ],
)
def test_envelope_api_rejects_malformed_types(tmp_path, monkeypatch, json_envelope, field):
    monkeypatch.setattr(artifact, "git", lambda *a, **kw: pytest.fail("Git ran on malformed input"))
    monkeypatch.setattr(artifact, "digest", lambda value: pytest.fail("Hashed malformed input"))
    invalid: Any
    for invalid in (None, [], True):
        candidate = json.loads(json.dumps(json_envelope))
        if field == "root":
            candidate = invalid
        elif field == "record":
            candidate["files"]["kvcached/compat.py"] = invalid
        elif field in ("data", "sha256", "mode"):
            candidate["files"]["kvcached/compat.py"][field] = invalid
        else:
            candidate[field] = invalid
        base, tag = json_envelope["base"], TAG
        with pytest.raises(ValueError):
            artifact.validate(candidate, base, tag, ALLOW)
        with pytest.raises(ValueError):
            artifact.materialize(tmp_path, candidate, base, tag, ALLOW)
        if field in ("base", "tag"):
            with pytest.raises(ValueError):
                artifact.validate(
                    candidate,
                    invalid if field == "base" else base,
                    invalid if field == "tag" else tag,
                    ALLOW,
                )


def test_envelope_api_bounds_encoded_content_before_decode(monkeypatch, json_envelope):
    monkeypatch.setattr(artifact, "MAX_BYTES", 2)
    monkeypatch.setattr(
        artifact.base64, "b64decode", lambda *a, **kw: pytest.fail("Decoded oversized data")
    )
    with pytest.raises(ValueError, match="Invalid file content"):
        artifact.validate(json_envelope, json_envelope["base"], TAG, ALLOW)


@pytest.fixture
def verified_target(payload, base, clone):
    target = clone()
    artifact.materialize(target, payload, base, TAG, ALLOW)
    return target


def test_verify_checkout_accepts_no_change(source, base):
    payload = artifact.pack(source, base, TAG, ALLOW)
    assert artifact.verify_checkout(source, payload, base, TAG, ALLOW) == base


def test_verify_checkout_rejects_clean_committed_changes(verified_target, payload, base):
    target = verified_target
    (target / "kvcached/compat.py").write_bytes(b"changed after CI\n")
    git(
        target,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qam",
        "post-CI edit",
    )
    assert not git(target, "status", "--porcelain")
    with pytest.raises(ValueError, match="Checkout HEAD"):
        artifact.verify_checkout(target, payload, base, TAG, ALLOW)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
@pytest.mark.parametrize("name", ["kvcached/compat.py", "README.md"])
def test_verify_checkout_detects_hidden_edits(verified_target, payload, base, flag, name):
    target = verified_target
    git(target, "update-index", flag, name)
    assert artifact.verify_checkout(target, payload, base, TAG, ALLOW) == payload["candidate_head"]
    (target / name).write_bytes(b"hidden post-CI edit\n")
    assert not git(target, "status", "--porcelain")
    with pytest.raises(ValueError, match="Tracked file content"):
        artifact.verify_checkout(target, payload, base, TAG, ALLOW)


def test_verify_checkout_rejects_index_only_edit(verified_target, payload, base):
    target = verified_target
    blob = artifact.git(target, "hash-object", "-w", "--stdin", data=b"index only").decode().strip()
    git(target, "update-index", "--cacheinfo", f"100644,{blob},kvcached/compat.py")
    assert (target / "kvcached/compat.py").read_bytes() == FIX
    with pytest.raises(ValueError, match="index tree"):
        artifact.verify_checkout(target, payload, base, TAG, ALLOW)


@pytest.mark.skipif(os.name == "nt", reason="NTFS does not expose POSIX executable-bit edits")
def test_verify_checkout_detects_mode_despite_git_filemode_false(verified_target, payload, base):
    target = verified_target
    git(target, "config", "core.filemode", "false")
    (target / "kvcached/compat.py").chmod(0o755)
    assert not git(target, "status", "--porcelain")
    with pytest.raises(ValueError, match="Tracked file mode"):
        artifact.verify_checkout(target, payload, base, TAG, ALLOW)


def test_verify_checkout_rejects_only_nonignored_untracked(verified_target, payload, base):
    target = verified_target
    (target / "generated.log").write_bytes(b"test output\n")
    with pytest.raises(ValueError, match="nonignored untracked"):
        artifact.verify_checkout(target, payload, base, TAG, ALLOW)
    (target / ".git/info/exclude").write_text("generated.log\n", encoding="utf-8")
    assert artifact.verify_checkout(target, payload, base, TAG, ALLOW) == payload["candidate_head"]


def test_verify_checkout_hashes_symlink_target_literal(source):
    path = source / "tests/test_added.py"
    literal = b"../README.md"
    if os.name == "nt":
        path.write_bytes(literal)
        git(source, "config", "core.symlinks", "false")
    else:
        path.symlink_to(literal.decode("ascii"))
    blob = artifact.git(source, "hash-object", "-w", "--stdin", data=literal).decode().strip()
    git(source, "update-index", "--add", "--cacheinfo", f"120000,{blob},tests/test_added.py")
    git(source, "commit", "-qm", "existing repository symlink")
    base = git(source, "rev-parse", "HEAD")
    payload = artifact.pack(source, base, TAG, ALLOW)
    assert artifact.verify_checkout(source, payload, base, TAG, ALLOW) == base
    git(source, "update-index", "--assume-unchanged", "tests/test_added.py")
    if os.name == "nt":
        path.write_bytes(b".././README.md")
    else:
        path.unlink()
        path.symlink_to(".././README.md")
    assert not git(source, "status", "--porcelain")
    with pytest.raises(ValueError, match="Tracked file content"):
        artifact.verify_checkout(source, payload, base, TAG, ALLOW)
