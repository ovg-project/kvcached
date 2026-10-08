#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Discover stable engine releases and track each engine/release/profile independently.

Public API: discover/claim/finish return JSON plans; GhAPI is injectable as api=.
Discovery orders numeric versions; the CLI atomically saves updated plans.
Writers MUST share repository-wide workflow concurrency: comment snapshot checks are
not atomic compare-and-swap. The workflow must block GITHUB_RUN_ATTEMPT > 1; retry needs
a new manual dispatch with --tag ... --retry and /actions/runs/ID/attempts/N URLs.
Never automatically retry ambiguous writes or adopt marked foreign comments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

UPSTREAM = "vllm-project/vllm"
UPSTREAMS = {"vllm": UPSTREAM, "sglang": "sgl-project/sglang"}
TRACKER_MARKER = "<!-- kvcached-vllm-release-tracker -->"
COMMENT_MARKER = "<!-- kvcached-vllm-release "
VERSION = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:\.post(0|[1-9][0-9]*))?"
TERMINAL = {"passed", "failed", "blocked"}
IDENTITY = ("release_id", "tag", "engine_sha")


class GateError(RuntimeError):
    """A fail-closed validation or infrastructure error, safe to print publicly."""


class APIError(GateError):
    def __init__(self, status: Optional[int] = None):
        self.status = status
        super().__init__("GitHub API request failed" + (f" (HTTP {status})" if status else ""))


class GhAPI:
    """JSON-only gh transport; no shell, implicit host, API cache, or write retries."""

    def request(self, endpoint: str, *, method="GET", payload=None, paginate=False) -> Any:
        argv = ["gh", "api", "--hostname", "github.com", "--method", method, endpoint]
        argv += ["--header", "Accept: application/vnd.github+json"]
        if paginate:
            if method != "GET":
                raise GateError("Only list reads may paginate")
            argv += ["--paginate", "--slurp"]
        if payload is not None:
            argv += ["--input", "-"]
        try:
            result = subprocess.run(
                argv,
                input=json.dumps(payload) if payload is not None else None,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired, UnicodeError):
            raise APIError() from None
        if result.returncode:
            status = re.search(r"\(HTTP ([0-9]{3})\)", result.stderr)
            raise APIError(int(status[1]) if status else None)
        try:
            data = json.loads(result.stdout)
        except (ValueError, TypeError):
            raise GateError("GitHub returned invalid JSON") from None
        if paginate:
            if not isinstance(data, list) or any(not isinstance(page, list) for page in data):
                raise GateError("GitHub returned an invalid paginated list")
            data = [item for page in data for item in page]
        return data

    def authenticated_login(self) -> str:
        try:
            user = self.request("user")
        except APIError as error:
            if error.status != 403:
                raise
            # Installation tokens may reject /user. Ask GitHub, never trust GITHUB_ACTOR:
            # the triggering human is not the author used by the Actions installation.
            result = self.request(
                "graphql", method="POST", payload={"query": "query { viewer { login } }"}
            )
            if not isinstance(result, dict) or result.get("errors"):
                raise GateError("Cannot establish authenticated comment author")
            data = result.get("data")
            user = data.get("viewer") if isinstance(data, dict) else None
        login = user.get("login") if isinstance(user, dict) else None
        if not isinstance(login, str) or not re.fullmatch(r"[A-Za-z0-9-]+(?:\[bot\])?", login):
            raise GateError("Cannot establish authenticated comment author")
        return login


def _require(condition, message):
    if not condition:
        raise GateError(message)


def _positive(value):
    return type(value) is int and value > 0


def _matches(pattern, value):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def _version(value):
    match = re.fullmatch(VERSION, value.removeprefix("v"))
    _require(match is not None, "Invalid stable version")
    return tuple(int(part) if part is not None else -1 for part in match.groups())


def _target(repository, tracker_issue):
    _require(
        _matches(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", repository)
        and repository.split("/")[1] not in {".", ".."}
        and _positive(tracker_issue),
        "Expected repository owner/repo and a positive tracker issue number",
    )
    return f"repos/{repository}/issues/{tracker_issue}"


def _url(value, repository, kind):
    suffix = r"actions/runs/[1-9][0-9]*/attempts/[1-9][0-9]*"
    if kind == "pr":
        suffix = r"pull/[1-9][0-9]*"
    _require(
        _matches(r"https://github\.com/" + re.escape(repository) + "/" + suffix, value),
        "Expected a public GitHub URL in the target repository",
    )


def _identity(data):
    _require(
        isinstance(data, dict)
        and _positive(data.get("release_id"))
        and _matches("v" + VERSION, data.get("tag"))
        and _matches(r"[0-9a-f]{40}", data.get("engine_sha")),
        "Invalid release identity",
    )
    return tuple(data[key] for key in IDENTITY)


def _same_identity(left, right):
    _require(_identity(left) == _identity(right), "Release identity changed: moved or replaced")
    _require(_engine(left) == _engine(right), "Release engine changed")


def _engine(data):
    name = data.get("engine", "vllm")
    _require(isinstance(name, str) and name in UPSTREAMS, "Unsupported release engine")
    return name


def _profile(data):
    # Historical ledger entries ran only the original profiling task.
    name = data.get("profile", "vllm")
    _require(_matches(r"[a-z][a-z0-9-]{0,47}", name), "Invalid ledger profile")
    return name


def resolve_tag(api, tag: str, engine="vllm") -> str:
    """Resolve an exact stable upstream tag, peeling annotated tags to a commit SHA."""
    _require(_matches("v" + VERSION, tag), "Expected a stable vX.Y.Z tag")
    upstream = UPSTREAMS[_engine({"engine": engine})]
    ref = api.request(f"repos/{upstream}/git/ref/tags/{tag}")
    _require(isinstance(ref, dict) and ref.get("ref") == f"refs/tags/{tag}", "Invalid tag ref")
    obj, seen = ref.get("object"), set()
    for _ in range(16):
        _require(
            isinstance(obj, dict) and _matches(r"[0-9a-f]{40}", obj.get("sha")),
            "Invalid tag object",
        )
        sha = obj["sha"]
        if obj.get("type") == "commit":
            return sha
        _require(obj.get("type") == "tag" and sha not in seen, "Invalid or cyclic annotated tag")
        seen.add(sha)
        annotated = api.request(f"repos/{upstream}/git/tags/{sha}")
        _require(isinstance(annotated, dict) and annotated.get("sha") == sha, "Invalid tag")
        obj = annotated.get("object")
    raise GateError("Annotated tag nesting limit exceeded")


def _release(api, release, engine="vllm"):
    _require(
        isinstance(release, dict)
        and _positive(release.get("id"))
        and release.get("draft") is False
        and release.get("prerelease") is False
        and _matches("v" + VERSION, release.get("tag_name")),
        "Expected a stable published engine release",
    )
    tag = release["tag_name"]
    url = f"https://github.com/{UPSTREAMS[engine]}/releases/tag/{tag}"
    _require(release.get("html_url") == url, "Invalid upstream release URL")
    return {
        "release_id": release["id"],
        "tag": tag,
        "engine_sha": resolve_tag(api, tag, engine),
        "release_url": url,
        **({"engine": engine} if engine != "vllm" else {}),
    }


def _list(api, endpoint):
    items = api.request(endpoint + "?per_page=100", paginate=True)
    _require(
        isinstance(items, list) and all(isinstance(item, dict) for item in items),
        "GitHub returned an invalid list",
    )
    return items


def _comment(comment, repository, issue_url):
    body = comment.get("body")
    _require(isinstance(body, str), "Invalid issue comment body")
    if COMMENT_MARKER not in body:
        return None
    _require(body.startswith(COMMENT_MARKER) and body.endswith(" -->"), "Malformed ledger marker")
    try:
        record = json.loads(body[len(COMMENT_MARKER) : -4])
    except ValueError:
        raise GateError("Malformed ledger JSON") from None
    _identity(record)
    _require(
        set(record) <= {*IDENTITY, "status", "run_url", "pr_url", "profile", "engine"}
        and isinstance(record.get("status"), str)
        and record["status"] in TERMINAL | {"running"},
        "Invalid ledger fields or status",
    )
    _profile(record)
    _engine(record)
    _url(record.get("run_url"), repository, "run")
    if "pr_url" in record:
        _url(record["pr_url"], repository, "pr")
    user = comment.get("user")
    _require(
        _positive(comment.get("id"))
        and comment.get("issue_url") == issue_url
        and isinstance(user, dict)
        and isinstance(user.get("login"), str)
        and bool(user["login"]),
        "Invalid ledger comment identity or author",
    )
    revision = hashlib.sha256(
        json.dumps([body, comment.get("updated_at")], ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {"id": comment["id"], "author": user["login"], "revision": revision, "record": record}


def _ledger(api, repository, tracker_issue):
    endpoint = _target(repository, tracker_issue)
    issue = api.request(endpoint)
    _require(
        isinstance(issue, dict)
        and issue.get("number") == tracker_issue
        and issue.get("url") == f"https://api.github.com/{endpoint}"
        and "pull_request" not in issue
        and isinstance(issue.get("body"), str)
        and TRACKER_MARKER in issue["body"],
        "Configured issue is not a marked vLLM release tracker",
    )
    entries, ids, tags, comments = [], set(), set(), set()
    for comment in _list(api, endpoint + "/comments"):
        entry = _comment(comment, repository, issue["url"])
        if entry is None:
            continue
        record = entry["record"]
        profile = _profile(record)
        _require(
            (_engine(record), record["release_id"], profile) not in ids
            and (_engine(record), record["tag"], profile) not in tags
            and entry["id"] not in comments,
            "Duplicate ledger release or comment",
        )
        ids.add((_engine(record), record["release_id"], profile))
        tags.add((_engine(record), record["tag"], profile))
        comments.add(entry["id"])
        entries.append(entry)
    return entries


def _entry(entries, release):
    matches = [
        entry
        for entry in entries
        if _engine(entry["record"]) == _engine(release)
        and (entry["record"]["release_id"] == release["release_id"]
             or entry["record"]["tag"] == release["tag"])
    ]
    for entry in matches:
        _same_identity(entry["record"], release)
    matches = [entry for entry in matches if _profile(entry["record"]) == _profile(release)]
    _require(len(matches) <= 1, "Conflicting release identities in ledger")
    if matches:
        _same_identity(matches[0]["record"], release)
        return matches[0]
    return None


def discover(repository, tracker_issue, first_version="0.28.0", *, tag=None, retry=False,
             profile="vllm", engine="vllm", api=None):
    """Return pending/idle; only explicit tag+retry may select an existing ledger entry."""
    _require(_matches(VERSION, first_version), "Expected first-version X.Y.Z")
    _require(type(retry) is bool and (not retry or tag is not None), "Retry needs an explicit tag")
    _require(tag is None or _matches("v" + VERSION, tag), "Expected a stable vX.Y.Z tag")
    _profile({"profile": profile})
    _engine({"engine": engine})
    floor = _version(first_version)
    api = api if api is not None else GhAPI()
    entries = _ledger(api, repository, tracker_issue)
    releases = _list(api, f"repos/{UPSTREAMS[engine]}/releases")
    candidates, ids, tags = [], set(), set()
    for release in releases:
        name = release.get("tag_name")
        if release.get("draft") is not False or release.get("prerelease") is not False:
            continue
        if not _matches("v" + VERSION, name):
            continue
        version = _version(name)
        if version < floor:
            continue
        _require(_positive(release.get("id")), "Invalid release ID")
        _require(release["id"] not in ids and name not in tags, "Duplicate upstream release")
        ids.add(release["id"])
        tags.add(name)
        candidates.append((version, release))
    if tag is not None:
        _require(tag in tags, "Requested tag is not an eligible published release")
    for _, candidate in sorted(candidates, key=lambda item: item[0]):
        if tag is not None and candidate["tag_name"] != tag:
            continue
        release = _release(api, candidate, engine)
        release["profile"] = profile
        entry = _entry(entries, release)
        if entry and not retry:
            continue
        plan = {
            "status": "pending",
            **release,
            "repository": repository,
            "tracker_issue": tracker_issue,
            "retry": retry,
        }
        if retry:
            plan["manual_tag"] = tag
        if entry:
            plan.update(ledger_comment_id=entry["id"], ledger_revision=entry["revision"])
        if profile == "auto":
            previous = [item for item in releases
                        if item.get("draft") is False and item.get("prerelease") is False
                        and _matches("v" + VERSION, item.get("tag_name"))
                        and _version(item["tag_name"]) < _version(release["tag"])]
            _require(bool(previous), "Automatic analysis needs a preceding stable release")
            predecessor = max(previous, key=lambda item: _version(item["tag_name"]))
            plan.update(old_tag=predecessor["tag_name"],
                        old_engine_sha=resolve_tag(api, predecessor["tag_name"], engine))
        return plan
    return {"status": "idle", **dict.fromkeys((*IDENTITY, "release_url"))}


def _prepare(api, repository, tracker_issue, plan, run_url):
    _target(repository, tracker_issue)
    _identity(plan)
    _require(
        plan.get("repository") == repository and plan.get("tracker_issue") == tracker_issue,
        "Plan belongs to another tracker",
    )
    _url(run_url, repository, "run")
    engine = _engine(plan)
    candidate = api.request(f"repos/{UPSTREAMS[engine]}/releases/{plan['release_id']}")
    _require(
        isinstance(candidate, dict)
        and candidate.get("id") == plan["release_id"]
        and candidate.get("tag_name") == plan["tag"],
        "Release identity changed (tag move or replacement)",
    )
    release = _release(api, candidate, engine)
    _same_identity(release, plan)
    _require(release["release_url"] == plan.get("release_url"), "Plan release URL changed")
    entry = _entry(_ledger(api, repository, tracker_issue), plan)
    login = api.authenticated_login()
    if entry:
        _require(entry["author"].casefold() == login.casefold(), "Cannot update a foreign comment")
        _require(
            entry["id"] == plan.get("ledger_comment_id")
            and entry["revision"] == plan.get("ledger_revision"),
            "Ledger claim changed since this plan was read",
        )
    else:
        _require("ledger_comment_id" not in plan, "Planned ledger comment disappeared")
    return entry, login


def _write(api, repository, tracker_issue, plan, entry, login, record):
    # Re-list even for POST: a new comment may have appeared since discovery/preflight.
    current = _entry(_ledger(api, repository, tracker_issue), plan)
    _require(current == entry, "Ledger claim changed before write")
    endpoint = _target(repository, tracker_issue) + "/comments"
    method = "POST"
    if entry:
        endpoint = f"repos/{repository}/issues/comments/{entry['id']}"
        method = "PATCH"
    body = COMMENT_MARKER + json.dumps(record, sort_keys=True, separators=(",", ":")) + " -->"
    response = api.request(endpoint, method=method, payload={"body": body})
    _require(isinstance(response, dict), "Invalid ledger write response")
    saved = _comment(
        response, repository, f"https://api.github.com/{_target(repository, tracker_issue)}"
    )
    _require(
        saved is not None
        and saved["author"].casefold() == login.casefold()
        and saved["record"] == record
        and (entry is None or saved["id"] == entry["id"]),
        "Ledger write identity mismatch",
    )
    _require(
        _entry(_ledger(api, repository, tracker_issue), plan) == saved,
        "Ledger changed after write; do not retry automatically",
    )
    return dict(plan, **record, ledger_comment_id=saved["id"], ledger_revision=saved["revision"])


def claim(repository: str, tracker_issue: int, plan: dict, run_url: str, *, api=None) -> dict:
    """Return idle unchanged, or claim a pending release with status running."""
    if isinstance(plan, dict) and plan.get("status") == "idle":
        return dict(plan)
    api = api if api is not None else GhAPI()
    entry, login = _prepare(api, repository, tracker_issue, plan, run_url)
    if plan.get("status") == "running":
        _require(
            entry is not None
            and entry["record"]["status"] == "running"
            and entry["record"]["run_url"] == plan.get("run_url") == run_url,
            "Running claim does not belong to this run",
        )
        return dict(plan)
    _require(plan.get("status") == "pending", "Claim requires a pending plan")
    if entry:
        _require(
            plan.get("retry") is True and plan.get("manual_tag") == plan["tag"],
            "Existing claim requires explicit manual tag and retry",
        )
        _require(entry["record"]["run_url"] != run_url, "Retry requires a new run URL")
    record = {key: plan[key] for key in IDENTITY}
    if _engine(plan) != "vllm":
        record["engine"] = _engine(plan)
    record["profile"] = _profile(plan)
    record.update(status="running", run_url=run_url)
    return _write(api, repository, tracker_issue, plan, entry, login, record)


def finish(repository, tracker_issue, plan, status, run_url, *, pr_url=None, api=None):
    """Finish only the owned, unchanged claim for this exact release and run."""
    _require(
        isinstance(status, str) and status in TERMINAL,
        "Finish status must be passed, failed, or blocked",
    )
    if pr_url is not None:
        _url(pr_url, repository, "pr")
    api = api if api is not None else GhAPI()
    entry, login = _prepare(api, repository, tracker_issue, plan, run_url)
    _require(
        entry is not None and entry["record"]["run_url"] == plan.get("run_url") == run_url,
        "Finish requires this run's existing claim",
    )
    record = {key: plan[key] for key in IDENTITY}
    if _engine(plan) != "vllm":
        record["engine"] = _engine(plan)
    record["profile"] = _profile(plan)
    record.update(status=status, run_url=run_url)
    if pr_url is not None:
        record["pr_url"] = pr_url
    if entry["record"] == record and plan.get("status") == status:
        return dict(plan)
    _require(
        entry["record"]["status"] == plan.get("status") == "running",
        "Only a running claim can be finished",
    )
    return _write(api, repository, tracker_issue, plan, entry, login, record)


def _save(path, plan):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(plan, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("discover", "claim", "finish"):
        command = commands.add_parser(name)
        command.add_argument("--repository", required=True)
        command.add_argument("--tracker-issue", type=int, required=True)
        if name == "discover":
            command.add_argument("--first-version", default="0.28.0")
            command.add_argument("--tag")
            command.add_argument("--profile", default="vllm")
            command.add_argument("--engine", choices=sorted(UPSTREAMS), default="vllm")
            command.add_argument("--retry", action="store_true")
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--plan", type=Path, required=True)
            command.add_argument("--run-url", required=True)
        if name == "finish":
            command.add_argument("--status", choices=sorted(TERMINAL), required=True)
            command.add_argument("--pr-url")
    args = vars(parser.parse_args(argv))
    operation = args.pop("command")
    path = args.pop("output" if operation == "discover" else "plan")
    try:
        if operation != "discover":
            args["plan"] = json.loads(path.read_text(encoding="utf-8"))
        if operation == "discover":
            plan = discover(**args)
        elif operation == "claim":
            plan = claim(**args)
        else:
            plan = finish(**args)
        _save(path, plan)
    except GateError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError, UnicodeError):
        print("error: Unable to read or write valid plan JSON", file=sys.stderr)
        return 1
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
