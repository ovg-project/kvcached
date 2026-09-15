# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""CPU-only release/ledger tests. All GitHub access is mocked, including subprocesses."""

import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Optional

import pytest

if TYPE_CHECKING:
    from tools import engine_release_discovery as discovery
else:
    spec = importlib.util.spec_from_file_location(
        "engine_release_discovery", Path(__file__).parents[1] / "tools/engine_release_discovery.py"
    )
    assert spec is not None and spec.loader is not None
    discovery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(discovery)

REPOSITORY = "owner/repo"
ISSUE = 17
ISSUE_PATH = f"repos/{REPOSITORY}/issues/{ISSUE}"
COMMENTS_PATH = ISSUE_PATH + "/comments?per_page=100"
RELEASES_PATH = f"repos/{discovery.UPSTREAM}/releases"
RUN = f"https://github.com/{REPOSITORY}/actions/runs/100/attempts/1"
RETRY_RUN = f"https://github.com/{REPOSITORY}/actions/runs/101/attempts/1"
PR = f"https://github.com/{REPOSITORY}/pull/42"
BOT = "github-actions[bot]"


def release(number=28, tag="v0.28.0", **overrides):
    return {
        "id": number,
        "tag_name": tag,
        "draft": False,
        "prerelease": False,
        "html_url": f"https://github.com/{discovery.UPSTREAM}/releases/tag/{tag}",
        "target_commitish": "main",
        **overrides,
    }


def record(number=28, tag="v0.28.0", status="running", **overrides):
    return {
        "release_id": number,
        "tag": tag,
        "engine_sha": f"{number:040x}",
        "status": status,
        "run_url": RUN,
        **overrides,
    }


def comment(data=None, comment_id=71, author=BOT, **overrides):
    data = record() if data is None else data
    return {
        "id": comment_id,
        "body": discovery.COMMENT_MARKER + json.dumps(data) + " -->",
        "user": {"login": author},
        "issue_url": f"https://api.github.com/{ISSUE_PATH}",
        "updated_at": "2026-09-16T00:00:00Z",
        **overrides,
    }


class FakeAPI(discovery.GhAPI):
    def __init__(self):
        self.releases = [release()]
        self.comments = []
        self.issue = {
            "number": ISSUE,
            "url": f"https://api.github.com/{ISSUE_PATH}",
            "body": "Externally configured\n" + discovery.TRACKER_MARKER,
            "title": "Unrelated title",
        }
        self.login = BOT
        self.responses = {}
        self.calls = []
        self.hook = lambda *args: None

    @property
    def writes(self):
        return [call for call in self.calls if call[1] != "GET" and call[0] != "graphql"]

    def request(self, endpoint, *, method="GET", payload=None, paginate=False):
        self.calls.append((endpoint, method, copy.deepcopy(payload), paginate))
        self.hook(endpoint, method)
        if endpoint in self.responses:
            response = self.responses[endpoint]
            if isinstance(response, Exception):
                raise response
            return copy.deepcopy(response)
        if endpoint == "user":
            return {"login": self.login}
        if endpoint == "graphql":
            return {"data": {"viewer": {"login": self.login}}}
        if method == "GET":
            if endpoint == ISSUE_PATH:
                return copy.deepcopy(self.issue)
            if endpoint == COMMENTS_PATH:
                assert paginate
                return copy.deepcopy(self.comments)
            if endpoint == RELEASES_PATH + "?per_page=100":
                assert paginate
                return copy.deepcopy(self.releases)
            for item in self.releases:
                if endpoint == RELEASES_PATH + f"/{item['id']}":
                    return copy.deepcopy(item)
                if endpoint == f"repos/{discovery.UPSTREAM}/git/ref/tags/{item['tag_name']}":
                    return {
                        "ref": f"refs/tags/{item['tag_name']}",
                        "object": {"type": "commit", "sha": f"{item['id']:040x}"},
                    }
            raise AssertionError(f"Unexpected read: {endpoint}")
        if method == "POST":
            assert endpoint == ISSUE_PATH + "/comments"
            item = comment(comment_id=101, author=self.login, body=payload["body"])
            self.comments.append(item)
        else:
            assert method == "PATCH"
            item = next(c for c in self.comments if endpoint.endswith(f"/comments/{c['id']}"))
            item["body"] = payload["body"]
            item["updated_at"] += "1"
        return copy.deepcopy(item)


@pytest.fixture(autouse=True)
def no_real_github(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Real subprocess/GitHub access is prohibited")

    monkeypatch.setattr(discovery.subprocess, "run", forbidden)


@pytest.fixture
def api():
    return FakeAPI()


def discover(api: FakeAPI, **kwargs: Any) -> Dict[str, Any]:
    return discovery.discover(REPOSITORY, ISSUE, api=api, **kwargs)


def claim(
    api: FakeAPI, plan: Optional[Dict[str, Any]] = None, run_url: str = RUN
) -> Dict[str, Any]:
    plan = discover(api) if plan is None else plan
    return discovery.claim(REPOSITORY, ISSUE, plan, run_url, api=api)


def finish(
    api: FakeAPI,
    plan: Dict[str, Any],
    status: str = "passed",
    run_url: str = RUN,
    **kwargs: Any,
) -> Dict[str, Any]:
    return discovery.finish(REPOSITORY, ISSUE, plan, status, run_url, api=api, **kwargs)


def test_discover_numeric_oldest_unprocessed_read_only(api):
    api.releases = [
        release(30, "v0.30.0"),
        release(310, "v0.31.0", prerelease=True),
        release(320, "v0.32.0", draft=True),
        release(33, "v0.33.0rc1"),
        release(34, "v0.34.0+build"),
        release(35, "0.35.0"),
        release(36, "v00.36.0"),
        release(27, "v0.27.0"),
        release(2810, "v0.28.10"),
        release(282, "v0.28.2"),
        release(),
    ]
    api.comments = [comment(record(status="passed"))]
    plan = discover(api)
    assert plan["status"] == "pending" and plan["tag"] == "v0.28.2"
    assert plan["engine_sha"] == f"{282:040x}"
    assert plan["repository"] == REPOSITORY and plan["tracker_issue"] == ISSUE
    assert "ledger_comment_id" not in plan
    assert not api.writes and all(call[0] != "user" for call in api.calls)


@pytest.mark.parametrize("status", ["running", "passed", "failed", "blocked"])
def test_existing_states_are_idle_unless_manual_retry(api, status):
    api.comments = [comment(record(status=status))]
    idle = discover(api)
    assert idle == {
        "status": "idle",
        "release_id": None,
        "tag": None,
        "engine_sha": None,
        "release_url": None,
    }
    assert discover(api, tag="v0.28.0")["status"] == "idle"
    plan = discover(api, tag="v0.28.0", retry=True)
    assert plan["status"] == "pending" and plan["ledger_comment_id"] == 71
    assert len(plan["ledger_revision"]) == 64
    assert plan["manual_tag"] == "v0.28.0"
    assert not api.writes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"retry": True},
        {"retry": "true"},
        {"tag": "v0.28.0rc1"},
        {"tag": "v0.28.0/extra"},
        {"tag": "v0.29.0"},
        {"first_version": "v0.28.0"},
        {"first_version": "0.28"},
        {"first_version": "00.28.0"},
        {"tag": "v0.28.0", "first_version": "0.29.0"},
    ],
)
def test_invalid_selection_fails(api, kwargs):
    with pytest.raises(discovery.GateError):
        discover(api, **kwargs)
    assert not api.writes


@pytest.mark.parametrize("flag", ["draft", "prerelease"])
def test_explicit_tag_cannot_select_unpublished_or_prerelease(api, flag):
    api.releases[0][flag] = True
    assert discover(api)["status"] == "idle"
    with pytest.raises(discovery.GateError):
        discover(api, tag="v0.28.0", retry=True)


@pytest.mark.parametrize(
    "change",
    [
        {"body": "No marker", "title": discovery.TRACKER_MARKER},
        {"body": None},
        {"pull_request": {}},
        {"number": 99},
        {"url": "https://api.github.com/repos/other/repo/issues/17"},
    ],
)
def test_tracker_marker_is_in_configured_issue_body(api, change):
    api.issue.update(change)
    with pytest.raises(discovery.GateError, match="tracker"):
        discover(api)
    assert not api.writes


def test_unmarked_foreign_comments_are_ignored(api):
    api.comments = [comment(author="someone", body="Discussion, not a ledger entry")]
    assert discover(api)["status"] == "pending"


@pytest.mark.parametrize("mode", ["same-id", "same-tag", "same-comment-id"])
def test_duplicate_ledger_is_not_silently_resolved(api, mode):
    second = record(29, "v0.29.0")
    second_id = 72
    if mode == "same-id":
        second["release_id"] = 28
    elif mode == "same-tag":
        second["tag"] = "v0.28.0"
    else:
        second_id = 71
    api.comments = [comment(), comment(second, comment_id=second_id)]
    with pytest.raises(discovery.GateError, match="Duplicate"):
        discover(api)


@pytest.mark.parametrize("second", [release(), release(29), release(28, "v0.29.0")])
def test_duplicate_release_list_fails_closed(api, second):
    api.releases.append(second)
    with pytest.raises(discovery.GateError, match="Duplicate upstream"):
        discover(api)


@pytest.mark.parametrize(
    "body",
    [
        discovery.COMMENT_MARKER + "not JSON -->",
        discovery.COMMENT_MARKER + "{} -->",
        discovery.COMMENT_MARKER + "[] -->",
        "quoted " + discovery.COMMENT_MARKER + json.dumps(record()) + " -->",
        discovery.COMMENT_MARKER + json.dumps(record()) + " --> trailing",
        discovery.COMMENT_MARKER + json.dumps(record(status="unknown")) + " -->",
        discovery.COMMENT_MARKER + json.dumps(record(status=[])) + " -->",
        discovery.COMMENT_MARKER + json.dumps(record(log="private path")) + " -->",
    ],
)
def test_malformed_marked_comments_are_not_treated_as_absent(api, body):
    api.comments = [comment(body=body)]
    with pytest.raises(discovery.GateError):
        discover(api)
    assert not api.writes


@pytest.mark.parametrize(
    "changes",
    [
        {"engine_sha": "a" * 40},
        {"release_id": 29},
        {"tag": "v0.29.0"},
    ],
)
def test_discover_rejects_moved_or_recreated_release_even_on_retry(api, changes):
    api.comments = [comment(record(**changes))]
    for kwargs in ({}, {"tag": "v0.28.0", "retry": True}):
        with pytest.raises(discovery.GateError, match="identity changed"):
            discover(api, **kwargs)


def test_annotated_tags_are_peeled_recursively(api):
    ref = f"repos/{discovery.UPSTREAM}/git/ref/tags/v0.28.0"
    api.responses[ref] = {
        "ref": "refs/tags/v0.28.0",
        "object": {"type": "tag", "sha": "a" * 40},
    }
    for current, following, kind in [("a", "b", "tag"), ("b", "c", "commit")]:
        api.responses[f"repos/{discovery.UPSTREAM}/git/tags/{current * 40}"] = {
            "sha": current * 40,
            "object": {"type": kind, "sha": following * 40},
        }
    assert discover(api)["engine_sha"] == "c" * 40
    assert not any("main" in call[0] for call in api.calls)


@pytest.mark.parametrize("mode", ["cycle", "tree", "bad-sha", "wrong-ref", "wrong-tag-sha"])
def test_invalid_tag_graphs_fail_closed(api, mode):
    obj = {"type": "tag", "sha": "a" * 40}
    ref = {"ref": "refs/tags/v0.28.0", "object": obj}
    if mode == "tree":
        obj["type"] = "tree"
    elif mode == "bad-sha":
        obj["sha"] = "main"
    elif mode == "wrong-ref":
        ref["ref"] = "refs/tags/v0.29.0"
    api.responses[f"repos/{discovery.UPSTREAM}/git/ref/tags/v0.28.0"] = ref
    api.responses[f"repos/{discovery.UPSTREAM}/git/tags/{'a' * 40}"] = {
        "sha": "b" * 40 if mode == "wrong-tag-sha" else "a" * 40,
        "object": obj,
    }
    with pytest.raises(discovery.GateError):
        discover(api)


@pytest.mark.parametrize("endpoint", [ISSUE_PATH, COMMENTS_PATH, RELEASES_PATH + "?per_page=100"])
@pytest.mark.parametrize("failure", [discovery.APIError(403), discovery.APIError(500), {}, None])
def test_infrastructure_list_failure_is_not_idle_or_an_empty_ledger(api, endpoint, failure):
    api.responses[endpoint] = failure
    with pytest.raises(discovery.GateError):
        discover(api)
    assert not api.writes


def test_claim_and_finish_reuse_single_owned_comment(api):
    pending = discover(api)
    running = claim(api, pending)
    assert pending["status"] == "pending" and "ledger_comment_id" not in pending
    assert running["status"] == "running" and running["ledger_comment_id"] == 101
    assert claim(api, running) == running
    done = finish(api, running, pr_url=PR)
    assert done["status"] == "passed" and done["pr_url"] == PR
    assert finish(api, done, pr_url=PR) == done
    assert [call[1] for call in api.writes] == ["POST", "PATCH"]
    assert len(api.comments) == 1
    data = json.loads(api.comments[0]["body"][len(discovery.COMMENT_MARKER) : -4])
    assert data == record(status="passed", pr_url=PR)
    assert discover(api)["status"] == "idle"


def test_claim_idle_is_noop_without_api_access(api):
    api.releases = []
    idle = discover(api)
    api.calls.clear()
    assert claim(api, idle) == idle
    assert not api.calls


@pytest.mark.parametrize("old_status", ["running", "failed", "passed", "blocked"])
def test_retry_updates_owned_comment_and_invalidates_old_run(api, old_status):
    old = claim(api)
    if old_status != "running":
        old = finish(api, old, status=old_status)
    pending = discover(api, tag="v0.28.0", retry=True)
    running = claim(api, pending, run_url=RETRY_RUN)
    assert running["ledger_comment_id"] == old["ledger_comment_id"]
    assert running["run_url"] == RETRY_RUN and len(api.comments) == 1
    with pytest.raises(discovery.GateError):
        finish(api, old)
    assert finish(api, running, status="blocked", run_url=RETRY_RUN)["status"] == "blocked"


@pytest.mark.parametrize("login", [BOT, "release-maintainer", "another-app[bot]"])
def test_author_is_authenticated_user_not_workflow_actor(api, monkeypatch, login):
    api.login = login
    monkeypatch.setenv("GITHUB_ACTOR", "unrelated-human")
    running = claim(api)
    assert api.comments[0]["user"]["login"] == login
    assert finish(api, running)["status"] == "passed"


def test_actions_token_403_uses_authenticated_graphql_viewer(api, monkeypatch):
    api.responses["user"] = discovery.APIError(403)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_ACTOR", "triggering-human")
    plan = claim(api)
    assert finish(api, plan)["status"] == "passed"
    queries = [call for call in api.calls if call[0] == "graphql"]
    assert queries and all(call[2] == {"query": "query { viewer { login } }"} for call in queries)


@pytest.mark.parametrize("status", [401, 404, 429, 500, None])
def test_user_infrastructure_errors_do_not_fallback_to_actor(api, monkeypatch, status):
    api.responses["user"] = discovery.APIError(status)
    monkeypatch.setenv("GITHUB_ACTOR", BOT)
    with pytest.raises(discovery.APIError):
        claim(api)
    assert not api.writes and not any(call[0] == "graphql" for call in api.calls)


@pytest.mark.parametrize(
    "response",
    [
        discovery.APIError(403),
        {"errors": [{"message": "private infrastructure error"}]},
        {"data": None},
        {"data": ["not an object"]},
        {"data": {"viewer": None}},
        {"data": {"viewer": {"login": ""}}},
    ],
)
def test_unknown_authenticated_author_fails_closed(api, monkeypatch, response):
    api.responses.update(user=discovery.APIError(403), graphql=response)
    monkeypatch.setenv("GITHUB_ACTOR", BOT)
    with pytest.raises(discovery.GateError):
        claim(api)
    assert not api.writes


def test_foreign_marked_comment_reserves_release_but_cannot_be_retried(api):
    api.comments = [comment(author="foreign-human")]
    assert discover(api)["status"] == "idle"
    pending = discover(api, tag="v0.28.0", retry=True)
    with pytest.raises(discovery.GateError, match="foreign"):
        claim(api, pending, run_url=RETRY_RUN)
    assert not api.writes


@pytest.mark.parametrize("change", ["author", "body", "id", "deleted", "issue"])
def test_finish_rejects_changed_or_foreign_comment(api, change):
    running = claim(api)
    api.calls.clear()
    if change == "author":
        api.comments[0]["user"]["login"] = "other-person"
    elif change == "body":
        api.comments[0]["body"] = comment(record(status="failed"))["body"]
    elif change == "id":
        api.comments[0]["id"] = 999
    elif change == "issue":
        api.comments[0]["issue_url"] = "https://api.github.com/repos/other/repo/issues/17"
    else:
        api.comments.clear()
    with pytest.raises(discovery.GateError):
        finish(api, running)
    assert not api.writes


def test_two_new_plans_cannot_claim_twice(api):
    first, second = discover(api), discover(api)
    claim(api, first)
    with pytest.raises(discovery.GateError, match="changed"):
        claim(api, second, run_url=RETRY_RUN)
    assert len(api.writes) == len(api.comments) == 1


def test_retry_requires_manual_intent_distinct_run_and_unchanged_snapshot(api):
    claim(api)
    plan = discover(api, tag="v0.28.0", retry=True)
    api.calls.clear()
    for changes in ({"retry": False}, {"manual_tag": "v0.29.0"}):
        with pytest.raises(discovery.GateError, match="manual"):
            claim(api, {**plan, **changes}, run_url=RETRY_RUN)
    with pytest.raises(discovery.GateError, match="new run URL"):
        claim(api, plan)
    api.comments[0]["updated_at"] += "1"
    with pytest.raises(discovery.GateError, match="changed"):
        claim(api, plan, run_url=RETRY_RUN)
    assert not api.writes


@pytest.mark.parametrize("operation", ["claim", "finish"])
@pytest.mark.parametrize("change", ["id", "tag", "sha", "draft", "prerelease"])
def test_writes_revalidate_upstream_identity(api, operation, change):
    plan = discover(api) if operation == "claim" else claim(api)
    api.calls.clear()
    if change == "sha":
        api.responses[f"repos/{discovery.UPSTREAM}/git/ref/tags/v0.28.0"] = {
            "ref": "refs/tags/v0.28.0",
            "object": {"type": "commit", "sha": "a" * 40},
        }
    else:
        replacement = release()
        replacement[{"tag": "tag_name"}.get(change, change)] = {
            "id": 29,
            "tag": "v0.29.0",
            "draft": True,
            "prerelease": True,
        }[change]
        replacement["html_url"] = (
            f"https://github.com/{discovery.UPSTREAM}/releases/tag/{replacement['tag_name']}"
        )
        api.responses[RELEASES_PATH + "/28"] = replacement
    with pytest.raises(discovery.GateError):
        if operation == "claim":
            claim(api, plan)
        else:
            finish(api, plan)
    assert not api.writes


@pytest.mark.parametrize("operation", ["claim", "finish"])
def test_rechecks_for_racing_comment_before_write(api, operation):
    plan = discover(api) if operation == "claim" else claim(api)
    api.calls.clear()
    reads = 0

    def race(endpoint, method):
        nonlocal reads
        if endpoint == COMMENTS_PATH:
            reads += 1
            if reads == 2:
                api.comments.append(comment(comment_id=999))

    api.hook = race
    with pytest.raises(discovery.GateError):
        if operation == "claim":
            claim(api, plan)
        else:
            finish(api, plan)
    assert reads == 2 and not api.writes


@pytest.mark.parametrize("operation", ["claim", "finish"])
def test_list_failure_during_write_preflight_never_mutates(api, operation):
    plan = discover(api) if operation == "claim" else claim(api)
    api.calls.clear()
    api.responses[COMMENTS_PATH] = discovery.APIError(500)
    with pytest.raises(discovery.APIError):
        if operation == "claim":
            claim(api, plan)
        else:
            finish(api, plan)
    assert not api.writes


def test_post_write_duplicate_is_reported_not_retried(api):
    def race(endpoint, method):
        if endpoint == COMMENTS_PATH and api.writes:
            api.comments.append(comment(comment_id=999))

    api.hook = race
    with pytest.raises(discovery.GateError, match="Duplicate"):
        claim(api)
    assert len(api.writes) == 1


@pytest.mark.parametrize("operation", ["claim", "finish"])
@pytest.mark.parametrize("change", ["status", "author", "tracker", "failure"])
def test_same_comment_is_rechecked_immediately_before_patch(api, operation, change):
    running = claim(api)
    plan = discover(api, tag="v0.28.0", retry=True) if operation == "claim" else running
    api.calls.clear()
    reads = 0

    def race(endpoint, method):
        nonlocal reads
        if endpoint == ISSUE_PATH:
            reads += 1
            if reads != 2:
                return
            if change == "status":
                api.comments[0]["body"] = comment(record(status="failed"))["body"]
            elif change == "author":
                api.comments[0]["user"]["login"] = "foreign"
            elif change == "tracker":
                api.issue["body"] = "Marker removed"
            else:
                raise discovery.APIError(500)

    api.hook = race
    with pytest.raises(discovery.GateError):
        if operation == "claim":
            claim(api, plan, run_url=RETRY_RUN)
        else:
            finish(api, plan)
    assert reads == 2 and not api.writes


@pytest.mark.parametrize("field", ["user", "id", "body", "issue_url"])
def test_write_response_must_match_authenticated_claim(api, field):
    response = comment(comment_id=101)
    response[field] = {
        "user": {"login": "unexpected-author"},
        "id": None,
        "body": comment(record(status="passed"))["body"],
        "issue_url": "https://api.github.com/repos/other/repo/issues/17",
    }[field]
    api.responses[ISSUE_PATH + "/comments"] = response
    with pytest.raises(discovery.GateError):
        claim(api)
    assert len(api.writes) == 1


def test_ambiguous_write_failure_is_not_retried(api):
    api.responses[ISSUE_PATH + "/comments"] = discovery.APIError(502)
    with pytest.raises(discovery.APIError):
        claim(api)
    assert len(api.writes) == 1


@pytest.mark.parametrize("field", ["repository", "tracker_issue", "release_id", "engine_sha"])
def test_tampered_plan_cannot_write(api, field):
    plan = discover(api)
    plan[field] = "invalid"
    with pytest.raises(discovery.GateError):
        claim(api, plan)
    assert not api.writes


@pytest.mark.parametrize("status", ["passed", "failed", "blocked"])
def test_terminal_states_require_matching_running_claim(api, status):
    plan = claim(api)
    api.calls.clear()
    with pytest.raises(discovery.GateError):
        finish(api, plan, status=status, run_url=RETRY_RUN)
    assert not api.writes
    done = finish(api, plan, status=status)
    assert done["status"] == status
    with pytest.raises(discovery.GateError):
        finish(api, done, status="failed" if status != "failed" else "passed")


@pytest.mark.parametrize(
    "url",
    [
        "C:/private/log.txt",
        "https://private.invalid/run/1",
        RUN + "?token=secret",
        RUN + "#private-log",
        RUN.removesuffix("/attempts/1"),
        RUN.replace("owner/repo", "other/repo"),
        RUN.replace("https://", "http://"),
        "https://github.com@private.invalid/owner/repo/actions/runs/1",
    ],
)
def test_private_or_noncanonical_urls_cannot_be_published(api, url):
    plan = discover(api)
    with pytest.raises(discovery.GateError):
        claim(api, plan, run_url=url)
    with pytest.raises(discovery.GateError):
        finish(api, plan, pr_url=url)
    assert not api.writes


def test_only_allowlisted_fields_are_published(api):
    plan = discover(api)
    plan.update(raw_logs="private output", private_path="D:/secret/project")
    running = claim(api, plan)
    finish(api, running)
    for _, _, payload, _ in api.writes:
        assert "private" not in payload["body"] and "secret" not in payload["body"]


def test_gh_transport_paginates_all_pages_and_uses_json_stdin(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        output = [[release(29, "v0.29.0")], [release()], []] if "--paginate" in argv else {}
        return SimpleNamespace(returncode=0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(discovery.subprocess, "run", run)
    transport = discovery.GhAPI()
    items = transport.request(RELEASES_PATH + "?per_page=100", paginate=True)
    assert [item["id"] for item in items] == [29, 28]
    transport.request(ISSUE_PATH + "/comments", method="POST", payload={"body": "safe"})
    argv, kwargs = calls[0]
    assert "--slurp" in argv and "--paginate" in argv
    assert argv[argv.index("--hostname") + 1] == "github.com"
    assert kwargs["timeout"] == 60
    argv, kwargs = calls[1]
    assert argv[-2:] == ["--input", "-"] and "--paginate" not in argv
    assert json.loads(kwargs["input"]) == {"body": "safe"}
    assert "shell" not in kwargs


def test_discovery_consumes_later_release_and_ledger_pages(monkeypatch):
    api = FakeAPI()
    responses = {
        ISSUE_PATH: api.issue,
        COMMENTS_PATH: [[comment(body="Discussion")], [comment(record(status="failed"))]],
        RELEASES_PATH
        + "?per_page=100": [[release(30, "v0.30.0")], [release(29, "v0.29.0"), release()]],
    }
    for number, tag in [(28, "v0.28.0"), (29, "v0.29.0")]:
        responses[f"repos/{discovery.UPSTREAM}/git/ref/tags/{tag}"] = {
            "ref": f"refs/tags/{tag}",
            "object": {"type": "commit", "sha": f"{number:040x}"},
        }

    def run(argv, **kwargs):
        assert argv[argv.index("--method") + 1] == "GET"
        return SimpleNamespace(returncode=0, stdout=json.dumps(responses[argv[6]]), stderr="")

    monkeypatch.setattr(discovery.subprocess, "run", run)
    assert discovery.discover(REPOSITORY, ISSUE)["tag"] == "v0.29.0"


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(returncode=1, stdout="[[{}]]", stderr="secret (HTTP 500)"),
        SimpleNamespace(returncode=0, stdout="invalid JSON", stderr=""),
        SimpleNamespace(returncode=0, stdout='[{"not": "a page"}]', stderr=""),
        OSError("private executable path"),
        subprocess.TimeoutExpired("private command", 60),
    ],
)
def test_gh_transport_discards_partial_lists_and_redacts_errors(monkeypatch, result):
    def run(*args, **kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(discovery.subprocess, "run", run)
    with pytest.raises(discovery.GateError) as error:
        discovery.GhAPI().request(RELEASES_PATH, paginate=True)
    assert "private" not in str(error.value) and "secret" not in str(error.value)


def test_cli_roundtrip_persists_claim_and_finish(api, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(discovery, "GhAPI", lambda: api)
    path = tmp_path / "plan.json"
    common = ["--repository", REPOSITORY, "--tracker-issue", str(ISSUE)]
    assert discovery.main(["discover", *common, "--output", str(path)]) == 0
    assert json.loads(path.read_text())["status"] == "pending"
    assert discovery.main(["claim", *common, "--plan", str(path), "--run-url", RUN]) == 0
    assert json.loads(path.read_text())["ledger_comment_id"] == 101
    assert (
        discovery.main(
            ["finish", *common, "--plan", str(path), "--run-url", RUN, "--status", "failed"]
        )
        == 0
    )
    assert json.loads(path.read_text())["status"] == "failed"
    assert not list(tmp_path.glob("*.tmp"))
    assert not capsys.readouterr().err


def test_cli_idle_claim_exits_cleanly(api, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(discovery, "GhAPI", lambda: api)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"status": "idle"}), encoding="utf-8")
    assert (
        discovery.main(
            [
                "claim",
                "--repository",
                REPOSITORY,
                "--tracker-issue",
                str(ISSUE),
                "--plan",
                str(path),
                "--run-url",
                RUN,
            ]
        )
        == 0
    )
    assert json.loads(path.read_text()) == {"status": "idle"}
    assert not api.calls and not capsys.readouterr().err


def test_cli_error_does_not_replace_plan_or_leak_paths(api, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(discovery, "GhAPI", lambda: api)
    path = tmp_path / "private-plan.json"
    path.write_text("invalid-json", encoding="utf-8")
    args = [
        "claim",
        "--repository",
        REPOSITORY,
        "--tracker-issue",
        str(ISSUE),
        "--plan",
        str(path),
        "--run-url",
        RUN,
    ]
    assert discovery.main(args) == 1
    assert path.read_text() == "invalid-json" and not api.writes
    assert str(path) not in capsys.readouterr().err


def test_cli_failed_discovery_does_not_write_idle(api, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(discovery, "GhAPI", lambda: api)
    api.responses[COMMENTS_PATH] = discovery.APIError(403)
    path = tmp_path / "plan.json"
    assert (
        discovery.main(
            [
                "discover",
                "--repository",
                REPOSITORY,
                "--tracker-issue",
                str(ISSUE),
                "--output",
                str(path),
            ]
        )
        == 1
    )
    assert not path.exists() and not api.writes
    assert "HTTP 403" in capsys.readouterr().err


def test_plan_replace_failure_preserves_previous_file(monkeypatch, tmp_path):
    path = tmp_path / "plan.json"
    path.write_text("previous-plan", encoding="utf-8")

    def fail_replace(*args):
        raise OSError("write failure")

    monkeypatch.setattr(discovery.os, "replace", fail_replace)
    with pytest.raises(OSError):
        discovery._save(path, {"status": "idle"})
    assert path.read_text() == "previous-plan" and not list(tmp_path.glob("*.tmp"))
