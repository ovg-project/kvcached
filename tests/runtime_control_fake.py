# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Constructed native boundary for engine-thread reclamation tests."""
from types import SimpleNamespace


def attach_control(manager, monkeypatch):
    import kvcached
    from kvcached import control
    from kvcached.runtime_control import RuntimeControl

    bundle = manager._memory_limit_page_bundle_bytes()
    state = dict(status="unpublished", convergence="unpublished", generation=42,
                 revision=0, published=False, ready=True, pool_owner=1, pool_slot=0,
                 gpu="test-gpu", instance="test", can_shrink=True,
                 pool_capacity_bytes=manager.mem_size // manager.page_size * bundle,
                 pool_target_bytes=manager.mem_size // manager.page_size * bundle,
                 page_bundle_bytes=bundle, expected_participants=1,
                 registered_participants=1, effective_limit_bytes=0, charged_bytes=0)
    acked = [-1]
    failed = [-1]

    def query(*args):
        pages = manager.page_allocator.get_page_state()
        state["charged_bytes"] = (pages["inuse_pages"] + pages["reserved_pages"]) * bundle
        if state["published"]:
            state["status"] = state["convergence"] = (
                "failed" if failed[0] == state["revision"] else
                "converged" if acked[0] == state["revision"] and
                state["charged_bytes"] <= min(state["pool_target_bytes"], state["requested_bytes"]) else "pending")
        return dict(state)

    def publish(path, name, limit, generation, revision):
        assert generation == 42
        state.update(revision=state["revision"] + 1 if revision is None else revision,
                     published=True, requested_bytes=limit,
                     pool_target_bytes=max(bundle if getattr(manager, "null_block", None) is not None else 0,
                                           min(limit // bundle * bundle, state["pool_capacity_bytes"])))
        state["effective_limit_bytes"] = state["pool_target_bytes"]
        return dict(query(), status="accepted")

    def ack(group, revision):
        if revision != state["revision"]:
            return False
        acked[0] = revision
        return True

    def fail(path, name, slot, owner, generation, revision):
        if (slot, owner, generation, revision) != tuple(
                state[key] for key in ("pool_slot", "pool_owner", "generation", "revision")):
            return False
        failed[0] = revision
        state["failure_reason"] = "quarantined_pages_prevent_resize"
        return True

    monkeypatch.setattr(kvcached, "_C", SimpleNamespace(
        local_instance_control=query, query_instance_control=query, observe_instance_control=query,
        publish_instance_control=publish, acknowledge_instance_control=ack,
        fail_instance_control=fail), raising=False)
    monkeypatch.setattr(control.glob, "glob", lambda _: ["test-gpu"])
    runtime = RuntimeControl(manager, worker_ipc=False, can_shrink=True, expected_participants=1)
    poll = runtime.poll
    monkeypatch.setattr(runtime, "poll", lambda **kwargs: poll(force=True))
    setattr(runtime, "test_state", state)
    manager._runtime_control = runtime
    return runtime
