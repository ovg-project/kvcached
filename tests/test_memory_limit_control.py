# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import threading
import types

import pytest

if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
    sys.modules.setdefault("torch", types.ModuleType("torch"))


def _load_manager_class():
    fake_module = types.ModuleType("kvcached.vmm_ops")
    setattr(fake_module, "PageAllocator", object)
    setattr(
        fake_module,
        "InternalPage",
        types.SimpleNamespace(
            get_num_blocks=lambda page_size, block_mem_size: page_size // block_mem_size
        ),
    )
    setattr(fake_module, "kv_tensors_created", lambda *args, **kwargs: True)
    setattr(fake_module, "map_to_kv_tensors", lambda *args, **kwargs: True)
    setattr(fake_module, "unmap_from_kv_tensors", lambda *args, **kwargs: True)
    previous = sys.modules.get("kvcached.vmm_ops")
    sys.modules["kvcached.vmm_ops"] = fake_module
    sys.modules.pop("kvcached.kv_cache_manager", None)
    try:
        from kvcached.kv_cache_manager import KVCacheManager

        return KVCacheManager
    finally:
        if previous is None:
            sys.modules.pop("kvcached.vmm_ops", None)
        else:
            sys.modules["kvcached.vmm_ops"] = previous


KVCacheManager = _load_manager_class()


class FakeAllocator:
    def __init__(
        self,
        *,
        page_size: int,
        total_pages: int = 16,
        inuse_pages: int = 0,
        reserved_pages: int = 0,
    ):
        self.page_size = page_size
        self.total_pages = total_pages
        self.inuse_pages = inuse_pages
        self.reserved_pages = reserved_pages
        self.resize_calls: list[int] = []
        # What the native resize watcher reports: the last kvctl quota it
        # observed, kept until a new one lands, or -1 before the first.
        self.resize_target = -1

    def get_resize_target(self):
        return self.resize_target

    def get_num_free_pages(self):
        return max(0, self.total_pages - self.inuse_pages)

    def get_num_reserved_pages(self):
        return self.reserved_pages

    def get_avail_physical_pages(self):
        return self.get_num_free_pages()

    def resize(self, new_mem_size):
        target_pages = new_mem_size // self.page_size
        self.resize_calls.append(target_pages)
        if target_pages < self.inuse_pages:
            return False
        self.total_pages = target_pages
        self.reserved_pages = min(
            self.reserved_pages,
            max(0, target_pages - self.inuse_pages),
        )
        return True

    def get_page_state(self):
        free_pages = max(0, self.total_pages - self.inuse_pages)
        return {
            "total_pages": self.total_pages,
            "free_pages": free_pages,
            "inuse_pages": self.inuse_pages,
            "reserved_pages": self.reserved_pages,
        }


def _manager(*, inuse_pages: int = 0, reserved_pages: int = 0):
    manager = object.__new__(KVCacheManager)
    manager.page_size = 2 * 1024 * 1024
    manager.num_layers = 2
    manager.num_kv_buffers = 2
    manager.mem_size = manager.page_size * 16
    manager.num_blocks = 16
    manager.group_id = 0
    manager._pool_name = "unified"
    manager._memory_limit_bytes = None
    manager._memory_limit_effective_bytes = None
    manager._memory_limit_revision = -1
    manager.page_allocator = FakeAllocator(
        page_size=manager.page_size,
        total_pages=16,
        inuse_pages=inuse_pages,
        reserved_pages=reserved_pages,
    )
    manager._lock = threading.RLock()
    manager._wait_post_init = lambda: None
    manager.num_avail_blocks = 0
    manager.reserved_blocks = []
    manager.in_shrink = False
    manager.target_num_blocks = None
    manager.block_mem_size = manager.page_size
    manager.avail_pages = {}
    manager.full_pages = {}
    manager._get_num_alloced_blocks = lambda: manager.page_allocator.inuse_pages
    manager._avail_physical_pages_cache = None
    manager._avail_physical_pages_ts = 0.0
    return manager


def _kvctl_limit(manager, pages):
    manager.page_allocator.resize_target = pages * manager.page_size


def _allocate(manager):
    # alloc() applies the published quota before it touches a page. Zero
    # blocks keeps the fake, which has no pages to hand out, on that path.
    assert manager.alloc(0) == []


class NativeControlFake:
    """Constructed frontend/engine tests; native concurrency is tested in C++."""

    def __init__(self, manager):
        bundle = manager._memory_limit_page_bundle_bytes()
        self.states = {
            gpu: dict(status="unpublished", convergence="unpublished", generation=42,
                      revision=0, published=False, pool_slot=0, pool_owner=123,
                      pool_target_bytes=16 * bundle, pool_capacity_bytes=16 * bundle,
                      page_bundle_bytes=bundle, can_shrink=True, ready=True,
                      expected_participants=2, registered_participants=1)
            for gpu in ("gpu-a", "gpu-b")
        }
        self.acks = []
        self.fail_gpu = None

    def query_instance_control(self, path, instance, slot=None):
        return dict(self.states[path])

    def publish_instance_control(self, path, instance, limit, generation, revision):
        if path == self.fail_gpu:
            raise RuntimeError("injected publication failure")
        state = self.states[path]
        if generation != state["generation"]:
            return dict(state, status="instance_changed")
        if revision is not None and state["published"]:
            if revision < state["revision"]:
                return dict(state, status="stale")
            if revision == state["revision"]:
                return dict(state, status="idempotent" if limit == state["limit_bytes"] else "conflict")
        revision = state["revision"] + 1 if revision is None else revision
        bundle = state["page_bundle_bytes"]
        state.update(status="pending", convergence="pending", revision=revision,
                     published=True, limit_bytes=limit,
                     pool_target_bytes=min(limit // bundle * bundle, state["pool_capacity_bytes"]))
        return dict(state, status="accepted")

    def fail_instance_control(self, path, instance, slot, owner, generation, revision):
        state = self.states[path]
        if (slot, owner, generation, revision) != tuple(
                state[key] for key in ("pool_slot", "pool_owner", "generation", "revision")):
            return False
        state.update(status="failed", convergence="failed", failed=1,
                     failure_reason="quarantined_pages_prevent_resize")
        return True

    def local_instance_control(self, group_id):
        return dict(self.states["gpu-a"])

    def acknowledge_instance_control(self, group_id, revision):
        self.acks.append(revision)
        return revision == self.states["gpu-a"]["revision"]


@pytest.fixture
def controls(monkeypatch):
    import kvcached
    from kvcached import control
    from kvcached.runtime_control import RuntimeControl

    manager = _manager()
    native = NativeControlFake(manager)
    monkeypatch.setattr(kvcached, "_C", native, raising=False)
    monkeypatch.setattr(control.glob, "glob", lambda pattern: list(native.states))
    runtime = object.__new__(RuntimeControl)
    runtime.manager = manager
    runtime.worker_ipc = False
    runtime.sources = (dict(native.states["gpu-a"], gpu="gpu-a", instance="model"),)
    runtime.current = ()
    runtime.executing = ()
    runtime.last_poll = 0.0
    manager._runtime_control = runtime
    return manager, native, control


def test_cli_api_same_value_is_a_new_native_publication(controls, capsys):
    from kvcached.cli.utils import update_kv_cache_limit

    manager, native, control = controls
    size = manager._memory_limit_page_bundle_bytes() * 4
    first = update_kv_cache_limit("model", size)
    second = control.set_instance_memory_limit(size, instance_name="model")
    third = update_kv_cache_limit("model", size)
    assert [r["revision"] for r in first["gpus"]] == [1, 1]
    assert [r["revision"] for r in second["gpus"]] == [2, 2]
    assert [r["revision"] for r in third["gpus"]] == [3, 3]
    assert all(r["limit_bytes"] == size for r in third["gpus"])
    assert '"converged": false' in capsys.readouterr().out


def test_explicit_retry_conflict_stale_and_restart(controls):
    _, native, control = controls
    generations = {gpu: 42 for gpu in native.states}
    assert control.set_instance_memory_limit(123, revision=7, generations=generations)["accepted"]
    same = control.set_instance_memory_limit(123, revision=7, generations=generations)
    assert {r["status"] for r in same["gpus"]} == {"idempotent"}
    assert {r["status"] for r in control.set_instance_memory_limit(
        124, revision=7, generations=generations)["gpus"]} == {"conflict"}
    assert {r["status"] for r in control.set_instance_memory_limit(
        123, revision=6, generations=generations)["gpus"]} == {"stale"}
    native.states["gpu-b"]["generation"] = 43
    retry = control.set_instance_memory_limit(123, revision=7, generations=generations)
    assert retry["status"] == "partial"
    assert retry["gpus"][1]["status"] == "instance_changed"


def test_partial_gpu_failure_can_be_repaired_by_caller(controls):
    _, native, control = controls
    native.fail_gpu = "gpu-b"
    result = control.set_instance_memory_limit(123)
    assert result["status"] == "partial"
    assert result["gpus"][0]["revision"] == 1
    assert result["gpus"][1]["status"] == "error"
    native.fail_gpu = None
    repair = control.set_instance_memory_limit(123, gpu_ids=["gpu-b"])
    assert repair["accepted"]
    assert native.states["gpu-a"]["revision"] == native.states["gpu-b"]["revision"] == 1


def test_missing_rank_does_not_report_complete_or_publish(controls):
    _, native, control = controls
    native.states["gpu-b"]["registered_participants"] = 0
    result = control.set_instance_memory_limit(123)
    assert result["status"] == "unavailable" and not result["accepted"]
    assert not result["converged"]
    assert native.states["gpu-a"]["revision"] == 0


def test_raw_legacy_watcher_is_never_consumed(controls):
    manager, _, _ = controls
    _kvctl_limit(manager, 1)
    _allocate(manager)
    assert manager.page_allocator.resize_calls == []


def test_revision_replaces_old_shrink_and_old_ack_is_rejected(controls):
    manager, native, control = controls
    bundle = manager._memory_limit_page_bundle_bytes()
    manager.page_allocator.inuse_pages = 6
    control.set_instance_memory_limit(bundle * 4)
    manager.apply_resize_target()
    assert manager.in_shrink
    old = manager._runtime_control.executing
    control.set_instance_memory_limit(bundle * 12)
    manager._runtime_control.poll(force=True)
    manager.apply_resize_target()
    assert not manager.in_shrink
    assert manager.page_allocator.total_pages == 12
    manager._runtime_control.executing = old
    manager._runtime_control.acknowledge()
    assert native.acks[-1] == 1
    assert not native.acknowledge_instance_control(0, 1)
    assert native.states["gpu-a"]["revision"] == 2


def test_deferred_shrink_finishes_without_republishing(controls):
    manager, native, control = controls
    manager.page_allocator.inuse_pages = 6
    control.set_instance_memory_limit(manager._memory_limit_page_bundle_bytes() * 4)
    manager.apply_resize_target()
    assert manager.in_shrink and native.acks == []
    manager.page_allocator.inuse_pages = 2
    manager._maybe_finish_shrink()
    assert not manager.in_shrink and native.acks == [1]
    assert native.states["gpu-a"]["revision"] == 1


def test_unsupported_pool_does_not_execute_shrink(controls):
    manager, native, control = controls
    native.states["gpu-a"]["can_shrink"] = False
    control.set_instance_memory_limit(0)
    manager.apply_resize_target()
    assert manager.page_allocator.resize_calls == [] and native.acks == []


def test_shared_logical_pool_uses_minimum_rank_page_target(controls):
    manager, native, control = controls
    runtime = manager._runtime_control
    runtime.sources += (dict(native.states["gpu-b"], gpu="gpu-b", instance="model"),)
    # Rank B has twice as many bytes per page bundle (e.g. asymmetric PP).
    native.states["gpu-b"]["page_bundle_bytes"] *= 2
    control.set_instance_memory_limit(manager._memory_limit_page_bundle_bytes() * 8)
    runtime.acknowledge = lambda: None
    manager.apply_resize_target()
    assert manager.page_allocator.total_pages == 4


def test_explicit_revision_requires_instance_generation(controls):
    _, _, control = controls
    with pytest.raises(ValueError, match="generations"):
        control.set_instance_memory_limit(123, revision=1)


def test_completed_revision_acknowledged_once(controls):
    manager, native, control = controls
    control.set_instance_memory_limit(manager._memory_limit_page_bundle_bytes() * 8)
    for _ in range(10):
        manager.apply_resize_target()
    assert native.acks == [1]


@pytest.mark.parametrize("generation", [-1, 1 << 64, "42"])
def test_invalid_generation_validated_before_any_gpu_write(controls, generation):
    _, native, control = controls
    with pytest.raises(ValueError, match="generation"):
        control.set_instance_memory_limit(123, generations={"gpu-a": 42, "gpu-b": generation})
    assert all(state["revision"] == 0 for state in native.states.values())


@pytest.mark.parametrize("buffers", [1, 2])
def test_runtime_uses_manager_startup_pages(controls, monkeypatch, buffers):
    from kvcached.runtime_control import RuntimeControl
    manager, native, _ = controls
    manager.num_kv_buffers = buffers
    calls = []
    def configure(group, shrink, participants, pages):
        calls.append((group, shrink, participants, pages))
        return dict(native.states["gpu-a"], gpu="gpu-a", instance="model")
    monkeypatch.setattr(native, "local_instance_control", configure)
    RuntimeControl(manager, worker_ipc=False, can_shrink=True, expected_participants=2)
    assert calls == [(0, True, 2, 16)]


@pytest.mark.parametrize("segment,exact,expected", [
    ("model_g1", False, "model"),
    ("/dev/shm/model_g12", False, "model"),
    ("model_g1", True, "model_g1"),
    ("model_g0", False, "model_g0"),
    ("model_group", False, "model_group"),
])
def test_group_segment_control_resolution(monkeypatch, segment, exact, expected):
    from kvcached import control
    from kvcached.cli.utils import get_segment_control, update_kv_cache_limit

    def query(name):
        return {"instance": name, "gpus": [{}] if name == "model" or exact else []}

    monkeypatch.setattr(control, "get_instance_memory_limit", query)
    monkeypatch.setattr(control, "set_instance_memory_limit",
                        lambda size, *, instance_name: {"instance": instance_name})
    assert get_segment_control(segment)["instance"] == expected
    assert update_kv_cache_limit(segment, 100)["instance"] == expected


def test_manager_controls_use_registered_instance(controls, monkeypatch):
    manager, native, control = controls
    monkeypatch.setattr(control, "get_instance_memory_limit", lambda name: name)
    monkeypatch.setattr(control, "set_instance_memory_limit",
                        lambda size, **kwargs: kwargs["instance_name"])
    assert manager.memory_limit_state() == "model"
    assert manager.set_memory_limit(100) == "model"


def test_listing_groups_uses_one_physical_instance_row(monkeypatch, capsys):
    import json

    from kvcached.cli import kvctl
    from kvcached.cli.utils import MemInfoStruct

    monkeypatch.setattr(kvctl.control_segment, "active", lambda path: True)
    monkeypatch.setattr(kvctl, "get_kv_cache_limit", lambda name: MemInfoStruct(999, 10, 5))
    monkeypatch.setattr(kvctl, "get_segment_control", lambda name: {
        "instance": "model", "gpus": [{"effective_limit_bytes": 200, "charged_bytes": 120}]})
    kvctl.cmd_list(["model", "model_g1"], json_out=True)
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["ipc"] == "model"
    assert rows[0]["used_bytes"] == 120
    assert rows[0]["limit_bytes"] == 200


@pytest.mark.parametrize("deferred", [False, True])
def test_quarantine_failure_is_queryable_without_retry(controls, monkeypatch, deferred):
    from kvcached.errors import QuarantinedResizeError

    manager, native, control = controls
    bundle = manager._memory_limit_page_bundle_bytes()
    control.set_instance_memory_limit(4 * bundle)
    if deferred:
        manager.page_allocator.inuse_pages = 6
        manager.apply_resize_target()
        assert manager.in_shrink
        manager.page_allocator.inuse_pages = 2
    attempts = []

    def fail(size):
        attempts.append(size)
        raise QuarantinedResizeError("injected quarantine")

    monkeypatch.setattr(manager.page_allocator, "resize", fail)
    monkeypatch.setattr(manager, "_record_operation_error", lambda *args: None)
    if deferred:
        manager._maybe_finish_shrink()
    else:
        manager.apply_resize_target()
    for _ in range(3):
        manager.apply_resize_target()
    state = control.get_instance_memory_limit()
    assert state["status"] == "failed" and not state["converged"]
    assert state["gpus"][0]["failure_reason"] == "quarantined_pages_prevent_resize"
    assert state["gpus"][0]["limit_bytes"] == 4 * bundle
    assert len(attempts) == 1 and not manager.in_shrink and native.acks == []


def test_deferred_failure_does_not_poison_new_publication(controls, monkeypatch):
    from kvcached.errors import QuarantinedResizeError

    manager, native, control = controls
    bundle = manager._memory_limit_page_bundle_bytes()
    manager.page_allocator.inuse_pages = 6
    control.set_instance_memory_limit(4 * bundle)
    manager.apply_resize_target()
    old = manager._runtime_control.executing
    control.set_instance_memory_limit(12 * bundle)
    manager.page_allocator.inuse_pages = 2

    def fail(size):
        raise QuarantinedResizeError("injected quarantine")

    resize = manager.page_allocator.resize
    monkeypatch.setattr(manager.page_allocator, "resize", fail)
    monkeypatch.setattr(manager, "_record_operation_error", lambda *args: None)
    manager._maybe_finish_shrink()
    assert native.states["gpu-a"]["convergence"] == "pending"
    assert native.states["gpu-a"]["revision"] == 2
    monkeypatch.setattr(manager.page_allocator, "resize", resize)
    manager._runtime_control.poll(force=True)
    manager.apply_resize_target()
    assert native.acks == [2] and not manager._resize_rejected
    manager._runtime_control.executing = old
    manager._runtime_control.fail()
    assert native.states["gpu-a"]["convergence"] == "pending"


def test_worker_control_failure_reports_each_bound_participant(controls):
    manager, native, control = controls
    runtime = manager._runtime_control
    runtime.worker_ipc = True
    runtime.sources += (dict(native.states["gpu-b"], gpu="gpu-b", instance="model"),)
    control.set_instance_memory_limit(0)
    runtime.executing = runtime.poll(force=True)
    # A replacement participant must not receive an old worker's failure.
    native.states["gpu-b"]["pool_owner"] += 1
    runtime.fail()
    assert native.states["gpu-a"]["convergence"] == "failed"
    assert native.states["gpu-b"]["convergence"] == "pending"
