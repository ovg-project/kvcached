# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import gc
import multiprocessing
import sys
import threading
import types
from queue import Queue
from typing import Any

import pytest
import torch

import kvcached
import kvcached.prefix_cache_retention as retention
from kvcached import control
from kvcached.integration.sglang import patches as sgl_patches
from kvcached.integration.vllm import patches as vllm_patches


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    policy = retention._PrefixCacheRetentionPolicy(16000)
    monkeypatch.setattr(retention, "_policy", policy)
    return policy


class Enforcer:
    def __init__(self, error=None):
        self.limits = []
        self.error = error

    def _enforce_prefix_cache_retention(self, limit):
        self.limits.append(limit)
        if self.error is not None:
            raise self.error


@pytest.mark.parametrize("limit", [-1, 0, 1, 16000])
def test_setter_updates_and_enforces_registered_caches(limit):
    cache = Enforcer()
    retention._policy.register(cache)

    retention._policy.set_limit(limit)

    assert control.get_prefix_cache_retention_limit() == limit
    assert cache.limits == [limit]


@pytest.mark.parametrize("limit", [-2, 1.5, "1", True, None])
def test_setter_rejects_invalid_limits(limit):
    with pytest.raises(ValueError, match="-1 or a non-negative integer"):
        retention._policy.set_limit(limit)


def test_registration_is_weak():
    cache = Enforcer()
    retention._policy.register(cache)
    assert len(retention._policy.enforcers) == 1

    del cache
    gc.collect()

    assert len(retention._policy.enforcers) == 0


def test_policy_stays_updated_when_enforcement_fails():
    healthy = Enforcer()
    broken = Enforcer(RuntimeError("eviction failed"))
    retention._policy.register(healthy)
    retention._policy.register(broken)

    with pytest.raises(RuntimeError, match="eviction failed"):
        retention._policy.set_limit(100)

    assert control.get_prefix_cache_retention_limit() == 100
    assert healthy.limits == [100]
    assert broken.limits == [100]


class NativeRetentionFake:
    def __init__(self, states=None):
        self.states = {} if states is None else states
        self.fail_gpu = None
        for instance in ("model", "other-model"):
            for gpu, generation in (("gpu-0", 42), ("gpu-1", 43)):
                key = (instance, gpu)
                if key not in self.states:
                    self.states[key] = dict(gpu=gpu, generation=generation,
                                            published=False, convergence="unpublished",
                                            revision=0, retention_tokens=-1, retention_revision=0)

    def snapshot(self, name, *, observe):
        name = name or "model"
        return {"instance": name, "complete_membership": True,
                "gpus": [dict(self.states[(name, gpu)]) for gpu in ("gpu-0", "gpu-1")]}

    def publish_instance_controls(self, gpu, name, generation, revision, *, retention_tokens):
        tokens = retention_tokens
        if gpu == self.fail_gpu:
            raise RuntimeError("publication failed")
        state = dict(self.states[(name, gpu)])
        if generation != state["generation"]:
            return dict(state, status="instance_changed")
        if state["revision"] and revision <= state["revision"]:
            status = ("stale" if revision < state["revision"] else
                      "idempotent" if tokens == state["retention_tokens"] else "conflict")
            return dict(state, status=status)
        state.update(revision=revision, retention_tokens=tokens, retention_revision=revision)
        self.states[(name, gpu)] = state
        return dict(state, status="accepted")


@pytest.fixture
def shared_control(monkeypatch):
    native = NativeRetentionFake()
    monkeypatch.setattr(kvcached, "_C", native, raising=False)
    monkeypatch.setattr(control, "_get_instance_memory_limit", native.snapshot)
    retention._policy.instance_name = "model"
    return native


def test_public_control_publishes_without_evicting_on_the_caller(shared_control):
    cache = Enforcer()
    retention._policy.register(cache)
    result = control.set_prefix_cache_retention_limit(123, instance_name="model")

    assert result["accepted"]
    assert not result["converged"]
    assert set(result["gpus"][0]) == {
        "gpu", "status", "generation", "retention_tokens", "retention_revision"}
    assert control.get_prefix_cache_retention_limit() == 16000
    assert cache.limits == []

    retention._apply_prefix_cache_retention_update(
        retention._read_prefix_cache_retention_update())

    assert control.get_prefix_cache_retention_limit() == 123
    assert cache.limits == [123]
    assert retention._read_prefix_cache_retention_update() is None


def test_replacement_engine_ignores_old_assignment(shared_control):
    control.set_prefix_cache_retention_limit(0)
    for gpu, generation in (("gpu-0", 50), ("gpu-1", 51)):
        shared_control.states[("model", gpu)] = dict(
            gpu=gpu, generation=generation, retention_revision=0, retention_tokens=-1)

    assert retention._read_prefix_cache_retention_update() is None
    assert control.get_prefix_cache_retention_limit() == 16000


def test_instance_controls_are_isolated(shared_control):
    control.set_prefix_cache_retention_limit(0, instance_name="other-model")
    assert retention._read_prefix_cache_retention_update() is None


def test_retention_uses_the_next_shared_control_revision(shared_control):
    shared_control.states[("model", "gpu-0")]["revision"] = 8
    shared_control.states[("model", "gpu-1")]["revision"] = 3

    result = control.set_prefix_cache_retention_limit(256)

    assert result["accepted"] and result["revision"] == 9
    assert all(state["retention_revision"] == 9 for state in result["gpus"])
    retention._apply_prefix_cache_retention_update(retention._read_prefix_cache_retention_update())
    assert control.get_prefix_cache_retention_limit() == 256


def test_partial_publication_waits_for_repair(shared_control):
    shared_control.fail_gpu = "gpu-1"
    result = control.set_prefix_cache_retention_limit(0)
    assert result["status"] == "partial"
    assert not result["accepted"]
    assert retention._read_prefix_cache_retention_update() is None

    shared_control.fail_gpu = None
    result = control.set_prefix_cache_retention_limit(0)
    assert result["accepted"] and result["revision"] == 2
    retention._apply_prefix_cache_retention_update(retention._read_prefix_cache_retention_update())
    assert control.get_prefix_cache_retention_limit() == 0


def test_interleaved_publication_is_not_consumed(shared_control):
    control.set_prefix_cache_retention_limit(0)
    shared_control.publish_instance_controls("gpu-0", "model", 42, 2, retention_tokens=-1)
    assert retention._read_prefix_cache_retention_update() is None
    control.set_prefix_cache_retention_limit(-1)
    retention._apply_prefix_cache_retention_update(retention._read_prefix_cache_retention_update())
    assert control.get_prefix_cache_retention_limit() == -1


def test_incomplete_membership_does_not_publish(shared_control, monkeypatch):
    monkeypatch.setattr(control, "_get_instance_memory_limit", lambda name, observe: {
        **shared_control.snapshot("model", observe=False), "complete_membership": False, "status": "unavailable"})
    assert not control.set_prefix_cache_retention_limit(0)["accepted"]
    assert retention._read_prefix_cache_retention_update() is None


@pytest.mark.parametrize("limit", [-2, 1.5, "1", True, None, 1 << 63])
def test_public_setter_rejects_invalid_limits(limit, shared_control):
    with pytest.raises(ValueError, match="-1 or a non-negative int64 integer"):
        control.set_prefix_cache_retention_limit(limit)


def _tp_retention_worker(rank, directory, states, results, attention_groups):
    native = NativeRetentionFake(states)
    kvcached._C = native
    control._get_instance_memory_limit = native.snapshot
    retention._policy = retention._PrefixCacheRetentionPolicy(16000)
    retention._policy.instance_name = "model"
    cache = Enforcer()
    retention._policy.register(cache)
    torch.distributed.init_process_group("gloo", init_method=f"file://{directory}/gloo",
                            rank=rank, world_size=2)
    try:
        sgl_patches._sglang_queue_cpu_groups = lambda: [torch.distributed.group.WORLD]
        sgl_patches._sglang_pipeline_stages = lambda: 1
        broadcasts = []
        collectives = []
        original_broadcast = torch.distributed.broadcast_object_list

        def counted_broadcast(*args, **kwargs):
            collectives.append(1)
            return original_broadcast(*args, **kwargs)

        torch.distributed.broadcast_object_list = counted_broadcast

        def broadcast_requests(data, *args, **kwargs):
            broadcasts.append(data)
            payload = [data]
            torch.distributed.broadcast_object_list(payload, src=0)
            return payload[0]

        receive = sgl_patches._wrap_sglang_retention_broadcast(
            broadcast_requests, attention_groups=attention_groups)
        sgl_patches._SGLANG_RETENTION_RECEIVE.active = True
        # Only the leader may read the ledger. Rank 1 must use the broadcast.
        if rank == 1:
            sgl_patches._read_prefix_cache_retention_update = lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("non-leader polled"))
        assert receive(["request"] if rank == 0 else None, rank) == ["request"]
        first = control.get_prefix_cache_retention_limit()
        # No update: use only the existing request broadcast and leave its
        # empty source payload unchanged, including after the throttle expires.
        for _ in range(20):
            retention._policy.last_poll = 0
            assert receive([] if rank == 0 else None, rank) == []
        assert len(broadcasts) == 21
        if rank == 0:
            assert broadcasts[1:] == [[]] * 20
        torch.distributed.barrier()
        if rank == 1:
            assert control.set_prefix_cache_retention_limit(-1)["accepted"]
            assert cache.limits == [0]  # Publication does not evict locally.
        torch.distributed.barrier()
        retention._policy.last_poll = 0
        assert receive([] if rank == 0 else None, rank) == []
        assert len(broadcasts) == 22
        assert len(collectives) == 22
        if rank == 0:
            sgl_patches._read_prefix_cache_retention_update = lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("ledger unavailable"))
        with pytest.raises(RuntimeError, match="ledger unavailable"):
            receive([] if rank == 0 else None, rank)
        assert len(collectives) == 23
        results.put((rank, first, control.get_prefix_cache_retention_limit(), cache.limits))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("attention_groups", [False, True])
def test_external_caller_and_nonleader_update_all_tp_ranks(
        shared_control, tmp_path, attention_groups):
    pytest.importorskip("torch.distributed")
    assert control.set_prefix_cache_retention_limit(0, instance_name="model")["accepted"]
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    shared_states = context.Manager()
    states = shared_states.dict(shared_control.states)
    workers = [context.Process(target=_tp_retention_worker,
                               args=(rank, str(tmp_path), states, results, attention_groups)) for rank in range(2)]
    try:
        for worker in workers:
            worker.start()
        observed = sorted(results.get(timeout=30) for _ in workers)
        assert observed == [(0, 0, -1, [0, -1]), (1, 0, -1, [0, -1])]
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        results.close()
        shared_states.shutdown()


@pytest.mark.parametrize("receive_name", ["recv_requests", "ingest_requests"])
def test_scheduler_consumes_update_after_receiving_requests(
        monkeypatch, shared_control, receive_name):
    events = []

    class Scheduler:
        def __init__(self):
            pass

        def on_idle(self):
            events.append("idle")

    def receive(self):
        events.append("receive")
        return ["request"]

    setattr(Scheduler, receive_name, receive)
    module = types.ModuleType("sglang.srt.managers.scheduler")
    setattr(module, "Scheduler", Scheduler)
    def no_reclaimers(scheduler):
        return []

    monkeypatch.setattr(sgl_patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(sgl_patches, "_register_shrink_reclaimers", no_reclaimers)
    monkeypatch.setattr(sgl_patches, "_sglang_queue_cpu_groups", list)
    monkeypatch.setattr(sgl_patches, "_sglang_pipeline_stages", lambda: 1)
    monkeypatch.setattr("kvcached.runtime_control.initialize_runtime_controls", lambda *a, **k: ())
    patch = sgl_patches.SchedulerIdleLimitPatch()
    assert patch.patch_idle_hook(module)
    assert patch.patch_idle_hook(module)
    scheduler = Scheduler()
    cache = Enforcer()
    retention._policy.register(cache)
    control.set_prefix_cache_retention_limit(0)

    assert getattr(scheduler, receive_name)() == ["request"]
    assert events == ["receive"]
    assert cache.limits == [0]
    scheduler.on_idle()
    assert cache.limits == [0]


@pytest.mark.parametrize("modern_receiver", [False, True])
def test_scheduler_reuses_request_broadcast_before_dispatch(
        monkeypatch, shared_control, modern_receiver):
    module = types.ModuleType("retention_test_scheduler")
    receiver_module = types.ModuleType("retention_test_receiver") if modern_receiver else module
    monkeypatch.setitem(sys.modules, receiver_module.__name__, receiver_module)
    payloads = []
    cache = Enforcer()
    retention._policy.register(cache)

    def broadcast(data, rank, **kwargs):
        payloads.append(list(data))
        return data

    setattr(receiver_module, "broadcast_pyobj", broadcast)

    class Scheduler:
        def __init__(self):
            pass

        def on_idle(self):
            pass

    def receive(self):
        requests = receiver_module.broadcast_pyobj(["request"], 0)
        assert requests == ["request"]  # Metadata never reaches dispatch.
        assert cache.limits == [0]
        return requests

    if modern_receiver:
        receiver = type("SchedulerRequestReceiver", (), {"__module__": receiver_module.__name__})
        setattr(module, "SchedulerRequestReceiver", receiver)
    setattr(Scheduler, "ingest_requests" if modern_receiver else "recv_requests", receive)
    setattr(module, "Scheduler", Scheduler)
    monkeypatch.setattr(sgl_patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(sgl_patches, "_sglang_queue_cpu_groups", lambda: [object()])
    monkeypatch.setattr(sgl_patches, "_sglang_pipeline_stages", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *a: 0)
    patch = sgl_patches.SchedulerIdleLimitPatch()
    assert patch.patch_idle_hook(module)
    assert patch.patch_idle_hook(module)
    control.set_prefix_cache_retention_limit(0)
    scheduler = Scheduler.__new__(Scheduler)
    receive_name = "ingest_requests" if modern_receiver else "recv_requests"

    assert getattr(scheduler, receive_name)() == ["request"]
    assert len(payloads) == 1
    assert len(payloads[0]) == 2
    retention._policy.last_poll = 0
    assert getattr(scheduler, receive_name)() == ["request"]
    assert payloads[1] == ["request"]
    assert not sgl_patches._SGLANG_RETENTION_RECEIVE.active


@pytest.mark.parametrize("active,pp_size", [(False, 1), (True, 2)])
def test_retention_leaves_other_broadcasts_unchanged(monkeypatch, active, pp_size):
    monkeypatch.setattr(sgl_patches._SGLANG_RETENTION_RECEIVE, "active", active, raising=False)
    monkeypatch.setattr(sgl_patches, "_sglang_pipeline_stages", lambda: pp_size)

    def forbidden_read(**kwargs):
        raise AssertionError("unexpected retention poll")

    monkeypatch.setattr(sgl_patches, "_read_prefix_cache_retention_update", forbidden_read)
    data = ["request"]
    calls = []

    def broadcast(requests, rank, **kwargs):
        calls.append((requests, rank, kwargs))
        return requests

    wrapped = sgl_patches._wrap_sglang_retention_broadcast(broadcast)
    assert wrapped(data, 3, src=3) is data
    assert calls == [(data, 3, {"src": 3})]


@pytest.mark.parametrize("step_name", ["step", "step_with_batch_queue"])
def test_vllm_consumes_update_before_scheduling(monkeypatch, shared_control, step_name):
    cache = Enforcer()
    retention._policy.register(cache)

    class EngineCore:
        pass

    def step(self):
        assert cache.limits == [0]
        return "scheduled"

    setattr(EngineCore, step_name, step)
    module = types.ModuleType("vllm.v1.engine.core")
    setattr(module, "EngineCore", EngineCore)
    monkeypatch.setattr(vllm_patches, "enable_kvcached", lambda: True)
    assert vllm_patches.EngineCorePatch().patch_engine_init(module)
    control.set_prefix_cache_retention_limit(0)

    engine = EngineCore.__new__(EngineCore)
    assert getattr(engine, step_name)() == "scheduled"


def test_vllm_idle_watch_leaves_eviction_on_engine_thread(monkeypatch, shared_control):
    class EngineCore:
        pass

    module = types.ModuleType("vllm.v1.engine.core")
    setattr(module, "EngineCore", EngineCore)
    assert vllm_patches.EngineCorePatch().patch_engine_init(module)
    wakeup = object()
    engine_module = types.ModuleType("vllm.v1.engine")
    setattr(engine_module, "EngineCoreRequestType", types.SimpleNamespace(WAKEUP=wakeup))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine_module)
    monkeypatch.setattr(vllm_patches, "_release_retired_pages_if_idle", lambda *a: None)
    monkeypatch.setattr(vllm_patches, "_publish_kv_cache_events_if_idle", lambda *a: None)
    threads = []
    cache = Enforcer()
    original_enforce = cache._enforce_prefix_cache_retention

    def enforce(limit):
        threads.append(threading.current_thread())
        original_enforce(limit)

    monkeypatch.setattr(cache, "_enforce_prefix_cache_retention", enforce)
    retention._policy.register(cache)
    engine: Any = EngineCore.__new__(EngineCore)
    engine._kvcached_control_managers = (types.SimpleNamespace(
        _runtime_control=types.SimpleNamespace(sources=()),
        apply_resize_target=lambda: None, reclaim_for_shrink=lambda: None),)
    engine._idle_state_callbacks = []
    engine.input_queue = Queue()
    control.set_prefix_cache_retention_limit(0)
    engine._kvcached_start_limit_watch()
    stop, thread = engine._kvcached_limit_watch
    try:
        assert engine.input_queue.get(timeout=5) == (wakeup, None)
        assert cache.limits == []
        engine._idle_state_callbacks.pop()(engine)
        assert cache.limits == [0]
        assert threads == [threading.current_thread()]
    finally:
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
