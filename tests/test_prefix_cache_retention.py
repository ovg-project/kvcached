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

import kvcached.prefix_cache_retention as retention
from kvcached import control


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
                                            retention_tokens=-1, retention_revision=0)

    def snapshot(self, name, *, observe):
        name = name or "model"
        return {"instance": name, "complete_membership": True,
                "gpus": [dict(self.states[(name, gpu)]) for gpu in ("gpu-0", "gpu-1")]}

    def publish_prefix_cache_retention(self, gpu, name, tokens, generation, revision):
        if gpu == self.fail_gpu:
            raise RuntimeError("publication failed")
        state = dict(self.states[(name, gpu)])
        if generation != state["generation"]:
            return dict(state, status="instance_changed")
        if state["retention_revision"] and revision <= state["retention_revision"]:
            status = ("stale" if revision < state["retention_revision"] else
                      "idempotent" if tokens == state["retention_tokens"] else "conflict")
            return dict(state, status=status)
        state.update(retention_tokens=tokens, retention_revision=revision)
        self.states[(name, gpu)] = state
        return dict(state, status="accepted")


@pytest.fixture
def shared_control(monkeypatch):
    import kvcached

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
    shared_control.publish_prefix_cache_retention("gpu-0", "model", -1, 42, 2)
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


def _tp_retention_worker(rank, directory, states, results):
    import torch.distributed as dist

    import kvcached
    from kvcached.integration.sglang import patches

    native = NativeRetentionFake(states)
    kvcached._C = native
    control._get_instance_memory_limit = native.snapshot
    retention._policy = retention._PrefixCacheRetentionPolicy(16000)
    retention._policy.instance_name = "model"
    cache = Enforcer()
    retention._policy.register(cache)
    dist.init_process_group("gloo", init_method=f"file://{directory}/gloo",
                            rank=rank, world_size=2)
    try:
        patches._sglang_queue_cpu_groups = lambda: [dist.group.WORLD]
        patches._sglang_pipeline_stages = lambda: 1
        # Only the leader may read the ledger. Rank 1 must use the broadcast.
        if rank == 1:
            retention._read_prefix_cache_retention_update = lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("non-leader polled"))
        patches._apply_sglang_prefix_retention()
        first = control.get_prefix_cache_retention_limit()
        dist.barrier()
        if rank == 1:
            assert control.set_prefix_cache_retention_limit(-1)["accepted"]
            assert cache.limits == [0]  # Publication does not evict locally.
        dist.barrier()
        retention._policy.last_poll = 0
        patches._apply_sglang_prefix_retention()
        results.put((rank, first, control.get_prefix_cache_retention_limit(), cache.limits))
    finally:
        dist.destroy_process_group()


def test_external_caller_and_nonleader_update_all_tp_ranks(shared_control, tmp_path):
    pytest.importorskip("torch.distributed")
    assert control.set_prefix_cache_retention_limit(0, instance_name="model")["accepted"]
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    shared_states = context.Manager()
    states = shared_states.dict(shared_control.states)
    workers = [context.Process(target=_tp_retention_worker,
                               args=(rank, str(tmp_path), states, results)) for rank in range(2)]
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
    from kvcached.integration.sglang import patches

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

    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(patches, "_register_shrink_reclaimers", no_reclaimers)
    monkeypatch.setattr(patches, "_sglang_queue_cpu_groups", list)
    monkeypatch.setattr(patches, "_sglang_pipeline_stages", lambda: 1)
    monkeypatch.setattr("kvcached.runtime_control.initialize_runtime_controls", lambda *a, **k: ())
    patch = patches.SchedulerIdleLimitPatch()
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


@pytest.mark.parametrize("step_name", ["step", "step_with_batch_queue"])
def test_vllm_consumes_update_before_scheduling(monkeypatch, shared_control, step_name):
    from kvcached.integration.vllm import patches

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
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patches.EngineCorePatch().patch_engine_init(module)
    control.set_prefix_cache_retention_limit(0)

    engine = EngineCore.__new__(EngineCore)
    assert getattr(engine, step_name)() == "scheduled"


def test_vllm_idle_watch_leaves_eviction_on_engine_thread(monkeypatch, shared_control):
    from kvcached.integration.vllm import patches

    class EngineCore:
        pass

    module = types.ModuleType("vllm.v1.engine.core")
    setattr(module, "EngineCore", EngineCore)
    assert patches.EngineCorePatch().patch_engine_init(module)
    wakeup = object()
    engine_module = types.ModuleType("vllm.v1.engine")
    setattr(engine_module, "EngineCoreRequestType", types.SimpleNamespace(WAKEUP=wakeup))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine_module)
    monkeypatch.setattr(patches, "_release_retired_pages_if_idle", lambda *a: None)
    monkeypatch.setattr(patches, "_publish_kv_cache_events_if_idle", lambda *a: None)
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
