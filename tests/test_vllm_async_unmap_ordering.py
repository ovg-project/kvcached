# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from collections import deque
from importlib.machinery import ModuleSpec
from queue import Queue
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest


def _load_patches(monkeypatch):
    torch = mock.MagicMock()
    torch.__version__ = "2.6.0"
    torch.__spec__ = ModuleSpec("torch", loader=None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())

    from kvcached.integration.vllm import patches

    return patches


class FakeManager:
    def __init__(self):
        self.defer_physical_release = True
        self.marker = 0
        self.released = []

    def retire(self):
        self.marker += 1

    def capture_physical_release_marker(self):
        return self.marker

    def release_retired_pages_through(self, marker):
        self.released.append(marker)


@pytest.mark.parametrize("legacy", [False, True])
def test_queue_api_waits_for_all_inflight_batches(monkeypatch, legacy):
    manager = FakeManager()
    queue: Any = Queue() if legacy else deque()
    put = queue.put_nowait if legacy else queue.appendleft
    take = queue.get_nowait if legacy else queue.pop
    for _ in range(3):
        put(object())

    def original_step(self):
        take()
        return ({}, True)

    EngineCore = _patch_engine(monkeypatch, original_step)
    engine = _engine(EngineCore, manager, queue)
    manager.retire()
    engine.step_with_batch_queue()
    engine.step_with_batch_queue()
    assert manager.released == []
    engine.step_with_batch_queue()
    assert manager.released == [1]


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("pending", [False, True])
def test_queue_api_reset_flushes_only_when_empty(monkeypatch, legacy, pending):
    manager = FakeManager()
    queue: Any = Queue() if legacy else deque()
    if pending:
        (queue.put_nowait if legacy else queue.appendleft)(object())

    def original_reset(self):
        manager.retire()
        return True

    EngineCore = _patch_engine(monkeypatch, mock.Mock(), original_reset)
    engine = _engine(EngineCore, manager, queue)
    assert engine.reset_prefix_cache() is True
    assert manager.released == ([] if pending else [1])


@pytest.mark.parametrize("legacy", [False, True])
def test_queue_api_submission_does_not_release_early(monkeypatch, legacy):
    manager = FakeManager()
    queue: Any = Queue() if legacy else deque()

    def original_step(self):
        (queue.put_nowait if legacy else queue.appendleft)(object())
        manager.retire()
        return (None, True)

    EngineCore = _patch_engine(monkeypatch, original_step)
    engine = _engine(EngineCore, manager, queue)
    engine.step_with_batch_queue()
    assert manager.released == []


@pytest.mark.parametrize("failed_patch", [None, "init", "lifetime", "shutdown"])
def test_engine_apply_keeps_lifetime_and_shutdown_patches(monkeypatch, failed_patch):
    patches = _load_patches(monkeypatch)
    patch = patches.EngineCorePatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    hooks = {}
    for name, method in (("init", "patch_engine_init"),
                         ("lifetime", "patch_async_batch_lifetime"),
                         ("shutdown", "patch_engine_shutdown")):
        hooks[name] = mock.Mock(return_value=name != failed_patch)
        monkeypatch.setattr(patch, method, hooks[name])
    engine_mod = types.ModuleType("vllm.v1.engine.core")

    assert patch.apply(engine_mod) is (failed_patch is None)
    for hook in hooks.values():
        hook.assert_called_once_with(engine_mod)


def _patch_engine(monkeypatch, original_step, original_reset=None):
    patches = _load_patches(monkeypatch)
    engine_mod = types.ModuleType("vllm.v1.engine.core")

    class EngineCore:
        step_with_batch_queue = original_step

    if original_reset is not None:
        setattr(EngineCore, "reset_prefix_cache", original_reset)
    setattr(engine_mod, "EngineCore", EngineCore)
    assert patches.EngineCorePatch().patch_async_batch_lifetime(engine_mod)
    return EngineCore


def _engine(EngineCore, manager, queue):
    engine = EngineCore()
    engine.batch_queue = queue
    engine.scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            block_pool=SimpleNamespace(kv_cache_manager=manager)
        )
    )
    return engine


@pytest.mark.parametrize("manager_mode", ["immediate", "legacy", "missing"])
def test_immediate_release_bypasses_batch_lifetime_tracking(monkeypatch, manager_mode):
    manager = FakeManager()
    manager.defer_physical_release = False
    if manager_mode == "legacy":
        del manager.defer_physical_release
    capture_marker = mock.Mock(
        side_effect=AssertionError("Immediate release must not read retirement epochs")
    )
    release_pages = mock.Mock(
        side_effect=AssertionError("Immediate release must not flush retired pages")
    )
    monkeypatch.setattr(manager, "capture_physical_release_marker", capture_marker)
    monkeypatch.setattr(manager, "release_retired_pages_through", release_pages)
    calls = []

    def original_step(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        return mock.sentinel.result

    EngineCore = _patch_engine(monkeypatch, original_step)
    engine = _engine(
        EngineCore,
        None if manager_mode == "missing" else manager,
        mock.sentinel.unused_batch_queue,
    )

    result = engine.step_with_batch_queue(mock.sentinel.argument, option=True)

    assert result is mock.sentinel.result
    assert calls == [(engine, (mock.sentinel.argument,), {"option": True})]
    capture_marker.assert_not_called()
    release_pages.assert_not_called()
    assert not hasattr(engine, "_kvcached_release_fences")
    assert not hasattr(engine, "_kvcached_last_fenced_release_marker")


def test_completed_batch_releases_only_pages_retired_before_call(monkeypatch):
    manager = FakeManager()
    manager.retire()

    def original_step(self):
        manager.retire()
        return ({}, True)

    EngineCore = _patch_engine(monkeypatch, original_step)
    engine = _engine(EngineCore, manager, deque([object()]))

    engine.step_with_batch_queue()

    assert manager.released == [1]


def test_final_completed_batch_releases_all_retired_pages(monkeypatch):
    manager = FakeManager()
    manager.retire()

    def original_step(self):
        manager.retire()
        self.batch_queue.clear()
        return ({}, True)

    EngineCore = _patch_engine(monkeypatch, original_step)
    engine = _engine(EngineCore, manager, deque([object()]))

    engine.step_with_batch_queue()

    assert manager.released == [2]


def test_queue_submission_without_completion_does_not_release_pages(monkeypatch):
    manager = FakeManager()

    def original_step(self):
        manager.retire()
        self.batch_queue.appendleft(object())
        return (None, True)

    EngineCore = _patch_engine(monkeypatch, original_step)
    engine = _engine(EngineCore, manager, deque())

    engine.step_with_batch_queue()

    assert manager.released == []


def test_retired_pages_wait_for_every_older_inflight_batch(monkeypatch):
    manager = FakeManager()

    def original_step(self):
        self.batch_queue.pop()
        return ({}, True)

    EngineCore = _patch_engine(monkeypatch, original_step)
    engine = _engine(EngineCore, manager, deque([object(), object(), object()]))
    manager.retire()

    engine.step_with_batch_queue()
    engine.step_with_batch_queue()
    assert manager.released == []

    engine.step_with_batch_queue()
    assert manager.released == [1]


def test_idle_prefix_reset_releases_pages_without_another_batch(monkeypatch):
    manager = FakeManager()
    calls = []

    def original_reset(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        manager.retire()
        return mock.sentinel.reset_result

    EngineCore = _patch_engine(monkeypatch, mock.Mock(), original_reset)
    engine = _engine(EngineCore, manager, deque())
    engine._kvcached_release_fences = [[0, 0]]

    result = engine.reset_prefix_cache(mock.sentinel.argument, reset_connector=True)

    assert result is mock.sentinel.reset_result
    assert calls == [(engine, (mock.sentinel.argument,), {"reset_connector": True})]
    assert manager.released == [1]
    assert engine._kvcached_release_fences == []


def test_prefix_reset_with_inflight_batches_keeps_retirement_fenced(monkeypatch):
    manager = FakeManager()

    def original_reset(self):
        manager.retire()
        return True

    def original_step(self):
        self.batch_queue.pop()
        return ({}, True)

    EngineCore = _patch_engine(monkeypatch, original_step, original_reset)
    engine = _engine(EngineCore, manager, deque([object(), object()]))

    assert engine.reset_prefix_cache() is True
    assert manager.released == []
    engine.step_with_batch_queue()
    assert manager.released == []
    engine.step_with_batch_queue()
    assert manager.released == [1]


@pytest.mark.parametrize("manager_mode", ["immediate", "legacy", "missing"])
def test_prefix_reset_bypasses_immediate_release_manager(monkeypatch, manager_mode):
    manager = FakeManager()
    manager.defer_physical_release = False
    if manager_mode == "legacy":
        del manager.defer_physical_release
    capture_marker = mock.Mock(side_effect=AssertionError("Unexpected epoch query"))
    release_pages = mock.Mock(side_effect=AssertionError("Unexpected physical release"))
    monkeypatch.setattr(manager, "capture_physical_release_marker", capture_marker)
    monkeypatch.setattr(manager, "release_retired_pages_through", release_pages)

    def original_reset(self):
        return False

    EngineCore = _patch_engine(monkeypatch, mock.Mock(), original_reset)
    engine = _engine(
        EngineCore,
        None if manager_mode == "missing" else manager,
        mock.sentinel.unused_batch_queue,
    )

    assert engine.reset_prefix_cache() is False
    capture_marker.assert_not_called()
    release_pages.assert_not_called()


def test_prefix_reset_does_not_assume_missing_queue_is_idle(monkeypatch):
    manager = FakeManager()

    def original_reset(self):
        manager.retire()
        return True

    EngineCore = _patch_engine(monkeypatch, mock.Mock(), original_reset)
    engine = _engine(EngineCore, manager, None)

    assert engine.reset_prefix_cache() is True
    assert manager.released == []


def test_failed_prefix_reset_preserves_exception_without_releasing(monkeypatch):
    manager = FakeManager()

    def original_reset(self):
        manager.retire()
        raise RuntimeError("reset failed")

    EngineCore = _patch_engine(monkeypatch, mock.Mock(), original_reset)
    engine = _engine(EngineCore, manager, deque())

    with pytest.raises(RuntimeError, match="reset failed"):
        engine.reset_prefix_cache()
    assert manager.released == []


def test_failed_release_preserves_completed_fence(monkeypatch):
    manager = FakeManager()
    manager.retire()

    def original_step(self):
        return ({}, True)

    EngineCore = _patch_engine(monkeypatch, original_step)
    engine = _engine(EngineCore, manager, deque([object()]))
    release = mock.Mock(side_effect=RuntimeError("worker barrier failed"))
    monkeypatch.setattr(manager, "release_retired_pages_through", release)
    with pytest.raises(RuntimeError, match="worker barrier failed"):
        engine.step_with_batch_queue()
    assert engine._kvcached_release_fences == [[1, 0]]
    release.side_effect = None
    engine.step_with_batch_queue()
    assert engine._kvcached_release_fences == []
    assert release.call_args_list == [mock.call(1), mock.call(1)]


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_no_batch_queue_does_not_leave_pages_waiting_for_a_queue_step(monkeypatch, async_scheduling):
    patches = _load_patches(monkeypatch)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    interfaces = __import__(
        "kvcached.integration.vllm.interfaces", fromlist=["init_kvcached"],
    )
    monkeypatch.setattr(interfaces, "init_kvcached", mock.Mock())
    manager = FakeManager()
    rpc = mock.Mock(side_effect=AssertionError("Synchronous execution needs no queue barrier"))

    class EngineCore:
        def __init__(self, config):
            self.vllm_config = config
            self.batch_queue = None
            self.model_executor = SimpleNamespace(collective_rpc=rpc)
            self.scheduler = SimpleNamespace(kv_cache_manager=SimpleNamespace(
                block_pool=SimpleNamespace(kv_cache_manager=manager)))

    target = types.ModuleType("engine")
    setattr(target, "EngineCore", EngineCore)
    assert patches.EngineCorePatch().patch_engine_init(target)
    EngineCore(SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    ))
    assert manager.defer_physical_release is False
    assert not hasattr(manager, "physical_release_barrier")
    rpc.assert_not_called()


@pytest.mark.parametrize("version", ["0.28.0", "0.29.0"])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_engine_release_barrier_preserves_transactional_unmap_callback(
    monkeypatch, version, async_scheduling,
):
    patches = _load_patches(monkeypatch)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)

    interfaces = __import__(
        "kvcached.integration.vllm.interfaces", fromlist=["init_kvcached"]
    )
    monkeypatch.setattr(interfaces, "init_kvcached", mock.Mock())

    existing_callback = mock.Mock()
    page_allocator = SimpleNamespace(
        callback=existing_callback,
        set_broadcast_unmap_callback=mock.Mock(),
    )
    manager = SimpleNamespace(
        group_id=17,
        pp_rank=-1,
        page_allocator=page_allocator,
        _increment_operation_counter=mock.Mock(),
    )

    class Executor:
        def collective_rpc(self, method, *, args):
            assert method is patches._worker_physical_release_barrier
            assert args == ()
            return [True] * 8

    engine_mod = types.ModuleType("vllm.v1.engine.core")

    class EngineCore:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.batch_queue: deque[Any] = deque()
            self.model_executor = Executor()
            self.scheduler = SimpleNamespace(
                kv_cache_manager=SimpleNamespace(
                    block_pool=SimpleNamespace(kv_cache_manager=manager)
                )
            )

    setattr(engine_mod, "EngineCore", EngineCore)
    patch = patches.EngineCorePatch()
    patch.detected_version = version
    assert patch.patch_engine_init(engine_mod)
    config = SimpleNamespace(
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )

    EngineCore(config)
    assert manager.defer_physical_release
    manager.physical_release_barrier()
    page_allocator.set_broadcast_unmap_callback.assert_not_called()
    assert page_allocator.callback is existing_callback



@pytest.mark.parametrize("responses", [[True, False], [True]])
def test_release_barrier_rejects_failed_or_missing_worker(monkeypatch, responses):
    patches = _load_patches(monkeypatch)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    interfaces = __import__(
        "kvcached.integration.vllm.interfaces", fromlist=["init_kvcached"]
    )
    monkeypatch.setattr(interfaces, "init_kvcached", mock.Mock())

    page_allocator = SimpleNamespace(callback=None)
    page_allocator.set_broadcast_unmap_callback = lambda callback: setattr(
        page_allocator, "callback", callback
    )
    manager = SimpleNamespace(
        group_id=3,
        pp_rank=0,
        page_allocator=page_allocator,
        _increment_operation_counter=mock.Mock(),
    )
    engine_mod = types.ModuleType("vllm.v1.engine.core")

    class EngineCore:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.batch_queue: deque[Any] = deque()
            self.model_executor = SimpleNamespace(
                collective_rpc=lambda method, args: responses
            )
            self.scheduler = SimpleNamespace(
                kv_cache_manager=SimpleNamespace(
                    block_pool=SimpleNamespace(kv_cache_manager=manager)
                )
            )

    setattr(engine_mod, "EngineCore", EngineCore)
    assert patches.EngineCorePatch().patch_engine_init(engine_mod)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=True),
    )
    EngineCore(config)

    with pytest.raises(RuntimeError, match="KV release barrier failed"):
        manager.physical_release_barrier()
    assert page_allocator.callback is None
