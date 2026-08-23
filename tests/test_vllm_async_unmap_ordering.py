# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from collections import deque
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
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


def test_engine_ordered_unmap_uses_worker_rpc(monkeypatch):
    patches = _load_patches(monkeypatch)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)

    interfaces = __import__(
        "kvcached.integration.vllm.interfaces", fromlist=["init_kvcached"]
    )
    monkeypatch.setattr(interfaces, "init_kvcached", mock.Mock())

    class PageAllocator:
        callback = None

        def set_broadcast_unmap_callback(self, callback):
            self.callback = callback

    manager = SimpleNamespace(
        group_id=17,
        pp_rank=-1,
        page_allocator=PageAllocator(),
        _increment_operation_counter=mock.Mock(),
    )

    class Executor:
        def collective_rpc(self, method, *, args):
            assert method is patches._worker_ordered_unmap
            assert args == ([64, 128], 17)
            return [True] * 8

    engine_mod = types.ModuleType("vllm.v1.engine.core")

    class EngineCore:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.model_executor = Executor()
            self.scheduler = SimpleNamespace(
                kv_cache_manager=SimpleNamespace(
                    block_pool=SimpleNamespace(kv_cache_manager=manager)
                )
            )

    setattr(engine_mod, "EngineCore", EngineCore)
    assert patches.EngineCorePatch().patch_engine_init(engine_mod)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=True),
    )

    EngineCore(config)
    manager.page_allocator.callback(4, [64, 128])



def test_ordered_unmap_raises_after_partial_failure(monkeypatch):
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
            self.model_executor = SimpleNamespace(
                collective_rpc=lambda method, args: [True, False]
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

    with pytest.raises(RuntimeError, match="Ordered KV unmap failed"):
        page_allocator.callback(2, [64])

