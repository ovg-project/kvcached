# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed engine's queue method without constructing a model."""

import ast
import importlib.util
import types
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest


def load_native_step(source):
    tree = ast.parse(source)
    engine = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "EngineCore")
    step = next(node for node in engine.body if isinstance(node, ast.FunctionDef)
                and node.name == "step_with_batch_queue")
    step.decorator_list = []
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        step,
    ], type_ignores=[])
    namespace = dict(Future=Future, ModelRunnerOutput=object, cast=cast)
    exec(compile(ast.fix_missing_locations(module), "<native-engine-queue>", "exec"), namespace)
    return namespace["step_with_batch_queue"]


def exercise_native_queue(native_step, deferred_sampling, worker_failure):
    from kvcached.integration.vllm.patches import EngineCorePatch

    events: list[Any] = []

    class Manager:
        defer_physical_release = True
        marker = 0

        def capture_physical_release_marker(self):
            return self.marker

        def release_retired_pages_through(self, marker):
            events.append(("unmap", marker))

    manager = Manager()

    class Engine(SimpleNamespace):
        step_with_batch_queue = native_step

    target = types.ModuleType("native_queue_contract")
    setattr(target, "EngineCore", Engine)
    assert EngineCorePatch().patch_async_batch_lifetime(target)
    engine = Engine()
    older, newer = Mock(), Mock()
    def result(event):
        events.append(event)
        return object()

    older.result.side_effect = lambda: result("older-result")
    newer.result.side_effect = (
        RuntimeError("worker failed") if worker_failure
        else lambda: result("newer-result")
    )
    next_output = SimpleNamespace(total_num_scheduled_tokens=1,
                                  pending_structured_output_tokens=deferred_sampling)

    def update(*_):
        manager.marker = 1
        events.append("logical-free")
        return {}

    engine.scheduler = SimpleNamespace(
        has_requests=Mock(return_value=True), schedule=Mock(return_value=next_output),
        get_grammar_bitmask=Mock(return_value=None), update_from_output=update,
        kv_cache_manager=SimpleNamespace(block_pool=SimpleNamespace(kv_cache_manager=manager)),
    )
    engine.model_executor = SimpleNamespace(execute_model=Mock(return_value=newer),
                                           sample_tokens=Mock(return_value=newer))
    engine.batch_queue = deque([(older, object(), older)])
    engine.batch_queue_size = 2
    engine.is_ec_consumer = True
    engine.is_pooling_model = False
    engine.check_for_draft_tokens = False
    engine._should_throttle_prefills = lambda: False
    engine.log_error_detail = lambda *_: nullcontext()
    engine.capture_iteration_details = lambda *_: nullcontext()
    engine._attach_iteration_details = lambda *_: None
    engine._process_aborts_queue = lambda: None

    assert engine.step_with_batch_queue() == ({}, True)
    assert events == ["older-result", "logical-free"]
    assert len(engine.batch_queue) == 1
    assert engine.batch_queue[0][0] is newer
    engine.scheduler.has_requests.return_value = False
    if worker_failure:
        with pytest.raises(RuntimeError, match="worker failed"):
            engine.step_with_batch_queue()
        assert not any(isinstance(event, tuple) for event in events)
    else:
        assert engine.step_with_batch_queue() == ({}, False)
        assert events[-3:] == ["newer-result", "logical-free", ("unmap", 1)]


@pytest.mark.parametrize("deferred_sampling", [False, True])
@pytest.mark.parametrize("worker_failure", [False, True])
def test_installed_native_queue_preserves_inflight_mapping(deferred_sampling, worker_failure):
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        pytest.skip("requires an installed vLLM engine source")
    assert spec is not None and spec.origin is not None
    source = Path(spec.origin).parent / "v1/engine/core.py"
    exercise_native_queue(load_native_step(source.read_text()), deferred_sampling, worker_failure)
