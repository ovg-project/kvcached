# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""CPU simulation of the pinned native PP loop; not model/GPU evidence."""

import ast
import copy
import importlib
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from queue import Queue
from threading import Barrier, Event
from types import SimpleNamespace as NS
from typing import Any

import pytest
import test_sglang_rank_coordination as consensus


class StopLoop(Exception):
    pass


class NoOp:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class Work:
    def __init__(self):
        self.received = Event()

    def wait(self):
        assert self.received.wait(3), "send was never received"


class Proxy:
    def __init__(self, tensors):
        self.tensors = tensors


class Batch:
    def __init__(self, reqs=None, batch_is_full=False):
        self.reqs = reqs or []
        self.batch_is_full = batch_is_full
        self.forward_mode = NS(is_prebuilt=lambda: False)


pipeline = importlib.import_module(consensus.package.__name__ + ".pipeline")
SOURCE = (
    Path(os.getenv("SGLANG_SOURCE", "engine_integration/sglang"))
    / "python/sglang/srt/managers/scheduler_pp_mixin.py"
)


def native_class():
    functions = {
        "event_loop_pp",
        "init_pp_loop_state",
        "_pp_send_output_to_next_stage",
        "_pp_send_recv_and_preprocess_output_tensors",
        "_pp_commit_send_output_work_and_preprocess_output_tensors",
        "_pp_send_dict_to_next_stage",
        "_pp_recv_typed_dict",
        "_pp_recv_dict_from_prev_stage",
        "_pp_recv_proxy_tensors",
    }
    source = ast.parse(SOURCE.read_text(encoding="utf-8"))
    owner = next(
        n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "SchedulerPPMixin"
    )
    nodes = [
        copy.deepcopy(n)
        for n in owner.body
        if isinstance(n, ast.FunctionDef) and n.name in functions
    ]
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *nodes,
        ],
        type_ignores=[],
    )
    globals_ = dict(
        torch=NS(profiler=NS(record_function=lambda *args: nullcontext())),
        get_parallel=lambda: NS(pp_async_batch_depth=0),
        _is_npu=False,
        is_xpu=lambda: False,
        ScheduleBatch=Batch,
        deque=deque,
        defaultdict=defaultdict,
        _pp_can_skip_output_comm=lambda batch: getattr(batch, "skip_output", False),
        PPProxyTensors=Proxy,
        logger=NoOp(),
    )
    exec(
        compile(ast.fix_missing_locations(module), "native_scheduler_pp_mixin.py", "exec"), globals_
    )
    return type("Native", (), {name: globals_[name] for name in functions})


def probe(size, early_flush, scenario="busy"):
    Native: Any = native_class()
    wire: list[Queue] = [Queue() for _ in range(size)]
    barrier = Barrier(size, timeout=3)
    signatures: list[Any] = [None] * size
    iterations = size * 4

    class Scheduler(Native):
        def __init__(self, rank):
            self.rank = rank
            self.ps = NS(pp_size=size)
            self.pp_group = NS(
                is_first_rank=rank == 0,
                is_last_rank=rank == size - 1,
                send_tensor_dict=self.send,
                recv_tensor_dict=self.recv,
            )
            self.attn_tp_group = None
            self.device_module = NS(current_stream=lambda: NoOp(), Event=NoOp)
            self.copy_stream, self.schedule_stream = NoOp(), NoOp()
            self.copy_stream_ctx = nullcontext()
            self.count = 0
            self.processed = []
            self.progress = pipeline.PipelineProgress(self)
            self.max_pending = 0

        def ingest_requests(self):
            return []

        def _pp_send_pyobj_to_next_stage(self, *args, **kwargs):
            return []

        def _pp_commit_comm_work(self, work):
            for item in work:
                item.wait()
            work.clear()

        def _pp_commit_send_output_work_and_preprocess_output_tensors(self, *args):
            if early_flush:
                self.progress.defer_output_work()
            return Native._pp_commit_send_output_work_and_preprocess_output_tensors(self, *args)

        def send(self, tensor_dict, **kwargs):
            receipt = Work()
            wire[(self.rank + 1) % size].put((dict(tensor_dict), receipt))
            return [receipt]

        def recv(self, **kwargs):
            tensors, receipt = wire[self.rank].get(timeout=3)
            receipt.received.set()
            return tensors

        def _pp_prep_batch_result(self, target, metadata, pp_outputs):
            return NS(logits_output=None, token=pp_outputs.tensors["step"])

        def _pp_process_batch_result(self, batch, result):
            batch.reqs[0].outputs += 1
            self.processed.append((batch.reqs[0].slot, result.token))

        def _pp_launch_batch(self, mb_id, batch, pp_proxy_tensors, metadata, queue):
            data = dict(step=self.count - 1, slot=mb_id)
            if pp_proxy_tensors is not None:
                assert {
                    k: v for k, v in pp_proxy_tensors.tensors.items() if k != "__msg_type__"
                } == data
            if self.pp_group.is_last_rank:
                queue.append((NoOp(), Proxy(data)))
            return NS(pp_hidden_states_proxy_tensors=Proxy(data)), NoOp()

        def get_next_batch_to_run(self, running_batch, last_batch):
            if early_flush:
                self.progress.before_admission()
            if scenario == "lag" and self.rank == size - 1:
                time.sleep(0.003)
            slot = next(i for i, batch in enumerate(self.running_mbs) if batch is running_batch)
            signatures[self.rank] = (
                self.count,
                slot,
                running_batch.reqs[0].outputs if running_batch.reqs else 0,
            )
            barrier.wait()
            assert all(s == signatures[0] for s in signatures), signatures
            barrier.wait()
            if early_flush:
                self.progress.after_agreement()
                self.max_pending = max(self.max_pending, len(self.progress.pending))
            if self.count >= iterations:
                raise StopLoop()
            self.count += 1
            if not running_batch.reqs:
                running_batch.reqs = [NS(slot=slot, outputs=0)]
            running_batch.skip_output = scenario == "chunk" and slot == 0
            batch = None if scenario == "idle" and self.count <= size else running_batch
            return NS(running_batch=running_batch, batch_to_run=batch)

        def _pp_make_skip_output_result(self, target, metadata):
            return None, NS(logits_output=None, token=-1), NoOp()

        def on_idle(self):
            pass

    def worker(rank):
        sched = Scheduler(rank)
        try:
            sched.event_loop_pp()
        except StopLoop:
            return dict(
                rank=rank, rounds=sched.count, processed=sched.processed, pending=sched.max_pending
            )
        except Exception as error:
            barrier.abort()
            return dict(
                rank=rank, rounds=sched.count, error=type(error).__name__ + ": " + str(error)
            )

    with ThreadPoolExecutor(max_workers=size) as pool:
        result = list(pool.map(worker, range(size)))
    return result


@pytest.mark.skipif(not SOURCE.is_file(), reason="native SGLang source unavailable")
@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("scenario", ["busy", "idle", "chunk", "lag"])
def test_native_pipeline_progress(size, scenario):
    results = probe(size, True, scenario)
    assert all("error" not in r for r in results), results
    assert all(r["rounds"] == size * 4 for r in results)
    assert all(r["processed"] == results[0]["processed"] for r in results), results
    assert all(r["pending"] <= 2 * (size + 2) for r in results)


@pytest.mark.skipif(not SOURCE.is_file(), reason="native SGLang source unavailable")
def test_admission_barrier_without_output_progress_deadlocks():
    results = probe(2, False)
    assert all(r["rounds"] == 2 for r in results)
    assert all("error" in r for r in results), results
