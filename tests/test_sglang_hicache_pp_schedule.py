# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Drive SGLang's pinned PP scheduler and output ring with Gloo on CPU."""
from __future__ import annotations

import ast
import logging
import os
import sys
from collections import defaultdict, deque
from collections.abc import Collection
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist
from test_sglang_hicache_distributed_retention import Leaf, _use_groups

from kvcached.integration.sglang import hicache_retention as retention


def _native_loop(path, depth):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    mixin = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "SchedulerPPMixin")
    methods = {"event_loop_pp", "init_pp_loop_state", "_pp_commit_comm_work",
               "_pp_commit_send_output_work_and_preprocess_output_tensors",
               "_pp_send_output_to_next_stage", "_pp_send_recv_and_preprocess_output_tensors",
               "_pp_recv_typed_dict", "_pp_recv_dict_from_prev_stage", "_pp_recv_proxy_tensors",
               "_pp_send_dict_to_next_stage"}
    functions = [m for m in mixin.body if isinstance(m, ast.FunctionDef) and m.name in methods]
    for method in functions:
        method.decorator_list = []
    mixin.body = list(functions)
    tree.body = [mixin]

    def skip(batch):
        return batch.skip_output

    ns = dict(torch=torch, deque=deque, defaultdict=defaultdict, ScheduleBatch=NS, logger=logging.getLogger(__name__),
              PPProxyTensors=lambda t: NS(tensors=t), _is_npu=False, is_xpu=lambda: False,
              _pp_can_skip_output_comm=skip,
              get_parallel=lambda: NS(pp_async_batch_depth=depth, enable_dsa_prefill_context_parallel=False))
    exec(compile(tree, path, "exec"), ns)
    module = ModuleType("sglang.srt.managers.scheduler_pp_mixin")
    setattr(module, "_pp_can_skip_output_comm", skip)
    sys.modules[module.__name__] = module
    return ns["SchedulerPPMixin"]()


def _run_loop(path, depth, attention, pipeline, dp_stage=None, follower_pressure=False, disputed=False):
    stage = dist.get_rank(pipeline)
    ranks = dist.get_process_group_ranks(pipeline)
    previous, following = ranks[(stage - 1) % len(ranks)], ranks[(stage + 1) % len(ranks)]
    scheduler = _native_loop(path, depth)
    scheduler.ps = NS(pp_size=len(ranks), pp_rank=stage,
                      attn_dp_size=dist.get_world_size(dp_stage) if dp_stage is not None else 1)
    scheduler.pp_group = NS(is_last_rank=stage == len(ranks) - 1, is_first_rank=stage == 0)
    scheduler.attn_tp_group, scheduler.require_attn_tp_allgather = attention, False
    scheduler.enable_hierarchical_cache = True
    scheduler.disaggregation_mode = NS(value="null")
    stream = NS(wait_event=lambda _: None, wait_stream=lambda _: None)
    event = NS(record=lambda _: None, synchronize=lambda: None)
    scheduler.schedule_stream = scheduler.copy_stream = stream
    scheduler.copy_stream_ctx = nullcontext()
    scheduler.device_module = NS(current_stream=lambda: stream, Event=lambda: event)

    def receive(tag):
        value = torch.empty(1, dtype=torch.int64)
        dist.recv(value, src=previous, group=pipeline, tag=tag)
        return int(value.item())

    def send(tag, value):
        data = torch.tensor([value])
        return [NS(work=dist.isend(data, dst=following, group=pipeline, tag=tag), tensor=data)]

    kinds = ("proxy", "output")

    def send_dict(tensor_dict, all_gather_group, async_send):
        packet = torch.tensor([kinds.index(tensor_dict["__msg_type__"]), tensor_dict["id"]])
        return [NS(work=dist.isend(packet, dst=following, group=pipeline, tag=2), tensor=packet)]

    def recv_dict(all_gather_group):
        packet = torch.empty(2, dtype=torch.int64)
        dist.recv(packet, src=previous, group=pipeline, tag=2)
        kind, value = packet.tolist()
        return {"__msg_type__": kinds[kind], "id": value}

    scheduler.pp_group.send_tensor_dict, scheduler.pp_group.recv_tensor_dict = send_dict, recv_dict
    # Force both kinds through the same channel in the opposite order to
    # the reads. A receiver must save the first arrival for a later slot.
    scheduler._pp_tensor_dict_inbox = defaultdict(deque)
    work = scheduler._pp_send_dict_to_next_stage({"id": stage}, msg_type="proxy")
    work.extend(scheduler._pp_send_dict_to_next_stage({"id": stage}, msg_type="output"))
    assert scheduler._pp_recv_dict_from_prev_stage()["id"] == (stage - 1) % len(ranks)
    assert len(scheduler._pp_tensor_dict_inbox["proxy"]) == 1
    assert scheduler._pp_recv_typed_dict("proxy")["id"] == (stage - 1) % len(ranks)
    scheduler._pp_commit_comm_work(work)

    # Native HiCache reduces within stage zero and forwards ACKs down PP.
    ack_work = []

    def reduce_attention(tensor, op):
        if attention is not None:
            dist.all_reduce(tensor, op=op, group=attention)

    def ready_counts(tensor, op):
        if stage == 0:
            reduce_attention(tensor, op)
        else:
            dist.recv(tensor, src=previous, group=pipeline, tag=4)
        if not scheduler.pp_group.is_last_rank:
            data = tensor.clone()
            ack_work.append(NS(work=dist.isend(data, dst=following, group=pipeline, tag=4), tensor=data))

    _use_groups(attention, pipeline)
    cache = NS(pp_size=len(ranks), pp_rank=stage, tp_world_size=1,
               cache_controller=NS(write_policy="write_through"), evictable_leaves=set(),
               evictable_size_=0, protected_size_=0,
               _all_reduce=ready_counts, _enforce_prefix_cache_retention=lambda _: None,
               eviction_strategy=NS(get_priority=lambda n: n.key[0]), update_eviction_metrics=lambda *_: None)
    evicted_at = []

    def evict(node):
        evicted_at.append(scheduler.tick)
        cache.evictable_leaves.remove(node)
        cache.evictable_size_ -= 1
        return 1

    def check_events():
        scheduler._pp_commit_comm_work(ack_work)
        cache._all_reduce(torch.zeros(2, dtype=torch.int), dist.ReduceOp.MIN)

    cache.check_hicache_events, cache._evict_backuped = check_events, evict
    scheduler.tree_cache = cache
    scheduler.tick = -1
    processed, idle = [], []
    busy = range(40 if follower_pressure else 20)
    pressure: Collection[int] = busy
    if follower_pressure:
        pressure = ()
    if dp_stage is not None:
        # Only replica zero exceeds the cap, while both have work in every busy slot.
        pressure = (7,) if dist.get_rank(dp_stage) == 0 else ()

    class Finished(Exception):
        pass

    def receive_requests():
        scheduler.tick += 1
        if scheduler.tick == len(busy) + 12:
            raise Finished
        if stage:
            assert receive(1) == scheduler.tick
        if dp_stage is not None:
            # Default DP-attention control-request broadcast spans the entire TP stage.
            control = torch.tensor([scheduler.tick])
            dist.broadcast(control, src=dist.get_global_rank(dp_stage, 0), group=dp_stage)
            assert control.item() == scheduler.tick
        return scheduler.tick

    def dispatch(tick):
        if follower_pressure and tick in (4, 5):
            policy = "write_back" if tick == 4 else "write_through"
            scheduler._request_dispatcher(NS(hicache_write_policy=policy))

    def attach(request):
        if stage == 0 or request.hicache_write_policy == "write_through":
            cache.cache_controller.write_policy = request.hicache_write_policy

    def ingest():
        tick = receive_requests()
        scheduler.process_input_requests(tick)
        return tick

    root = NS(parent=None)
    shared = 2 * retention._VICTIMS_PER_BATCH + 1
    limit = shared + 1 if disputed else int(follower_pressure)

    def schedule(running_batch, last_batch):
        i = scheduler.tick
        if i in pressure:
            cache.evictable_leaves.add(Leaf(i, root))
            cache.evictable_size_ += 1
        if follower_pressure and i == 7:
            if disputed:
                # More than two proposal pages have different namespaces.
                cache.evictable_leaves.update(Leaf(k, root, str(stage)) for k in range(shared))
            cache.evictable_leaves.add(Leaf(shared if disputed else 0, root))
            if scheduler.pp_group.is_last_rank:
                cache.evictable_leaves.add(Leaf(shared + 1, root))
            cache.evictable_size_ = len(cache.evictable_leaves)
        cache.check_hicache_events()
        if dp_stage is not None:
            # DP attention gathers MLP batch metadata after HiCache polling.
            ticks = torch.empty(scheduler.ps.attn_dp_size, dtype=torch.int64)
            dist.all_gather_into_tensor(ticks, torch.tensor([i]), group=dp_stage)
            assert ticks.tolist() == [i] * scheduler.ps.attn_dp_size
        elif i in busy and not follower_pressure:
            assert not cache.evictable_leaves
        batch = NS(id=i, forward_mode=NS(is_prebuilt=lambda: False), skip_output=(i % 5 == 2))
        return NS(batch_to_run=batch if i in busy else None, running_batch=running_batch)

    def launch(mb_id, batch, proxy, metadata, queue):
        if stage:
            assert proxy.tensors["id"] == batch.id
        payload = NS(tensors={"id": batch.id})
        if scheduler.pp_group.is_last_rank:
            queue.append((event, payload))
        return NS(pp_hidden_states_proxy_tensors=payload), event

    def process(batch, result):
        assert batch.id == result.id
        processed.append(batch.id)

    scheduler.ingest_requests = ingest
    scheduler.request_receiver = NS(recv_requests=receive_requests)
    scheduler.process_input_requests = dispatch
    scheduler._request_dispatcher = attach
    scheduler.get_next_batch_to_run = schedule
    scheduler.on_idle = lambda: idle.append(scheduler.tick)
    scheduler._pp_send_pyobj_to_next_stage = lambda value, async_send: send(1, value)
    scheduler._pp_launch_batch = launch
    scheduler._pp_prep_batch_result = lambda batch, metadata, outputs: NS(id=outputs.tensors["id"], logits_output=None)
    scheduler._pp_make_skip_output_result = lambda batch, metadata: (None, NS(id=batch.id), event)
    scheduler._pp_process_batch_result = process
    with pytest.MonkeyPatch.context() as patch:
        # Only the follower needs to reclaim. Its finite cap also applies
        # when stage zero explicitly requested unlimited retention.
        local_limit = -1 if disputed and stage == 0 else limit
        patch.setattr(retention, "get_prefix_cache_retention_limit", lambda: local_limit)
        retention.install_parallel_hicache_retention(scheduler)
        with pytest.raises(Finished):
            scheduler.event_loop_pp()
    assert processed == list(busy) and idle and min(idle) >= len(busy)
    if follower_pressure:
        assert len(evicted_at) == 1 and evicted_at[0] in busy
        assert cache.evictable_size_ == limit - 1 + int(scheduler.pp_group.is_last_rank)
        if disputed:
            assert all(node.key[0] != shared for node in cache.evictable_leaves)
    if dp_stage is not None:
        assert cache.check_hicache_events is check_events
        assert cache.evictable_size_ == int(dist.get_rank(dp_stage) == 0)
    for work in (scheduler.send_req_work, scheduler.send_output_work, scheduler.send_proxy_work, ack_work):
        scheduler._pp_commit_comm_work(work)


def _worker(rank, uri, path, dp_attention):
    timeout = timedelta(seconds=15) if dp_attention else timedelta(seconds=30)
    dist.init_process_group("gloo", init_method=uri, rank=rank, world_size=4, timeout=timeout)
    try:
        groups = {r: dist.new_group(list(r), timeout=timeout) for r in ((0, 1), (2, 3), (0, 2), (1, 3))}
        stage, lane = divmod(rank, 2)
        tp_stage = groups[(2 * stage, 2 * stage + 1)]
        pipeline = groups[(lane, lane + 2)]
        if dp_attention:
            # TP=DP=2 gives each replica an attention group of size one.
            _run_loop(path, 0, None, pipeline, dp_stage=tp_stage)
            return
        for depth in (0, 1):
            _run_loop(path, depth, None, dist.group.WORLD)
            _run_loop(path, depth, tp_stage, pipeline)
        _run_loop(path, 1, None, dist.group.WORLD, follower_pressure=True)
        _run_loop(path, 1, tp_stage, pipeline, follower_pressure=True, disputed=True)
    finally:
        dist.destroy_process_group()


@pytest.fixture
def pp_source():
    root = Path(os.environ.get("SGLANG_SOURCE", "engine_integration/sglang"))
    path = root / "python/sglang/srt/managers/scheduler_pp_mixin.py"
    if "SGLANG_SOURCE" not in os.environ and not path.is_file():
        pytest.skip("native SGLang scheduler source unavailable")
    return str(path)


def test_busy_pipeline_returns_each_output_and_trims_before_idle(tmp_path, pp_source):
    torch.multiprocessing.spawn(_worker, args=((tmp_path / "store").as_uri(), pp_source, False), nprocs=4)


def test_dp_attention_pipeline_finishes_with_one_replica_over_cap(tmp_path, pp_source):
    torch.multiprocessing.spawn(_worker, args=((tmp_path / "store").as_uri(), pp_source, True), nprocs=4)
