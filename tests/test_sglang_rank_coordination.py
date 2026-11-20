# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""CPU protocol and native-admission regressions; no model weights or CUDA."""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import os
import sys
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1] / "kvcached/integration/sglang/scheduler"
package = types.ModuleType("_rank_consensus_tests")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
coord = importlib.import_module(package.__name__ + ".rank_coordination")
plans = importlib.import_module(package.__name__ + ".rank_plan")


class Req:
    def __init__(self, rid, tokens=16384, host=12288, output=0):
        self.rid = rid
        self.origin_input_ids = list(range(tokens))
        self.full_untruncated_fill_ids = self.origin_input_ids + [1] * output
        self.output_ids = [1] * output
        self.prefix_indices = []
        self.host_hit_length = host
        self.storage_hit_length = 0
        self.mamba_pool_idx = None
        self.swa_host_hit_length = 0
        self.is_retracted = self.retracted_stain = False
        self.beam_group = None
        self.kv = NS(kv_committed_len=tokens, cache_protected_len=0)
        self.sampling_params = NS(ignore_eos=False, max_new_tokens=200)
        self.last_node = self.best_match_node = rid
        self.set_extend_range(0, tokens)

    def set_extend_range(self, start, end):
        self.extend_range = NS(start=start, end=end, length=end - start)

    def init_next_round_input(self, tree=None, cow_mamba=False):
        assert not cow_mamba

    def needs_host_load_back(self):
        return self.host_hit_length > 0

    def finished(self):
        return False


def mode(name):
    return NS(name=name, is_decode=lambda: name == "DECODE", is_extend=lambda: name == "EXTEND")


class ThreadWorld:
    """Ordered, blocking object collectives with a timeout, not canned results."""

    def __init__(self, size):
        self.size = size
        self.local = threading.local()
        self.condition = threading.Condition()
        self.rounds = {}

    def gather(self, output, value, group=None):
        rank, step = (self.local.rank, getattr(self.local, "step", 0))
        self.local.step = step + 1
        with self.condition:
            rows = self.rounds.setdefault(step, {})
            rows[rank] = copy.deepcopy(value)
            self.condition.notify_all()
            if not self.condition.wait_for(lambda: len(rows) == self.size, timeout=5):
                raise TimeoutError(f"collective {step} hung: {rows}")
            output[:] = [copy.deepcopy(rows[i]) for i in range(self.size)]

    def run(self, worker):

        def invoke(rank):
            self.local.rank, self.local.step = (rank, 0)
            try:
                return worker(rank)
            except Exception as exc:
                return exc

        with ThreadPoolExecutor(max_workers=self.size) as pool:
            return list(pool.map(invoke, range(self.size)))


class ContextTestCase(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)


class CoordinationTests(ContextTestCase):
    def setUp(self):
        super().setUp()
        self.world = ThreadWorld(4)
        torch: Any = types.ModuleType("torch")
        dist: Any = types.ModuleType("torch.distributed")
        torch.distributed = dist
        dist.all_gather_object = self.world.gather
        self.context.enter_context(mock.patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}))
        self.context.enter_context(
            mock.patch.dict(
                os.environ, {"KVCACHED_RANK_COORDINATION_ENABLE": "true", "ENABLE_KVCACHED": "true"}
            )
        )
        self.context.enter_context(mock.patch.object(coord, "_PATCHED", False))
        self.context.enter_context(
            mock.patch.object(coord.importlib.metadata, "version", return_value="0.5.20")
        )

        class Result(Enum):
            CONTINUE = auto()
            OTHER = auto()
            NO_TOKEN = auto()

        class Batch:
            prepare_error: bool
            allocate: Any
            token_to_kv_pool_allocator: Any
            req_to_token_pool: Any
            capacity_check: Any

            def __init__(self, reqs, name="DECODE", capacity=True):
                self.reqs = reqs
                self.forward_mode = mode(name)
                self.batch_is_full = self.is_prefill_only = False
                self.seq_lens_cpu = [r.kv.kv_committed_len for r in reqs]
                self.capacity = capacity
                self.checks = 0

            def check_decode_mem(self, selected_indices=None):
                self.checks += 1
                if getattr(self, "check_error", False):
                    raise RuntimeError("simulated decode capacity failure")
                callback = getattr(self, "capacity_check", None)
                if callback is not None:
                    return callback()
                return self.capacity

            def prepare_for_extend(self):
                if getattr(self, "prepare_error", False):
                    raise RuntimeError("simulated prefill allocation failure")
                callback = getattr(self, "allocate", None)
                if callback is not None:
                    callback(self)

            def prepare_for_decode(self):
                if getattr(self, "prepare_error", False):
                    raise RuntimeError("simulated decode allocation failure")

        class Adder:
            def __init__(self, available):
                self.available = available
                self.accepted: list[Any] = []
                self.can_run_list = self.accepted

            @property
            def rem_total_tokens(self):
                return self.available

            @property
            def cur_rem_tokens(self):
                return self.available

            @property
            def rem_swa_tokens(self):
                return self.available

            def add_one_req(self, req, **kwargs):
                if self.rem_total_tokens < len(req.origin_input_ids):
                    return Result.NO_TOKEN
                admission = self._select_prefill_admission(req)
                if admission == Result.OTHER:
                    return Result.OTHER
                self._commit_prefill_admission(req, admission)
                return Result.CONTINUE

            def _select_prefill_admission(self, req, **kwargs):
                return NS(
                    prefix_len=req.host_hit_length,
                    extend_len=len(req.origin_input_ids) - req.host_hit_length,
                    max_new_tokens=200,
                    is_chunked=False,
                )

            def _commit_prefill_admission(self, req, admission):
                req.set_extend_range(
                    admission.prefix_len, admission.prefix_len + admission.extend_len
                )
                self.accepted.append(req)

            def add_chunked_req(self, req):
                self.accepted.append(req)
                return None

        class Policy:
            def __init__(self):
                self.calls = 0

            def calc_priority(self, *args, **kwargs):
                self.calls += 1

        class Grammar:
            def has_waiting_grammars(self):
                return False

            def get_ready_grammar_requests(self):
                return []

        class Scheduler:
            token_to_kv_pool_allocator: Any
            req_to_token_pool: Any
            available_slots: int
            enable_dp_attention: bool
            tree_cache: Any
            allocate: Any

            def __init__(self, rank, available, actual_available=None, bypass=False, capacity=True):
                self.tp_group = NS(world_size=4, rank_in_group=rank, ranks=list(range(4)))
                self.tp_cpu_group: Any = "instance-group"
                self.ps = NS(tp_size=4, pp_size=1)
                self.waiting_queue = [Req("prefill-request")]
                self.chunked_req = None
                self.forward_ct = 10
                self.grammar_manager = Grammar()
                self.policy = Policy()
                self.processed_tokens_counter = 0
                self.available = available
                self.actual_available = available if actual_available is None else actual_available
                self.bypass, self.capacity = (bypass, capacity)
                self.forwards = 0

            def get_next_batch_to_run(self, running_batch, last_batch):
                if self.bypass:
                    return NS(batch_to_run=running_batch)
                p = self.get_new_batch_prefill(running_batch)
                if hasattr(p, "batch_to_run"):
                    p = p.batch_to_run
                if p is not None:
                    return NS(batch_to_run=p)
                if not running_batch.check_decode_mem():
                    return NS(batch_to_run=None)
                running_batch.prepare_error = getattr(self, "decode_prepare_error", False)
                running_batch.prepare_for_decode()
                return NS(batch_to_run=running_batch)

            def get_new_batch_prefill(self, running_batch):
                self.policy.calc_priority(self.waiting_queue, running_batch)
                self.native_slots_seen = self.get_num_allocatable_reqs(len(running_batch.reqs))
                if self.native_slots_seen <= 0:
                    return None
                adder = Adder(self.actual_available)
                for req in self.waiting_queue:
                    if adder.add_one_req(req) != Result.CONTINUE:
                        break
                if adder.accepted:
                    batch = Batch(adder.accepted, "EXTEND")
                    batch.prepare_error = getattr(self, "prefill_prepare_error", False)
                    if hasattr(self, "allocate"):
                        batch.allocate = self.allocate
                        batch.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
                        batch.req_to_token_pool = self.req_to_token_pool
                    batch.prepare_for_extend()
                    return batch
                return None

            def run_batch(self, batch):
                self.forwards += 1

            def get_num_allocatable_reqs(self, *args, **kwargs):
                limit = getattr(self, "available_slots", 1)
                pool = getattr(self, "req_to_token_pool", None)
                return min(limit, pool.available_size()) if pool is not None else limit

        self.Batch, self.Scheduler, self.Adder, self.Result = (Batch, Scheduler, Adder, Result)
        self.context.enter_context(
            mock.patch.dict(
                sys.modules,
                {
                    "sglang.srt.managers.scheduler": NS(Scheduler=Scheduler, NextBatchPlan=NS),
                    "sglang.srt.managers.schedule_batch": NS(ScheduleBatch=Batch),
                    "sglang.srt.managers.schedule_policy": NS(
                        PrefillAdder=Adder, SchedulePolicy=Policy, AddReqResult=Result
                    ),
                    "sglang.srt.constrained.grammar_manager": NS(GrammarManager=Grammar),
                },
            )
        )

        def preview(scheduler, running, **kwargs):
            req = scheduler.waiting_queue[0]
            p = plans.plan(
                "P", [dict(plans.request_row(req, "P", start=12288, tokens=4096), host_load=True)]
            )
            d = plans.plan("D", [plans.request_row(r, "D") for r in running.reqs])
            return {"p": p if scheduler.available >= 16384 else None, "d": d}

        self.context.enter_context(mock.patch.object(coord, "preview", side_effect=preview))
        self.assertEqual(coord.patch_rank_coordination(), ["rank-coordination"])

    def execute(
        self, capacities, actual=None, bypass=None, decode=None, configure=None, before_forward=None
    ):
        schedulers = [
            self.Scheduler(
                r,
                capacities[r],
                actual[r] if actual else None,
                bypass[r] if bypass else False,
                decode[r] if decode else True,
            )
            for r in range(4)
        ]

        def worker(rank):
            s = schedulers[rank]
            b = self.Batch([Req("decode", tokens=100, host=0, output=1)], capacity=s.capacity)
            if configure is not None:
                configure(rank, s, b)
            result = s.get_next_batch_to_run(b, None)
            if result.batch_to_run is not None:
                if before_forward is not None:
                    before_forward(rank, result.batch_to_run)
                s.run_batch(result.batch_to_run)
            return result.batch_to_run

        return (schedulers, self.world.run(worker))

    def test_asymmetric_capacity_all_choose_decode(self):
        schedulers, results = self.execute([8192, 8192, 8192, 65536])
        self.assertEqual([b.forward_mode.name for b in results], ["DECODE"] * 4)
        self.assertEqual([s.policy.calls for s in schedulers], [1] * 4)
        self.assertTrue(all((s.waiting_queue[0].extend_range.start == 0 for s in schedulers)))

    def test_common_prefill_preserves_native_selection(self):
        schedulers, results = self.execute([65536] * 4)
        self.assertEqual([b.forward_mode.name for b in results], ["EXTEND"] * 4)
        self.assertEqual([b.reqs[0].extend_range.length for b in results], [4096] * 4)

    def test_registration_is_idempotent(self):
        next_batch = self.Scheduler.get_next_batch_to_run
        run_batch = self.Scheduler.run_batch
        budget = self.Adder.rem_total_tokens
        self.assertEqual(coord.patch_rank_coordination(), [])
        self.assertIs(self.Scheduler.get_next_batch_to_run, next_batch)
        self.assertIs(self.Scheduler.run_batch, run_batch)
        self.assertIs(self.Adder.rem_total_tokens, budget)

    def test_budget_shrinks_after_preview_all_fall_back_to_decode(self):
        schedulers, results = self.execute([65536] * 4, actual=[65536, 65536, 65536, 0])
        self.assertEqual([b.forward_mode.name for b in results], ["DECODE"] * 4)
        self.assertEqual([s.forwards for s in schedulers], [1] * 4)

    def test_different_scheduler_branches_fail_without_collective_hang(self):
        schedulers, results = self.execute([65536] * 4, bypass=[False, False, False, True])
        self.assertTrue(all((isinstance(r, coord.RankConsensusError) for r in results)), results)
        self.assertEqual([s.forwards for s in schedulers], [0] * 4)

    def test_decode_capacity_requires_all_ranks(self):
        schedulers, results = self.execute([8192] * 4, decode=[True, True, True, False])
        self.assertEqual(results, [None] * 4)

    def test_one_rank_extend_allocation_failure_stops_all_forward(self):
        schedulers, results = self.execute(
            [65536] * 4,
            configure=lambda rank, s, b: setattr(s, "prefill_prepare_error", rank == 3),
        )
        self.assertTrue(all((isinstance(r, coord.RankConsensusError) for r in results)), results)
        self.assertTrue(all(("prefill allocation failure" in str(r) for r in results)))
        self.assertEqual([s.forwards for s in schedulers], [0] * 4)

    def test_one_rank_decode_allocation_failure_stops_all_forward(self):
        schedulers, results = self.execute(
            [8192] * 4, configure=lambda rank, s, b: setattr(s, "decode_prepare_error", rank == 0)
        )
        self.assertTrue(all((isinstance(r, coord.RankConsensusError) for r in results)), results)
        self.assertTrue(all(("decode allocation failure" in str(r) for r in results)))
        self.assertEqual([s.forwards for s in schedulers], [0] * 4)

    def test_one_rank_decode_checker_error_is_propagated(self):
        schedulers, results = self.execute(
            [8192] * 4, configure=lambda rank, s, b: setattr(b, "check_error", rank == 2)
        )
        self.assertTrue(all((isinstance(r, coord.RankConsensusError) for r in results)), results)
        self.assertTrue(all(("decode capacity failure" in str(r) for r in results)))
        self.assertEqual([s.forwards for s in schedulers], [0] * 4)

    def test_mutation_after_selection_is_blocked_at_forward(self):

        def mutate(rank, batch):
            if rank == 3:
                batch.forward_mode = mode("DECODE")

        schedulers, results = self.execute([65536] * 4, before_forward=mutate)
        self.assertTrue(all((isinstance(r, coord.RankConsensusError) for r in results)), results)
        self.assertTrue(all(("stage=forward" in str(r) for r in results)))
        self.assertEqual([s.forwards for s in schedulers], [0] * 4)

    def test_native_defer_path_still_checks_actual_batch(self):
        schedulers, results = self.execute([8192] * 4, bypass=[True] * 4)
        self.assertEqual([b.forward_mode.name for b in results], ["DECODE"] * 4)

    def test_preview_error_is_propagated_to_every_rank(self):
        original = coord.preview.side_effect

        def preview(s, b, **kwargs):
            if s.tp_group.rank_in_group == 2:
                raise ValueError("simulated local preview error")
            return original(s, b)

        coord.preview.side_effect = preview
        schedulers, results = self.execute([65536] * 4)
        self.assertTrue(all((isinstance(r, coord.RankConsensusError) for r in results)), results)
        self.assertTrue(all(("simulated local preview error" in str(r) for r in results)))

    def test_leader_decision_error_is_propagated_to_every_rank(self):
        with mock.patch.object(coord, "common_plan", side_effect=ValueError("invalid leader plan")):
            schedulers, results = self.execute([65536] * 4)
        self.assertTrue(all((isinstance(r, coord.RankConsensusError) for r in results)), results)
        self.assertTrue(all(("invalid leader plan" in str(r) for r in results)))
        self.assertEqual([s.forwards for s in schedulers], [0] * 4)

    def test_slot_transport_failure_stops_before_the_next_collective(self):
        calls: list[list[str]] = [[] for _ in range(4)]

        def gather(output, value, group=None):
            calls[self.world.local.rank].append(value["stage"])
            if value["stage"] == "slots":
                raise OSError("TP peer disconnected")
            self.world.gather(output, value, group)

        with mock.patch.object(sys.modules["torch.distributed"], "all_gather_object", gather):
            _, results = self.execute([65536] * 4)
        for events, error in zip(calls, results):
            self.assertIsInstance(error, coord.RankConsensusError)
            self.assertIsInstance(error.__cause__, OSError)
            self.assertEqual(events[-1], "slots")

    def test_slot_capacity_uses_minimum_and_is_frozen_for_native_pass(self):
        schedulers = [self.Scheduler(r, 65536) for r in range(4)]
        for s, available in zip(schedulers, [3, 2, 1, 3]):
            s.available_slots = available
        original = coord.preview.side_effect
        observed = []

        def preview(s, b, *, slots, **kwargs):
            observed.append((slots, s.get_num_allocatable_reqs(1)))
            return original(s, b)

        coord.preview.side_effect = preview

        def worker(rank):
            s = schedulers[rank]
            b = self.Batch([Req("d", tokens=100, host=0)])
            result = s.get_next_batch_to_run(b, None)
            return result.batch_to_run

        results = self.world.run(worker)
        self.assertTrue(all((not isinstance(r, Exception) for r in results)), results)
        self.assertEqual(sorted(observed), [(1, 1), (1, 2), (1, 3), (1, 3)])
        self.assertEqual([s.native_slots_seen for s in schedulers], [1] * 4)

    def test_singleton_preserves_native_path_without_preview(self):
        s = self.Scheduler(0, 65536)
        s.tp_group.world_size = 1
        s.ps.tp_size = 1
        s.tp_cpu_group = None
        result = s.get_next_batch_to_run(self.Batch([]), None)
        self.assertEqual(result.batch_to_run.forward_mode.name, "EXTEND")
        coord.preview.assert_not_called()

    def test_unsupported_topology_fails_on_every_rank(self):

        def worker(rank):
            s = self.Scheduler(rank, 65536)
            s.enable_dp_attention = rank == 3
            return s.get_next_batch_to_run(self.Batch([]), None)

        results = self.world.run(worker)
        self.assertTrue(all((isinstance(r, coord.RankConsensusError) for r in results)), results)
        self.assertTrue(all(("attention-DP" in str(r) for r in results)))


class PurePlanTests(unittest.TestCase):
    def test_signature_does_not_read_cuda_tensors(self):

        class Tensor:
            device = NS(type="cuda")

            def tolist(self):
                raise AssertionError("must not synchronize with GPU")

        batch = NS(
            reqs=[Req("d", tokens=1, host=0)], forward_mode=mode("DECODE"), seq_lens_cpu=Tensor()
        )
        with self.assertRaisesRegex(ValueError, "must not materialize a GPU tensor"):
            plans.batch_signature(batch)

    def test_common_ordered_prefix_and_strict_decode(self):
        p = plans.plan(
            "P",
            [{"rid": "a", "context": 0, "tokens": 10}, {"rid": "b", "context": 0, "tokens": 20}],
        )
        q = copy.deepcopy(p)
        q["rows"][1]["tokens"] = 19
        self.assertEqual(len(plans.common_plan([p, q], "P")["rows"]), 1)
        self.assertIsNone(plans.common_plan([p, q], "D"))
        self.assertIsNone(plans.common_plan([p, None], "P"))

    def test_common_prefill_merges_host_materialization_state(self):
        host = plans.plan("P", [{"rid": "a", "context": 1024, "tokens": 256, "host_load": True}])
        device = plans.plan("P", [{"rid": "a", "context": 1024, "tokens": 256, "host_load": False}])
        self.assertEqual(
            plans.common_plan([host, device], "P"),
            plans.plan("P", [{"rid": "a", "context": 1024, "tokens": 256, "host_load": True}]),
        )

    def test_prefill_backoff_is_suppressed_by_running_decode(self):
        scheduler = NS(
            tp_group=NS(world_size=1, rank_in_group=0, ranks=[0]),
            tp_cpu_group=None,
            ps=NS(tp_size=1, pp_size=1),
        )
        runtime = coord.RankCoordinator(scheduler)
        runtime.prefill_backoff_until = float("inf")
        runtime.has_running_decode = True
        self.assertFalse(runtime.prefill_backoff_active())
        runtime.has_running_decode = False
        self.assertTrue(runtime.prefill_backoff_active())

    def test_disabled_registration_is_noop(self):
        with (
            mock.patch.dict(os.environ, {"KVCACHED_RANK_COORDINATION_ENABLE": "0"}),
            mock.patch.object(coord, "_PATCHED", False),
        ):
            self.assertEqual(coord.patch_rank_coordination(), [])

    def test_library_defaults_to_disabled_without_importing_sglang(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(coord, "_PATCHED", False),
            mock.patch.object(coord.importlib.metadata, "version") as version,
        ):
            self.assertEqual(coord.patch_rank_coordination(), [])
            version.assert_not_called()

    def test_unsupported_sglang_version_fails_before_registration(self):
        for version in ("0.5.17", "0.5.19", "0.5.21"):
            with (
                self.subTest(version=version),
                mock.patch.dict(
                    os.environ, {"KVCACHED_RANK_COORDINATION_ENABLE": "1", "ENABLE_KVCACHED": "1"}
                ),
                mock.patch.object(coord, "_PATCHED", False),
                mock.patch.object(coord.importlib.metadata, "version", return_value=version),
            ):
                with self.assertRaisesRegex(RuntimeError, "requires SGLang 0.5.20"):
                    coord.patch_rank_coordination()
                self.assertFalse(coord._PATCHED)


SOURCE = Path(os.environ.get("SGLANG_SOURCE", "__sglang_source_unavailable__"))


def load_native_policy():
    """Execute the real source classes; stub imports, not admission control flow."""
    source = SOURCE / "python/sglang/srt/managers/schedule_policy.py"
    tree = ast.parse(source.read_text())
    nodes: list[ast.stmt] = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ]
    nodes += [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name in {"AddReqResult", "_PrefillAdmission", "PrefillAdder"}
    ]
    ns = {
        "__name__": __name__,
        "dataclass": dataclass,
        "Enum": Enum,
        "auto": auto,
        "contextmanager": contextmanager,
        "is_swa_req_ring": lambda a: False,
        "_use_exact_chunk_fill": lambda: False,
        "_IS_HIP": False,
        "is_dsa_prefill_cp_in_seq_split": lambda: False,
        "is_prefill_context_parallel_enabled": lambda: False,
        "split_cached_prefix_by_tier": lambda **k: (k["prefix_len"], 0, 0),
        "CLIP_MAX_NEW_TOKENS": 4096,
        "InitLoadBackParams": NS,
        "torch": NS(cat=lambda arrays: sum(arrays, [])),
    }
    for name in (
        "SWATokenToKVPoolAllocator",
        "DeepSeekV4HiSparseTokenToKVPoolAllocator",
        "PureSWATokenToKVPoolAllocator",
        "UnifiedMambaTokenToKVPoolAllocator",
        "UnifiedMambaSWATokenToKVPoolAllocator",
    ):
        ns[name] = type(name, (), {})
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"
        ),
        ns,
    )
    return NS(**ns)


@unittest.skipUnless(SOURCE.is_dir(), "native SGLang source unavailable")
class NativePreviewTests(ContextTestCase):
    def setUp(self):
        super().setUp()
        self.native = load_native_policy()
        self.pins = 0

        def acquire(node):
            self.pins += 1

        def release(node):
            self.pins -= 1

        cache = NS(
            disable=False,
            supports_mamba=lambda: False,
            evictable_size=lambda: 0,
            inc_lock_ref=acquire,
            dec_lock_ref=release,
            is_tree_cache=lambda: False,
            init_load_back=mock.Mock(side_effect=AssertionError("preview must not load KV")),
        )
        self.available = 8192
        self.req = Req("8591b688c56745baafbc1f74d2723f49")
        self.scheduler = NS(
            waiting_queue=[self.req],
            chunked_req=None,
            tree_cache=cache,
            token_to_kv_pool_allocator=NS(available_size=lambda: self.available),
            page_size=256,
            chunked_prefill_size=32768,
            max_prefill_tokens=16384,
            max_prefill_bs=15,
            max_running_requests=15,
            truncation_align_size=None,
            new_token_ratio_tracker=NS(current=1.0),
            get_num_allocatable_reqs=lambda *a, **k: 15,
            tp_worker=NS(model_runner=NS(attn_backend=NS())),
        )
        self.running = NS(reqs=[], batch_is_full=False, is_prefill_only=False)
        self.context.enter_context(
            mock.patch.dict(
                sys.modules,
                {
                    "sglang.srt.managers.schedule_policy": self.native,
                    "sglang.srt.runtime_context": NS(
                        get_schedule=lambda: NS(prefill_max_requests=None)
                    ),
                },
            )
        )

    def test_native_gate_rejects_before_h2d(self):
        self.assertIsNone(plans.preview(self.scheduler, self.running)["p"])
        self.available = 65536
        p = plans.preview(self.scheduler, self.running)["p"]
        self.assertEqual(
            p["rows"][0], {"rid": self.req.rid, "context": 12288, "tokens": 4096, "host_load": True}
        )
        self.assertEqual(self.req.prefix_indices, [])
        self.assertEqual(self.req.extend_range.start, 0)
        self.assertEqual(self.pins, 0)
        self.scheduler.tree_cache.init_load_back.assert_not_called()

    def test_fixed_match_traversal_is_bounded_by_common_running_limit(self):
        self.scheduler.max_running_requests = 2
        self.scheduler.waiting_queue = [Req(str(i)) for i in range(5)]
        self.running.reqs = [Req("running", host=0)]
        touched = []
        with mock.patch.object(
            Req, "init_next_round_input", lambda req, *a, **k: touched.append(req.rid)
        ):
            matches = plans.match_preview_requests(self.scheduler, self.running)
        self.assertEqual(touched, ["0"])
        self.assertEqual(list(matches), ["0"])


@unittest.skipUnless(SOURCE.is_dir(), "native SGLang source unavailable")
class NativeAdmissionTests(ContextTestCase):
    """Four ranks run actual native gate/select/H2D/commit/budget control flow."""

    def setUp(self):
        super().setUp()
        self.native = load_native_policy()
        self.world = getattr(self, "world", None) or ThreadWorld(4)
        torch: Any = types.ModuleType("torch")
        dist: Any = types.ModuleType("torch.distributed")
        torch.distributed = dist
        dist.all_gather_object = self.world.gather

        def noop(*args, **kwargs):
            return None

        scheduler = type(
            "Scheduler",
            (),
            {
                name: noop
                for name in (
                    "get_next_batch_to_run",
                    "get_new_batch_prefill",
                    "get_num_allocatable_reqs",
                    "run_batch",
                )
            },
        )
        batch = type(
            "Batch",
            (),
            {
                name: noop
                for name in ("check_decode_mem", "prepare_for_extend", "prepare_for_decode")
            },
        )
        self.native.SchedulePolicy = type("Policy", (), {"calc_priority": noop})
        grammar = type("Grammar", (), {"get_ready_grammar_requests": noop})
        modules = {
            "sglang.srt.managers.schedule_policy": self.native,
            "sglang.srt.managers.scheduler": NS(Scheduler=scheduler),
            "sglang.srt.managers.schedule_batch": NS(ScheduleBatch=batch),
            "sglang.srt.constrained.grammar_manager": NS(GrammarManager=grammar),
            "sglang.srt.mem_cache.base_prefix_cache": NS(InitLoadBackParams=NS),
        }
        if isinstance(self.world, ThreadWorld):
            torch.cat = lambda arrays: sum(arrays, [])
            modules.update({"torch": torch, "torch.distributed": dist})
        self.context.enter_context(mock.patch.dict(sys.modules, modules))
        self.context.enter_context(
            mock.patch.dict(
                os.environ, {"KVCACHED_RANK_COORDINATION_ENABLE": "1", "ENABLE_KVCACHED": "1"}
            )
        )
        self.context.enter_context(mock.patch.object(coord, "_PATCHED", False))
        self.context.enter_context(
            mock.patch.object(
                coord.importlib.metadata, "version", return_value=getattr(self, "version", "0.5.20")
            )
        )
        coord.patch_rank_coordination()
        self.native.PrefillAdder._account_prefill_cache_admission = noop

    def execute(
        self,
        capacities,
        *,
        after_pin=None,
        loads=None,
        host=0,
        count=1,
        chunk=False,
        after_commit=None,
        swa=None,
        budget_error=None,
        device_resident=None,
    ):
        self.adders, self.loads, self.forwards = ([], [0] * 4, [0] * 4)
        for rank in range(4):
            state = NS(available=capacities[rank], pins=0)

            def available(rank=rank, state=state):
                if budget_error == rank:
                    raise RuntimeError("simulated live budget failure")
                return state.available

            def acquire(node, state=state, rank=rank):
                state.pins += 1
                if after_pin is not None:
                    state.available = after_pin[rank]
                return NS(to_dec_params=lambda: None)

            def release(node, state=state):
                state.pins -= 1

            def load(params, rank=rank):
                self.loads[rank] += 1
                n = host if loads is None else loads[rank]
                return None if n is None else ([0] * n, params.best_match_node)

            cache = NS(
                disable=False,
                supports_mamba=lambda: False,
                evictable_size=lambda: 0,
                full_evictable_size=lambda: 0,
                swa_evictable_size=lambda: 0,
                sliding_window_size=4096,
                inc_lock_ref=acquire,
                dec_lock_ref=release,
                is_tree_cache=lambda: False,
                init_load_back=load,
            )
            if swa is None:
                allocator = NS(available_size=available)
            else:
                allocator = self.native.SWATokenToKVPoolAllocator()
                allocator.full_available_size = available
                allocator.swa_available_size = lambda rank=rank: swa[rank]
                allocator.size_swa = 1000000
            adder = self.native.PrefillAdder(256, cache, allocator, NS(reqs=[]), 1.0, 16384, 32768)
            if after_commit is not None:
                commit = adder._update_prefill_budget

                def account(*args, state=state, rank=rank, commit=commit, **kwargs):
                    commit(*args, **kwargs)
                    state.available = after_commit[rank]

                adder._update_prefill_budget = account
            self.adders.append(adder)

        def worker(rank):
            runtime = coord.RankCoordinator(
                NS(
                    tp_group=NS(world_size=4, rank_in_group=rank, ranks=list(range(4))),
                    tp_cpu_group="instance",
                    ps=NS(tp_size=4, pp_size=1),
                )
            )
            token = coord._ACTIVE.set(runtime)
            try:
                adder = self.adders[rank]
                verdicts = []
                for i in range(count):
                    req = Req(str(i), tokens=16384 if host else 1024, host=host)
                    if device_resident is not None and device_resident[rank]:
                        req.prefix_indices = [0] * host
                        req.host_hit_length = 0
                    if chunk:
                        result = adder.add_chunked_req(req)
                        verdicts.append(result is not None)
                    else:
                        result = adder.add_one_req(
                            req, has_chunked_req=False, truncation_align_size=None
                        )
                        verdicts.append(result.name)
                        if result != self.native.AddReqResult.CONTINUE:
                            break
                runtime.agree("test_complete", verdicts)
                self.forwards[rank] += bool(adder.can_run_list)
                return verdicts
            except coord.RankConsensusError:
                raise
            except Exception as exc:
                runtime.exchange("test_complete", None, error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                coord._ACTIVE.reset(token)

        return self.world.run(worker)

    def assert_no_admission(self, results, verdict):
        self.assertEqual(results, [[verdict]] * 4)
        self.assertEqual(self.loads, [0] * 4)
        self.assertEqual([len(a.can_run_list) for a in self.adders], [0] * 4)
        self.assertEqual(self.forwards, [0] * 4)

    def test_real_outer_gate_uses_minimum_before_any_host_load(self):
        results = self.execute([8192, 8192, 8192, 65536], host=12288)
        self.assert_no_admission(results, "NO_TOKEN")

    def test_capacity_drops_after_outer_gate_before_select(self):
        results = self.execute([65536] * 4, after_pin=[65536, 65536, 0, 65536], host=12288)
        self.assert_no_admission(results, "NO_TOKEN")

    def test_swa_rejection_inside_select_is_unanimous(self):
        results = self.execute([65536] * 4, swa=[10000, 10000, 10000, 0], host=12288)
        self.assert_no_admission(results, "NO_TOKEN")

    def test_native_success_loads_and_commits_same_shape(self):
        results = self.execute([65536] * 4, host=12288)
        self.assertEqual(results, [["CONTINUE"]] * 4)
        self.assertEqual(self.loads, [1] * 4)
        self.assertEqual([a.can_run_list[0].extend_range.length for a in self.adders], [4096] * 4)
        self.assertEqual(self.forwards, [1] * 4)

    def test_host_load_none_on_one_rank_defers_prefill_without_commit(self):
        results = self.execute([65536] * 4, host=12288, loads=[12288, 12288, 12288, None])
        self.assertEqual(results, [["OTHER"]] * 4)
        self.assertEqual(self.loads, [1] * 4)
        self.assertEqual([len(a.can_run_list) for a in self.adders], [0] * 4)
        self.assertEqual(self.forwards, [0] * 4)

    def test_host_load_zero_on_one_rank_defers_without_recompute(self):
        results = self.execute([65536] * 4, host=12288, loads=[12288, 12288, 0, 12288])
        self.assertEqual(results, [["OTHER"]] * 4)
        self.assertEqual([len(a.can_run_list) for a in self.adders], [0] * 4)
        self.assertEqual(self.forwards, [0] * 4)

    def test_host_load_shape_difference_defers_before_any_commit(self):
        results = self.execute([65536] * 4, host=12288, loads=[12288, 12288, 12288, 55553])
        self.assertEqual(results, [["OTHER"]] * 4)
        self.assertEqual([len(a.can_run_list) for a in self.adders], [0] * 4)
        self.assertEqual(self.forwards, [0] * 4)

    def test_deferred_cache_tier_difference_waits_for_last_rank(self):
        results = self.execute(
            [65536] * 4,
            host=12288,
            loads=[None, None, None, 12288],
            device_resident=[True, True, True, False],
        )
        self.assertEqual(results, [["CONTINUE"]] * 4)
        self.assertEqual(self.loads, [0, 0, 0, 1])
        self.assertEqual([len(a.can_run_list) for a in self.adders], [1] * 4)
        self.assertEqual([a.can_run_list[0].extend_range.length for a in self.adders], [4096] * 4)

    def test_deferred_cache_tier_difference_keeps_running_decode_on_retry_failure(self):
        results = self.execute(
            [65536] * 4,
            host=12288,
            loads=[None, None, None, None],
            device_resident=[True, True, True, False],
        )
        self.assertEqual(results, [["OTHER"]] * 4)
        self.assertEqual(self.loads, [0, 0, 0, 1])
        self.assertEqual([len(a.can_run_list) for a in self.adders], [0] * 4)

    def test_budget_state_stops_all_after_same_first_candidate(self):
        results = self.execute([65536] * 4, count=2, after_commit=[65536, 65536, 0, 65536])
        self.assertEqual(results, [["NO_TOKEN"]] * 4)
        self.assertEqual([len(a.can_run_list) for a in self.adders], [1] * 4)

    def test_multiple_candidates_keep_order(self):
        results = self.execute([65536] * 4, count=3)
        self.assertEqual(results, [["CONTINUE"] * 3] * 4)
        self.assertEqual(
            [[r.rid for r in a.can_run_list] for a in self.adders], [["0", "1", "2"]] * 4
        )

    def test_chunk_continuation_uses_common_live_token_budget(self):
        results = self.execute([65536, 65536, 512, 65536], chunk=True)
        self.assertEqual(results, [[True]] * 4)
        self.assertEqual([a.can_run_list[0].extend_range.length for a in self.adders], [512] * 4)

    def test_property_error_is_propagated_to_all_ranks(self):
        results = self.execute([65536] * 4, budget_error=3)
        self.assertTrue(all((isinstance(r, coord.RankConsensusError) for r in results)), results)
        self.assertTrue(all(("live budget failure" in str(r) for r in results)))
        self.assertEqual(self.forwards, [0] * 4)


if __name__ == "__main__":
    unittest.main()
