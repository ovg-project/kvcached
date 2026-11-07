# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""The demand an engine posts for each scheduling step (#608).

#585's watermarks keep free memory from lower priorities for a
higher-priority instance. The engine posts a level every step: the new pages
the step's need would map beyond the pool's free blocks, where the need is
what the step allocated plus its first request that did not fit.
"""
from __future__ import annotations

import importlib
import sys
import threading
import types
from typing import Any
from unittest import mock

import pytest


class _FakeInternalPage:
    @staticmethod
    def get_num_blocks(page_size: int, block_mem_size: int) -> int:
        return page_size // block_mem_size


class _PageAllocator:
    def __init__(self, reserved_pages: int) -> None:
        self.reserved_pages = reserved_pages

    def get_num_reserved_pages(self) -> int:
        return self.reserved_pages

    def get_avail_physical_pages(self) -> int:
        return 50


class _WorkerControl:
    """Runtime control over two workers whose pages are 16 and 24 bytes."""

    worker_ipc = True
    sources = ({"gpu": "gpu0", "instance": "model", "generation": 7},
               {"gpu": "gpu1", "instance": "model", "generation": 8})

    def poll(self) -> tuple:
        return ({"page_bundle_bytes": 16, "available_bytes": 16 * 9},
                {"page_bundle_bytes": 24, "available_bytes": 24 * 5 + 3})


@pytest.fixture
def manager_module(monkeypatch):
    try:
        import kvcached.vmm_ops  # noqa: F401
    except ImportError:
        stub = types.ModuleType("kvcached.vmm_ops")
        stub.PageAllocator = object  # type: ignore[attr-defined]
        stub.InternalPage = _FakeInternalPage  # type: ignore[attr-defined]
        stub.kv_tensors_created = lambda group_id=0: True  # type: ignore[attr-defined]
        stub.map_to_kv_tensors = lambda *a, **k: None  # type: ignore[attr-defined]
        stub.unmap_from_kv_tensors = lambda *a, **k: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", stub)
    import kvcached.kv_cache_manager as module

    demand = mock.MagicMock()
    monkeypatch.setattr(sys.modules["kvcached.vmm_ops"], "set_kv_demand",
                        demand, raising=False)
    monkeypatch.setattr(module, "InternalPage", _FakeInternalPage)
    return module, demand


def _manager(module: Any, *, world_size: int = 1, worker_ipc: bool = False) -> Any:
    from kvcached.locks import NoOpLock

    manager = object.__new__(module.KVCacheManager)
    # 4 blocks per page; a page bundle is 4 bytes x 2 layers x K and V.
    manager.page_size = 4
    manager.block_mem_size = 1
    manager.num_layers = 2
    manager.num_kv_buffers = 2
    manager.num_avail_blocks = 1
    manager.reserved_blocks = [7]
    manager.page_allocator = _PageAllocator(reserved_pages=1)
    manager.world_size = world_size
    manager.group_id = 0
    manager._uses_worker_ipc = worker_ipc
    manager._avail_physical_pages_cache = None
    manager._avail_physical_pages_ts = 0.0
    manager._lock = NoOpLock()
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()
    return manager


def test_demand_counts_only_new_pages(manager_module):
    module, _ = manager_module
    manager = _manager(module)
    # 1 free block, 1 reserved block and a pre-mapped page of 4 need no grant.
    assert manager.demand_pages(6) == 0
    assert manager.demand_pages(7) == 1
    assert manager.demand_pages(15) == 3
    assert manager.demand_pages(0) == 0


def test_a_step_counts_each_new_block_once(manager_module):
    module, _ = manager_module
    manager = _manager(module)
    manager.reserved_blocks = [7, 8, 9]
    fresh = iter(range(100, 200))

    def alloc_impl(need, **kwargs):
        taken = manager.reserved_blocks[:need]
        manager.reserved_blocks = manager.reserved_blocks[len(taken):]
        return taken + [next(fresh) for _ in range(need - len(taken))]

    manager._alloc_impl = alloc_impl
    manager._operation_counters = None
    # try_to_reserve(): 3 reserved blocks and 2 new ones, set aside again.
    reserved = module.KVCacheManager._alloc(manager, 5)
    manager.reserved_blocks.extend(reserved)
    # The allocation that takes them maps nothing new; one more block does.
    module.KVCacheManager._alloc(manager, 6)
    module.KVCacheManager._alloc(manager, 1, _skip_wait=True)  # the null block
    assert manager.take_step_alloc_blocks() == 3
    assert manager.take_step_alloc_blocks() == 0


def test_demand_is_written_only_when_the_level_changes(manager_module):
    module, demand = manager_module
    manager = _manager(module)
    manager.post_demand(2)
    manager.post_demand(2)
    manager.post_demand(0)
    # A page bundle is 4 bytes x 2 layers x K and V.
    assert demand.call_args_list == [mock.call(32, 0), mock.call(0, 0)]


@pytest.fixture
def native(monkeypatch):
    import kvcached

    fake = mock.MagicMock()
    monkeypatch.setattr(kvcached, "_C", fake, raising=False)
    return fake


@pytest.mark.parametrize("world_size,worker_ipc", [(2, False), (1, True)])
def test_worker_ledgers_get_the_demand_in_their_page_size(
        manager_module, native, world_size, worker_ipc):
    module, demand = manager_module
    manager = _manager(module, world_size=world_size, worker_ipc=worker_ipc)
    manager.post_demand(2)  # workers not registered yet: nothing, not cached
    assert not native.publish_instance_demand.called
    manager._runtime_control = _WorkerControl()
    manager.post_demand(2)
    assert native.publish_instance_demand.call_args_list == [
        mock.call("gpu0", "model", 7, 32), mock.call("gpu1", "model", 8, 48)]
    assert not demand.called


def test_worker_ledgers_bound_the_forecast(manager_module):
    module, _ = manager_module
    manager = _manager(module, world_size=2)
    assert manager._get_cached_avail_physical_pages() == 50
    manager._runtime_control = _WorkerControl()
    manager._avail_physical_pages_cache = None
    # The second GPU can still grant 5 of its pages.
    assert manager._get_cached_avail_physical_pages() == 5


@pytest.fixture
def vllm_patches(monkeypatch):
    torch = mock.MagicMock()
    torch.__version__ = "2.6.0"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", torch.cuda)
    monkeypatch.setitem(sys.modules, "torch.utils", torch.utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension",
                        torch.utils.cpp_extension)
    monkeypatch.setitem(sys.modules, "posix_ipc", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())
    monkeypatch.delitem(sys.modules, "kvcached.integration.vllm.patches",
                        raising=False)
    patches = importlib.import_module("kvcached.integration.vllm.patches")
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    return patches


class _Coordinator:
    def __init__(self) -> None:
        self.needs: list[int] = []

    def get_num_blocks_to_allocate(self, **kwargs: Any) -> int:
        return self.needs.pop(0)


class _BlockPool:
    def __init__(self, fail: bool = False) -> None:
        self.posted: list[int] = []
        self.steps = 0
        self.fail = fail

    def begin_demand_step(self) -> None:
        self.steps += 1

    def post_demand_blocks(self, need_blocks: int) -> None:
        if self.fail:
            raise RuntimeError("ledger unavailable")
        self.posted.append(need_blocks)


def _engine(vllm_patches, monkeypatch, free_blocks: int, pool: _BlockPool):
    class KVCacheManager:
        def __init__(self) -> None:
            self.coordinator = _Coordinator()
            self.block_pool = pool

        def allocate_slots(self, request: Any, num_new_tokens: int) -> Any:
            need = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id)
            return None if need > free_blocks else ["blocks"]

    class Scheduler:
        def __init__(self) -> None:
            self.kv_cache_manager = KVCacheManager()
            self.needs: list[int] = []

        def schedule(self) -> list[Any]:
            self.kv_cache_manager.coordinator.needs = list(self.needs)
            return [self.kv_cache_manager.allocate_slots(
                types.SimpleNamespace(request_id=str(i)), 1)
                for i in range(len(self.needs))]

    manager_mod = types.ModuleType("vllm.v1.core.kv_cache_manager")
    manager_mod.KVCacheManager = KVCacheManager  # type: ignore[attr-defined]
    scheduler_mod = types.ModuleType("vllm.v1.core.sched.scheduler")
    scheduler_mod.Scheduler = Scheduler  # type: ignore[attr-defined]
    for patch, module in ((vllm_patches.KVCacheManagerAllocateSlotsPatch(), manager_mod),
                          (vllm_patches.SchedulerDemandPatch(), scheduler_mod)):
        monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
        patch.detected_version = None
        method = (patch.patch_allocate_slots if module is manager_mod
                  else patch.patch_schedule)
        assert method(module) is True
    return Scheduler()


def test_the_first_unfit_request_of_a_step_is_posted(vllm_patches, monkeypatch):
    pool = _BlockPool()
    scheduler = _engine(vllm_patches, monkeypatch, free_blocks=4, pool=pool)
    scheduler.needs = [2, 9, 12]
    assert scheduler.schedule() == [["blocks"], None, None]
    scheduler.needs = [3]
    scheduler.schedule()
    # The pool adds what each step allocated to its unfit request's blocks.
    assert pool.posted == [9, 0]
    assert pool.steps == 2


def test_a_failed_post_does_not_fail_the_step(vllm_patches, monkeypatch):
    scheduler = _engine(vllm_patches, monkeypatch, free_blocks=4,
                        pool=_BlockPool(fail=True))
    scheduler.needs = [9]
    assert scheduler.schedule() == [None]


class _DemandManager:
    """A kvcached manager stand-in: every missing block is one page."""

    def __init__(self) -> None:
        self.posted: list[int] = []
        self.allocated = 0

    def take_step_alloc_blocks(self) -> int:
        allocated, self.allocated = self.allocated, 0
        return allocated

    def demand_pages(self, need_blocks: int) -> int:
        return max(0, need_blocks)

    def post_demand(self, pages: int) -> None:
        self.posted.append(pages)


def _sglang_scheduler(evictable_tokens: int = 0):
    manager = _DemandManager()
    scheduler = types.SimpleNamespace(
        token_to_kv_pool_allocator=types.SimpleNamespace(kvcached_allocator=manager),
        tree_cache=types.SimpleNamespace(evictable_size=lambda: evictable_tokens),
        page_size=2)
    return scheduler, manager


def test_sglang_round_posts_what_it_allocated_and_its_first_unfit_request():
    from kvcached.integration.sglang import patches

    # Evictable radix cache is not subtracted: lower priorities' empty pages
    # and cache go first.
    scheduler, manager = _sglang_scheduler(evictable_tokens=4)
    manager.allocated = 2  # before the round: not this round's
    with patches._sglang_demand_round(scheduler):
        manager.allocated += 4
        with patches._sglang_demand_round(scheduler):  # nested: posts nothing
            patches._sglang_note_unfit(5)
        patches._sglang_note_unfit(9)
    assert manager.posted == [9]
    with patches._sglang_demand_round(scheduler):
        pass
    assert manager.posted == [9, 0]


def test_sglang_prefill_budget_miss_is_recorded(monkeypatch):
    import enum

    from kvcached.integration.sglang import patches

    class AddReqResult(enum.Enum):
        CONTINUE = 1
        NO_TOKEN = 2

    class PrefillAdder:
        page_size = 2

        def add_one_req(self, req: Any, has_chunked_req: bool,
                        truncation_align_size: Any) -> Any:
            return AddReqResult.NO_TOKEN

    module = types.ModuleType("sglang.srt.managers.schedule_policy")
    module.PrefillAdder = PrefillAdder  # type: ignore[attr-defined]
    patch = patches.PrefillAdderDemandPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patch.patch_add_one_req(module) is True

    # 8 input + 4 output + 2 alignment tokens, strictly below available: 8 blocks.
    req = types.SimpleNamespace(
        full_untruncated_fill_ids=list(range(10)), prefix_indices=[0, 1],
        sampling_params=types.SimpleNamespace(max_new_tokens=4), output_ids=[])
    scheduler, manager = _sglang_scheduler()
    with patches._sglang_demand_round(scheduler):
        assert PrefillAdder().add_one_req(req, False, None) is AddReqResult.NO_TOKEN
    assert manager.posted == [8]
    # Tokens the adder set aside for running requests' output count too.
    adder = PrefillAdder()
    adder.rem_total_token_offset = 4  # type: ignore[attr-defined]
    with patches._sglang_demand_round(scheduler):
        adder.add_one_req(req, False, None)
    assert manager.posted == [8, 10]


def test_sglang_decode_gate_miss_is_recorded():
    from kvcached.integration.sglang import patches

    allocator = types.SimpleNamespace(kvcached_allocator=types.SimpleNamespace(
        try_to_reserve=lambda need, packed=False: False))
    scheduler, manager = _sglang_scheduler()
    with patches._sglang_demand_round(scheduler):
        assert not patches._sglang_reserve_decode_capacity(
            allocator, scheduler.tree_cache, num_tokens=6, need_blocks=3)
    assert manager.posted == [3]


def test_sglang_carries_a_prefill_miss_over_rounds_that_skip_prefill(monkeypatch):
    import enum

    from kvcached.integration.sglang import patches

    class AddReqResult(enum.Enum):
        CONTINUE = 1
        NO_TOKEN = 2

    class PrefillAdder:
        page_size = 2
        result = AddReqResult.NO_TOKEN

        def add_one_req(self, req: Any, has_chunked_req: bool,
                        truncation_align_size: Any) -> Any:
            return self.result

    module = types.ModuleType("sglang.srt.managers.schedule_policy")
    module.PrefillAdder = PrefillAdder  # type: ignore[attr-defined]
    patch = patches.PrefillAdderDemandPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patch.patch_add_one_req(module) is True
    req = types.SimpleNamespace(
        full_untruncated_fill_ids=list(range(10)), prefix_indices=[0, 1],
        sampling_params=types.SimpleNamespace(max_new_tokens=4), output_ids=[])
    scheduler, manager = _sglang_scheduler()
    scheduler.waiting_queue = [req]
    with patches._sglang_demand_round(scheduler):
        PrefillAdder().add_one_req(req, False, None)
    # SGLang then skips prefill while its batch is full; the request waits.
    with patches._sglang_demand_round(scheduler):
        pass
    PrefillAdder.result = AddReqResult.CONTINUE
    with patches._sglang_demand_round(scheduler):
        PrefillAdder().add_one_req(req, False, None)
    scheduler.waiting_queue = []
    with patches._sglang_demand_round(scheduler):
        pass
    assert manager.posted == [8, 8, 0, 0]


def test_sglang_retries_prefill_once_the_pool_backs_its_last_miss():
    from kvcached.integration.sglang import patches

    scheduler, manager = _sglang_scheduler()
    manager.available_size = lambda: manager.available  # type: ignore[attr-defined]
    manager.available = 5  # type: ignore[attr-defined]
    scheduler.waiting_queue = ["req"]
    scheduler.running_batch = types.SimpleNamespace(batch_is_full=True)
    scheduler._kvcached_prefill_miss = 6
    with patches._sglang_demand_round(scheduler):
        assert scheduler.running_batch.batch_is_full  # 5 blocks cannot back 6
    # Another instance released memory: SGLang may try the prefill again.
    manager.available = 6  # type: ignore[attr-defined]
    with patches._sglang_demand_round(scheduler):
        assert not scheduler.running_batch.batch_is_full


@pytest.mark.parametrize("page_size,offset,mamba_reserve", [(2, 0, 0), (2, 4, 0),
                                                           (2, 0, 4), (1, 0, 0),
                                                           (4, 1, 3)])
def test_sglang_native_admission_boundary_posts_and_waits(
        monkeypatch, page_size, offset, mamba_reserve):
    """Mirror the native >= total-budget gate, including shared Mamba cost."""
    import enum

    from kvcached.integration.sglang import patches

    class AddReqResult(enum.Enum):
        CONTINUE = 1
        NO_TOKEN = 2

    req = types.SimpleNamespace(
        full_untruncated_fill_ids=list(range(10)), prefix_indices=[0, 1],
        sampling_params=types.SimpleNamespace(max_new_tokens=4), output_ids=[],
        mamba_pool_idx=None)
    scheduler, manager = _sglang_scheduler()
    scheduler.page_size = page_size
    scheduler.waiting_queue = [req]
    scheduler.running_batch = types.SimpleNamespace(batch_is_full=True)
    required_tokens = 8 + 4 + page_size + mamba_reserve + offset
    manager.available = required_tokens // page_size
    manager.available_size = lambda: manager.available
    manager.demand_pages = lambda blocks: max(0, blocks - manager.available)

    class PrefillAdder:
        def __init__(self):
            self.page_size = page_size
            self.rem_total_token_offset = offset

        def _mamba_gap_budget_for_req(self, req):
            return mamba_reserve if req.mamba_pool_idx is None else 0

        def add_one_req(self, req, *args):
            total = (len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
                     + min(max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
                           4096) + self.page_size + self._mamba_gap_budget_for_req(req))
            if total >= manager.available * page_size - self.rem_total_token_offset:
                return AddReqResult.NO_TOKEN
            return AddReqResult.CONTINUE

    module = types.ModuleType("sglang.srt.managers.schedule_policy")
    module.PrefillAdder = PrefillAdder  # type: ignore[attr-defined]
    patch = patches.PrefillAdderDemandPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patch.patch_add_one_req(module)
    with patches._sglang_demand_round(scheduler):
        assert PrefillAdder().add_one_req(req) is AddReqResult.NO_TOKEN
    assert manager.posted == [1]
    assert scheduler._kvcached_prefill_miss == required_tokens // page_size + 1
    with patches._sglang_demand_round(scheduler):
        assert scheduler.running_batch.batch_is_full
    manager.available += 1
    with patches._sglang_demand_round(scheduler):
        assert not scheduler.running_batch.batch_is_full
        assert PrefillAdder().add_one_req(req) is AddReqResult.CONTINUE
    assert scheduler._kvcached_prefill_miss is None


def test_sglang_captures_mamba_reserve_before_admission_mutates_request(monkeypatch):
    from kvcached.integration.sglang import patches

    class PrefillAdder:
        page_size = 2
        rem_total_token_offset = 0

        def _mamba_gap_budget_for_req(self, req):
            return 4 if req.mamba_pool_idx is None else 0

        def add_one_req(self, req):
            req.mamba_pool_idx = 1
            self.rem_total_token_offset = 20
            return types.SimpleNamespace(name="NO_TOKEN")

    module = types.ModuleType("sglang.srt.managers.schedule_policy")
    module.PrefillAdder = PrefillAdder  # type: ignore[attr-defined]
    patch = patches.PrefillAdderDemandPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patch.patch_add_one_req(module)
    scheduler, manager = _sglang_scheduler()
    req = types.SimpleNamespace(
        full_untruncated_fill_ids=list(range(8)), prefix_indices=[],
        sampling_params=types.SimpleNamespace(max_new_tokens=4), output_ids=[],
        mamba_pool_idx=None)
    with patches._sglang_demand_round(scheduler):
        PrefillAdder().add_one_req(req)
    assert manager.posted == [10]  # ceil((8 + 4 + 2 + 4 + 1) / 2)


@pytest.mark.parametrize("admitted", [False, True])
@pytest.mark.parametrize("ignore_eos", [False, True])
def test_sglang_keeps_ignore_eos_estimate_and_does_not_count_admitted_req(
        monkeypatch, admitted, ignore_eos):
    from kvcached.integration.sglang import patches

    class PrefillAdder:
        page_size = 2
        tree_cache = types.SimpleNamespace(disable=True)

        def __init__(self):
            self.can_run_list = []

        def add_one_req(self, req):
            if admitted:
                self.can_run_list.append(req)
            # budget_state() can say NO_TOKEN after successfully admitting req.
            return types.SimpleNamespace(name="NO_TOKEN")

    module = types.ModuleType("sglang.srt.managers.schedule_policy")
    module.PrefillAdder = PrefillAdder  # type: ignore[attr-defined]
    patch = patches.PrefillAdderDemandPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patch.patch_add_one_req(module)
    scheduler, manager = _sglang_scheduler()
    req = types.SimpleNamespace(
        full_untruncated_fill_ids=list(range(8)), prefix_indices=[],
        sampling_params=types.SimpleNamespace(max_new_tokens=4, ignore_eos=ignore_eos), output_ids=[])
    with patches._sglang_demand_round(scheduler):
        PrefillAdder().add_one_req(req)
    need = 6 if ignore_eos else 8
    assert manager.posted == ([0] if admitted else [need])
    assert scheduler._kvcached_prefill_miss == (None if admitted else need)


def test_sglang_admitted_no_token_retries_without_reposting_admitted_demand(monkeypatch):
    from kvcached.integration.sglang import patches

    scheduler, manager = _sglang_scheduler()
    manager.available = 8
    manager.available_size = lambda: manager.available
    manager.demand_pages = lambda blocks: max(0, blocks - manager.available)
    scheduler.running_batch = types.SimpleNamespace(batch_is_full=False)
    req_a = types.SimpleNamespace(
        full_untruncated_fill_ids=list(range(9)), prefix_indices=[],
        sampling_params=types.SimpleNamespace(max_new_tokens=4), output_ids=[])
    req_b = types.SimpleNamespace(
        full_untruncated_fill_ids=[0], prefix_indices=[],
        sampling_params=types.SimpleNamespace(max_new_tokens=1), output_ids=[])
    scheduler.waiting_queue = [req_a, req_b]

    class PrefillAdder:
        page_size = 2

        def __init__(self, offset=0):
            self.can_run_list = []
            self.rem_total_token_offset = offset

        def add_one_req(self, req):
            total = len(req.full_untruncated_fill_ids) + req.sampling_params.max_new_tokens + 2
            if total >= manager.available * 2 - self.rem_total_token_offset:
                return types.SimpleNamespace(name="NO_TOKEN")
            self.can_run_list.append(req)
            # A: initial total15 < available16, but ceil(input9)=10 plus
            # output4 and page2 consumes the whole prefill budget.
            return types.SimpleNamespace(name="NO_TOKEN" if req is req_a else "CONTINUE")

    module = types.ModuleType("sglang.srt.managers.schedule_policy")
    module.PrefillAdder = PrefillAdder  # type: ignore[attr-defined]
    patch = patches.PrefillAdderDemandPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patch.patch_add_one_req(module)
    with patches._sglang_demand_round(scheduler):
        assert PrefillAdder().add_one_req(req_a).name == "NO_TOKEN"
        scheduler.running_batch.batch_is_full = True
        scheduler.waiting_queue = [req_b]
        manager.allocated, manager.available = 5, 3
    assert manager.posted == [2]  # Only five allocated blocks minus three free.
    assert scheduler._kvcached_prefill_miss is None
    # A one-shot attempt measures B, rather than waiting for A's eight blocks.
    with patches._sglang_demand_round(scheduler):
        assert not scheduler.running_batch.batch_is_full
        assert PrefillAdder(offset=4).add_one_req(req_b).name == "NO_TOKEN"
        scheduler.running_batch.batch_is_full = True
    assert scheduler._kvcached_prefill_miss == 5
    assert not scheduler._kvcached_prefill_retry
    with patches._sglang_demand_round(scheduler):
        assert scheduler.running_batch.batch_is_full  # No repeated blind retries.
    manager.available = 5  # Peer release, enough for B but still not A.
    with patches._sglang_demand_round(scheduler):
        assert not scheduler.running_batch.batch_is_full
        assert PrefillAdder(offset=4).add_one_req(req_b).name == "CONTINUE"
        scheduler.waiting_queue = []
    assert scheduler._kvcached_prefill_miss is None
    assert not scheduler._kvcached_prefill_retry


def test_sglang_admitted_retry_is_disabled_under_tensor_parallelism(monkeypatch):
    from kvcached.integration.sglang import patches

    scheduler, manager = _sglang_scheduler()
    scheduler.waiting_queue = ["next request"]
    scheduler.running_batch = types.SimpleNamespace(batch_is_full=True)
    scheduler._kvcached_prefill_retry = True
    manager.available_size = lambda: 100
    monkeypatch.setattr(patches, "_sglang_parallel_size", lambda scheduler: 2)
    patches._sglang_retry_prefill_if_backed(scheduler, manager)
    assert scheduler.running_batch.batch_is_full
    assert scheduler._kvcached_prefill_retry  # No rank-local scheduling decision.
    monkeypatch.setattr(patches, "_sglang_parallel_size", lambda scheduler: 1)
    patches._sglang_retry_prefill_if_backed(scheduler, manager)
    assert not scheduler.running_batch.batch_is_full
    assert not scheduler._kvcached_prefill_retry


@pytest.mark.parametrize("cache_kind", ["plain", "full", "hybrid", "swa"])
@pytest.mark.parametrize("cache_tokens", [1, 4])
def test_sglang_retry_counts_the_matching_evictable_cache(monkeypatch, cache_kind, cache_tokens):
    from kvcached.integration.sglang import patches

    scheduler, manager = _sglang_scheduler(evictable_tokens=cache_tokens)
    initial_free = (14 - cache_tokens) // 2
    manager.available = initial_free
    manager.available_size = lambda: manager.available
    manager.demand_pages = lambda blocks: max(0, blocks - manager.available)
    scheduler.running_batch = types.SimpleNamespace(batch_is_full=True)
    # Poison other cache counters to detect selecting the wrong pool/unit.
    counter = {"plain": "evictable_size", "full": "full_evictable_size",
               "hybrid": "full_evictable_size", "swa": "swa_evictable_size"}[cache_kind]
    for name in ("evictable_size", "full_evictable_size", "swa_evictable_size"):
        setattr(scheduler.tree_cache, name, lambda: 1000)
    setattr(scheduler.tree_cache, counter, lambda: cache_tokens)
    available_counter = {"plain": "available_size", "full": "available_size",
                         "hybrid": "full_available_size", "swa": "swa_available_size"}[cache_kind]
    setattr(scheduler.token_to_kv_pool_allocator, available_counter,
            lambda: manager.available * 2)
    req = types.SimpleNamespace(
        full_untruncated_fill_ids=list(range(8)), prefix_indices=[],
        sampling_params=types.SimpleNamespace(max_new_tokens=4), output_ids=[])
    scheduler.waiting_queue = [req]

    class PrefillAdder:
        page_size = 2
        is_all_swa = cache_kind == "swa"
        is_hybrid_swa = cache_kind == "hybrid"
        is_hybrid_ssm_cache = cache_kind == "full"
        rem_swa_token_offset = 0
        rem_swa_tokens = 1000

        @property
        def rem_total_tokens(self):
            return manager.available * 2 + getattr(scheduler.tree_cache, counter)()

        def add_one_req(self, req):
            available = manager.available * 2 + getattr(scheduler.tree_cache, counter)()
            return types.SimpleNamespace(name="NO_TOKEN" if 8 + 4 + 2 >= available else "CONTINUE")

    module = types.ModuleType("sglang.srt.managers.schedule_policy")
    module.PrefillAdder = PrefillAdder  # type: ignore[attr-defined]
    patch = patches.PrefillAdderDemandPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patch.patch_add_one_req(module)
    with patches._sglang_demand_round(scheduler):
        assert PrefillAdder().add_one_req(req).name == "NO_TOKEN"
    # Demand still counts radix-backed need: eviction is not free demand.
    assert manager.posted == [8 - initial_free]
    with patches._sglang_demand_round(scheduler):
        assert scheduler.running_batch.batch_is_full
    manager.available += 1  # Cross native token gate, still fewer than eight free blocks.
    with patches._sglang_demand_round(scheduler):
        assert not scheduler.running_batch.batch_is_full
        assert PrefillAdder().add_one_req(req).name == "CONTINUE"



def test_sglang_retry_respects_native_allocator_size_cap():
    from kvcached.integration.sglang import patches

    scheduler, manager = _sglang_scheduler(evictable_tokens=4)
    manager.available_size = lambda: 100  # Raw manager count is not the native cap.
    scheduler._kvcached_prefill_retry_budget = [("available_size", "evictable_size", 14)]
    scheduler.token_to_kv_pool_allocator.available_size = lambda: 10
    assert not patches._sglang_prefill_miss_is_backed(scheduler, manager, 8)
    scheduler.token_to_kv_pool_allocator.available_size = lambda: 12
    assert patches._sglang_prefill_miss_is_backed(scheduler, manager, 8)


@pytest.mark.parametrize("growth_source", ["allocator", "cache"])
@pytest.mark.parametrize("offset", [0, 0.5])
def test_sglang_prefix_unlock_does_not_retry_without_budget_growth(monkeypatch, growth_source, offset):
    from kvcached.integration.sglang import patches

    scheduler, manager = _sglang_scheduler()
    manager.available = 6
    manager.available_size = lambda: manager.available
    manager.demand_pages = lambda blocks: max(0, blocks - manager.available)
    cache = types.SimpleNamespace(tokens=4, pinned=False)
    cache.evictable_size = lambda: 0 if cache.pinned else cache.tokens
    scheduler.tree_cache = cache
    scheduler.token_to_kv_pool_allocator.available_size = lambda: manager.available * 2
    scheduler.running_batch = types.SimpleNamespace(batch_is_full=True)
    req = types.SimpleNamespace(
        full_untruncated_fill_ids=list(range(12)), prefix_indices=list(range(4)),
        sampling_params=types.SimpleNamespace(max_new_tokens=4), output_ids=[])
    scheduler.waiting_queue = [req]
    calls = []

    class PrefillAdder:
        page_size = 2
        rem_total_token_offset = offset

        @property
        def rem_total_tokens(self):
            return manager.available * 2 + cache.evictable_size() - offset

        def add_one_req(self, req):
            calls.append(req)
            if 8 + 4 + 2 >= self.rem_total_tokens:
                return types.SimpleNamespace(name="NO_TOKEN")
            cache.pinned = True
            try:
                result = "NO_TOKEN" if 8 + 4 + 2 >= self.rem_total_tokens else "CONTINUE"
                return types.SimpleNamespace(name=result)
            finally:
                cache.pinned = False

    module = types.ModuleType("sglang.srt.managers.schedule_policy")
    module.PrefillAdder = PrefillAdder  # type: ignore[attr-defined]
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patches.PrefillAdderDemandPatch().patch_add_one_req(module)
    with patches._sglang_demand_round(scheduler):
        assert PrefillAdder().add_one_req(req).name == "NO_TOKEN"
    # Prefix is evictable again, but retrying would pin and reject it again.
    for _ in range(3):
        with patches._sglang_demand_round(scheduler):
            if not scheduler.running_batch.batch_is_full:
                PrefillAdder().add_one_req(req)
                scheduler.running_batch.batch_is_full = True
    assert len(calls) == 1
    if growth_source == "allocator":
        manager.available += 2
    else:
        # Newly evictable *other* prefixes remain available under this lock.
        cache.tokens += 4
        cache.evictable_size = lambda: cache.tokens - (4 if cache.pinned else 0)
    with patches._sglang_demand_round(scheduler):
        assert not scheduler.running_batch.batch_is_full
        assert PrefillAdder().add_one_req(req).name == "CONTINUE"
    assert len(calls) == 2


@pytest.mark.parametrize("change", ["decode", "ratio", "legacy_ratio"])
@pytest.mark.parametrize("prefix_tokens", [0, 4])
@pytest.mark.parametrize("initial_ratio", [1.0, 0.5])
def test_sglang_retry_updates_running_reservation(monkeypatch, change, prefix_tokens, initial_ratio):
    from kvcached.integration.sglang import patches

    scheduler, manager = _sglang_scheduler()
    manager.available_size = lambda: 6
    manager.demand_pages = lambda blocks: max(0, blocks - 6)
    scheduler.token_to_kv_pool_allocator.available_size = lambda: 12
    cache = types.SimpleNamespace(pinned=False)
    cache.evictable_size = lambda: 0 if cache.pinned else prefix_tokens
    scheduler.tree_cache = cache
    running = types.SimpleNamespace(sampling_params=types.SimpleNamespace(max_new_tokens=4), output_ids=[])
    scheduler.running_batch = types.SimpleNamespace(batch_is_full=True, reqs=[running])
    if change == "legacy_ratio":
        scheduler.new_token_ratio = initial_ratio
    else:
        scheduler.new_token_ratio_tracker = types.SimpleNamespace(current=initial_ratio)
    req = types.SimpleNamespace(
        full_untruncated_fill_ids=list(range(4 + prefix_tokens)),
        prefix_indices=list(range(prefix_tokens)),
        sampling_params=types.SimpleNamespace(max_new_tokens=4), output_ids=[])
    scheduler.waiting_queue = [req]

    class PrefillAdder:
        page_size = 2

        def __init__(self):
            self.new_token_ratio = (scheduler.new_token_ratio if change == "legacy_ratio"
                                    else scheduler.new_token_ratio_tracker.current)
            self.rem_total_token_offset = self._get_running_request_total_token_offset(running)

        def _get_running_request_total_token_offset(self, request):
            return min(request.sampling_params.max_new_tokens - len(request.output_ids), 4096) * self.new_token_ratio

        @property
        def rem_total_tokens(self):
            return 12 + cache.evictable_size() - self.rem_total_token_offset

        def add_one_req(self, req):
            if 10 >= self.rem_total_tokens:
                return types.SimpleNamespace(name="NO_TOKEN")
            cache.pinned = True
            try:
                return types.SimpleNamespace(name="NO_TOKEN" if 10 >= self.rem_total_tokens else "CONTINUE")
            finally:
                cache.pinned = False

    module = types.ModuleType("sglang.srt.managers.schedule_policy")
    module.PrefillAdder = PrefillAdder  # type: ignore[attr-defined]
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patches.PrefillAdderDemandPatch().patch_add_one_req(module)
    with patches._sglang_demand_round(scheduler):
        assert PrefillAdder().add_one_req(req).name == "NO_TOKEN"
    # Neither restored evictability nor unchanged running requests wake it.
    for _ in range(3):
        with patches._sglang_demand_round(scheduler):
            assert scheduler.running_batch.batch_is_full
    if initial_ratio == 1.0:
        running.output_ids.append(0)  # Reservation falls, but still cannot fit.
        with patches._sglang_demand_round(scheduler):
            assert not scheduler.running_batch.batch_is_full
            assert PrefillAdder().add_one_req(req).name == "NO_TOKEN"
            scheduler.running_batch.batch_is_full = True
        with patches._sglang_demand_round(scheduler):
            assert scheduler.running_batch.batch_is_full  # New baseline consumed.
    if change == "decode":
        running.output_ids.extend([1, 2, 3] if initial_ratio == 1.0 else [1])
    elif change == "ratio":
        scheduler.new_token_ratio_tracker.current = 0.25 if initial_ratio == 1.0 else 0.375
    else:
        scheduler.new_token_ratio = 0.25 if initial_ratio == 1.0 else 0.375
    # Raw allocator/cache capacity has not changed, only native reservation.
    with patches._sglang_demand_round(scheduler):
        assert not scheduler.running_batch.batch_is_full
        assert PrefillAdder().add_one_req(req).name == "CONTINUE"


def test_sglang_cache_disabled_ignore_eos_keeps_legacy_retry(monkeypatch):
    from kvcached.integration.sglang import patches

    scheduler, manager = _sglang_scheduler()
    manager.available_size = lambda: 5000
    manager.demand_pages = lambda blocks: max(0, blocks - 5000)
    scheduler.token_to_kv_pool_allocator.available_size = lambda: 10000
    scheduler.running_batch = types.SimpleNamespace(batch_is_full=True)
    scheduler.tree_cache.disable = True
    req = types.SimpleNamespace(
        full_untruncated_fill_ids=list(range(8)), prefix_indices=[],
        sampling_params=types.SimpleNamespace(max_new_tokens=9000, ignore_eos=True), output_ids=[])
    scheduler.waiting_queue = [req]
    lifetime_gate = types.SimpleNamespace(can_fit=False)

    class PrefillAdder:
        page_size = 2
        tree_cache = scheduler.tree_cache
        rem_total_tokens = 10000

        def add_one_req(self, req):
            # Native ignore_eos checks the lifetimes of all running requests,
            # not the clipped per-candidate normal-admission estimate.
            return types.SimpleNamespace(name="CONTINUE" if lifetime_gate.can_fit else "NO_TOKEN")

    module = types.ModuleType("sglang.srt.managers.schedule_policy")
    module.PrefillAdder = PrefillAdder  # type: ignore[attr-defined]
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patches.PrefillAdderDemandPatch().patch_add_one_req(module)
    with patches._sglang_demand_round(scheduler):
        assert PrefillAdder().add_one_req(req).name == "NO_TOKEN"
    assert scheduler._kvcached_prefill_miss == (8 + 4096) // 2
    assert scheduler._kvcached_prefill_retry_budget is None
    assert scheduler._kvcached_prefill_retry_reservation is None
    # Native lifetime eligibility can change without raw pool growth.
    lifetime_gate.can_fit = True
    with patches._sglang_demand_round(scheduler):
        assert not scheduler.running_batch.batch_is_full
        assert PrefillAdder().add_one_req(req).name == "CONTINUE"


def test_sglang_admitted_retry_survives_empty_queue(monkeypatch):
    from kvcached.integration.sglang import patches

    scheduler, manager = _sglang_scheduler()
    scheduler.running_batch = types.SimpleNamespace(batch_is_full=True)
    req = types.SimpleNamespace(full_untruncated_fill_ids=[1], prefix_indices=[],
                                sampling_params=types.SimpleNamespace(max_new_tokens=1), output_ids=[])
    scheduler.waiting_queue = [req]

    class PrefillAdder:
        def __init__(self):
            self.can_run_list = []

        def add_one_req(self, req):
            self.can_run_list.append(req)
            return types.SimpleNamespace(name="NO_TOKEN")

    module = types.ModuleType("sglang.srt.managers.schedule_policy")
    module.PrefillAdder = PrefillAdder  # type: ignore[attr-defined]
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patches.PrefillAdderDemandPatch().patch_add_one_req(module)
    with patches._sglang_demand_round(scheduler):
        PrefillAdder().add_one_req(req)
        scheduler.waiting_queue = []
    for _ in range(3):
        with patches._sglang_demand_round(scheduler):
            assert scheduler._kvcached_prefill_retry
            assert scheduler.running_batch.batch_is_full
    scheduler.waiting_queue = [req]
    with patches._sglang_demand_round(scheduler):
        assert not scheduler.running_batch.batch_is_full
        assert not scheduler._kvcached_prefill_retry
        scheduler.running_batch.batch_is_full = True
    with patches._sglang_demand_round(scheduler):
        assert scheduler.running_batch.batch_is_full  # One shot, not an always-retry flag.


@pytest.mark.parametrize("pool", ["swa", "ring", "mamba"])
def test_sglang_retry_on_independent_native_budget_progress(monkeypatch, pool):
    from kvcached.integration.sglang import patches

    scheduler, manager = _sglang_scheduler()
    manager.available_size = lambda: 50
    scheduler.running_batch = types.SimpleNamespace(batch_is_full=True)
    capacity = types.SimpleNamespace(value=2)
    allocator = scheduler.token_to_kv_pool_allocator
    allocator.full_available_size = lambda: 100
    allocator.available_size = lambda: 100
    allocator.swa_available_size = lambda: capacity.value
    allocator.mamba_allocator = types.SimpleNamespace(schedulable_available_size=lambda: capacity.value)
    scheduler.tree_cache.full_evictable_size = lambda: 0
    scheduler.tree_cache.swa_evictable_size = lambda: 1000 if pool == "ring" else 0
    scheduler.tree_cache.mamba_evictable_size = lambda: 0
    req = types.SimpleNamespace(full_untruncated_fill_ids=[1], prefix_indices=[],
                                sampling_params=types.SimpleNamespace(max_new_tokens=1), output_ids=[])
    scheduler.waiting_queue = [req]

    class PrefillAdder:
        page_size = 2
        rem_total_tokens = 100
        is_hybrid_swa = pool in ("swa", "ring")
        is_hybrid_ssm_cache = pool == "mamba"
        _swa_req_ring = pool == "ring"
        rem_swa_token_offset = 0

        @property
        def rem_swa_tokens(self):
            return capacity.value

        @property
        def rem_mamba_slots(self):
            return capacity.value if pool == "mamba" else None

        def add_one_req(self, req):
            return types.SimpleNamespace(name="CONTINUE" if capacity.value >= 6 else "NO_TOKEN")

    module = types.ModuleType("sglang.srt.managers.schedule_policy")
    module.PrefillAdder = PrefillAdder  # type: ignore[attr-defined]
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patches.PrefillAdderDemandPatch().patch_add_one_req(module)
    with patches._sglang_demand_round(scheduler):
        assert PrefillAdder().add_one_req(req).name == "NO_TOKEN"
    with patches._sglang_demand_round(scheduler):
        assert scheduler.running_batch.batch_is_full
    capacity.value = 4  # Progress, but not yet sufficient for native admission.
    with patches._sglang_demand_round(scheduler):
        assert not scheduler.running_batch.batch_is_full
        assert PrefillAdder().add_one_req(req).name == "NO_TOKEN"
        scheduler.running_batch.batch_is_full = True
    with patches._sglang_demand_round(scheduler):
        assert scheduler.running_batch.batch_is_full  # Fresh miss consumes progress.
    capacity.value = 6
    with patches._sglang_demand_round(scheduler):
        assert not scheduler.running_batch.batch_is_full
        assert PrefillAdder().add_one_req(req).name == "CONTINUE"


@pytest.mark.parametrize("guard", ["promised", "surplus", "shrink"])
def test_demand_excludes_pages_waiting_for_physical_release(manager_module, guard):
    module, _ = manager_module
    manager = _manager(module)
    manager.num_avail_blocks = 4
    manager.reserved_blocks = []
    manager.page_allocator.reserved_pages = 0
    manager.avail_pages = {1: types.SimpleNamespace(num_free_blocks=lambda: 4)}
    manager._retired_pages = [(1, [1])]
    manager.in_shrink = guard == "shrink"
    if guard == "promised":
        manager._priority_retired_ids = {1}
    elif guard == "surplus":
        manager._priority_retained_ids = {1}
    assert manager.demand_pages(4) == 1
    manager._priority_retired_ids = set()
    manager._retired_pages = []
    manager.in_shrink = False
    assert manager.demand_pages(4) == 0  # Barrier-safe surplus is reusable.


def test_demand_excludes_native_reserve_awaiting_priority_refund(manager_module):
    module, _ = manager_module
    manager = _manager(module)
    manager.num_avail_blocks = 0
    manager.reserved_blocks = []
    manager._priority_reserve_debt = 1
    assert manager.demand_pages(4) == 1
    manager._priority_reserve_debt = 0
    assert manager.demand_pages(4) == 0


@pytest.mark.parametrize("page_size,prefix,end,offset,allocated,need,miss", [
    (1, 0, 4, 6, 4, 11, 11),  # Jeff's accepted A, then rejected B.
    (2, 0, 3, 7, 2, 7, 7),    # Round each admitted request's actual pages.
    (2, 1, 2, 5, 0, 6, 6),    # Reuse the partial prefix page; keep padding.
    (1, 0, 4, 8.5, 4, 13, 13),  # Keep running output reservation.
    (1, 0, 4, 6, 0, 11, 11),  # No allocation: no overlap can be deducted.
])
def test_sglang_admitted_input_is_counted_once(
        monkeypatch, page_size, prefix, end, offset, allocated, need, miss):
    from kvcached.integration.sglang import patches

    accepted = types.SimpleNamespace(extend_range=types.SimpleNamespace(start=prefix, end=end))
    rejected = types.SimpleNamespace(
        full_untruncated_fill_ids=[0, 1], prefix_indices=[], output_ids=[],
        sampling_params=types.SimpleNamespace(max_new_tokens=1))

    class PrefillAdder:
        def __init__(self):
            self.page_size = page_size
            self.rem_total_token_offset = offset
            self.can_run_list = [accepted]

        def add_one_req(self, req):
            return types.SimpleNamespace(name="NO_TOKEN")

    module = types.ModuleType("sglang.srt.managers.schedule_policy")
    module.PrefillAdder = PrefillAdder  # type: ignore[attr-defined]
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patches.PrefillAdderDemandPatch().patch_add_one_req(module)
    scheduler, manager = _sglang_scheduler()
    scheduler.waiting_queue = [rejected]
    with patches._sglang_demand_round(scheduler):
        PrefillAdder().add_one_req(rejected)
        manager.allocated = allocated
    assert manager.posted == [need]
    assert scheduler._kvcached_prefill_miss == miss  # Keep the original admission estimate.
    with patches._sglang_demand_round(scheduler):
        pass
    assert manager.posted == [need, need - allocated]
    scheduler.waiting_queue = []
    with patches._sglang_demand_round(scheduler):
        pass
    assert manager.posted[-1] == 0


def test_sglang_ignore_eos_retry_keeps_output_reservations_after_admitted_input(monkeypatch):
    from kvcached.integration.sglang import patches

    accepted = types.SimpleNamespace(extend_range=types.SimpleNamespace(start=0, end=40))
    rejected = types.SimpleNamespace(
        full_untruncated_fill_ids=list(range(40)), prefix_indices=[], output_ids=[],
        sampling_params=types.SimpleNamespace(max_new_tokens=5, ignore_eos=True))

    class PrefillAdder:
        page_size = 1
        tree_cache = types.SimpleNamespace(disable=True)
        # Existing running output 20, accepted input 40 + output 10 + page 1.
        rem_total_token_offset = 71
        can_run_list = [accepted]

        def add_one_req(self, req):
            return types.SimpleNamespace(name="NO_TOKEN")

    module = types.ModuleType("sglang.srt.managers.schedule_policy")
    module.PrefillAdder = PrefillAdder  # type: ignore[attr-defined]
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    assert patches.PrefillAdderDemandPatch().patch_add_one_req(module)
    scheduler, manager = _sglang_scheduler()
    scheduler.waiting_queue = [rejected]
    scheduler.running_batch = types.SimpleNamespace(batch_is_full=True)
    manager.available = 60
    manager.available_size = lambda: manager.available
    manager.demand_pages = lambda blocks: max(0, blocks - manager.available)
    with patches._sglang_demand_round(scheduler):
        PrefillAdder().add_one_req(rejected)
        manager.allocated = 40
    assert scheduler._kvcached_prefill_miss == 116
    assert scheduler._kvcached_prefill_demand == 76
    assert scheduler._kvcached_prefill_retry_budget is None
    with patches._sglang_demand_round(scheduler):
        assert scheduler.running_batch.batch_is_full
    assert manager.posted[-1] == 16
    manager.available = 75
    with patches._sglang_demand_round(scheduler):
        assert scheduler.running_batch.batch_is_full
    manager.available = 76
    with patches._sglang_demand_round(scheduler):
        assert not scheduler.running_batch.batch_is_full


def test_sglang_prefill_input_pages_include_legacy_and_chunked_ranges():
    from kvcached.integration.sglang import patches

    adder = types.SimpleNamespace(can_run_list=[
        types.SimpleNamespace(prefix_indices=[0, 1], extend_input_len=3),
        types.SimpleNamespace(extend_range=types.SimpleNamespace(start=4, end=6)),
    ])
    assert patches._sglang_prefill_input_blocks(adder, 2) == 3


def test_sglang_decode_miss_does_not_deduct_later_prefill_inputs():
    from kvcached.integration.sglang import patches

    scheduler, manager = _sglang_scheduler()
    with patches._sglang_demand_round(scheduler):
        patches._sglang_note_unfit(5)
        patches._sglang_note_unfit(11, allocated_overlap=4)
        manager.allocated = 4
    assert manager.posted == [9]
