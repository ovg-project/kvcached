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

    # 8 input tokens beyond the cached prefix plus 4 output tokens: 6 blocks.
    req = types.SimpleNamespace(
        full_untruncated_fill_ids=list(range(10)), prefix_indices=[0, 1],
        sampling_params=types.SimpleNamespace(max_new_tokens=4), output_ids=[])
    scheduler, manager = _sglang_scheduler()
    with patches._sglang_demand_round(scheduler):
        assert PrefillAdder().add_one_req(req, False, None) is AddReqResult.NO_TOKEN
    assert manager.posted == [6]
    # Tokens the adder set aside for running requests' output count too.
    adder = PrefillAdder()
    adder.rem_total_token_offset = 4  # type: ignore[attr-defined]
    with patches._sglang_demand_round(scheduler):
        adder.add_one_req(req, False, None)
    assert manager.posted == [6, 8]


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
    assert manager.posted == [6, 6, 0, 0]


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
