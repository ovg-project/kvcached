# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""A kvcached pool miss after admission must reach SGLang as a retry.

SGLang's allocators report a miss by returning None, and every caller inside
``alloc_for_extend`` turns that None into a fatal RuntimeError or assert,
because native pools are static and the admission budget cannot be wrong.
Under kvcached the budget is a snapshot of device-wide physical memory that a
second pool in the same process (hybrid attention plus Mamba, #547) or another
process (#467) can consume first. These tests pin the translation into
SGLang's own retry path, retraction to the waiting queue, and pin that it
stays as narrow as the vLLM one (#453, #511): only a kvcached capacity miss is
translated, and only what the failed batch was handed out is released.
"""
from __future__ import annotations

import contextlib
import inspect
import sys
import types
from typing import Any, List, Optional

import pytest
import torch

from kvcached.integration.sglang import patches
from kvcached.integration.sglang.patches import (
    ElasticAllocatorPatch,
    ElasticMambaPoolPatch,
    ElasticSWAAllocatorPatch,
    ScheduleBatchCapacityMissPatch,
    SchedulerCapacityMissPatch,
    SGLangPrefillCapacityMiss,
    _sglang_batch_allocation_zone,
)
from kvcached.utils import KVCachePoolExhausted

MAMBA_PING_PONG = "Not enough space for mamba ping pong idx"
MAMBA_SLOT = "Not enough space for mamba cache"
TOKEN_OOM = "Out of memory. Try to lower your batch size."
ROW_OOM = "alloc_req_slots runs out of memory"


# --------------------------------------------------------------------------
# kvcached side: a KVCacheManager stand-in over block ids 1..size
# --------------------------------------------------------------------------


class FakeManager:
    def __init__(self, size: int) -> None:
        self.size = size
        self.free_ids = list(range(1, size + 1))
        self.reserved: List[int] = []
        self.alloc_calls: List[int] = []
        self.free_calls: List[List[int]] = []
        self.reserve_calls: List[int] = []

    def available_size(self) -> int:
        return len(self.free_ids) + len(self.reserved)

    def alloc(self, count: int):
        # Reserved blocks are served first, as KVCacheManager._alloc_impl does.
        self.alloc_calls.append(count)
        if count > len(self.free_ids) + len(self.reserved):
            return None
        from_reserved = min(count, len(self.reserved))
        result, self.reserved = self.reserved[:from_reserved], self.reserved[from_reserved:]
        remaining = count - from_reserved
        result += self.free_ids[:remaining]
        self.free_ids = self.free_ids[remaining:]
        return result

    alloc_packed = alloc

    def free(self, ids) -> None:
        ids = list(ids)
        self.free_calls.append(ids)
        self.free_ids.extend(ids)

    def try_to_reserve(self, need: int) -> bool:
        self.reserve_calls.append(need)
        got = self.alloc(need)
        if got is None:
            return False
        self.reserved.extend(got)
        return True

    def clear(self) -> None:
        self.free_ids = list(range(1, self.size + 1))
        self.reserved = []


class FakeMambaPool:
    """ElasticMambaPool shape as ElasticMambaSlotAllocator reads it."""

    def __init__(self, size: int) -> None:
        self.kvcached_allocator = FakeManager(size)
        self.size = size
        self.device = "cpu"

    def available_size(self) -> int:
        return self.kvcached_allocator.available_size()

    def alloc(self, count: int):
        result = self.kvcached_allocator.alloc(count)
        if result is None:
            return None
        return torch.tensor(result, dtype=torch.int64)

    def free(self, slots) -> None:
        self.kvcached_allocator.free(slots.tolist())

    def clear(self) -> None:
        self.kvcached_allocator.clear()


class FakeBaseTokenToKVPoolAllocator:
    def __init__(self, size, page_size, dtype, device, kvcache, *args, **kwargs):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.free_group = None
        self.evicted: List[Any] = []

    def evict_to_free_tokens(self, tree_cache, num_tokens):
        # The 0.5.19+ base helper the decode gate calls before deciding.
        self.evicted.append((tree_cache, num_tokens))


class FakeKernelFn:
    def __init__(self, parameter_names):
        self.__signature__ = inspect.Signature([
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in parameter_names
        ])

    def __call__(self, *args, **kwargs):
        pass


class FakeTritonKernel:
    def __init__(self, fn):
        self.fn = fn
        self.calls: List[Any] = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def _install_sglang_utils(monkeypatch, num_new_pages: int) -> None:
    sglang: Any = types.ModuleType("sglang")
    srt: Any = types.ModuleType("sglang.srt")
    utils: Any = types.ModuleType("sglang.srt.utils")
    utils.get_num_new_pages = lambda **kwargs: num_new_pages
    utils.next_power_of_2 = lambda value: 1 << (max(value, 1) - 1).bit_length()
    for name, module in (("sglang", sglang), ("sglang.srt", srt),
                         ("sglang.srt.utils", utils)):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, module)


def _make_allocator_module() -> Any:
    alloc_mod: Any = types.ModuleType("sglang.srt.mem_cache.allocator")
    alloc_mod.BaseTokenToKVPoolAllocator = FakeBaseTokenToKVPoolAllocator
    alloc_mod.alloc_extend_kernel = FakeTritonKernel(FakeKernelFn((
        "pre_lens_ptr", "seq_lens_ptr", "last_loc_ptr", "free_page_ptr",
        "out_indices", "bs_upper", "page_size")))
    alloc_mod.alloc_decode_kernel = FakeTritonKernel(FakeKernelFn((
        "seq_lens_ptr", "last_loc_ptr", "free_page_ptr", "out_indices",
        "bs_upper", "page_size")))
    return alloc_mod


@pytest.fixture
def cpu_tensors(monkeypatch):
    # The elastic allocators refuse non-GPU devices; CPU tensors stand in here.
    monkeypatch.setattr(patches, "_is_supported_gpu_device", lambda device: True)
    monkeypatch.setenv("ENABLE_KVCACHED", "true")


def _token_allocator(manager_size: int):
    alloc_mod = _make_allocator_module()
    assert ElasticAllocatorPatch().inject_elastic_allocator(alloc_mod)
    kvcache = types.SimpleNamespace(kvcached_allocator=FakeManager(manager_size))
    allocator = alloc_mod.ElasticTokenToKVPoolAllocator(
        size=manager_size, dtype=torch.int64, device="cpu", kvcache=kvcache)
    return allocator, kvcache.kvcached_allocator


def _paged_allocator(monkeypatch, manager_size: int, page_size: int,
                     num_new_pages: int):
    _install_sglang_utils(monkeypatch, num_new_pages)
    alloc_mod = _make_allocator_module()
    assert ElasticAllocatorPatch().inject_elastic_paged_allocator(alloc_mod)
    kvcache = types.SimpleNamespace(kvcached_allocator=FakeManager(manager_size))
    allocator = alloc_mod.ElasticPagedTokenToKVPoolAllocator(
        size=manager_size * page_size, page_size=page_size, dtype=torch.int64,
        device="cpu", kvcache=kvcache)
    return allocator, kvcache.kvcached_allocator


class MambaSlotAllocator:
    """Module-level so the SGLang 0.5.13 allocator split is detectable in
    the fake ``_init_mamba_pool``'s globals, as the production patch checks."""


@pytest.fixture
def mamba_slot_allocator_cls(monkeypatch):
    class FakeHybridReqToTokenPool:
        def _init_mamba_pool(self):
            self.mamba_allocator = MambaSlotAllocator()

    memory_pool: Any = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    memory_pool.MambaPool = type("FakeMambaPool", (), {"State": object})
    memory_pool.HybridReqToTokenPool = FakeHybridReqToTokenPool
    patch = ElasticMambaPoolPatch()
    monkeypatch.setattr(patch.version_manager, "detect_version",
                        lambda library: "0.5.20")
    assert patch.apply(memory_pool)
    return memory_pool.ElasticMambaSlotAllocator


# --------------------------------------------------------------------------
# allocator contract: None outside the batch window, typed inside it
# --------------------------------------------------------------------------


def _lens(count: int):
    return torch.zeros(count, dtype=torch.int64)


def _paged_calls(allocator):
    return {
        "alloc": lambda: allocator.alloc(allocator.page_size * 2),
        "alloc_extend": lambda: allocator.alloc_extend(
            _lens(1), _lens(1), _lens(1), _lens(1), _lens(1), extend_num_tokens=4),
        "alloc_decode": lambda: allocator.alloc_decode(_lens(1), _lens(1), _lens(1)),
    }


@pytest.mark.parametrize("method", ["alloc", "alloc_extend", "alloc_decode"])
def test_paged_allocator_miss_is_none_outside_and_typed_inside_the_window(
        monkeypatch, cpu_tensors, method):
    allocator, manager = _paged_allocator(
        monkeypatch, manager_size=1, page_size=4, num_new_pages=2)
    call = _paged_calls(allocator)[method]

    assert call() is None  # the #545 contract outside the window
    with _sglang_batch_allocation_zone():
        with pytest.raises(KVCachePoolExhausted, match="cannot back 2 pages"):
            call()
    assert call() is None  # the window closed with the exception
    assert manager.free_ids == [1]  # a miss hands out nothing


def test_token_allocator_miss_is_none_outside_and_typed_inside_the_window(
        cpu_tensors):
    allocator, manager = _token_allocator(manager_size=1)

    assert allocator.alloc(2) is None
    with _sglang_batch_allocation_zone():
        with pytest.raises(KVCachePoolExhausted, match="cannot back 2 tokens"):
            allocator.alloc(2)
        # Inside the window a fit still allocates.
        assert allocator.alloc(1).tolist() == [1]
    assert manager.free_ids == []


def test_mamba_slot_allocator_miss_is_typed_only_inside_the_window(
        mamba_slot_allocator_cls):
    pool = FakeMambaPool(2)
    allocator = mamba_slot_allocator_cls(2, "cpu", pool)

    # alloc_group_begin runs before the batch window; a miss there stays the
    # best-effort None it is natively.
    allocator.alloc_group_begin(3)
    assert allocator._alloc_iter is None
    assert allocator.alloc(3) is None
    with _sglang_batch_allocation_zone():
        with pytest.raises(KVCachePoolExhausted, match="cannot back 3 mamba slots"):
            allocator.alloc(3)
        assert allocator.alloc(2).tolist() == [1, 2]
    assert pool.kvcached_allocator.free_ids == []


def test_swa_composite_precheck_miss_is_typed_inside_the_window(cpu_tensors):
    class SWATokenToKVPoolAllocator:
        """The native composite: its own pre-check returns None before any
        sub-allocator runs (swa.py alloc/alloc_extend)."""

        def __init__(self, elastic: bool) -> None:
            self.full_attn_allocator = (
                types.SimpleNamespace(kvcached_allocator=FakeManager(1))
                if elastic else types.SimpleNamespace())
            self.swa_attn_allocator = types.SimpleNamespace()

        def available_size(self) -> int:
            return 0

        def alloc(self, need_size: int):
            return None

        def alloc_extend(self, *args, **kwargs):
            return None

        def alloc_decode(self, *args, **kwargs):
            return None

    swa_mod: Any = types.ModuleType("sglang.srt.mem_cache.allocator.swa")
    swa_mod.SWATokenToKVPoolAllocator = SWATokenToKVPoolAllocator
    patch = ElasticSWAAllocatorPatch()
    assert patch.patch_swa_composite_misses(swa_mod) is True
    assert patch.patch_swa_composite_misses(swa_mod) is True  # idempotent

    elastic = SWATokenToKVPoolAllocator(elastic=True)
    native = SWATokenToKVPoolAllocator(elastic=False)
    assert elastic.alloc(4) is None
    with _sglang_batch_allocation_zone():
        for method in ("alloc", "alloc_extend", "alloc_decode"):
            with pytest.raises(KVCachePoolExhausted, match=method):
                getattr(elastic, method)(4)
            # A composite over native sub-allocators keeps the None contract.
            assert getattr(native, method)(4) is None


# --------------------------------------------------------------------------
# decode gate: reserve the step where a shortfall retracts
# --------------------------------------------------------------------------


def test_decode_gate_reserves_the_step_after_evicting(monkeypatch, cpu_tensors):
    allocator, manager = _paged_allocator(
        monkeypatch, manager_size=3, page_size=4, num_new_pages=3)
    tree_cache = object()

    assert allocator.check_decode_capacity(
        num_tokens=9, tree_cache=tree_cache, requests=[], spec_algorithm=None)
    assert allocator.evicted == [(tree_cache, 9)]
    assert manager.reserve_calls == [3]  # ceil(9 / 4) pages
    assert manager.reserved == [1, 2, 3]
    assert allocator.available_size() == 12  # reserved blocks stay counted

    # The following decode allocation draws the reserved pages, so it cannot
    # run out of physical pages inside this process; the retract loop's
    # repeated gate re-reserves the same pages instead of growing the set.
    manager.free_ids = []
    assert allocator.check_decode_capacity(num_tokens=4, tree_cache=tree_cache) is True
    assert sorted(manager.reserved) == [1, 2, 3]
    assert allocator.alloc_decode(_lens(3), _lens(3), _lens(3)) is not None
    assert manager.reserved == []


def test_decode_gate_reports_a_shortfall_instead_of_allocating(cpu_tensors):
    allocator, manager = _token_allocator(manager_size=1)

    assert allocator.check_decode_capacity(num_tokens=2, tree_cache=None) is False
    assert manager.reserved == []
    assert manager.free_ids == [1]
    # No need, no reservation.
    assert allocator.check_decode_capacity(num_tokens=0, tree_cache=None) is True
    assert manager.reserve_calls == [2]


# --------------------------------------------------------------------------
# the scheduler path, mirrored from SGLang 0.5.20
# --------------------------------------------------------------------------


class FakeKV:
    """The request's ReqKVState fields the allocation path reads and writes."""

    def __init__(self) -> None:
        self.req_pool_idx: Optional[int] = None
        self.mamba_pool_idx: Optional[torch.Tensor] = None
        self.mamba_ping_pong_track_buffer: Optional[torch.Tensor] = None
        self.mamba_next_track_idx: Optional[int] = None
        self.mamba_last_track_idx: Optional[int] = None
        self.mamba_needs_clear = False
        self.mamba_cow_src_index: Optional[torch.Tensor] = None

    @property
    def holds_kv(self) -> bool:
        return self.req_pool_idx is not None

    @property
    def holds_mamba(self) -> bool:
        return self.mamba_pool_idx is not None


class FakeReceipt:
    """DecLockRefParams stand-in: the admission lock's release receipt."""

    def __init__(self, node: Any = None) -> None:
        self.node = node

    def to_dec_params(self) -> "FakeReceipt":
        return self


class FakeReq:
    def __init__(self, rid: str, *, chunked: bool = False,
                 session: Any = None) -> None:
        self.rid = rid
        self.kv = FakeKV()
        self.last_node = f"node-{rid}"
        self.lock_receipt = FakeReceipt()
        self.session = session
        self.prefix_indices: List[int] = []
        self.extend_range: Any = None
        self.inflight_middle_chunks = 0
        self.chunked = chunked

    def set_extend_range(self, start: int, end: int) -> None:
        self.extend_range = (start, end)

    def __repr__(self) -> str:
        return f"FakeReq({self.rid})"


class FakeTreeCache:
    def __init__(self, token_allocator: Any) -> None:
        self.token_to_kv_pool_allocator = token_allocator
        self.lock_calls: List[Any] = []
        self.printed = 0

    def inc_lock_ref(self, node: Any) -> FakeReceipt:
        self.lock_calls.append(("inc", node))
        return FakeReceipt(node)

    def dec_lock_ref(self, node: Any, params: Any = None) -> None:
        self.lock_calls.append(("dec", node, params))

    def evict_for_alloc(self, params: Any) -> None:
        pass

    def pretty_print(self) -> None:
        self.printed += 1


class FakeReqToTokenPool:
    """ReqToTokenPool: static request rows, popped from the tail."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.free_slots = list(range(1, size + 1))
        self.freed: List[int] = []

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self, reqs: List[FakeReq]):
        fresh = [req for req in reqs if not req.kv.holds_kv]
        if len(fresh) > len(self.free_slots):
            return None
        select = self.free_slots[len(self.free_slots) - len(fresh):]
        del self.free_slots[len(self.free_slots) - len(fresh):]
        for req, idx in zip(fresh, select):
            req.kv.req_pool_idx = idx
        return [req.kv.req_pool_idx for req in reqs]

    def free(self, req: FakeReq) -> None:
        idx = req.kv.req_pool_idx
        assert idx is not None
        self.freed.append(idx)
        self.free_slots.append(idx)
        req.kv.req_pool_idx = None


class FakeHybridReqToTokenPool(FakeReqToTokenPool):
    """HybridReqToTokenPool.alloc: rows first, then one Mamba slot and the
    ping-pong pair per request, each guarded by the native assert."""

    def __init__(self, size: int, mamba_allocator: Any) -> None:
        super().__init__(size)
        self.mamba_allocator = mamba_allocator
        self.enable_mamba_extra_buffer = True

    def alloc(self, reqs: List[FakeReq]):
        select_index = super().alloc(reqs)
        if select_index is None:
            return None
        for req in reqs:
            if not req.kv.holds_mamba:
                mid = self.mamba_allocator.alloc(1)
                assert mid is not None, MAMBA_SLOT
                req.kv.mamba_pool_idx = mid[0]
                req.kv.mamba_needs_clear = True
            if req.kv.mamba_ping_pong_track_buffer is None:
                slots = self.mamba_allocator.alloc(2)
                assert slots is not None, MAMBA_PING_PONG
                req.kv.mamba_ping_pong_track_buffer = slots.clone()
                req.kv.mamba_next_track_idx = 0
                req.kv.mamba_last_track_idx = 1
        return select_index


class FakeBatch:
    def __init__(self, reqs, pool, tree_cache, module, extend_num_tokens):
        self.reqs = reqs
        self.req_to_token_pool = pool
        self.tree_cache = tree_cache
        self.extend_num_tokens = extend_num_tokens
        self._module = module

    def prepare_for_extend(self):
        # Resolved through the module, as ScheduleBatch does, so the patched
        # binding is the one that runs.
        return self._module.alloc_for_extend(self)


def _make_schedule_batch_module() -> Any:
    module: Any = types.ModuleType("sglang.srt.managers.schedule_batch")

    def alloc_req_slots(pool, reqs, tree_cache):
        req_pool_indices = pool.alloc(reqs)
        if req_pool_indices is None:
            raise RuntimeError(ROW_OOM)
        return req_pool_indices

    def alloc_token_slots(tree_cache, num_tokens):
        out_cache_loc = tree_cache.token_to_kv_pool_allocator.alloc(num_tokens)
        if out_cache_loc is None:
            tree_cache.pretty_print()
            raise RuntimeError(f"{TOKEN_OOM}\nTry to allocate {num_tokens} tokens.")
        return out_cache_loc

    def alloc_for_extend(batch):
        req_pool_indices = alloc_req_slots(
            batch.req_to_token_pool, batch.reqs, batch.tree_cache)
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
        return out_cache_loc, req_pool_indices

    module.alloc_for_extend = alloc_for_extend
    return module


class FakeScheduleBag:
    def __init__(self, prefill_max_requests: Optional[int] = None) -> None:
        self.prefill_max_requests = prefill_max_requests
        self.override_calls: List[dict] = []

    @contextlib.contextmanager
    def override(self, **kwargs):
        self.override_calls.append(dict(kwargs))
        saved = {name: getattr(self, name) for name in kwargs}
        for name, value in kwargs.items():
            setattr(self, name, value)
        try:
            yield self
        finally:
            for name, value in saved.items():
                setattr(self, name, value)


class FakeScheduler:
    """Scheduler._get_new_batch_prefill_raw from admission to prepare."""

    CHUNK = 8

    def __init__(self, module, pool, tree_cache, bag) -> None:
        self.waiting_queue: List[FakeReq] = []
        self.chunked_req: Optional[FakeReq] = None
        self.req_to_token_pool = pool
        self.tree_cache = tree_cache
        self._module = module
        self._bag = bag
        self.caps_seen: List[Optional[int]] = []
        self.extend_num_tokens = 4

    def _get_new_batch_prefill_raw(self, prefill_delayer_single_pass, running_batch):
        if (running_batch.batch_is_full
                or len(self.waiting_queue) == 0) and self.chunked_req is None:
            return None, running_batch
        cap = self._bag.prefill_max_requests  # read when the adder is built
        self.caps_seen.append(cap)
        can_run_list: List[FakeReq] = []
        new_chunked_req = None
        if self.chunked_req is not None:
            req = self.chunked_req
            prefix_len = len(req.prefix_indices)
            req.set_extend_range(prefix_len, prefix_len + self.CHUNK)
            can_run_list.append(req)
            self.chunked_req = req if req.chunked else None
        for req in self.waiting_queue:
            if cap is not None and len(can_run_list) >= cap:
                break
            req.set_extend_range(0, self.CHUNK)
            req.lock_receipt = self.tree_cache.inc_lock_ref(req.last_node).to_dec_params()
            can_run_list.append(req)
            if req.chunked and self.chunked_req is None:
                # add_one_req chunks a request only while none is in flight.
                new_chunked_req = req
                break
        if len(can_run_list) == 0:
            return None, running_batch
        can_run_set = set(can_run_list)
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]
        if new_chunked_req is not None:
            assert self.chunked_req is None
            self.chunked_req = new_chunked_req
        if self.chunked_req is not None:
            self.chunked_req.inflight_middle_chunks += 1
        batch = FakeBatch(can_run_list, self.req_to_token_pool, self.tree_cache,
                          self._module, self.extend_num_tokens)
        batch.prepare_for_extend()
        return batch, running_batch


def _install_runtime_context(monkeypatch, bag: FakeScheduleBag) -> None:
    _install_sglang_utils(monkeypatch, 0)
    runtime_context: Any = types.ModuleType("sglang.srt.runtime_context")
    runtime_context.get_schedule = lambda: bag
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", runtime_context)


class Pipeline:
    def __init__(self, monkeypatch, *, rows: int, mamba_slots: Optional[int],
                 tokens: int, prefill_max_requests: Optional[int] = None,
                 version: str = "0.5.20", patched: bool = True,
                 mamba_slot_allocator_cls=None) -> None:
        self.token_allocator, self.token_manager = _token_allocator(tokens)
        self.tree_cache = FakeTreeCache(self.token_allocator)
        if mamba_slots is None:
            self.mamba_manager = None
            self.pool: Any = FakeReqToTokenPool(rows)
        else:
            mamba_pool = FakeMambaPool(mamba_slots)
            self.mamba_manager = mamba_pool.kvcached_allocator
            self.pool = FakeHybridReqToTokenPool(
                rows, mamba_slot_allocator_cls(mamba_slots, "cpu", mamba_pool))
        self.batch_module = _make_schedule_batch_module()
        self.bag = FakeScheduleBag(prefill_max_requests)
        _install_runtime_context(monkeypatch, self.bag)
        sched_module: Any = types.ModuleType("sglang.srt.managers.scheduler")
        # A fresh class per pipeline: the patch rebinds a class attribute.
        scheduler_cls = type("Scheduler", (FakeScheduler,), {})
        sched_module.Scheduler = scheduler_cls
        self.sched_module = sched_module
        self.applied = (False, False)
        if patched:
            batch_patch = ScheduleBatchCapacityMissPatch()
            sched_patch = SchedulerCapacityMissPatch()
            for patch in (batch_patch, sched_patch):
                monkeypatch.setattr(patch.version_manager, "detect_version",
                                    lambda library: version)
            self.applied = (batch_patch.apply(self.batch_module),
                            sched_patch.apply(sched_module))
        self.scheduler = scheduler_cls(
            self.batch_module, self.pool, self.tree_cache, self.bag)
        self.running_batch = types.SimpleNamespace(batch_is_full=False, reqs=[])

    def enqueue(self, *reqs: FakeReq) -> None:
        self.scheduler.waiting_queue.extend(reqs)

    def round(self):
        return self.scheduler._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=None, running_batch=self.running_batch)

    @property
    def mamba(self) -> FakeManager:
        assert self.mamba_manager is not None
        return self.mamba_manager

    def free_state(self):
        return (
            sorted(self.pool.free_slots),
            None if self.mamba_manager is None else sorted(self.mamba_manager.free_ids),
            sorted(self.token_manager.free_ids),
        )


def _dec_calls(tree_cache: FakeTreeCache):
    return [call for call in tree_cache.lock_calls if call[0] == "dec"]


def test_alloc_for_extend_wrapper_releases_and_types_the_miss(
        monkeypatch, cpu_tensors, mamba_slot_allocator_cls):
    """The batch seam alone: the typed miss carries the batch, and the rows
    and slots handed out before the miss are back before it is raised."""
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=4, tokens=16,
                        mamba_slot_allocator_cls=mamba_slot_allocator_cls)
    reqs = [FakeReq("a"), FakeReq("b")]
    batch = FakeBatch(reqs, pipeline.pool, pipeline.tree_cache,
                      pipeline.batch_module, extend_num_tokens=4)
    before = pipeline.free_state()

    with pytest.raises(SGLangPrefillCapacityMiss, match="mamba slots") as info:
        batch.prepare_for_extend()

    assert info.value.batch is batch
    assert isinstance(info.value.__cause__, KVCachePoolExhausted)
    assert pipeline.free_state() == before
    assert pipeline.pool.freed == [3, 4]
    assert pipeline.mamba.free_calls == [[2, 3], [1], [4]]
    assert all(not req.kv.holds_kv and not req.kv.holds_mamba for req in reqs)


def test_hybrid_miss_exits_the_scheduler_without_the_patches(
        monkeypatch, cpu_tensors, mamba_slot_allocator_cls):
    """The #547 crash: admitted requests, then the Mamba pool misses."""
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=4, tokens=16,
                        patched=False,
                        mamba_slot_allocator_cls=mamba_slot_allocator_cls)
    reqs = [FakeReq("a"), FakeReq("b")]
    pipeline.enqueue(*reqs)

    with pytest.raises(AssertionError, match=MAMBA_PING_PONG):
        pipeline.round()
    # The requests left the queue and keep the rows and slots they were
    # handed; nothing can run them.
    assert pipeline.scheduler.waiting_queue == []
    assert reqs[0].kv.holds_kv and reqs[0].kv.holds_mamba
    assert pipeline.free_state() == ([1, 2], [], list(range(1, 17)))


def test_token_miss_exits_the_scheduler_without_the_patches(
        monkeypatch, cpu_tensors, mamba_slot_allocator_cls):
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=8, tokens=2,
                        patched=False,
                        mamba_slot_allocator_cls=mamba_slot_allocator_cls)
    pipeline.enqueue(FakeReq("a"))

    with pytest.raises(RuntimeError, match=TOKEN_OOM):
        pipeline.round()
    assert pipeline.tree_cache.printed == 1


def test_mamba_miss_returns_the_batch_to_the_queue(
        monkeypatch, cpu_tensors, mamba_slot_allocator_cls):
    """Rixin's first error: the ping-pong pair of the second request."""
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=7, tokens=16,
                        mamba_slot_allocator_cls=mamba_slot_allocator_cls)
    assert pipeline.applied == (True, True)
    reqs = [FakeReq("a"), FakeReq("b"), FakeReq("c")]
    pipeline.enqueue(*reqs)
    before = pipeline.free_state()

    assert pipeline.round() == (None, pipeline.running_batch)

    # Everything the batch was handed is back, in both pools.
    assert pipeline.free_state() == before
    for req in reqs:
        assert not req.kv.holds_kv and not req.kv.holds_mamba
        assert req.kv.mamba_ping_pong_track_buffer is None
        assert req.kv.mamba_next_track_idx is None
        assert req.kv.mamba_needs_clear is False
    # The admission locks are released with their receipts and the requests
    # are back at the head of the queue in order.
    assert [(call[1], call[2].node) for call in _dec_calls(pipeline.tree_cache)] == [
        ("node-a", "node-a"), ("node-b", "node-b"), ("node-c", "node-c")]
    assert all(req.lock_receipt.node is None for req in reqs)
    assert pipeline.scheduler.waiting_queue == reqs
    assert pipeline.running_batch.batch_is_full is False
    assert pipeline.tree_cache.printed == 0
    # The next round admits one request fewer and fits.
    batch, _ = pipeline.round()
    assert pipeline.bag.override_calls == [{"prefill_max_requests": 2}]
    assert pipeline.scheduler.caps_seen == [None, 2]
    assert [req.rid for req in batch.reqs] == ["a", "b"]
    assert pipeline.scheduler.waiting_queue == [reqs[2]]
    # A round that did not miss lifts the cap.
    assert pipeline.scheduler._kvcached_prefill_request_cap is None


def test_token_miss_returns_rows_and_mamba_slots(
        monkeypatch, cpu_tensors, mamba_slot_allocator_cls):
    """Rixin's second error: Mamba fits, the attention pool misses."""
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=8, tokens=2,
                        mamba_slot_allocator_cls=mamba_slot_allocator_cls)
    reqs = [FakeReq("a"), FakeReq("b")]
    pipeline.enqueue(*reqs)
    before = pipeline.free_state()

    assert pipeline.round() == (None, pipeline.running_batch)

    assert pipeline.free_state() == before
    assert pipeline.pool.freed == [3, 4]
    assert pipeline.scheduler.waiting_queue == reqs
    assert pipeline.tree_cache.printed == 0
    assert pipeline.scheduler._kvcached_prefill_request_cap == 1


def test_attention_only_miss_returns_rows(monkeypatch, cpu_tensors):
    """The pure-attention control: one kvcached pool, no Mamba allocator."""
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=None, tokens=2)
    reqs = [FakeReq("a"), FakeReq("b")]
    pipeline.enqueue(*reqs)
    before = pipeline.free_state()

    assert pipeline.round() == (None, pipeline.running_batch)

    assert pipeline.free_state() == before
    assert pipeline.scheduler.waiting_queue == reqs
    assert [call[1] for call in _dec_calls(pipeline.tree_cache)] == [
        "node-a", "node-b"]
    # With enough tokens the same queue runs under the cap.
    pipeline.token_manager.free_ids = list(range(1, 17))
    batch, _ = pipeline.round()
    assert [req.rid for req in batch.reqs] == ["a"]
    assert reqs[0].kv.holds_kv


def _hold_chunk_state(pipeline, continuing: FakeReq) -> None:
    """Give a continuation the row and slots its earlier chunks allocated."""
    continuing.prefix_indices = list(range(8))
    continuing.kv.req_pool_idx = 4
    pipeline.pool.free_slots.remove(4)
    continuing.kv.mamba_pool_idx = torch.tensor(7)
    pipeline.mamba.free_ids.remove(7)
    continuing.kv.mamba_ping_pong_track_buffer = torch.tensor([5, 6])
    pipeline.mamba.free_ids.remove(5)
    pipeline.mamba.free_ids.remove(6)
    pipeline.scheduler.chunked_req = continuing


def test_last_chunk_and_a_new_chunked_request_are_both_unwound(
        monkeypatch, cpu_tensors, mamba_slot_allocator_cls):
    """The continuation's last chunk hands the chunked slot to a new request;
    a miss restores the continuation and makes the new one a candidate."""
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=8, tokens=2,
                        mamba_slot_allocator_cls=mamba_slot_allocator_cls)
    continuing = FakeReq("chunk", chunked=False)
    _hold_chunk_state(pipeline, continuing)
    fresh = FakeReq("new", chunked=True)
    pipeline.enqueue(fresh)
    before = pipeline.free_state()

    assert pipeline.round() == (None, pipeline.running_batch)

    # The continuation keeps row 4 and slots 5, 6, 7, stays the scheduler's
    # chunked request, and is parked with nothing new to cache.
    assert pipeline.free_state() == before
    assert continuing.kv.req_pool_idx == 4
    assert continuing.kv.mamba_pool_idx is not None
    assert continuing.kv.mamba_pool_idx.item() == 7
    assert continuing.extend_range == (8, 8)
    assert continuing.inflight_middle_chunks == 0
    assert pipeline.scheduler.chunked_req is continuing
    # The request that had become the new chunked one is a candidate again:
    # no lock, no slots, no inflight chunk, back in the queue.
    assert fresh.inflight_middle_chunks == 0
    assert not fresh.kv.holds_kv and not fresh.kv.holds_mamba
    assert [call[1] for call in _dec_calls(pipeline.tree_cache)] == ["node-new"]
    assert pipeline.scheduler.waiting_queue == [fresh]


def test_mid_stream_chunk_is_parked_with_its_inflight_count(
        monkeypatch, cpu_tensors, mamba_slot_allocator_cls):
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=8, tokens=2,
                        mamba_slot_allocator_cls=mamba_slot_allocator_cls)
    continuing = FakeReq("chunk", chunked=True)
    _hold_chunk_state(pipeline, continuing)
    before = pipeline.free_state()

    assert pipeline.round() == (None, pipeline.running_batch)

    assert pipeline.free_state() == before
    assert continuing.extend_range == (8, 8)
    assert continuing.inflight_middle_chunks == 0
    assert pipeline.scheduler.chunked_req is continuing
    assert _dec_calls(pipeline.tree_cache) == []
    assert pipeline.scheduler.waiting_queue == []


def test_match_time_mamba_state_is_reverted_except_for_sessions(
        monkeypatch, cpu_tensors, mamba_slot_allocator_cls):
    """A COW slot from match_prefix is released like the scheduler releases
    it for a candidate it did not add; a session-held slot is not."""
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=8, tokens=2,
                        mamba_slot_allocator_cls=mamba_slot_allocator_cls)
    cow = FakeReq("cow")
    cow.kv.mamba_pool_idx = torch.tensor(8)
    cow.kv.mamba_cow_src_index = torch.tensor(1)
    pipeline.mamba.free_ids.remove(8)
    held = FakeReq("session", session=object())
    held.kv.mamba_pool_idx = torch.tensor(7)
    pipeline.mamba.free_ids.remove(7)
    pipeline.enqueue(cow, held)

    assert pipeline.round() == (None, pipeline.running_batch)

    assert not cow.kv.holds_mamba
    assert cow.kv.mamba_cow_src_index is None
    assert 8 in pipeline.mamba.free_ids
    assert held.kv.mamba_pool_idx is not None
    assert held.kv.mamba_pool_idx.item() == 7
    assert 7 not in pipeline.mamba.free_ids


def test_cap_keeps_a_user_ceiling(monkeypatch, cpu_tensors):
    pipeline = Pipeline(monkeypatch, rows=8, mamba_slots=None, tokens=2,
                        prefill_max_requests=2)
    pipeline.enqueue(*(FakeReq(str(i)) for i in range(5)))

    assert pipeline.round() == (None, pipeline.running_batch)
    pipeline.round()
    # The batch that missed had 2 requests (the user's ceiling), so the cap
    # is min(2, 1) = 1 and the user's value is restored afterwards.
    assert pipeline.bag.override_calls == [{"prefill_max_requests": 1}]
    assert pipeline.bag.prefill_max_requests == 2


@pytest.mark.parametrize("failure", [
    pytest.param("rows", id="static request rows exhausted"),
    pytest.param("value_error", id="contract violation"),
])
def test_other_failures_still_exit_the_scheduler(
        monkeypatch, cpu_tensors, mamba_slot_allocator_cls, failure):
    """Only a kvcached capacity miss is translated (#511 rule)."""
    pipeline = Pipeline(monkeypatch, rows=1, mamba_slots=8, tokens=16,
                        mamba_slot_allocator_cls=mamba_slot_allocator_cls)
    reqs = [FakeReq("a"), FakeReq("b")]
    pipeline.enqueue(*reqs)
    expected: Any
    if failure == "value_error":
        pipeline.pool.free_slots = [1, 2]
        pipeline.token_manager.alloc = lambda count: (_ for _ in ()).throw(
            ValueError("invalid block metadata"))
        expected, match = ValueError, "invalid block metadata"
    else:
        expected, match = RuntimeError, ROW_OOM

    with pytest.raises(expected, match=match):
        pipeline.round()
    assert pipeline.scheduler.waiting_queue == []
    assert getattr(pipeline.scheduler, "_kvcached_prefill_request_cap", None) is None


def test_passthrough_when_kvcached_is_disabled(
        monkeypatch, cpu_tensors, mamba_slot_allocator_cls):
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=4, tokens=16,
                        mamba_slot_allocator_cls=mamba_slot_allocator_cls)
    monkeypatch.setenv("ENABLE_KVCACHED", "false")
    pipeline.enqueue(FakeReq("a"), FakeReq("b"))

    with pytest.raises(AssertionError, match=MAMBA_PING_PONG):
        pipeline.round()


def test_patches_leave_older_sglang_alone(monkeypatch, cpu_tensors):
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=None, tokens=2,
                        version="0.5.19")
    assert pipeline.applied == (True, True)
    assert not hasattr(pipeline.batch_module.alloc_for_extend,
                       "__kvcached_alloc_for_extend_patched__")
    assert not hasattr(pipeline.sched_module.Scheduler._get_new_batch_prefill_raw,
                       "__kvcached_prefill_capacity_miss_patched__")
