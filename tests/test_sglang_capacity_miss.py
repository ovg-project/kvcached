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
translated, and only what the failed batch was handed out is released. Under
TP every rank drives its own pool, so the ranks that share a waiting queue
must read one budget before they admit (the budget round) and reach one
verdict before any of them runs the batch; the two-rank tests run each rank
in its own thread over a rendezvous that stands in for the gloo all-reduce.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import sys
import threading
import types
from typing import Any, Callable, List, Optional, Tuple

import pytest
import torch

from kvcached.integration.sglang import patches
from kvcached.integration.sglang.patches import (
    ElasticAllocatorPatch,
    ElasticMambaPoolPatch,
    ElasticSWAAllocatorPatch,
    ScheduleBatchCapacityMissPatch,
    SchedulerCapacityMissPatch,
    SchedulerSharedBudgetPatch,
    SGLangPrefillCapacityMiss,
    _sglang_batch_allocation_zone,
    _sglang_budget_round,
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
        self.reserve_packed: List[bool] = []

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

    def try_to_reserve(self, need: int, packed: bool = False) -> bool:
        self.reserve_calls.append(need)
        self.reserve_packed.append(packed)
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
        self.evict_shortfalls: List[int] = []

    def available_size(self) -> int:
        raise NotImplementedError  # the elastic subclasses define it

    def evict_to_free_tokens(self, tree_cache, num_tokens):
        # The 0.5.19+ base helper the decode gate calls before deciding;
        # evict_from_tree_cache evicts the shortfall against available_size().
        self.evicted.append((tree_cache, num_tokens))
        self.evict_shortfalls.append(max(0, num_tokens - self.available_size()))

    def check_decode_capacity(self, *, num_tokens, tree_cache, requests=None,
                              spec_algorithm=None):
        # SGLang 0.5.19+'s native gate: evict, then compare the free count.
        self.evict_to_free_tokens(tree_cache, num_tokens)
        return self.available_size() >= num_tokens  # type: ignore[attr-defined]


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


def test_window_journals_token_blocks_and_returns_them(cpu_tensors):
    """Blocks handed out inside the window are journaled per manager, and a
    rollback gives exactly those back, newest first."""
    allocator, manager = _token_allocator(manager_size=8)
    return_blocks = patches._sglang_return_journaled_blocks

    with _sglang_batch_allocation_zone() as journal:
        assert allocator.alloc(3).tolist() == [1, 2, 3]
        assert allocator.alloc(2).tolist() == [4, 5]
        assert journal == [(manager, [1, 2, 3]), (manager, [4, 5])]
        return_blocks(journal)
    assert journal == []
    assert manager.free_calls == [[4, 5], [1, 2, 3]]
    assert sorted(manager.free_ids) == list(range(1, 9))
    # Outside the window nothing is journaled.
    assert allocator.alloc(1).tolist() == [6]
    assert getattr(patches._SGLANG_BATCH_ALLOCATION, "journal", None) is None


def test_paged_window_journals_the_new_pages_only(monkeypatch, cpu_tensors):
    """alloc_extend hands out token slots that continue the prefix's last
    page; the journal holds the new page ids the manager returned, so a
    rollback never touches the page the prefix already owns."""
    allocator, manager = _paged_allocator(
        monkeypatch, manager_size=4, page_size=4, num_new_pages=2)

    with _sglang_batch_allocation_zone() as journal:
        out = allocator.alloc_extend(
            _lens(1), _lens(1), _lens(1), _lens(1), _lens(1), extend_num_tokens=6)
        assert out.numel() == 6
        assert journal == [(manager, [1, 2])]
    assert manager.free_ids == [3, 4]


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


def _decode_gate(allocator: Any, num_tokens: int) -> Callable[[], bool]:
    return lambda: allocator.check_decode_capacity(
        num_tokens=num_tokens, tree_cache=None)


def test_decode_gate_agrees_across_ranks(monkeypatch, cpu_tensors):
    """Two ranks, the second one short: the gate says no on both, so both
    take the native retract path instead of one running alloc_for_decode
    into a miss while the other retracts."""
    gloo = FakeGloo(2)
    _install_queue_groups(monkeypatch, gloo, world_size=2)
    ranks = [_token_allocator(manager_size=4), _token_allocator(manager_size=1)]

    verdicts = _run_ranks([_decode_gate(allocator, 2) for allocator, _ in ranks])

    assert verdicts == [False, False]
    assert gloo.rounds == [[0, 1]]
    # The rank that could reserve keeps its blocks; the next gate re-reserves
    # the same set, and a rank with no need still joins the collective.
    assert ranks[0][1].reserved == [1, 2]
    assert ranks[1][1].reserved == []
    verdicts = _run_ranks([_decode_gate(allocator, 0) for allocator, _ in ranks])
    assert verdicts == [True, True]
    assert gloo.rounds == [[0, 1], [0, 0]]


def test_group_verdict_is_local_without_peers(monkeypatch, cpu_tensors):
    """A group of one is tp_size == 1: no collective runs at all."""
    _install_queue_groups(monkeypatch, None, world_size=1)
    allocator, manager = _token_allocator(manager_size=4)

    assert allocator.check_decode_capacity(num_tokens=2, tree_cache=None) is True
    assert manager.reserved == [1, 2]
    allocator, manager = _token_allocator(manager_size=1)
    assert allocator.check_decode_capacity(num_tokens=2, tree_cache=None) is False


def test_decode_gate_reserves_as_the_decode_allocation_places(
        monkeypatch, cpu_tensors):
    """The decode allocation takes the reserved blocks as they are, so the
    gate reserves them the way that allocation would place them: packed for
    the paged allocator, which decodes with alloc_packed."""
    paged, paged_manager = _paged_allocator(
        monkeypatch, manager_size=3, page_size=4, num_new_pages=1)
    token, token_manager = _token_allocator(manager_size=4)

    assert paged.check_decode_capacity(num_tokens=4, tree_cache=None)
    assert token.check_decode_capacity(num_tokens=2, tree_cache=None)
    assert paged_manager.reserve_packed == [True]
    assert token_manager.reserve_packed == [False]


def _install_pipeline_stages(monkeypatch, stages: int) -> None:
    """A parallel state with ``stages`` pipeline stages of one rank each."""
    _install_queue_groups(monkeypatch, None, world_size=1)
    parallel_state = sys.modules["sglang.srt.distributed.parallel_state"]
    parallel_state.get_pipeline_model_parallel_world_size = (  # type: ignore[attr-defined]
        lambda: stages)


def test_pipeline_keeps_the_native_decode_gate(monkeypatch, cpu_tensors):
    """No verdict is agreed across pipeline stages, so a pipeline neither
    reserves nor agrees: the gate is SGLang's own."""
    _install_pipeline_stages(monkeypatch, 2)
    allocator, manager = _token_allocator(manager_size=1)

    assert allocator.check_decode_capacity(num_tokens=1, tree_cache=None) is True
    assert allocator.check_decode_capacity(num_tokens=2, tree_cache=None) is False
    assert manager.reserve_calls == []
    assert allocator.evicted == [(None, 1), (None, 2)]


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
        # Written at the end of a successful alloc_for_extend (allocation.py).
        self.kv_allocated_len = 0
        self.kv_committed_len = 0

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
        self.evict_calls: List[Any] = []
        self.printed = 0

    def inc_lock_ref(self, node: Any) -> FakeReceipt:
        self.lock_calls.append(("inc", node))
        return FakeReceipt(node)

    def dec_lock_ref(self, node: Any, params: Any = None) -> None:
        self.lock_calls.append(("dec", node, params))

    def evict_for_alloc(self, params: Any) -> None:
        self.evict_calls.append(params)

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


MAMBA_SLOTS_PER_REQ = 3  # MAMBA_STATE_PER_REQ_PREFIX_CACHE: the slot and the ping-pong pair


def _make_schedule_batch_module() -> Any:
    module: Any = types.ModuleType("sglang.srt.managers.schedule_batch")

    def alloc_req_slots(pool, reqs, tree_cache):
        mamba_allocator = getattr(pool, "mamba_allocator", None)
        if mamba_allocator is not None:
            # allocation.py:301-322 evicts the Mamba shortfall by the
            # planner's count before the rows are bound.
            available = mamba_allocator.schedulable_available_size()
            needed = len(reqs) * MAMBA_SLOTS_PER_REQ
            if available < needed:
                tree_cache.evict_for_alloc(("mamba", needed - available))
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
        # The native tail: the request's KV lengths follow the allocation.
        for req in batch.reqs:
            end = req.extend_range[1] if req.extend_range is not None else 0
            req.kv.kv_allocated_len = req.kv.kv_committed_len = end
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
    """Scheduler._get_new_batch_prefill_raw from admission to prepare, and
    the entry points around it that a budget round wraps."""

    CHUNK = 8
    TOKENS_PER_REQ = 4

    def __init__(self, module, pool, tree_cache, bag,
                 admit_by_budget: bool = False) -> None:
        self.waiting_queue: List[FakeReq] = []
        self.chunked_req: Optional[FakeReq] = None
        self.req_to_token_pool = pool
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = tree_cache.token_to_kv_pool_allocator
        self._module = module
        self._bag = bag
        self.caps_seen: List[Optional[int]] = []
        self.extend_num_tokens = 4
        # With admit_by_budget the adder's token budget decides admission,
        # as PrefillAdder.rem_total_tokens does; without it every queued
        # request is admitted, the stale-snapshot case of the miss tests.
        self.admit_by_budget = admit_by_budget
        self.budgets_seen: List[int] = []
        self.admitted: List[List[str]] = []
        self.decode_verdicts: List[bool] = []

    def get_next_batch_to_run(self, running_batch, last_batch):
        # scheduler.py:3655-3681: the prefill pass first, the decode update
        # only when it yields no batch.
        batch, running_batch = self.get_new_batch_prefill(running_batch)
        if batch is None and running_batch.reqs:
            running_batch = self.update_running_batch(running_batch)
        return batch, running_batch

    def get_new_batch_prefill(self, running_batch):
        return self._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=None, running_batch=running_batch)

    def update_running_batch(self, batch):
        # scheduler.py:4101-4185: the decode gate, then retraction when it
        # says no.
        fits = batch.check_decode_mem()
        self.decode_verdicts.append(fits)
        if not fits:
            batch.retract_decode()
        return batch

    def _get_new_batch_prefill_raw(self, prefill_delayer_single_pass, running_batch):
        if (running_batch.batch_is_full
                or len(self.waiting_queue) == 0) and self.chunked_req is None:
            return None, running_batch
        cap = self._bag.prefill_max_requests  # read when the adder is built
        self.caps_seen.append(cap)
        # PrefillAdder.rem_total_tokens: the allocator's count plus the
        # evictable tree, which these fakes keep empty.
        rem_total_tokens = self.token_to_kv_pool_allocator.available_size()
        self.budgets_seen.append(rem_total_tokens)
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
            if self.admit_by_budget:
                # add_one_req: NO_TOKEN once the request's tokens reach the
                # remaining budget.
                if self.TOKENS_PER_REQ >= rem_total_tokens:
                    break
                rem_total_tokens -= self.TOKENS_PER_REQ
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
        self.admitted.append([req.rid for req in can_run_list])
        extend_num_tokens = (self.TOKENS_PER_REQ * len(can_run_list)
                             if self.admit_by_budget else self.extend_num_tokens)
        batch = FakeBatch(can_run_list, self.req_to_token_pool, self.tree_cache,
                          self._module, extend_num_tokens)
        batch.prepare_for_extend()
        return batch, running_batch


class FakeRunningBatch:
    """The running batch as update_running_batch drives it: check_decode_mem
    asks the allocator's gate (schedule_batch.py:3149-3166) and a shortfall
    retracts."""

    def __init__(self, allocator: Any, tree_cache: Any, reqs: List[FakeReq],
                 num_tokens: int) -> None:
        self.token_to_kv_pool_allocator = allocator
        self.tree_cache = tree_cache
        self.reqs = reqs
        self.num_tokens = num_tokens
        self.batch_is_full = False
        self.retracted = 0

    def check_decode_mem(self) -> bool:
        return self.token_to_kv_pool_allocator.check_decode_capacity(
            num_tokens=self.num_tokens, tree_cache=self.tree_cache,
            requests=self.reqs)

    def retract_decode(self) -> None:
        self.retracted += 1


_ACTIVE_BAG = threading.local()


def _install_runtime_context(monkeypatch) -> None:
    """``get_schedule`` answers with the bag of the pipeline whose round runs
    on the calling thread, so two fake ranks keep separate config bags."""
    _install_sglang_utils(monkeypatch, 0)
    runtime_context: Any = types.ModuleType("sglang.srt.runtime_context")

    def get_schedule() -> Any:
        return _ACTIVE_BAG.bag

    runtime_context.get_schedule = get_schedule
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", runtime_context)


class FakeGloo:
    """The CPU group's all-reduce as a rendezvous between rank threads.

    Each rank deposits its flag and waits; the barrier action reduces once
    all ``world_size`` ranks have arrived and every rank reads the result.
    A rank that never calls it leaves the others waiting, as a missing
    collective would on a real group, which the timeout turns into a
    failure instead of a hang.
    """

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.rounds: List[Any] = []
        self._deposits: List[Tuple[Any, List[int]]] = []
        self._result: List[int] = []
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(world_size, action=self._reduce,
                                          timeout=10)

    def _reduce(self) -> None:
        ops = [op for op, _ in self._deposits]
        assert all(op == ops[0] for op in ops), "the ranks issued different collectives"
        values = [value for _, value in self._deposits]
        pick = max if ops[0] == torch.distributed.ReduceOp.MAX else min
        self._result = [pick(column) for column in zip(*values)]
        # Arrival order is thread scheduling; record the deposits sorted so
        # the assertions do not depend on who deposited first: a MAX of miss
        # flags as ints, a MIN of counts as lists.
        flags = ops[0] == torch.distributed.ReduceOp.MAX
        self.rounds.append(sorted(value[0] if flags else value for value in values))
        self._deposits = []

    def all_reduce(self, tensor, op) -> None:
        assert op in (torch.distributed.ReduceOp.MAX, torch.distributed.ReduceOp.MIN)
        with self._lock:
            self._deposits.append((op, [int(v) for v in tensor.reshape(-1).tolist()]))
        self._barrier.wait()
        tensor.copy_(torch.tensor(self._result, dtype=tensor.dtype).reshape(tensor.shape))


def _install_queue_groups(monkeypatch, gloo: Optional[FakeGloo],
                          world_size: int) -> None:
    """A parallel state whose attention TP group has ``world_size`` ranks and
    no CP group, as SGLang's without DP attention or context parallelism."""
    _install_sglang_utils(monkeypatch, 0)
    parallel_state: Any = types.ModuleType("sglang.srt.distributed.parallel_state")
    group = types.SimpleNamespace(world_size=world_size, cpu_group=gloo)

    def get_attn_tp_group() -> Any:
        return group

    def no_cp_group() -> Any:
        raise AssertionError(
            "attention context model parallel group is not initialized")

    parallel_state.get_attn_tp_group = get_attn_tp_group
    parallel_state.get_attn_cp_group = no_cp_group
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed",
                        types.ModuleType("sglang.srt.distributed"))
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed.parallel_state",
                        parallel_state)

    def all_reduce(tensor, op, group):
        assert group is not None, "all_reduce on a group of one"
        group.all_reduce(tensor, op)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)


def _run_ranks(steps: List[Callable[[], Any]]) -> List[Any]:
    """Run one step per rank on its own thread and return their results."""
    results: List[Any] = [None] * len(steps)
    errors: List[Optional[BaseException]] = [None] * len(steps)

    def run(index: int) -> None:
        try:
            results[index] = steps[index]()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors[index] = exc

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(steps))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), "a rank hung"
    for error in errors:
        if error is not None:
            raise error
    return results


class Pipeline:
    def __init__(self, monkeypatch, *, rows: int, mamba_slots: Optional[int],
                 tokens: int, prefill_max_requests: Optional[int] = None,
                 version: str = "0.5.20", patched: bool = True,
                 mamba_slot_allocator_cls=None,
                 token_allocator: Optional[Tuple[Any, FakeManager]] = None,
                 admit_by_budget: bool = False) -> None:
        if token_allocator is None:
            token_allocator = _token_allocator(tokens)
        self.token_allocator: Any = token_allocator[0]
        self.token_manager: FakeManager = token_allocator[1]
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
        _install_runtime_context(monkeypatch)
        sched_module: Any = types.ModuleType("sglang.srt.managers.scheduler")
        # A fresh class per pipeline: the patch rebinds a class attribute.
        scheduler_cls = type("Scheduler", (FakeScheduler,), {})
        sched_module.Scheduler = scheduler_cls
        self.sched_module = sched_module
        self.applied = (False, False)
        self.budget_applied = False
        if patched:
            batch_patch = ScheduleBatchCapacityMissPatch()
            sched_patch = SchedulerCapacityMissPatch()
            budget_patch = SchedulerSharedBudgetPatch()
            for patch in (batch_patch, sched_patch, budget_patch):
                monkeypatch.setattr(patch.version_manager, "detect_version",
                                    lambda library: version)
            self.applied = (batch_patch.apply(self.batch_module),
                            sched_patch.apply(sched_module))
            self.budget_applied = budget_patch.apply(sched_module)
        self.scheduler = scheduler_cls(
            self.batch_module, self.pool, self.tree_cache, self.bag,
            admit_by_budget=admit_by_budget)
        self.running_batch = types.SimpleNamespace(batch_is_full=False, reqs=[])

    def enqueue(self, *reqs: FakeReq) -> None:
        self.scheduler.waiting_queue.extend(reqs)

    def round(self):
        """The prefill pass alone, as the miss tests drive it."""
        _ACTIVE_BAG.bag = self.bag
        return self.scheduler._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=None, running_batch=self.running_batch)

    def iteration(self, running_batch: Any = None):
        """One scheduler iteration through get_next_batch_to_run, the entry
        the budget round wraps."""
        _ACTIVE_BAG.bag = self.bag
        if running_batch is None:
            running_batch = self.running_batch
        return self.scheduler.get_next_batch_to_run(
            running_batch=running_batch, last_batch=None)

    def state(self, reqs: List[FakeReq], result=None):
        """What one rank ends a round with, for comparing ranks."""
        batch = None if result is None else result[0]
        return {
            "has_batch": batch is not None,
            "batch": None if batch is None else [req.rid for req in batch.reqs],
            "queued": [req.rid for req in self.scheduler.waiting_queue],
            "holds_row": [req.kv.holds_kv for req in reqs],
            "kv_allocated_len": [req.kv.kv_allocated_len for req in reqs],
            "cap": getattr(self.scheduler, "_kvcached_prefill_request_cap", None),
        }

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


def test_tp_peer_miss_unwinds_every_rank(monkeypatch, cpu_tensors):
    """shipiyouniao's asymmetric probe: only rank 1's token pool misses.

    Without the group verdict rank 0 returns a batch and keeps its row while
    rank 1 requeues, and the ranks walk into the forward with different
    batches. With it both ranks requeue, return everything their own call
    handed out, apply the same cap, and build the same batch once the
    pressure on rank 1 lifts.
    """
    gloo = FakeGloo(2)
    _install_queue_groups(monkeypatch, gloo, world_size=2)
    ranks = [Pipeline(monkeypatch, rows=4, mamba_slots=None, tokens=16),
             Pipeline(monkeypatch, rows=4, mamba_slots=None, tokens=2)]
    reqs = [[FakeReq("a"), FakeReq("b")] for _ in ranks]
    for rank, rank_reqs in zip(ranks, reqs):
        rank.enqueue(*rank_reqs)
    before = [rank.free_state() for rank in ranks]

    results = _run_ranks([rank.round for rank in ranks])

    states = [rank.state(rank_reqs, result)
              for rank, rank_reqs, result in zip(ranks, reqs, results)]
    assert states[0] == states[1]
    assert states[0]["has_batch"] is False
    assert states[0]["queued"] == ["a", "b"]
    assert states[0]["holds_row"] == [False, False]
    assert states[0]["kv_allocated_len"] == [0, 0]
    assert states[0]["cap"] == 1
    assert [rank.free_state() for rank in ranks] == before
    # One collective per attempt, rank 1 the only local miss; rank 0's own
    # allocation of the batch's 4 tokens went back through the journal.
    assert gloo.rounds == [[0, 1]]
    assert ranks[0].token_manager.free_calls == [[1, 2, 3, 4]]
    assert ranks[1].token_manager.free_calls == []
    assert [rank.pool.freed for rank in ranks] == [[3, 4], [3, 4]]
    # The pressure lifts on rank 1: the same capped batch runs on both.
    ranks[1].token_manager.free_ids = list(range(1, 17))
    results = _run_ranks([rank.round for rank in ranks])
    states = [rank.state(rank_reqs, result)
              for rank, rank_reqs, result in zip(ranks, reqs, results)]
    assert states[0] == states[1]
    assert states[0]["batch"] == ["a"]
    assert states[0]["queued"] == ["b"]
    assert states[0]["cap"] is None
    assert gloo.rounds == [[0, 1], [0, 0]]


def test_tp_peer_miss_unwinds_mamba_slots_too(
        monkeypatch, cpu_tensors, mamba_slot_allocator_cls):
    """The hybrid pools on the rank whose peer missed: rows, the Mamba slot
    and the ping-pong pair go back along with the tokens."""
    gloo = FakeGloo(2)
    _install_queue_groups(monkeypatch, gloo, world_size=2)
    ranks = [Pipeline(monkeypatch, rows=4, mamba_slots=8, tokens=16,
                      mamba_slot_allocator_cls=mamba_slot_allocator_cls),
             Pipeline(monkeypatch, rows=4, mamba_slots=4, tokens=16,
                      mamba_slot_allocator_cls=mamba_slot_allocator_cls)]
    reqs = [[FakeReq("a"), FakeReq("b")] for _ in ranks]
    for rank, rank_reqs in zip(ranks, reqs):
        rank.enqueue(*rank_reqs)
    before = [rank.free_state() for rank in ranks]

    results = _run_ranks([rank.round for rank in ranks])

    states = [rank.state(rank_reqs, result)
              for rank, rank_reqs, result in zip(ranks, reqs, results)]
    for state in states:
        assert state["has_batch"] is False
        assert state["queued"] == ["a", "b"]
        assert state["holds_row"] == [False, False]
        assert state["cap"] == 1
    assert [rank.free_state() for rank in ranks] == before
    for rank_reqs in reqs:
        assert all(not req.kv.holds_mamba for req in rank_reqs)
        assert all(req.kv.mamba_ping_pong_track_buffer is None for req in rank_reqs)
    assert gloo.rounds == [[0, 1]]
    assert ranks[0].token_manager.free_calls == [[1, 2, 3, 4]]


# --------------------------------------------------------------------------
# the budget round: one agreed count per pool per scheduling round
# --------------------------------------------------------------------------


def _budget_rank(tokens: int) -> Tuple[Any, FakeManager, Any]:
    """A token allocator and the scheduler shape _sglang_budget_round walks."""
    allocator, manager = _token_allocator(tokens)
    scheduler = types.SimpleNamespace(token_to_kv_pool_allocator=allocator,
                                      req_to_token_pool=None)
    return allocator, manager, scheduler


def test_budget_round_serves_the_group_minimum_plus_its_own_moves(
        monkeypatch, cpu_tensors):
    """One MIN reduce per outermost round; inside it every count read is the
    agreed minimum plus what the round itself allocated and freed, and a
    nested round adds no collective. Outside the round the live count is
    back."""
    gloo = FakeGloo(2)
    _install_queue_groups(monkeypatch, gloo, world_size=2)
    ranks = [_budget_rank(16), _budget_rank(6)]

    def round_on(allocator, manager, scheduler):
        seen = []
        with _sglang_budget_round(scheduler):
            seen.append(allocator.available_size())
            allocator.alloc(2)
            seen.append(allocator.available_size())
            allocator.free(torch.tensor([1]))
            seen.append(allocator.available_size())
            with _sglang_budget_round(scheduler):
                seen.append(allocator.available_size())
        seen.append(allocator.available_size())
        return seen

    seen = _run_ranks([functools.partial(round_on, *rank) for rank in ranks])

    assert seen[0][:4] == seen[1][:4] == [6, 4, 5, 5]
    assert [rank_seen[4] for rank_seen in seen] == [15, 5]
    assert gloo.rounds == [[[6], [16]]]


def test_budget_round_keeps_the_paged_allocator_unit(monkeypatch, cpu_tensors):
    """The reduce carries manager blocks; the paged allocator still reports
    tokens."""
    gloo = FakeGloo(2)
    _install_queue_groups(monkeypatch, gloo, world_size=2)
    ranks = []
    for pages in (3, 2):
        allocator, _ = _paged_allocator(
            monkeypatch, manager_size=pages, page_size=4, num_new_pages=1)
        ranks.append((allocator, types.SimpleNamespace(
            token_to_kv_pool_allocator=allocator, req_to_token_pool=None)))

    def round_on(allocator, scheduler):
        with _sglang_budget_round(scheduler):
            before = allocator.available_size()
            allocator.alloc(4)
            return before, allocator.available_size()

    assert _run_ranks([functools.partial(round_on, *rank)
                       for rank in ranks]) == [(8, 4), (8, 4)]
    assert gloo.rounds == [[[2], [3]]]


def test_budget_round_ignores_the_mamba_group_preallocation(
        monkeypatch, cpu_tensors, mamba_slot_allocator_cls):
    """alloc_group_begin pre-allocates through the pool and succeeds on one
    rank only, so the tally follows the slots requests receive instead, and
    the counts stay equal through the group."""
    gloo = FakeGloo(2)
    _install_queue_groups(monkeypatch, gloo, world_size=2)
    ranks = []
    for slots in (8, 2):
        pool = FakeMambaPool(slots)
        allocator = mamba_slot_allocator_cls(slots, "cpu", pool)
        scheduler = types.SimpleNamespace(
            token_to_kv_pool_allocator=None,
            req_to_token_pool=types.SimpleNamespace(mamba_allocator=allocator))
        ranks.append((allocator, pool, scheduler))

    def round_on(allocator, pool, scheduler):
        with _sglang_budget_round(scheduler):
            seen = [allocator.schedulable_available_size()]
            allocator.alloc_group_begin(3)  # 3 slots on one rank, None on the other
            seen.append(allocator.available_size())
            slot = allocator.alloc(1)  # from the group or the pool, one slot either way
            seen.append(allocator.available_size())
            allocator.alloc_group_end()
            seen.append(allocator.available_size())
            allocator.free(slot)
            seen.append(allocator.available_size())
        return seen, pool.available_size()

    results = _run_ranks([functools.partial(round_on, *rank) for rank in ranks])

    assert results[0][0] == results[1][0] == [2, 2, 1, 1, 2]
    assert [live for _, live in results] == [8, 2]
    assert gloo.rounds == [[[2], [8]]]


def test_budget_is_local_at_tp_size_one(monkeypatch, cpu_tensors):
    """A group of one opens no round: admission reads the local count and
    no collective runs."""
    _install_queue_groups(monkeypatch, None, world_size=1)
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=None, tokens=6,
                        admit_by_budget=True)
    assert pipeline.budget_applied is True
    pipeline.enqueue(FakeReq("a"), FakeReq("b"))

    batch, _ = pipeline.iteration()

    assert [req.rid for req in batch.reqs] == ["a"]
    assert pipeline.scheduler.budgets_seen == [6]
    assert pipeline.token_allocator.available_size() == 2


def test_pipeline_keeps_the_budget_local(monkeypatch, cpu_tensors):
    """A pipeline opens no round, as the miss patches keep SGLang's own
    behaviour there: with a queue group present every rank still reads its
    own count and no collective runs."""
    gloo = FakeGloo(2)
    _install_queue_groups(monkeypatch, gloo, world_size=2)
    parallel_state = sys.modules["sglang.srt.distributed.parallel_state"]
    parallel_state.get_pipeline_model_parallel_world_size = (  # type: ignore[attr-defined]
        lambda: 2)
    ranks = [_budget_rank(16), _budget_rank(6)]

    def round_on(allocator, manager, scheduler):
        with _sglang_budget_round(scheduler):
            return allocator.available_size()

    assert _run_ranks([functools.partial(round_on, *rank)
                       for rank in ranks]) == [16, 6]
    assert gloo.rounds == []


def test_tp_admission_reads_one_budget(monkeypatch, cpu_tensors):
    """The weekly case of #577: rank 1's GPU is short and rank 0's is not.
    With each rank admitting against its own count, rank 0 builds [a, b]
    while rank 1 builds [a], both allocate, and the forward hangs on the
    mismatch. Under the round rank 1's count is the budget of both: the
    ranks build the same batch, and when rank 1's pool later cannot back
    an agreed batch they miss together, requeue identically and run the
    same batch once the pressure lifts."""
    gloo = FakeGloo(2)
    _install_queue_groups(monkeypatch, gloo, world_size=2)
    ranks = [Pipeline(monkeypatch, rows=4, mamba_slots=None, tokens=16,
                      admit_by_budget=True),
             Pipeline(monkeypatch, rows=4, mamba_slots=None, tokens=6,
                      admit_by_budget=True)]
    reqs = [[FakeReq("a"), FakeReq("b")] for _ in ranks]
    for rank, rank_reqs in zip(ranks, reqs):
        rank.enqueue(*rank_reqs)

    def states_after(results):
        return [rank.state(rank_reqs, result)
                for rank, rank_reqs, result in zip(ranks, reqs, results)]

    # Rank 0 could fit both requests; the group's smallest count admits one.
    states = states_after(_run_ranks([rank.iteration for rank in ranks]))
    assert states[0] == states[1]
    assert states[0]["batch"] == ["a"]
    assert states[0]["queued"] == ["b"]
    assert [rank.scheduler.budgets_seen for rank in ranks] == [[6], [6]]
    assert [rank.scheduler.admitted for rank in ranks] == [[["a"]], [["a"]]]
    # One MIN reduce of the counts, then the MAX reduce of the miss flags.
    assert gloo.rounds == [[[6], [16]], [0, 0]]

    # The batch finishes and its tokens come back. Then the other process
    # on rank 1's GPU takes the memory: rank 1's count still admits b, but
    # its pool backs nothing while the pressure is on.
    for rank in ranks:
        rank.token_manager.clear()
    pressure = {"on": True}
    pool_alloc = ranks[1].token_manager.alloc
    monkeypatch.setattr(ranks[1].token_manager, "alloc",
                        lambda count: None if pressure["on"] else pool_alloc(count))
    before = [rank.free_state() for rank in ranks]

    states = states_after(_run_ranks([rank.iteration for rank in ranks]))
    assert states[0] == states[1]
    assert states[0]["has_batch"] is False
    assert states[0]["queued"] == ["b"]
    assert states[0]["cap"] == 1
    assert [rank.scheduler.admitted for rank in ranks] == [[["a"], ["b"]], [["a"], ["b"]]]
    assert gloo.rounds[2:] == [[[6], [16]], [0, 1]]
    assert [rank.free_state() for rank in ranks] == before
    assert ranks[0].token_manager.free_calls == [[1, 2, 3, 4]]

    # The pressure lifts: the same capped batch runs on both.
    pressure["on"] = False
    states = states_after(_run_ranks([rank.iteration for rank in ranks]))
    assert states[0] == states[1]
    assert states[0]["batch"] == ["b"]
    assert states[0]["queued"] == []
    assert states[0]["cap"] is None
    assert gloo.rounds[4:] == [[[6], [16]], [0, 0]]


def test_tp_admission_agrees_on_the_mamba_count(
        monkeypatch, cpu_tensors, mamba_slot_allocator_cls):
    """alloc_req_slots evicts the Mamba shortfall by the planner's count
    (allocation.py:301-322), so the ranks must read one count there as
    well, or their radix trees drift apart. Rank 1's Mamba pool is short:
    both ranks evict the same shortfall, and the miss that follows unwinds
    both."""
    gloo = FakeGloo(2)
    _install_queue_groups(monkeypatch, gloo, world_size=2)
    ranks = [Pipeline(monkeypatch, rows=4, mamba_slots=8, tokens=16,
                      admit_by_budget=True,
                      mamba_slot_allocator_cls=mamba_slot_allocator_cls),
             Pipeline(monkeypatch, rows=4, mamba_slots=2, tokens=16,
                      admit_by_budget=True,
                      mamba_slot_allocator_cls=mamba_slot_allocator_cls)]
    reqs = [[FakeReq("a"), FakeReq("b")] for _ in ranks]
    for rank, rank_reqs in zip(ranks, reqs):
        rank.enqueue(*rank_reqs)
    before = [rank.free_state() for rank in ranks]

    results = _run_ranks([rank.iteration for rank in ranks])

    states = [rank.state(rank_reqs, result)
              for rank, rank_reqs, result in zip(ranks, reqs, results)]
    assert states[0] == states[1]
    assert states[0]["has_batch"] is False
    assert states[0]["queued"] == ["a", "b"]
    assert states[0]["cap"] == 1
    assert [rank.scheduler.admitted for rank in ranks] == [[["a", "b"]], [["a", "b"]]]
    # Two requests need six slots against the agreed count of two, on both.
    assert [rank.tree_cache.evict_calls for rank in ranks] == [
        [("mamba", 4)], [("mamba", 4)]]
    assert gloo.rounds == [[[16, 2], [16, 8]], [0, 1]]
    assert [rank.free_state() for rank in ranks] == before
    for rank_reqs in reqs:
        assert all(not req.kv.holds_mamba for req in rank_reqs)


def test_decode_gate_evicts_the_same_amount_on_every_rank(monkeypatch, cpu_tensors):
    """The gate evicts the shortfall against available_size() before it
    reserves (allocator/base.py:72-93 through evict_from_tree_cache). With
    per-rank counts rank 0 evicts nothing while rank 1 evicts two tokens,
    and the trees drift; under the round both evict the agreed shortfall and
    retract together."""
    gloo = FakeGloo(2)
    _install_queue_groups(monkeypatch, gloo, world_size=2)
    ranks = [Pipeline(monkeypatch, rows=4, mamba_slots=None, tokens=16),
             Pipeline(monkeypatch, rows=4, mamba_slots=None, tokens=2)]
    batches = [FakeRunningBatch(rank.token_allocator, rank.tree_cache,
                                [FakeReq("r")], num_tokens=4) for rank in ranks]

    _run_ranks([functools.partial(rank.iteration, batch)
                for rank, batch in zip(ranks, batches)])

    assert [rank.scheduler.decode_verdicts for rank in ranks] == [[False], [False]]
    assert [batch.retracted for batch in batches] == [1, 1]
    assert [rank.token_allocator.evict_shortfalls for rank in ranks] == [[2], [2]]
    assert gloo.rounds == [[[2], [16]], [0, 1]]
    # The rank that could reserve keeps its pages for the next gate.
    assert ranks[0].token_manager.reserved == [1, 2, 3, 4]


class FakeSWAComposite:
    """SWATokenToKVPoolAllocator.alloc at v0.5.20 (swa.py:219-232): the
    pre-check reads both children's counts, then the full child allocates,
    then the SWA child, and the full-to-SWA mapping is written after both."""

    def __init__(self, full: Any, swa: Any) -> None:
        self.full_attn_allocator = full
        self.swa_attn_allocator = swa
        self.page_size = 1
        self.mapping_writes: List[Any] = []

    def available_size(self) -> int:
        return min(self.full_attn_allocator.available_size(),
                   self.swa_attn_allocator.available_size())

    def alloc(self, need_size: int):
        if need_size > self.full_attn_allocator.available_size():
            return None
        if need_size > self.swa_attn_allocator.available_size():
            return None
        alloc_full_indices = self.full_attn_allocator.alloc(need_size)
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None
        self.mapping_writes.append(
            (alloc_full_indices.tolist(), alloc_swa_indices.tolist()))
        return alloc_full_indices

    def free(self, free_index) -> None:
        self.swa_attn_allocator.free(free_index)
        self.full_attn_allocator.free(free_index)


def test_swa_partial_allocation_is_returned_before_retry(monkeypatch, cpu_tensors):
    """shipiyouniao's probe: the SWA child misses after the composite's
    pre-check passed. Three retries used to leave the full child's blocks
    allocated and unreachable, 16 to 12 to 8 to 4; now every retry returns
    them, with the mapping untouched and both counts unchanged."""
    full_allocator, full_manager = _token_allocator(manager_size=16)
    swa_allocator, swa_manager = _token_allocator(manager_size=16)
    taken: List[List[int]] = []
    full_alloc = full_manager.alloc

    def record_full_alloc(count):
        ids = full_alloc(count)
        taken.append(list(ids))
        return ids

    monkeypatch.setattr(full_manager, "alloc", record_full_alloc)
    # Enough by the count, nothing by the pool: the miss after the pre-check.
    monkeypatch.setattr(swa_manager, "alloc", lambda count: None)
    swa_mod: Any = types.ModuleType("sglang.srt.mem_cache.allocator.swa")
    swa_mod.SWATokenToKVPoolAllocator = FakeSWAComposite
    assert ElasticSWAAllocatorPatch().patch_swa_composite_misses(swa_mod)
    composite = FakeSWAComposite(full_allocator, swa_allocator)
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=None, tokens=16,
                        token_allocator=(composite, full_manager))
    req = FakeReq("a")
    pipeline.enqueue(req)

    for attempt in range(1, 4):
        assert pipeline.round() == (None, pipeline.running_batch)
        assert pipeline.scheduler.waiting_queue == [req]
        assert not req.kv.holds_kv
        assert full_manager.available_size() == 16, attempt
        assert swa_manager.available_size() == 16, attempt
        # The full child's blocks from this very attempt went back, each time.
        assert full_manager.free_calls == taken
        assert [len(ids) for ids in taken] == [4] * attempt
    assert composite.mapping_writes == []
    # With the SWA child back, the same request runs and the mapping follows.
    monkeypatch.undo()
    monkeypatch.setattr(patches, "_is_supported_gpu_device", lambda device: True)
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    _install_runtime_context(monkeypatch)
    batch, _ = pipeline.round()
    assert [r.rid for r in batch.reqs] == ["a"]
    assert len(composite.mapping_writes) == 1
    assert full_manager.available_size() == 12


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
        monkeypatch.setattr(
            pipeline.token_manager, "alloc",
            lambda count: (_ for _ in ()).throw(ValueError("invalid block metadata")))
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


def test_pipeline_keeps_a_prefill_miss_fatal(
        monkeypatch, cpu_tensors, mamba_slot_allocator_cls):
    """A miss recovered on one stage alone would leave the other stages
    waiting on its sends, so a pipeline keeps SGLang's own behaviour."""
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=4, tokens=16,
                        mamba_slot_allocator_cls=mamba_slot_allocator_cls)
    _install_pipeline_stages(monkeypatch, 2)
    pipeline.enqueue(FakeReq("a"), FakeReq("b"))

    with pytest.raises(AssertionError, match=MAMBA_PING_PONG):
        pipeline.round()


def test_miss_patches_leave_older_sglang_alone(monkeypatch, cpu_tensors):
    """The two miss patches read 0.5.20 shapes and register from 0.5.20; the
    budget round registers from 0.5.13, so at 0.5.19 only the round is in."""
    pipeline = Pipeline(monkeypatch, rows=4, mamba_slots=None, tokens=2,
                        version="0.5.19")
    assert pipeline.applied == (True, True)
    assert not hasattr(pipeline.batch_module.alloc_for_extend,
                       "__kvcached_alloc_for_extend_patched__")
    assert not hasattr(pipeline.sched_module.Scheduler._get_new_batch_prefill_raw,
                       "__kvcached_prefill_capacity_miss_patched__")
    assert pipeline.budget_applied is True
    assert hasattr(pipeline.sched_module.Scheduler.get_next_batch_to_run,
                   "__kvcached_budget_round_patched__")


@pytest.mark.parametrize("version, budget, miss", [
    ("0.5.12", False, False),
    ("0.5.13", True, False),
    ("0.5.19", True, False),
    ("0.5.20", True, True),
])
def test_autopatch_selects_budget_and_miss_patches(monkeypatch, version, budget, miss):
    monkeypatch.setenv("KVCACHED_AUTOPATCH", "false")
    from kvcached.integration.sglang import autopatch

    monkeypatch.setenv("KVCACHED_AUTOPATCH", "true")
    manager = autopatch.PatchManager("sglang")
    monkeypatch.setattr(manager.version_manager, "detect_version", lambda _: version)
    applied = []

    def record_patch(self, patch):
        applied.append(type(patch).__name__)
        return True

    # Keep the real registration and apply_all_patches version filtering;
    # only replace the engine imports and individual patch implementations.
    monkeypatch.setattr(autopatch.PatchManager, "_apply_single_patch", record_patch)
    autopatch._patch_sglang(types.ModuleType("sglang"))

    assert ("SchedulerSharedBudgetPatch" in applied) is budget
    for name in ("ScheduleBatchCapacityMissPatch", "SchedulerCapacityMissPatch"):
        assert (name in applied) is miss


@pytest.mark.parametrize("version, wrapped", [("0.5.12", False), ("0.5.13", True)])
def test_budget_round_registers_from_the_allocator_split(
        monkeypatch, cpu_tensors, version, wrapped):
    """The round registers from SGLang 0.5.13, where the elastic SWA and
    Mamba slot allocators it reads first exist, ahead of the miss patches.
    At 0.5.13 the ranks of a queue group admit the same batch from one MIN
    reduce and no MAX reduce follows; at 0.5.12 nothing is wrapped and each
    rank admits from its own count."""
    gloo = FakeGloo(2)
    _install_queue_groups(monkeypatch, gloo, world_size=2)
    ranks = [Pipeline(monkeypatch, rows=4, mamba_slots=None, tokens=tokens,
                      version=version, admit_by_budget=True)
             for tokens in (16, 6)]
    for rank in ranks:
        assert rank.applied == (True, True)
        assert rank.budget_applied is True
        assert hasattr(rank.sched_module.Scheduler.get_next_batch_to_run,
                       "__kvcached_budget_round_patched__") is wrapped
        assert not hasattr(rank.batch_module.alloc_for_extend,
                           "__kvcached_alloc_for_extend_patched__")
        rank.enqueue(FakeReq("a"), FakeReq("b"))

    _run_ranks([rank.iteration for rank in ranks])

    admitted = [rank.scheduler.admitted for rank in ranks]
    budgets = [rank.scheduler.budgets_seen for rank in ranks]
    if wrapped:
        assert admitted == [[["a"]], [["a"]]]
        assert budgets == [[6], [6]]
        assert gloo.rounds == [[[6], [16]]]
    else:
        assert admitted == [[["a", "b"]], [["a"]]]
        assert budgets == [[16], [6]]
        assert gloo.rounds == []
