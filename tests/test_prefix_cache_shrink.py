# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""A lower KV memory limit must evict the prefix cache that holds it up.

Wires the real KVCacheManager under the real vLLM ElasticBlockPool. Only the
C++ page allocator is faked; like PageAllocator::resize it refuses to shrink
below the number of in-use pages. Pages held only by evictable (ref_cnt == 0)
prefix-cache blocks count as in use, and allocations reuse their slots one LRU
victim at a time, so without eviction a lower limit stayed deferred while the
engine idled and under traffic alike.

No GPU, vLLM or C++ extension required.
"""

import queue
import random
import sys
import threading
import types
from types import SimpleNamespace
from typing import Dict, List
from unittest import mock

import pytest

sys.modules.setdefault("posix_ipc", mock.MagicMock())

BLOCKS_PER_PAGE = 4
MAX_PAGES = 16
NUM_BLOCKS = MAX_PAGES * BLOCKS_PER_PAGE
# KVCACHED_MAX_CACHED_TOKENS default (16000) // block_size 16. Larger than this
# pool, so the static cap never fires: only the memory limit should matter.
DEFAULT_MAX_CACHED_BLOCKS = 1000


class FakePage:
    def __init__(self, page_id: int):
        self.page_id = page_id
        self.free_list: List[int] = []

    def init(self, block_mem_size: int) -> None:
        start = self.page_id * BLOCKS_PER_PAGE
        self.free_list = list(range(start, start + BLOCKS_PER_PAGE))

    def num_free_blocks(self) -> int:
        return len(self.free_list)

    def alloc(self, num: int) -> List[int]:
        out, self.free_list = self.free_list[:num], self.free_list[num:]
        return out

    def free_batch(self, idxs: List[int]) -> None:
        self.free_list.extend(idxs)

    def full(self) -> bool:
        return not self.free_list

    def empty(self) -> bool:
        return len(self.free_list) == BLOCKS_PER_PAGE

    @staticmethod
    def get_num_blocks(page_size: int, block_mem_size: int) -> int:
        return page_size // block_mem_size

    @staticmethod
    def get_block_range(page_id, page_size, block_mem_size):
        return page_id * BLOCKS_PER_PAGE, (page_id + 1) * BLOCKS_PER_PAGE


try:
    import kvcached.vmm_ops as _vmm_ops
except ImportError:
    _vmm_ops = types.ModuleType("kvcached.vmm_ops")
    sys.modules["kvcached.vmm_ops"] = _vmm_ops
# Without the compiled extension, other test files may have installed a stub
# that lacks what kv_cache_manager imports; add only the missing names.
for _name, _value in (
    ("PageAllocator", object),
    ("InternalPage", FakePage),
    ("kv_tensors_created", lambda group_id=0: True),
    ("map_to_kv_tensors", lambda *args, **kwargs: None),
    ("unmap_from_kv_tensors", lambda *args, **kwargs: None),
):
    if not hasattr(_vmm_ops, _name):
        setattr(_vmm_ops, _name, _value)

import kvcached.kv_cache_manager as kcm  # noqa: E402
from kvcached.kv_cache_manager import KVCacheManager  # noqa: E402
from kvcached.lifecycle import LifecycleState  # noqa: E402
from kvcached.locks import NoOpLock  # noqa: E402
from kvcached.utils import KVCachePoolExhausted  # noqa: E402


class ShrinkablePageAllocator:
    """Page accounting with PageAllocator::resize's shrink rule."""

    def __init__(self, page_size: int):
        self.page_size = page_size
        self.total_pages = MAX_PAGES
        self.inuse: set = set()
        # What kvctl's resize watcher publishes; -1 = nothing, as in C++.
        self.resize_target = -1
        self.resize_calls: List[int] = []

    def alloc_page(self) -> FakePage:
        if len(self.inuse) >= self.total_pages:
            raise RuntimeError("no free page")
        page_id = min(set(range(MAX_PAGES)) - self.inuse)
        self.inuse.add(page_id)
        return FakePage(page_id)

    def free_pages(self, page_ids: List[int]) -> None:
        self.inuse.difference_update(page_ids)

    def resize(self, new_mem_size: int) -> bool:
        new_pages = new_mem_size // self.page_size
        self.resize_calls.append(new_pages)
        if new_pages < len(self.inuse):
            return False
        self.total_pages = new_pages
        return True

    def group_indices_by_page(self, indices, block_mem_size) -> Dict[int, List[int]]:
        grouped: Dict[int, List[int]] = {}
        for idx in indices:
            grouped.setdefault(idx // BLOCKS_PER_PAGE, []).append(idx)
        return grouped

    def get_resize_target(self) -> int:
        return self.resize_target

    def get_num_free_pages(self) -> int:
        return max(0, self.total_pages - len(self.inuse))

    def get_avail_physical_pages(self) -> int:
        return MAX_PAGES

    def get_num_reserved_pages(self) -> int:
        return 0

    def get_page_state(self):
        return {
            "total_pages": self.total_pages,
            "free_pages": self.get_num_free_pages(),
            "inuse_pages": len(self.inuse),
            "reserved_pages": 0,
        }


def _make_manager(defer_physical_release: bool) -> KVCacheManager:
    manager = object.__new__(KVCacheManager)
    manager.defer_physical_release = defer_physical_release
    manager.physical_release_barrier = None
    manager._physical_release_epoch = 0
    manager._retired_pages = []
    manager.block_mem_size = 1
    manager.page_size = BLOCKS_PER_PAGE
    manager.num_layers = 1
    manager.num_kv_buffers = 1
    manager.mem_size = manager.page_size * MAX_PAGES
    manager.group_id = 0
    manager._pool_name = "block_pool"
    manager.page_allocator = ShrinkablePageAllocator(manager.page_size)
    manager.num_avail_blocks = 0
    manager.avail_pages = {}
    manager.full_pages = {}
    manager.reserved_blocks = []
    manager.in_shrink = False
    manager.target_num_blocks = None
    manager._resize_rejected = False
    manager._rejected_resize_target = None
    manager._memory_limit_bytes = None
    manager._memory_limit_effective_bytes = None
    manager._memory_limit_revision = -1
    manager._avail_physical_pages_cache = None
    manager._avail_physical_pages_ts = 0.0
    manager._page_release_callbacks = ()
    manager._lock = NoOpLock()
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()
    manager._lifecycle = LifecycleState("prefix-shrink-test")
    manager._lifecycle.mark_ready()
    # reserve_null_block=True: block 0 never enters circulation.
    manager.null_block = manager.alloc(1)
    assert manager.null_block == [0]
    return manager


class MockBlockPool:
    pass


class MockKVCacheBlock:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_cnt = 0
        self.is_null = False
        self.block_hash = None


@pytest.fixture(params=[False, True], ids=["immediate_release", "deferred_release"])
def pool(request, monkeypatch):
    monkeypatch.setattr(kcm, "InternalPage", FakePage)
    manager = _make_manager(defer_physical_release=request.param)
    mod = types.ModuleType("mock_block_pool")
    mod.BlockPool = MockBlockPool  # type: ignore[attr-defined]
    mod.KVCacheBlock = MockKVCacheBlock  # type: ignore[attr-defined]
    # Stub the module itself: other test files re-import the real one, which
    # rebinds the package attribute that mock.patch() would resolve.
    interfaces = types.ModuleType("kvcached.integration.vllm.interfaces")
    interfaces.get_kv_cache_manager = lambda *args, **kwargs: manager  # type: ignore[attr-defined]
    interfaces.init_kvcached = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kvcached.integration.vllm.interfaces", interfaces)
    from kvcached.integration.vllm.patches import ElasticBlockPoolPatch

    ElasticBlockPoolPatch().inject_elastic_block_pool(mod)
    return mod.ElasticBlockPool(  # type: ignore[attr-defined]
        num_gpu_blocks=NUM_BLOCKS,
        block_size=16,
        cell_size=1024,
        num_layers=1,
        enable_caching=True,
        max_cached_blocks=DEFAULT_MAX_CACHED_BLOCKS,
    )


_next_request = 0


def _start(pool, full_blocks: int, partial_blocks: int = 0):
    """Admit a request with a fresh prefix; return its blocks and hashes."""
    global _next_request
    _next_request += 1
    blocks = pool.get_new_blocks(full_blocks + partial_blocks)
    hashes = [f"r{_next_request}-{i}".encode() for i in range(full_blocks)]
    return blocks, hashes


def _finish(pool, blocks, hashes) -> None:
    """Cache the full blocks, release the request and end the engine step."""
    pool.cache_full_blocks(SimpleNamespace(block_hashes=hashes), blocks, 0, len(hashes), 16, 0)
    pool.free_blocks(blocks)
    _end_step(pool)


def _end_step(pool) -> None:
    # The engine drains retired pages after each batch (and on idle wakeups).
    manager = pool.kv_cache_manager
    if manager.defer_physical_release:
        manager.release_retired_pages_through(manager.capture_physical_release_marker())


def _serve(pool, full_blocks: int = BLOCKS_PER_PAGE, partial_blocks: int = 0) -> None:
    """Run one request with a fresh prefix to completion.

    Its full blocks go into the prefix cache and stay mapped as evictable;
    a partial tail block is freed on completion.
    """
    _finish(pool, *_start(pool, full_blocks, partial_blocks))


def _mapped_pages(pool) -> int:
    return len(pool.kv_cache_manager.page_allocator.inuse)


def _fill_prefix_cache(pool) -> None:
    """8 finished requests -> 32 cached blocks + null block on 9 pages."""
    for _ in range(8):
        _serve(pool)
    assert len(pool._evictable_blocks) == 32
    assert _mapped_pages(pool) == 9


LIMIT_PAGES = 4


def _limit_report(pool) -> str:
    state = pool.kv_cache_manager.memory_limit_state()
    return (
        f"status={state['status']} mapped_pages={_mapped_pages(pool)} "
        f"limit_pages={LIMIT_PAGES} "
        f"cached_blocks={len(pool._evictable_blocks)}"
    )


def _assert_within_limit(pool) -> None:
    assert _mapped_pages(pool) <= LIMIT_PAGES, _limit_report(pool)
    assert pool.kv_cache_manager.memory_limit_state()["status"] == "applied"


def _record_eviction_threads(pool, monkeypatch) -> List[str]:
    threads: List[str] = []
    evict = pool._evict_block_ids

    def recording(block_ids):
        threads.append(threading.current_thread().name)
        return evict(block_ids)

    monkeypatch.setattr(pool, "_evict_block_ids", recording)
    return threads


def _set_limit_from_controller(manager, revision: int = 1) -> dict:
    """Call set_memory_limit() the way a control handler does: off-thread."""
    result: dict = {}

    def handler():
        result.update(manager.set_memory_limit(manager.page_size * LIMIT_PAGES, revision=revision))

    thread = threading.Thread(target=handler, name="control-handler")
    thread.start()
    thread.join()
    return result


def test_set_memory_limit_leaves_eviction_to_the_engine_thread(pool, monkeypatch):
    """Only the engine thread may change the prefix cache; a controller's
    set_memory_limit() just records the shrink."""
    _fill_prefix_cache(pool)
    manager = pool.kv_cache_manager
    threads = _record_eviction_threads(pool, monkeypatch)

    state = _set_limit_from_controller(manager)

    assert threads == []
    assert state["status"] == "deferred"
    assert _mapped_pages(pool) == 9

    manager.reclaim_for_shrink()  # what the engine thread does next
    _end_step(pool)

    assert set(threads) == {threading.current_thread().name}
    _assert_within_limit(pool)
    # Only the excess pages are evicted; the rest of the cache survives.
    assert len(pool._evictable_blocks) == 15


def test_kvctl_limit_converges_under_traffic(pool):
    """A `kvctl limit` target is applied by the next allocation. Requests
    shorter than a page used to reuse cached slots forever instead."""
    rng = random.Random(0)
    _fill_prefix_cache(pool)
    manager = pool.kv_cache_manager
    manager.page_allocator.resize_target = manager.page_size * LIMIT_PAGES

    failures = 0
    for _ in range(50):
        try:
            # 1-2 full blocks plus an uncached partial tail, under one page.
            _serve(pool, rng.randint(1, 2), partial_blocks=1)
        except KVCachePoolExhausted:
            failures += 1

    _assert_within_limit(pool)
    assert failures == 0
    assert pool._evictable_blocks


def test_running_request_defers_shrink_until_it_finishes(pool):
    """Pages a running request uses are never revoked; they are reclaimed
    once the request finishes and leaves them to the prefix cache."""
    _fill_prefix_cache(pool)
    manager = pool.kv_cache_manager
    # Six pages' worth: with the null block's page this alone exceeds the limit.
    running = _start(pool, full_blocks=6 * BLOCKS_PER_PAGE)

    manager.set_memory_limit(manager.page_size * LIMIT_PAGES, revision=1)
    manager.reclaim_for_shrink()
    _end_step(pool)
    assert manager.memory_limit_state()["status"] == "deferred"

    _finish(pool, *running)

    _assert_within_limit(pool)


def test_shrink_does_not_reallocate_retiring_pages(pool):
    """Under deferred release, emptied pages stay mapped until the batch
    ends; a pending shrink must not hand them out again meanwhile."""
    _fill_prefix_cache(pool)
    manager = pool.kv_cache_manager
    manager.set_memory_limit(manager.page_size * LIMIT_PAGES, revision=1)
    manager.reclaim_for_shrink()
    retiring = manager._retiring_page_ids()
    assert bool(retiring) == manager.defer_physical_release

    blocks, hashes = _start(pool, full_blocks=BLOCKS_PER_PAGE)

    assert not {b.block_id // BLOCKS_PER_PAGE for b in blocks} & retiring
    _finish(pool, blocks, hashes)
    _assert_within_limit(pool)


def test_zero_kvctl_limit_is_applied(pool):
    """0 is a limit, not the "nothing published" value (-1). The null
    block's page keeps it deferred; every other page is evicted."""
    _fill_prefix_cache(pool)
    manager = pool.kv_cache_manager
    manager.page_allocator.resize_target = 0

    manager.apply_resize_target()
    manager.reclaim_for_shrink()
    _end_step(pool)

    assert manager.page_allocator.resize_calls == [0]
    assert _mapped_pages(pool) == 1
    # Blocks sharing the null block's page cannot free it and stay cached.
    assert {bid // BLOCKS_PER_PAGE for bid in pool._evictable_blocks} == {0}
    assert manager.memory_limit_state()["status"] == "deferred"


def test_unpublished_kvctl_limit_is_a_no_op(pool):
    _fill_prefix_cache(pool)
    manager = pool.kv_cache_manager
    assert manager.page_allocator.resize_target == -1

    manager.apply_resize_target()

    assert manager.page_allocator.resize_calls == []
    assert not manager.in_shrink


@pytest.mark.parametrize("source", ["kvctl", "kvctl_during_startup", "set_memory_limit"])
def test_vllm_idle_engine_applies_lower_limit(pool, monkeypatch, source):
    """An idle engine never allocates, so the limit watch wakes it and the
    engine thread evicts and drains the retired pages."""
    from kvcached.integration.vllm import patches

    wakeup = object()
    engine_mod = types.ModuleType("vllm.v1.engine")
    engine_mod.EngineCoreRequestType = SimpleNamespace(WAKEUP=wakeup)  # type: ignore[attr-defined]
    for name, module in (
        ("vllm", types.ModuleType("vllm")),
        ("vllm.v1", types.ModuleType("vllm.v1")),
        ("vllm.v1.engine", engine_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setenv("KVCACHED_PP_SIZE", "1")

    class EngineCore:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            # An idle async engine: nothing left in its batch queue.
            self.batch_queue: queue.Queue = queue.Queue()
            self.model_executor = SimpleNamespace(collective_rpc=lambda method, args: [True])
            self.scheduler = SimpleNamespace(kv_cache_manager=SimpleNamespace(block_pool=pool))
            self._idle_state_callbacks: list = []
            self.input_queue: queue.Queue = queue.Queue()

    core_mod = types.ModuleType("vllm.v1.engine.core")
    core_mod.EngineCore = EngineCore  # type: ignore[attr-defined]
    assert patches.EngineCorePatch().patch_engine_init(core_mod)

    _fill_prefix_cache(pool)
    manager = pool.kv_cache_manager
    defer = manager.defer_physical_release
    threads = _record_eviction_threads(pool, monkeypatch)
    if source == "kvctl_during_startup":
        manager.page_allocator.resize_target = manager.page_size * LIMIT_PAGES
    engine = EngineCore(
        SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
            scheduler_config=SimpleNamespace(async_scheduling=False),
        )
    )
    # The patched init derives deferred release from the batch queue; keep
    # the fixture's mode so both release paths stay covered.
    manager.defer_physical_release = defer
    stop, thread = engine._kvcached_limit_watch  # type: ignore[attr-defined]
    try:
        if source == "set_memory_limit":
            _set_limit_from_controller(manager)
        else:
            manager.page_allocator.resize_target = manager.page_size * LIMIT_PAGES

        # What EngineCore._process_input_queue does with a wakeup while idle.
        assert engine.input_queue.get(timeout=5) == (wakeup, None)
        while engine._idle_state_callbacks:
            engine._idle_state_callbacks.pop()(engine)
    finally:
        stop.set()
        thread.join()

    assert set(threads) == {threading.current_thread().name}
    _assert_within_limit(pool)


class FakeSGLangCache:
    """Prefix cache with SGLang's evict() contract: LRU tokens, page_size 1."""

    page_size = 1

    def __init__(self, manager):
        self.manager = manager
        self.cached: List[int] = []
        self.token_to_kv_pool_allocator = SimpleNamespace(
            kvcached_allocator=manager, free_group=None
        )
        self.evicting_threads: List[str] = []

    def full_evictable_size(self) -> int:
        return len(self.cached)

    def evict(self, num_tokens: int) -> None:
        self.evicting_threads.append(threading.current_thread().name)
        victims, self.cached = self.cached[:num_tokens], self.cached[num_tokens:]
        free_group = self.token_to_kv_pool_allocator.free_group
        if free_group is None:
            self.manager.free(victims)
        else:
            free_group.append(victims)


@pytest.fixture
def sglang_scheduler(monkeypatch):
    """A scheduler class patched by SchedulerIdleLimitPatch, built on a
    prefix cache holding 32 tokens of 8 finished requests on 9 pages."""
    from kvcached.integration.sglang import patches

    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kcm, "InternalPage", FakePage)
    manager = _make_manager(defer_physical_release=False)
    cache = FakeSGLangCache(manager)
    for _ in range(8):
        blocks = manager.alloc(BLOCKS_PER_PAGE)
        assert blocks is not None
        cache.cached.extend(blocks)
    assert len(manager.page_allocator.inuse) == 9

    class Scheduler:
        def __init__(self, tp_size: int = 1):
            self.tree_cache = cache
            self.token_to_kv_pool_allocator = cache.token_to_kv_pool_allocator
            self.ps = SimpleNamespace(tp_size=tp_size, pp_size=1)

        def on_idle(self):
            pass

    sched_mod = types.ModuleType("sglang.srt.managers.scheduler")
    sched_mod.Scheduler = Scheduler  # type: ignore[attr-defined]
    assert patches.SchedulerIdleLimitPatch().patch_idle_hook(sched_mod)
    return Scheduler


def _sglang_report(manager, cache) -> str:
    return (
        f"status={manager.memory_limit_state()['status']} "
        f"mapped_pages={len(manager.page_allocator.inuse)} "
        f"cached_tokens={len(cache.cached)}"
    )


def _assert_sglang_within_limit(scheduler) -> None:
    cache = scheduler.tree_cache
    manager = cache.manager
    assert len(manager.page_allocator.inuse) <= LIMIT_PAGES, _sglang_report(manager, cache)
    assert manager.memory_limit_state()["status"] == "applied"
    assert cache.cached


def test_sglang_idle_scheduler_applies_kvctl_limit(sglang_scheduler):
    scheduler = sglang_scheduler()
    manager = scheduler.tree_cache.manager
    manager.page_allocator.resize_target = manager.page_size * LIMIT_PAGES

    scheduler.on_idle()

    _assert_sglang_within_limit(scheduler)


def test_sglang_idle_scheduler_applies_set_memory_limit(sglang_scheduler):
    scheduler = sglang_scheduler()
    cache = scheduler.tree_cache

    _set_limit_from_controller(cache.manager)
    assert cache.evicting_threads == []

    scheduler.on_idle()

    assert set(cache.evicting_threads) == {threading.current_thread().name}
    _assert_sglang_within_limit(scheduler)


def test_sglang_allocation_reclaims_before_the_first_idle_pass(sglang_scheduler):
    """Reclaimers exist once the scheduler is built: a busy server may never
    have been idle when an allocation applies a lower limit."""
    scheduler = sglang_scheduler()
    manager = scheduler.tree_cache.manager
    manager.page_allocator.resize_target = manager.page_size * LIMIT_PAGES

    assert manager.alloc(1) is not None

    _assert_sglang_within_limit(scheduler)


def test_sglang_tensor_parallel_keeps_previous_behavior(sglang_scheduler):
    """TP ranks see a new limit at different iterations; evicting then would
    make their radix caches diverge, so nothing changes with TP > 1."""
    scheduler = sglang_scheduler(tp_size=2)
    manager = scheduler.tree_cache.manager
    manager.page_allocator.resize_target = manager.page_size * LIMIT_PAGES

    scheduler.on_idle()

    assert manager.page_allocator.resize_calls == []
    assert manager._shrink_reclaimers == ()


def test_sglang_reclaim_stops_while_frees_are_grouped(sglang_scheduler):
    """Grouped frees reach kvcached only at free_group_end; evicting until
    pages empty would drain the whole cache first."""
    scheduler = sglang_scheduler()
    cache = scheduler.tree_cache
    manager = cache.manager
    cache.token_to_kv_pool_allocator.free_group = []
    manager.page_allocator.resize_target = manager.page_size * LIMIT_PAGES

    scheduler.on_idle()

    excess_pages = 9 - LIMIT_PAGES
    assert len(cache.cached) == 32 - excess_pages * BLOCKS_PER_PAGE
    assert manager.memory_limit_state()["status"] == "deferred"
