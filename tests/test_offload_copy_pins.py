# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Pages under a vLLM OffloadingConnector copy stay mapped until it completes.

The connector frees a finished request's blocks at once and stores them one
build later, on its own CUDA stream. kvcached must neither unmap a page that a
pending copy uses nor let a copy start on a page it has handed back.
"""

import sys
import threading
import types
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

import pytest

_torch_mock = mock.MagicMock()
_torch_mock.__version__ = "2.6.0"
sys.modules.setdefault("torch", _torch_mock)
sys.modules.setdefault("torch.cuda", _torch_mock.cuda)
sys.modules.setdefault("torch.utils", _torch_mock.utils)
sys.modules.setdefault("torch.utils.cpp_extension", _torch_mock.utils.cpp_extension)
sys.modules.setdefault("posix_ipc", mock.MagicMock())
sys.modules.setdefault("kvcached.vmm_ops", mock.MagicMock())

BLOCKS_PER_PAGE = 10


class FakeAllocator:
    def __init__(self):
        self.freed_pages = []
        self.resize = mock.Mock(return_value=True)

    def free_pages(self, page_ids):
        self.freed_pages.extend(page_ids)

    def group_indices_by_page(self, indices, block_mem_size):
        pages: dict[int, list[int]] = {}
        for index in indices:
            pages.setdefault(index // BLOCKS_PER_PAGE, []).append(index)
        return pages


class FakePage:
    def __init__(self, page_id, used):
        self.page_id = page_id
        self.used = set(used)

    def free_batch(self, indices):
        self.used.difference_update(indices)

    def num_free_blocks(self):
        return BLOCKS_PER_PAGE - len(self.used)

    def empty(self):
        return not self.used

    def full(self):
        return len(self.used) == BLOCKS_PER_PAGE


def _manager(pages):
    """A deferred-release manager whose pages hold the given used blocks."""
    from kvcached.kv_cache_manager import KVCacheManager
    from kvcached.locks import NoOpLock

    manager = object.__new__(KVCacheManager)
    manager._lock = NoOpLock()
    manager.page_allocator = FakeAllocator()
    manager.block_mem_size = 1
    manager.defer_physical_release = False
    manager.physical_release_barrier = None
    manager._physical_release_epoch = 0
    manager._retired_pages = []
    manager.in_shrink = False
    manager.target_num_pages = None
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()
    manager.reserved_blocks = []
    manager.avail_pages = {page_id: FakePage(page_id, used)
                           for page_id, used in pages.items()}
    manager.full_pages = {}
    manager.num_avail_blocks = 0
    manager.enable_copy_fence()
    return manager


def _release(manager):
    manager.release_retired_pages_through(manager.capture_physical_release_marker())


def test_a_freed_page_waits_for_the_next_build():
    manager = _manager({1: [10, 11]})

    manager.free([10, 11])
    _release(manager)
    assert manager.page_allocator.freed_pages == []
    assert 1 in manager.avail_pages, "a retired page stays mapped and reusable"

    manager.advance_copy_fence()
    _release(manager)
    assert manager.page_allocator.freed_pages == [1]
    assert manager._retire_fence == {}


def test_a_pinned_page_stays_mapped_until_every_job_unpins():
    manager = _manager({1: [10, 11], 2: [20]})
    manager.pin_copy_job(5, [10, 11, 20])
    manager.pin_copy_job(6, [11])

    manager.free([10, 11, 20])
    manager.advance_copy_fence()
    _release(manager)
    assert manager.page_allocator.freed_pages == []
    assert manager.has_pending_copy_release()

    assert manager.unpin_copy_job(5)
    _release(manager)
    assert manager.page_allocator.freed_pages == [2], "page 1 is still under job 6"

    assert manager.unpin_copy_job(6)
    _release(manager)
    assert manager.page_allocator.freed_pages == [2, 1]
    assert not manager.has_pending_copy_release()


def test_a_copy_cannot_start_on_a_released_page():
    from kvcached.errors import StateConsistencyError

    manager = _manager({1: [10]})
    manager.free([10])
    manager.advance_copy_fence()
    _release(manager)

    with pytest.raises(StateConsistencyError, match="already released"):
        manager.pin_copy_job(7, [10])
    assert manager._copy_pins == {}


def test_pinning_is_idempotent_per_job():
    manager = _manager({1: [10]})
    manager.pin_copy_job(3, [10])
    manager.pin_copy_job(3, [10])
    assert manager._copy_pins == {1: 1}
    assert manager.unpin_copy_job(3)
    assert not manager.unpin_copy_job(3)


def test_without_a_fence_release_is_unchanged():
    manager = _manager({1: [10]})
    manager._copy_fence = None
    manager.free([10])
    _release(manager)
    assert manager.page_allocator.freed_pages == [1]


def test_clear_refuses_pending_copies():
    from kvcached.errors import StateConsistencyError

    manager = _manager({1: [10]})
    manager.pin_copy_job(1, [10])
    with pytest.raises(StateConsistencyError, match="pending"):
        manager._clear_locked()


# --- the vLLM adapter -------------------------------------------------------


class FakeGPUSpec:
    def __init__(self, block_ids):
        self.block_ids = mock.Mock(tolist=lambda: list(block_ids))


@dataclass
class FakeJob:
    src_spec: object
    dst_spec: object


@dataclass
class FakeMeta:
    load_jobs: dict = field(default_factory=dict)
    store_jobs: dict = field(default_factory=dict)


@dataclass
class FakeSchedulerOutput:
    # (req_id, new block ids per group or None, resumed after preemption)
    reqs: list = field(default_factory=list)
    finished_req_ids: set = field(default_factory=set)


def _pins(manager, num_workers=1, windowed=()):
    from kvcached.integration.vllm.offload_pins import OffloadCopyPins

    return OffloadCopyPins(manager, num_workers, FakeGPUSpec, windowed,
                           yield_req_data=lambda output: iter(output.reqs))


def _build(pins, meta=None, reqs=(), finished=()):
    pins.after_build(FakeSchedulerOutput(list(reqs), set(finished)), meta or FakeMeta())


def _store(job_id, block_ids):
    return FakeMeta(store_jobs={job_id: FakeJob(FakeGPUSpec(block_ids), object())})


def test_a_store_built_after_the_free_keeps_the_page_mapped():
    """The window this exists for: the free, then the store's build."""
    manager = _manager({1: [10, 11]})
    pins = _pins(manager)

    manager.free([10, 11])  # the request finished
    _build(pins, _store(9, [10, 11]))
    assert manager.page_allocator.freed_pages == [], "the store reads page 1 later"

    pins.completed({9: 1})
    _build(pins)
    assert manager.page_allocator.freed_pages == [1]


def test_loads_pin_their_destination_blocks():
    manager = _manager({1: [10]})
    pins = _pins(manager)
    _build(pins, FakeMeta(load_jobs={4: FakeJob(object(), FakeGPUSpec([10]))}))
    assert manager._copy_pins == {1: 1}


def test_a_job_unpins_only_when_every_worker_reported_it():
    manager = _manager({1: [10]})
    pins = _pins(manager, num_workers=2)
    _build(pins, _store(9, [10]))

    pins.completed({9: 1})
    assert manager._copy_pins == {1: 1}
    pins.completed({9: 1})
    assert manager._copy_pins == {}


def test_a_batch_queue_drains_retired_pages_itself():
    manager = _manager({1: [10]})
    manager.physical_release_barrier = mock.Mock()
    pins = _pins(manager)
    manager.free([10])
    _build(pins)
    manager.physical_release_barrier.assert_not_called()
    assert manager.page_allocator.freed_pages == []

    _release(manager)  # the step drain
    assert manager.page_allocator.freed_pages == [1]


def test_a_freed_window_block_stays_mapped_until_its_request_last_build():
    """Group 0 is full attention, group 1 a sliding window."""
    manager = _manager({1: [10], 2: [20]})
    pins = _pins(manager, windowed=(1,))
    _build(pins, reqs=[("r", ([20], [10]), False)])

    manager.free([10])  # slid out of the window
    for _ in range(3):
        _build(pins, reqs=[("r", None, False)])
    assert manager.page_allocator.freed_pages == []

    # The connector stores it builds later, e.g. after a failed CPU allocation.
    _build(pins, _store(9, [10]), reqs=[("r", None, False)])
    manager.free([20])
    _build(pins, finished={"r"})
    assert manager.page_allocator.freed_pages == [2], "page 1 is under store 9"
    pins.completed({9: 1})
    _build(pins)
    assert manager.page_allocator.freed_pages == [2, 1]
    assert not manager.has_pending_copy_release()


def test_without_holds_a_late_window_store_finds_its_page_released():
    from kvcached.errors import StateConsistencyError

    manager = _manager({1: [10]})
    pins = _pins(manager)
    _build(pins, reqs=[("r", ([], [10]), False)])
    manager.free([10])
    _build(pins, reqs=[("r", None, False)])
    assert manager.page_allocator.freed_pages == [1]
    with pytest.raises(StateConsistencyError, match="already released"):
        _build(pins, _store(9, [10]), reqs=[("r", None, False)])


def test_a_resumed_request_drops_the_blocks_it_held_before_preemption():
    manager = _manager({1: [10], 3: [30]})
    pins = _pins(manager, windowed=(1,))
    _build(pins, reqs=[("r", ([], [10]), False)])
    manager.free([10])  # preempted
    _build(pins)
    assert manager.page_allocator.freed_pages == []

    _build(pins, reqs=[("r", ([], [30]), True)])
    assert manager.page_allocator.freed_pages == [1]
    assert manager._copy_pins == {3: 1}


@pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1), (1, 3), (2, 3)])
@pytest.mark.parametrize("pending", [False, True])
def test_matching_drops_obsolete_window_holds_before_resumed_admission(
        monkeypatch, pending, tp, pp):
    """Constructed lower-limit/reset-running-requests regression, not serving."""
    from kvcached.integration.vllm.offload_pins import install_offload_copy_pins

    @dataclass
    class WorkerMeta:
        completed_jobs: dict = field(default_factory=dict)

    modules: dict[str, dict[str, Any]] = {
        "vllm.distributed.kv_transfer.kv_connector.utils": {
            "yield_req_data": lambda output: iter(output.reqs)},
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.common": {
            "OffloadingConnectorMetadata": FakeMeta, "OffloadingWorkerMetadata": WorkerMeta},
        "vllm.v1.kv_offload.base": {"GPULoadStoreSpec": FakeGPUSpec},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    manager = _manager({1: [10], 2: [20]})
    old_blocks = [10, 20]

    def match(request, num_computed_tokens, **kwargs):
        # vLLM 0.29/0.30 discard these before checking pending transfer jobs.
        old_blocks.clear()
        return (None if pending else 0), False

    connector = type("OffloadingConnector", (), {})()
    connector.connector_scheduler = types.SimpleNamespace(config=types.SimpleNamespace(
        kv_group_configs=[_group(0, window=8)], num_workers=tp * pp))
    connector.build_connector_meta = lambda output: FakeMeta()
    connector.update_connector_output = lambda output: None
    connector.has_pending_push_work = lambda: False
    connector.get_num_new_matched_tokens = match
    engine = types.SimpleNamespace(
        scheduler=types.SimpleNamespace(connector=connector),
        vllm_config=_config(tp=tp, pp=pp),
        model_executor=types.SimpleNamespace(world_size=tp * pp))
    assert install_offload_copy_pins(engine, manager)
    pins = engine._kvcached_offload_copy_pins
    _build(pins, reqs=[("r", ([10, 20],), False)])
    if pending:
        _build(pins, _store(9, [10]))

    # Lowering the limit and reset_running_requests=True retire the old pages.
    manager.in_shrink, manager.target_num_pages = True, 1
    manager.num_avail_blocks = 18
    manager.page_size = BLOCKS_PER_PAGE
    manager.page_allocator.get_num_free_pages = lambda: 2
    manager.free([10, 20])
    _build(pins)
    assert manager.available_size() == 0
    assert manager.page_allocator.freed_pages == []

    result = connector.get_num_new_matched_tokens(
        types.SimpleNamespace(request_id="r"), 0, max_num_new_tokens=20)
    assert result == ((None if pending else 0), False)
    assert old_blocks == []
    assert pins.held == {}
    assert manager._copy_pins == ({1: 1} if pending else {})
    _build(pins)
    assert set(manager.page_allocator.freed_pages) == ({2} if pending else {1, 2})
    if pending:
        pins.completed({9: tp * pp - 1})
        _build(pins)
        assert manager.page_allocator.freed_pages == [2]
        pins.completed({9: 1})
        _build(pins)
    assert set(manager.page_allocator.freed_pages) == {1, 2}
    assert not manager.in_shrink
    manager.page_allocator.get_avail_physical_pages = lambda: 1
    manager.page_allocator.get_num_reserved_pages = lambda: 0
    monkeypatch.setattr("kvcached.kv_cache_manager.InternalPage", types.SimpleNamespace(
        get_num_blocks=lambda page_size, block_size: page_size // block_size))
    assert manager.available_size() >= BLOCKS_PER_PAGE


def test_full_attention_blocks_are_not_held():
    manager = _manager({1: [10]})
    pins = _pins(manager, windowed=(1,))
    _build(pins, reqs=[("r", ([10], [0]), False)])
    assert manager._copy_pins == {}, "block 0 is the null block"


def _group(idx, window=None, cow=False):
    return types.SimpleNamespace(
        group_idx=idx, sliding_window_size_in_chunks=window, requires_cow_source=cow)


def test_full_attention_and_mamba_align_groups_are_covered_by_the_fence():
    from kvcached.integration.vllm.offload_pins import windowed_groups

    assert windowed_groups([_group(0), _group(1, window=1, cow=True)]) == ()
    assert windowed_groups([_group(0), _group(1, window=8), _group(2, window=1)]) == (1, 2)


# --- the startup gate --------------------------------------------------------


def _config(connector="OffloadingConnector", tp=1, pp=1, v2=True, nested=None,
            **parallel_options):
    extra = {"connectors": [{"kv_connector": name} for name in nested]} if nested else {}
    parallel = dict(tensor_parallel_size=tp, pipeline_parallel_size=pp,
                    distributed_executor_backend="mp" if tp * pp > 1 else "uni")
    parallel.update(parallel_options)
    return types.SimpleNamespace(
        use_v2_model_runner=v2,
        parallel_config=types.SimpleNamespace(**parallel),
        kv_transfer_config=types.SimpleNamespace(
            kv_connector=connector, kv_connector_extra_config=extra),
    )


@pytest.mark.parametrize("version", ["0.29.0", "0.30.0"])
@pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1), (3, 1), (5, 1), (8, 1),
                                  (1, 2), (1, 3), (3, 2), (2, 3)])
def test_offloading_is_allowed_where_copies_are_pinned(version, tp, pp):
    from kvcached.integration.vllm.offload_pins import check_offloading_config

    check_offloading_config(_config(tp=tp, pp=pp), version)


@pytest.mark.parametrize("version,config", [
    ("0.28.0", _config()),
    ("0.31.0", _config()),
    ("0.30.0", _config(v2=False)),
    ("0.30.0", _config(tp=2, data_parallel_size=2)),
    ("0.30.0", _config(tp=2, prefill_context_parallel_size=2)),
    ("0.30.0", _config(tp=2, decode_context_parallel_size=2)),
    ("0.30.0", _config(tp=2, nnodes=2)),
    ("0.30.0", _config(pp=2, nnodes_within_dp=2)),
    ("0.30.0", _config(pp=2, distributed_executor_backend="external_launcher")),
    ("0.30.0", _config(tp=2, distributed_executor_backend="external_launcher")),
    ("0.30.0", _config(tp=2, distributed_executor_backend="ray")),
    ("0.30.0", _config(tp=2, distributed_executor_backend="uni")),
    ("0.30.0", _config(tp=2, distributed_executor_backend=None)),
    ("0.30.0", _config(distributed_executor_backend="ray")),
    ("0.30.0", _config(distributed_executor_backend="external_launcher")),
    ("0.30.0", _config("MultiConnector", nested=["NixlConnector", "OffloadingConnector"])),
])
def test_offloading_is_rejected_elsewhere(version, config):
    from kvcached.integration.vllm.offload_pins import check_offloading_config
    from kvcached.utils import KVCachedConfigError

    with pytest.raises(KVCachedConfigError, match="--kv-offloading-size"):
        check_offloading_config(config, version)


def test_other_connectors_are_untouched():
    from kvcached.integration.vllm.offload_pins import check_offloading_config

    check_offloading_config(_config("NixlConnector", tp=4), "0.20.0")
    check_offloading_config(types.SimpleNamespace(kv_transfer_config=None), "0.30.0")


@pytest.mark.parametrize("num_workers", [2, 3, 4, 6, 10])
@pytest.mark.parametrize("queued", [False, True])
def test_hybrid_copies_wait_for_the_slowest_rank_after_free(num_workers, queued):
    """Scheduler-side lifecycle harness; no CUDA execution or real TP ranks."""
    from kvcached.integration.vllm.offload_pins import windowed_groups

    manager = _manager({1: [10], 2: [20], 3: [30]})
    if queued:
        manager.physical_release_barrier = mock.Mock()
    groups = [_group(0), _group(1, window=1, cow=True)]
    pins = _pins(manager, num_workers, windowed_groups(groups))
    # Finished attention, the exact align boundary, and a cancelled load.
    manager.free([10, 20, 30])
    _build(pins, FakeMeta(
        store_jobs={9: FakeJob(FakeGPUSpec([10]), object()),
                    10: FakeJob(FakeGPUSpec([20]), object())},
        load_jobs={11: FakeJob(object(), FakeGPUSpec([30]))}))
    pins.completed({9: num_workers - 1, 10: 1, 11: num_workers - 1})
    _build(pins)
    _release(manager)
    assert manager.page_allocator.freed_pages == []

    pins.completed({9: 1, 10: num_workers - 1})
    _build(pins)
    _release(manager)
    assert set(manager.page_allocator.freed_pages) == {1, 2}
    assert manager._copy_pins == {3: 1}
    assert manager.has_pending_copy_release()

    pins.completed({11: 1})
    _build(pins)
    _release(manager)
    assert set(manager.page_allocator.freed_pages) == {1, 2, 3}
    assert not manager.has_pending_copy_release()
    assert pins.reports == {}


@pytest.fixture
def offload_engine(monkeypatch):
    """Exercise wrapper installation against the native metadata contract."""
    @dataclass
    class WorkerMeta:
        completed_jobs: dict = field(default_factory=dict)

    modules: dict[str, dict[str, object]] = {
        "vllm.distributed.kv_transfer.kv_connector.utils": {
            "yield_req_data": lambda output: iter(output.reqs)},
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.common": {
            "OffloadingConnectorMetadata": FakeMeta,
            "OffloadingWorkerMetadata": WorkerMeta},
        "vllm.v1.kv_offload.base": {"GPULoadStoreSpec": FakeGPUSpec},
    }
    for path, attributes in modules.items():
        module = types.ModuleType(path)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, path, module)

    class OffloadingConnector:
        def __init__(self, workers):
            self.connector_scheduler = types.SimpleNamespace(config=types.SimpleNamespace(
                num_workers=workers, kv_group_configs=[_group(0), _group(1, 1, True)]))
            self.build_connector_meta = mock.Mock(return_value=_store(9, [10]))
            self.update_connector_output = mock.Mock()
            self.has_pending_push_work = mock.Mock(return_value=False)
            self.get_num_new_matched_tokens = mock.Mock(return_value=(0, False))

    def make(tp, workers, executor_workers, queued=False, pp=1):
        return types.SimpleNamespace(
            vllm_config=_config(tp=tp, pp=pp),
            scheduler=types.SimpleNamespace(connector=OffloadingConnector(workers)),
            model_executor=types.SimpleNamespace(world_size=executor_workers),
            batch_queue=[] if queued else None,
        )
    return make


@pytest.mark.parametrize("workers", [2, 4])
def test_installed_wrapper_drains_only_after_all_workers(offload_engine, workers):
    from kvcached.integration.vllm.offload_pins import install_offload_copy_pins

    engine = offload_engine(workers, workers, workers)
    manager = _manager({1: [10]})
    assert install_offload_copy_pins(engine, manager)
    connector = engine.scheduler.connector
    manager.free([10])
    connector.build_connector_meta(FakeSchedulerOutput())
    for _ in range(workers - 1):
        connector.update_connector_output(types.SimpleNamespace(
            kv_connector_worker_meta=types.SimpleNamespace(completed_jobs={9: 1})))
    _release(manager)
    assert manager.page_allocator.freed_pages == []
    assert connector.has_pending_push_work()

    connector.update_connector_output(types.SimpleNamespace(
        kv_connector_worker_meta=types.SimpleNamespace(completed_jobs={9: 1})))
    _release(manager)
    assert manager.page_allocator.freed_pages == [1]
    assert not connector.has_pending_push_work()


@pytest.mark.parametrize("tp,workers,executor_workers", [
    (2, 1, 2), (2, 4, 2), (4, 4, 1), (4, 4, None), (1, 0, 1),
])
def test_worker_mismatch_fails_before_enabling_copies(
        offload_engine, tp, workers, executor_workers):
    from kvcached.integration.vllm.offload_pins import install_offload_copy_pins
    from kvcached.utils import KVCachedConfigError

    engine = offload_engine(tp, workers, executor_workers)
    manager = _manager({1: [10]})
    manager.enable_copy_fence = mock.Mock()
    original_build = engine.scheduler.connector.build_connector_meta
    with pytest.raises(KVCachedConfigError, match="worker count mismatch"):
        install_offload_copy_pins(engine, manager)
    manager.enable_copy_fence.assert_not_called()
    assert engine.scheduler.connector.build_connector_meta is original_build


@pytest.mark.parametrize("tp,pp", [(2, 1), (1, 3), (3, 2)])
def test_queued_offloading_requires_the_release_barrier(offload_engine, tp, pp):
    from kvcached.integration.vllm.offload_pins import install_offload_copy_pins
    from kvcached.utils import KVCachedConfigError

    engine = offload_engine(tp, tp * pp, tp * pp, queued=True, pp=pp)
    engine.vllm_config.scheduler_config = types.SimpleNamespace(async_scheduling=False)
    manager = _manager({1: [10]})
    with pytest.raises(KVCachedConfigError, match="physical release barrier"):
        install_offload_copy_pins(engine, manager)
    manager.physical_release_barrier = mock.Mock()
    assert install_offload_copy_pins(engine, manager)


# --- rank-private CPU capacity and native initialization contract -------------


@pytest.fixture
def offload_capacity(monkeypatch):
    """Native metadata-shaped inputs; this fixture does not run vLLM or CUDA."""
    from kvcached.integration.vllm import offload_config

    class Attention:
        def __init__(self, page=16):
            self.page_size_bytes = page

    class State:
        def __init__(self, page=16):
            self.page_size_bytes = page

    interface = types.ModuleType("vllm.v1.kv_cache_interface")
    setattr(interface, "AttentionSpec", Attention)
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    native = types.ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.config")
    setattr(native, "build_offloading_config", lambda config, cache: types.SimpleNamespace(
        cache=types.SimpleNamespace(
            blocks_per_chunk=config.kv_transfer_config.kv_connector_extra_config.get(
                "blocks_per_chunk", 1))))
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(offload_config, "enable_kvcached", lambda: True)

    def cache(attention=1, state=0, packed=False, padding=0):
        pools = max(attention, state)
        blocks, page = 8, 16
        size = blocks * page * pools + padding
        tensors, groups = [], []
        for prefix, count, spec in (("a", attention, Attention()), ("s", state, State())):
            names = [f"{prefix}{i}" for i in range(count)]
            groups.append(types.SimpleNamespace(layer_names=names, kv_cache_spec=spec))
            if names:
                tensors.append(types.SimpleNamespace(
                    size=size, layers=names, offset=0,
                    layer_stride=page if packed else blocks * page,
                    block_stride=page * pools if packed else page))
        return types.SimpleNamespace(
            num_blocks=blocks, kv_cache_tensors=tensors, kv_cache_groups=groups)

    def config(tp=1, pp=1, budget=1024, chunk=1):
        cfg = _config(tp=tp, pp=pp)
        cfg.kv_transfer_config.kv_connector_extra_config.update(
            cpu_bytes_to_use=budget, blocks_per_chunk=chunk)
        return cfg

    return types.SimpleNamespace(adapter=offload_config, cache=cache, config=config)


@pytest.mark.parametrize("tp,pp", [(3, 1), (1, 3), (3, 2), (2, 3)])
def test_cpu_budget_counts_every_private_rank_copy(offload_capacity, tp, pp):
    cap = offload_capacity
    caches = [cap.cache(attention=1) if rank < tp else cap.cache(attention=0, state=2)
              for rank in range(tp * pp)]
    per_chunk = 2 * sum(16 if rank < tp else 32 for rank in range(tp * pp))
    cfg = cap.config(tp, pp, budget=per_chunk * 7 + per_chunk - 1, chunk=2)
    plan = cap.adapter.build_offload_plan(cfg, caches)
    assert plan["num_chunks"] == 7
    assert plan["payloads"] == [16 if rank < tp else 32 for rank in range(tp * pp)]
    assert plan["groups"] == [[True, False] if rank < tp else [False, True]
                              for rank in range(tp * pp)]
    assert plan["num_chunks"] * per_chunk <= cfg.kv_transfer_config.kv_connector_extra_config["cpu_bytes_to_use"]


@pytest.mark.parametrize("packed", [False, True])
def test_cpu_budget_deduplicates_attention_state_aliases(offload_capacity, packed):
    cap = offload_capacity
    cache = cap.cache(attention=2, state=2, packed=packed)
    assert cap.adapter.declared_payload(cache) == 32


def test_cpu_budget_uses_payload_not_backing_padding(offload_capacity):
    cap = offload_capacity
    assert cap.adapter.declared_payload(cap.cache(padding=4096)) == 16


def test_state_only_stage_keeps_native_layout_safety_boundary(offload_capacity):
    from kvcached.utils import KVCachedConfigError

    cap = offload_capacity
    assert cap.adapter.declared_payload(cap.cache(0, 2)) == 32
    with pytest.raises(KVCachedConfigError, match="page-contiguous state-only ranks"):
        cap.adapter.declared_payload(cap.cache(0, 2, packed=True))


def test_cpu_budget_rejects_an_empty_stage(offload_capacity):
    from kvcached.utils import KVCachedConfigError

    cap = offload_capacity
    with pytest.raises(KVCachedConfigError, match="without KV tensors"):
        cap.adapter.build_offload_plan(cap.config(pp=2), [cap.cache(), cap.cache(0, 0)])


@pytest.mark.parametrize("budget,expected", [(31, None), (32, 1), (63, 1), (64, 2)])
def test_cpu_budget_chunk_boundary(offload_capacity, budget, expected):
    from kvcached.utils import KVCachedConfigError

    cap = offload_capacity
    cfg = cap.config(pp=2, budget=budget)
    if expected is None:
        with pytest.raises(KVCachedConfigError, match="cannot hold one chunk"):
            cap.adapter.build_offload_plan(cfg, [cap.cache(), cap.cache()])
    else:
        assert cap.adapter.build_offload_plan(cfg, [cap.cache(), cap.cache()])["num_chunks"] == expected


@pytest.mark.parametrize("gib", [0.5, 8.0])
def test_cpu_budget_accepts_native_cli_float_bytes(offload_capacity, gib):
    cap = offload_capacity
    # vLLM's --kv-offloading-size is float GiB, multiplied by 1 << 30.
    budget = gib * (1 << 30)
    assert isinstance(budget, float)
    plan = cap.adapter.build_offload_plan(
        cap.config(pp=2, budget=budget), [cap.cache(), cap.cache()])
    assert plan["num_chunks"] == int(budget) // 32


def test_cpu_budget_truncates_fractional_bytes_like_native(offload_capacity):
    cap = offload_capacity
    plan = cap.adapter.build_offload_plan(
        cap.config(pp=2, budget=63.9), [cap.cache(), cap.cache()])
    assert plan["num_chunks"] == 1


@pytest.mark.parametrize("budget", [
    0.0, -1.0, 0.9, True, "1024", None, float("nan"), float("inf"), -float("inf"),
])
def test_cpu_budget_rejects_invalid_numeric_values(offload_capacity, budget):
    from kvcached.utils import KVCachedConfigError

    cap = offload_capacity
    with pytest.raises(KVCachedConfigError, match="cpu_bytes_to_use"):
        cap.adapter.build_offload_plan(cap.config(budget=budget), [cap.cache()])


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
@pytest.mark.parametrize("axis", ["tp", "pp"])
def test_offload_topology_must_be_positive_integral(axis, value):
    from kvcached.integration.vllm.offload_pins import check_offloading_config
    from kvcached.utils import KVCachedConfigError

    cfg = _config()
    setattr(cfg.parallel_config, f"{'tensor' if axis == 'tp' else 'pipeline'}_parallel_size", value)
    with pytest.raises(KVCachedConfigError, match="positive integer"):
        check_offloading_config(cfg, "0.30.0")


def _capacity_engine(cap, monkeypatch, tp=1, pp=2, failure=None, native=False):
    """Workers already exist with independent config copies when planning runs."""
    from copy import deepcopy

    cfg = cap.config(tp=tp, pp=pp)
    configs = [cap.cache(attention=1) if rank < tp else cap.cache(attention=0, state=2)
               for rank in range(tp * pp)]
    workers = []
    for rank in range(tp * pp):
        worker_cfg = deepcopy(cfg)
        worker_cfg.parallel_config.rank = rank
        workers.append(types.SimpleNamespace(vllm_config=worker_cfg))
    events = []

    class Executor:
        world_size = tp * pp

        def collective_rpc(self, method, args):
            events.append("configure")
            if failure == "rpc":
                raise RuntimeError("RPC failed")
            results = [method(worker, *args) for worker in workers]
            return results[:-1] if failure == "missing_rank" else results

        def initialize_from_config(self, received):
            events.append("initialize")
            assert received is configs
            for worker in workers if not native else ():
                extra = worker.vllm_config.kv_transfer_config.kv_connector_extra_config
                assert extra["spec_name"] == cap.adapter.SPEC_NAME
                assert extra[cap.adapter.PLAN_KEY] == cfg.kv_transfer_config.kv_connector_extra_config[cap.adapter.PLAN_KEY]
            if failure == "initialize":
                raise RuntimeError("KV initialization failed")
            return "initialized"

    class Engine:
        def __init__(self):
            self.model_executor = Executor()

        def _initialize_kv_caches(self, config):
            return self.model_executor.initialize_from_config(configs)

    engine = Engine()
    cap.adapter.patch_offload_initialization(Engine)
    return cfg, engine, workers, events


@pytest.mark.parametrize("tp,pp", [(1, 3), (3, 2)])
def test_capacity_reaches_existing_workers_before_native_initialization(
        offload_capacity, monkeypatch, tp, pp):
    cap = offload_capacity
    cfg, engine, workers, events = _capacity_engine(cap, monkeypatch, tp, pp)
    assert engine._initialize_kv_caches(cfg) == "initialized"
    assert events == ["configure", "initialize"]
    assert "initialize_from_config" not in vars(engine.model_executor)
    plan = cfg.kv_transfer_config.kv_connector_extra_config[cap.adapter.PLAN_KEY]
    assert len(plan["payloads"]) == len(workers) == tp * pp


@pytest.mark.parametrize("tp", [1, 3])
def test_without_pp_native_spec_and_shared_region_are_kept(offload_capacity, monkeypatch, tp):
    cap = offload_capacity
    cfg, engine, workers, events = _capacity_engine(cap, monkeypatch, tp, pp=1, native=True)
    previous_extra = cfg.kv_transfer_config.kv_connector_extra_config
    assert engine._initialize_kv_caches(cfg) == "initialized"
    assert events == ["initialize"]
    assert cfg.kv_transfer_config.kv_connector_extra_config is previous_extra
    for worker in workers:
        extra = worker.vllm_config.kv_transfer_config.kv_connector_extra_config
        assert cap.adapter.PLAN_KEY not in extra and "spec_name" not in extra


@pytest.mark.parametrize("failure", ["rpc", "missing_rank", "initialize"])
def test_failed_initialization_restores_parent_configuration_and_executor(
        offload_capacity, monkeypatch, failure):
    cap = offload_capacity
    cfg, engine, _, events = _capacity_engine(cap, monkeypatch, failure=failure)
    previous_extra = cfg.kv_transfer_config.kv_connector_extra_config
    with pytest.raises((RuntimeError, ValueError)):
        engine._initialize_kv_caches(cfg)
    assert cfg.kv_transfer_config.kv_connector_extra_config is previous_extra
    assert "initialize_from_config" not in vars(engine.model_executor)
    if failure != "initialize":
        assert events == ["configure"]


@dataclass
class FakeCanonicalCaches:
    tensors: list
    group_data_refs: list


def _private_cpu_spec(monkeypatch, capacity_attribute):
    """Fake only the native constructor boundary; no real native/GPU execution."""
    import importlib

    class NativeCPU:
        def _uses_shared_region(self):
            return True

        def __init__(self, config):
            self.config = config
            self.blocks_per_chunk = config.cache.blocks_per_chunk
            self.replicated_layout = config.replicated_layout and self._uses_shared_region()
            setattr(self, capacity_attribute, 12345)

        def create_worker(self, caches):
            self.received = caches
            return caches

        def get_manager(self):
            return types.SimpleNamespace(capacity=getattr(self, capacity_attribute))

    native_module = types.ModuleType("vllm.v1.kv_offload.cpu.spec")
    setattr(native_module, "CPUOffloadingSpec", NativeCPU)
    monkeypatch.setitem(sys.modules, native_module.__name__, native_module)
    path = "kvcached.integration.vllm.offload_spec"
    monkeypatch.delitem(sys.modules, path, raising=False)
    return importlib.import_module(path).KVCachedCPUOffloadingSpec


@pytest.mark.parametrize("capacity_attribute", ["num_blocks", "num_chunks"])
def test_private_spec_capacity_and_empty_groups_match_every_rank(
        offload_capacity, monkeypatch, capacity_attribute):
    from copy import deepcopy

    cap = offload_capacity
    cfg = cap.config(pp=2, budget=959, chunk=2)
    plan = cap.adapter.build_offload_plan(cfg, [cap.cache(), cap.cache(0, 2)])
    spec_cls = _private_cpu_spec(monkeypatch, capacity_attribute)
    for rank, payload in enumerate([16, 32]):
        native = types.SimpleNamespace(
            extra_config={cap.adapter.PLAN_KEY: deepcopy(plan)},
            parallel=types.SimpleNamespace(rank=rank, world_size=2),
            cache=types.SimpleNamespace(blocks_per_chunk=2), replicated_layout=True)
        spec = spec_cls(native)
        assert spec.get_manager().capacity == 9
        assert not spec._uses_shared_region()
        assert not spec.replicated_layout, "every private replica must be written"
        tensor = types.SimpleNamespace(
            numel=lambda: 8 * payload, element_size=lambda: 1,
            untyped_storage=mock.Mock(side_effect=AssertionError("VMM backing is not payload")))
        caches = FakeCanonicalCaches(
            [types.SimpleNamespace(tensor=tensor, page_size_bytes=payload)],
            [["packed"], ["packed"]])
        received = spec.create_worker(caches)
        assert received.group_data_refs == ([["packed"], []] if rank == 0 else [[], ["packed"]])
        assert caches.group_data_refs == [["packed"], ["packed"]]
        tensor.untyped_storage.assert_not_called()


@pytest.mark.parametrize("actual_page,logical_bytes", [(32, 256), (16, 127)])
def test_private_spec_rejects_payload_mismatch_before_allocating_cpu_buffers(
        offload_capacity, monkeypatch, actual_page, logical_bytes):
    from kvcached.utils import KVCachedConfigError

    cap = offload_capacity
    plan = cap.adapter.build_offload_plan(cap.config(), [cap.cache()])
    spec_cls = _private_cpu_spec(monkeypatch, "num_chunks")
    spec = spec_cls(types.SimpleNamespace(
        extra_config={cap.adapter.PLAN_KEY: plan},
        parallel=types.SimpleNamespace(rank=0, world_size=1),
        cache=types.SimpleNamespace(blocks_per_chunk=1), replicated_layout=False))
    tensor = types.SimpleNamespace(numel=lambda: logical_bytes, element_size=lambda: 1)
    caches = FakeCanonicalCaches(
        [types.SimpleNamespace(tensor=tensor, page_size_bytes=actual_page)],
        [["a0"], []])
    with pytest.raises(KVCachedConfigError, match="differs from"):
        spec.create_worker(caches)
    assert not hasattr(spec, "received")
