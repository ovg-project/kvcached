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


def _config(connector="OffloadingConnector", tp=1, pp=1, v2=True, nested=None):
    extra = {"connectors": [{"kv_connector": name} for name in nested]} if nested else {}
    return types.SimpleNamespace(
        use_v2_model_runner=v2,
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp),
        kv_transfer_config=types.SimpleNamespace(
            kv_connector=connector, kv_connector_extra_config=extra),
    )


@pytest.mark.parametrize("version", ["0.29.0", "0.30.0"])
def test_offloading_is_allowed_where_copies_are_pinned(version):
    from kvcached.integration.vllm.offload_pins import check_offloading_config

    check_offloading_config(_config(), version)


@pytest.mark.parametrize("version,config", [
    ("0.28.0", _config()),
    ("0.31.0", _config()),
    ("0.30.0", _config(v2=False)),
    ("0.30.0", _config(tp=2)),
    ("0.30.0", _config(pp=2)),
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
