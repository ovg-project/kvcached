# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""PP release fences and initialization, without a CUDA runtime."""

import importlib
from types import SimpleNamespace as NS
from unittest import mock

import pytest
import test_sglang_rank_coordination as consensus
from test_sglang_dsv4_kv_pool_patch import _load_modules

pipeline = importlib.import_module(consensus.package.__name__ + ".pipeline")


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_pipeline_startup_preserves_native_or_warms_return_edge(monkeypatch, managed, rank):
    interfaces, _, torch_mock = _load_modules(monkeypatch)
    install = mock.Mock()
    monkeypatch.setattr(interfaces, "set_pipeline_release_barrier", install)
    monkeypatch.setattr(pipeline, "sum_pipeline_counts", lambda sched, count: count * 3)

    class Scheduler:
        ps = NS(tp_size=1, pp_size=3)
        pp_group = mock.Mock(is_last_rank=rank == 2, is_first_rank=rank == 0, device="cuda")
        token_to_kv_pool_allocator = NS(_kvcached_backing_demand=lambda: 0) if managed else None

        def init_pp_loop_state(self):
            self.native_initialized = True

        def _pp_commit_send_output_work_and_preprocess_output_tensors(self):
            pass

    pipeline.patch_pipeline_progress(Scheduler)
    scheduler = Scheduler()
    scheduler.init_pp_loop_state()
    assert scheduler.native_initialized
    assert getattr(scheduler, "_kvcached_native_pipeline", False) == (not managed)
    assert install.call_count == int(managed)
    assert scheduler.pp_group.send_tensor_dict.call_count == int(managed and rank == 2)
    assert scheduler.pp_group.recv_tensor_dict.call_count == int(managed and rank == 0)
    if managed and rank == 2:
        scheduler.pp_group.send_tensor_dict.assert_called_once_with({"warmup": torch_mock.empty.return_value}, dst=0)
    if managed and rank == 0:
        scheduler.pp_group.recv_tensor_dict.assert_called_once_with(src=2)


def test_scheduler_fence_precedes_release_without_device_barrier(monkeypatch):
    from test_kvcache_manager_retired_pages import _manager, _retire

    from kvcached.pool_registry import register_kv_cache_pool

    manager = _manager()
    _retire(manager, [(1, [10])])
    manager._physical_release_epoch = 1
    events: list[object] = []
    scheduler = NS(
        ps=NS(pp_size=2),
        forward_stream=NS(synchronize=lambda: events.append("forward")),
        schedule_stream=NS(synchronize=lambda: events.append("schedule")),
    )
    progress = pipeline.PipelineProgress(scheduler)
    interfaces, _, torch_mock = _load_modules(monkeypatch)
    interfaces.clear_registered_kv_cache_pools(integration="sglang")
    register_kv_cache_pool(manager, integration="sglang")
    try:
        monkeypatch.setattr(interfaces, "_defer_physical_release", True)
        interfaces.set_pipeline_release_barrier(progress.fence_storage)
        manager.page_allocator.free_pages = lambda ids: events.append(("unmap", ids))
        progress.release_storage()
        assert events == ["forward", "schedule", ("unmap", [10])]
        progress.release_storage()
        assert len(events) == 3
        torch_mock.cuda.synchronize.assert_not_called()
    finally:
        interfaces.clear_registered_kv_cache_pools(integration="sglang")


def test_clear_fences_before_mutating_live_pages():
    from test_kvcache_manager_retired_pages import _manager, _retire

    manager = _manager()
    _retire(manager, [(1, [10])])
    manager._pending_kv_copies = {}
    manager.physical_release_barrier = mock.Mock(side_effect=RuntimeError("fence failed"))
    with pytest.raises(RuntimeError, match="fence failed"):
        manager._clear_locked()
    assert list(manager.avail_pages) == [10]
    assert manager._retired_pages == [(1, [10])]
    assert manager.page_allocator.freed_pages == []


def test_pipeline_init_policy_and_future_managers(monkeypatch):
    interfaces, _, torch_mock = _load_modules(monkeypatch)
    interfaces.clear_registered_kv_cache_pools(integration="sglang")
    interfaces.init_kvcached(device="cuda:0", async_sched=True, defer_physical_release=True)
    with pytest.raises(RuntimeError, match="policy"):
        interfaces.init_kvcached(device="cuda:0", async_sched=True)
    calls = []

    def construct(*args, **kwargs):
        calls.append(kwargs)
        return mock.Mock(
            defer_physical_release=kwargs["defer_physical_release"], _uses_worker_ipc=False
        )

    monkeypatch.setattr(interfaces, "KVCacheManager", construct)
    try:
        first = interfaces.get_kv_cache_manager(16, 4, 8, 2)
        first.physical_release_barrier()
        torch_mock.cuda.synchronize.assert_called_once_with("cuda:0")
        fence = mock.Mock()
        interfaces.set_pipeline_release_barrier(fence)
        second = interfaces.get_kv_cache_manager(16, 4, 8, 2)
        assert first.physical_release_barrier is fence
        assert second.physical_release_barrier is fence
        assert all(not call["async_sched"] and call["defer_physical_release"] for call in calls)
    finally:
        interfaces.clear_registered_kv_cache_pools(integration="sglang")


@pytest.mark.parametrize("deferred,ipc", [(False, False), (True, True)])
def test_incompatible_pools_reject_before_changing_barriers(monkeypatch, deferred, ipc):
    interfaces, _, _ = _load_modules(monkeypatch)
    first = NS(defer_physical_release=True, _uses_worker_ipc=False, physical_release_barrier=None)
    second = NS(defer_physical_release=deferred, _uses_worker_ipc=ipc)
    monkeypatch.setattr(interfaces, "_defer_physical_release", True)
    monkeypatch.setattr(
        interfaces,
        "get_registered_kv_cache_pools",
        lambda **kwargs: [(first, "sglang"), (second, "sglang")],
    )
    with pytest.raises(RuntimeError, match="rank-local"):
        interfaces.set_pipeline_release_barrier(lambda: None)
    assert first.physical_release_barrier is None
