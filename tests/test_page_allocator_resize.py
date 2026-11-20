# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Real single-GPU controls; these do not validate TP/PP engine execution."""
import gc
import threading
import time
from typing import Any

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from kvcached import _C, control, vmm_ops as vmm  # noqa: E402
from kvcached.kv_cache_manager import KVCacheManager  # noqa: E402
from kvcached.runtime_control import RuntimeControl  # noqa: E402

PAGE = 2 * 1024 * 1024

@pytest.fixture(params=[False, True], ids=["split", "contiguous"])
def pools(monkeypatch, tmp_path, request):
    monkeypatch.setenv("KVCACHED_PHYSICAL_GROWTH_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr("kvcached.kv_cache_manager.PAGE_PREALLOC_ENABLED", False)
    monkeypatch.setattr("kvcached.kv_cache_manager.CONTIGUOUS_LAYOUT", request.param)
    vmm.init_kvcached("cuda:0", PAGE, request.param)
    managers, tensors = [], []
    def create(buffers=2, group=0, null=False, blocks_per_page=1, prealloc=False):
        monkeypatch.setattr("kvcached.kv_cache_manager.PAGE_PREALLOC_ENABLED", prealloc)
        tensors.extend(vmm.create_kv_tensors(8 * PAGE * buffers, 1, "cuda:0", 2,
                                            buffers, group_id=group, page_size=PAGE))
        manager = KVCacheManager(num_blocks=8 * blocks_per_page, block_size=1,
                                 cell_size=PAGE // blocks_per_page,
                                 num_layers=2, num_kv_buffers=buffers, group_id=group,
                                 page_size=PAGE, reserve_null_block=null)
        manager.wait_ready(timeout=10)
        managers.append(manager)
        return manager
    yield create, managers
    for manager in managers:
        manager.shutdown()
    tensors.clear()
    gc.collect()
    vmm.shutdown_kvcached()

def configure(managers, supported=True):
    for manager in managers:
        manager._runtime_control = RuntimeControl(manager, worker_ipc=False,
            can_shrink=supported, expected_participants=len(managers))

def apply(manager):
    manager._runtime_control.poll(force=True)
    manager.apply_resize_target()

@pytest.mark.parametrize("buffers", [1, 2])
def test_real_geometry_revision_shrink_regrow(pools, buffers):
    create, managers = pools
    manager = create(buffers)
    configure(managers)
    source = manager._runtime_control.sources[0]
    bundle = PAGE * 2 * buffers
    assert source["pool_capacity_bytes"] == 8 * bundle
    blocks = manager.alloc(4)
    assert blocks is not None
    result = manager.set_memory_limit(2 * bundle)
    assert result["accepted"] and not result["converged"]
    apply(manager)
    assert manager.in_shrink
    assert not control.get_instance_memory_limit()["converged"]
    manager.free(blocks)
    apply(manager)
    assert control.get_instance_memory_limit()["converged"]
    old_revision = result["gpus"][0]["revision"]
    same = manager.set_memory_limit(2 * bundle)
    assert same["gpus"][0]["revision"] == old_revision + 1
    assert not _C.acknowledge_instance_control(0, old_revision)
    manager.set_memory_limit(100 * bundle)
    apply(manager)
    assert manager.page_allocator.get_num_total_pages() == 8
    assert manager.page_allocator.get_resize_target() == -1

def test_real_mixed_pools_share_cap(pools):
    create, managers = pools
    create(1, 0)
    create(2, 1)
    configure(managers)
    assert control.set_instance_memory_limit(9 * PAGE + 1)["accepted"]
    for manager in managers:
        apply(manager)
    state = control.get_instance_memory_limit()
    assert len(state["gpus"]) == 1
    assert state["gpus"][0]["effective_limit_bytes"] <= 9 * PAGE + 1
    assert [m.page_allocator.get_num_total_pages() for m in managers] == [1, 1]
    assert state["converged"]

def test_real_zero_clear_preserves_pending_revision(pools):
    create, managers = pools
    manager = create(null=True)
    configure(managers)
    result = manager.set_memory_limit(0)
    apply(manager)
    assert not manager.in_shrink
    assert manager.page_allocator.get_num_total_pages() == 1
    manager.clear()
    assert manager.null_block == [0] and not manager.in_shrink
    assert manager.page_allocator.get_num_total_pages() == 1
    state = control.get_instance_memory_limit()
    assert not state["converged"]
    assert state["gpus"][0]["revision"] == result["gpus"][0]["revision"]
    assert state["gpus"][0]["charged_bytes"] > 0
    manager.set_memory_limit(16 * PAGE)
    apply(manager)
    blocks = manager.alloc(1)
    assert blocks and 0 not in blocks
    manager.free(blocks)

def test_real_unsupported_shrink_blocks_growth(pools):
    create, managers = pools
    manager = create()
    configure(managers, supported=False)
    result = manager.set_memory_limit(0)
    assert result["gpus"][0]["convergence"] == "unsupported"
    assert control.get_instance_memory_limit()["status"] == "unsupported"
    apply(manager)
    assert not manager.in_shrink
    assert manager.alloc(1) is None


def test_real_cli_api_share_publication_and_observation(pools, capsys):
    import json

    from kvcached.cli.kvctl import cmd_list
    from kvcached.cli.utils import update_kv_cache_limit
    from kvcached.utils import DEFAULT_IPC_NAME
    create, managers = pools
    manager = create()
    configure(managers)
    first = update_kv_cache_limit(DEFAULT_IPC_NAME, 8 * PAGE)
    second = manager.set_memory_limit(8 * PAGE)
    assert second["gpus"][0]["revision"] == first["gpus"][0]["revision"] + 1
    apply(manager)
    capsys.readouterr()
    cmd_list([DEFAULT_IPC_NAME], json_out=True)
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["gpus"][0]["revision"] == second["gpus"][0]["revision"]
    assert rows[0]["limit_bytes"] == 8 * PAGE


@pytest.mark.parametrize("limit", [0, 1])
def test_real_mixed_null_pools_keep_physical_floor_and_exact_cap(pools, limit):
    create, managers = pools
    create(1, 0, null=True)
    create(2, 1, null=True)
    create(1, 2, null=False)
    configure(managers)
    accepted = control.set_instance_memory_limit(limit)
    assert accepted["accepted"]
    for manager in managers:
        apply(manager)
    state = control.get_instance_memory_limit()
    assert state["status"] == "pending"
    assert state["gpus"][0]["limit_bytes"] == limit
    assert state["gpus"][0]["charged_bytes"] == 6 * PAGE
    assert state["gpus"][0]["effective_limit_bytes"] == 6 * PAGE
    assert [m.page_allocator.get_num_total_pages() for m in managers] == [1, 1, 0]
    assert managers[2].alloc(1) is None
    control.set_instance_memory_limit(12 * PAGE)
    for manager in managers:
        apply(manager)
    assert control.get_instance_memory_limit()["converged"]


@pytest.mark.parametrize("phase", ["unpublished", "converged", "pending"])
def test_real_policy_preserves_resize_execution(pools, phase):
    create, managers = pools
    manager = create()
    configure(managers)
    blocks = manager.alloc(4) if phase == "pending" else []
    if phase != "unpublished":
        manager.set_memory_limit(8 * PAGE)
        apply(manager)
    before = control.get_instance_memory_limit()["gpus"][0]
    counters = dict(manager._operation_counters)
    manager._avail_physical_pages_cache = 321
    policies: tuple[dict[str, Any], ...] = (
        {"priority": 9}, {"min_bytes": PAGE}, {"headroom_bytes": PAGE})
    for policy in policies:
        assert control.set_instance_controls(**policy)["accepted"]
        apply(manager)
        after = control.get_instance_memory_limit()["gpus"][0]
        assert after["limit_revision"] == before["limit_revision"]
        assert after["convergence"] == before["convergence"]
        assert after["published"] == before["published"]
        assert manager._operation_counters == counters
        assert manager._avail_physical_pages_cache == 321
    if blocks:
        manager.free(blocks)
        apply(manager)
        assert control.get_instance_memory_limit()["converged"]
    # Even an unchanged target is a fresh request when max is explicit.
    manager.set_memory_limit(8 * PAGE)
    apply(manager)
    assert control.get_instance_memory_limit()["converged"]
    assert manager._operation_counters["resize_requests_total"] == counters.get("resize_requests_total", 0) + 1


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("fail_resize", [False, True])
def test_real_policy_interleaves_with_execution_report(pools, monkeypatch, deferred, fail_resize):
    """Real ledger/allocator, with a constructed publication race and failure."""
    from kvcached.errors import QuarantinedResizeError

    create, managers = pools
    manager = create()
    configure(managers)
    blocks = manager.alloc(4) if deferred else []
    manager.set_memory_limit(8 * PAGE)
    if deferred:
        apply(manager)
        assert manager.in_shrink
    allocator_type = type(manager.page_allocator)
    resize = allocator_type.resize

    def interleaved(allocator, size):
        control.set_instance_controls(priority=9)
        if fail_resize:
            raise QuarantinedResizeError("injected quarantine")
        return resize(allocator, size)

    monkeypatch.setattr(allocator_type, "resize", interleaved)
    if deferred:
        manager.free(blocks)
    else:
        apply(manager)
    expected = "failed" if fail_resize else "converged"
    assert control.get_instance_memory_limit()["status"] == expected
    assert not manager.in_shrink
    counters = dict(manager._operation_counters)
    policies: tuple[dict[str, Any], ...] = (
        {"priority": 7}, {"min_bytes": PAGE}, {"headroom_bytes": PAGE})
    for policy in policies:
        control.set_instance_controls(**policy)
        apply(manager)
        assert control.get_instance_memory_limit()["status"] == expected
        assert manager._operation_counters == counters
    monkeypatch.setattr(allocator_type, "resize", resize)
    manager.set_memory_limit(8 * PAGE)
    apply(manager)
    assert control.get_instance_memory_limit()["converged"]
    assert manager._operation_counters["resize_requests_total"] == 2


@pytest.mark.parametrize("fatal", [False, True])
def test_real_pending_observes_late_prealloc_failure(pools, monkeypatch, fatal):
    """Native background thread and ledger, with a constructed IPC failure."""
    from kvcached.errors import MapQuarantinedError, StateConsistencyError

    create, managers = pools
    allocator_type = vmm.PageAllocator
    start = allocator_type.start_prealloc_thread
    monkeypatch.setattr(allocator_type, "start_prealloc_thread", lambda self: None)
    manager = create(prealloc=True)
    configure(managers)
    blocks = manager.alloc(2)
    assert blocks is not None
    entered, release = threading.Event(), threading.Event()

    def fail(_world_size, _offsets):
        entered.set()
        assert release.wait(10)
        error = StateConsistencyError if fatal else MapQuarantinedError
        raise error("injected late preallocation failure")

    allocator = manager.page_allocator
    allocator.set_use_worker_ipc(True)
    allocator.set_broadcast_map_callback(fail)
    try:
        start(allocator)
        assert entered.wait(10)
        manager.set_memory_limit(4 * PAGE)
        apply(manager)
        assert manager.in_shrink
        assert manager._get_num_alloced_blocks() == 2
    finally:
        release.set()
    deadline = time.monotonic() + 10
    while allocator.get_transaction_state()["state"] == "HEALTHY" and time.monotonic() < deadline:
        time.sleep(0.01)
    allocator.stop_prealloc_thread()
    assert allocator.get_transaction_state()["state"] == ("FAILED" if fatal else "DEGRADED")
    control.set_instance_controls(priority=7)
    if fatal:
        with pytest.raises(StateConsistencyError):
            apply(manager)
        assert manager._lifecycle.phase == "failed"
    else:
        apply(manager)
        assert not manager.in_shrink
        assert control.get_instance_memory_limit()["status"] == "failed"
        assert manager._operation_counters["resize_errors_total"] == 1
        for _ in range(2):
            control.set_instance_controls(headroom_bytes=PAGE)
            apply(manager)
        assert manager._operation_counters["resize_errors_total"] == 1
        manager.free(blocks)
    assert manager._operation_counters["resize_requests_total"] == 1


def test_real_pending_releases_late_reserved_blocks(pools):
    create, managers = pools
    manager = create(blocks_per_page=2)
    configure(managers)
    blocks = manager.alloc(1)
    assert blocks is not None
    manager.set_memory_limit(0)
    apply(manager)
    assert manager.in_shrink
    assert manager.try_to_reserve(1)
    manager.free(blocks)
    assert manager.in_shrink and manager.reserved_blocks
    control.set_instance_controls(priority=7)
    apply(manager)
    assert not manager.in_shrink and not manager.reserved_blocks
    assert control.get_instance_memory_limit()["converged"]
    assert manager._operation_counters["resize_requests_total"] == 1


def test_real_policy_revision_cannot_acknowledge_a_max_retry(pools):
    create, managers = pools
    manager = create()
    configure(managers)
    manager.set_memory_limit(8 * PAGE)
    apply(manager)
    state = control.set_instance_controls(priority=7)["gpus"][0]
    generations = {state["gpu"]: state["generation"]}
    rejected = control.set_instance_memory_limit(
        8 * PAGE, revision=state["revision"], generations=generations)
    assert not rejected["accepted"]
    assert rejected["gpus"][0]["status"] == "conflict"
    apply(manager)
    assert manager._operation_counters["resize_requests_total"] == 1
    fresh = manager.set_memory_limit(8 * PAGE)
    apply(manager)
    assert fresh["accepted"] and control.get_instance_memory_limit()["converged"]
    repeated = control.set_instance_memory_limit(
        8 * PAGE, revision=fresh["gpus"][0]["revision"], generations=generations)
    assert repeated["accepted"] and repeated["gpus"][0]["status"] == "idempotent"
    apply(manager)
    assert manager._operation_counters["resize_requests_total"] == 2
