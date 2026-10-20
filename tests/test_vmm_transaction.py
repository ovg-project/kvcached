# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.mark.parametrize("phase", ["prepare", "commit"])
@pytest.mark.parametrize("num_pages", [1, 2])
def test_unknown_map_outcome_survives_native_callback_without_unmap(monkeypatch, phase, num_pages):
    """Exercise Python -> C++ alloc_page -> Python IPC -> C++ -> Python."""
    import uuid

    from kvcached import tp_ipc_util as ipc
    from kvcached.errors import StateConsistencyError

    vmm_ops = _compiled_vmm_ops()
    offsets, _ = _create_layout(vmm_ops, False, False)
    allocator = vmm_ops.PageAllocator(
        num_layers=2, mem_size_per_layer=8 * 1024 * 1024,
        page_size=2 * 1024 * 1024, world_size=1, pp_rank=0,
        async_sched=False, contiguous_layout=False, enable_page_prealloc=False,
        num_kv_buffers=2, group_id=0, ipc_name="UNKNOWN_" + uuid.uuid4().hex[:8],
    )
    allocator.set_use_worker_ipc(True)
    allocator.set_broadcast_map_callback(ipc.broadcast_map_to_kv_tensors)
    monkeypatch.setattr(ipc, "_UNRESOLVED_PHYSICAL_GROWTH_TRANSACTIONS", {})
    monkeypatch.setattr(ipc, "_PHYSICAL_DEVICE_ID_CACHE", {(0, 0): "gpu-0"})
    transactions = []
    commands = []

    async def exchange(rank, message, pp_rank=0):
        command = message["cmd"]
        commands.append(command)
        transaction = message["transaction_id"]
        if command == "prepare_map_to_kv_tensors":
            transactions.append(transaction)
            assert vmm_ops.prepare_map_to_kv_tensors(transaction, message["offsets"], 0)["success"]
            if phase == "prepare":
                raise ConnectionError("lost prepare response")
            return {"status": "success", "transaction_state": "reserved"}
        if command == "commit_prepared_map":
            assert vmm_ops.commit_prepared_map(transaction, 0)["success"]
            raise ConnectionError("lost commit response")
        if command == "get_map_transaction_state":
            raise ConnectionError("lost state query response")
        pytest.fail(f"unexpected cleanup or retry: {command}")

    monkeypatch.setattr(ipc, "_send_and_receive_message", exchange)
    try:
        with pytest.raises(StateConsistencyError, match="restart"):
            allocator.alloc_pages(num_pages)
        state = allocator.get_transaction_state()
        assert state["state"] == "FAILED"
        assert state["quarantined_page_ids"] == list(range(num_pages))
        with pytest.raises(StateConsistencyError):
            allocator.get_num_free_pages()
        assert len(transactions) == 1
        assert commands.count("get_map_transaction_state") == 2
        calls = len(commands)
        with pytest.raises(StateConsistencyError):
            allocator.alloc_pages(num_pages)
        assert len(commands) == calls
        if phase == "prepare":
            assert vmm_ops.has_prepared_map(transactions[0], 0)
        else:
            # Idempotent native map reports no newly mapped offsets: the
            # original mapping was retained, not silently rolled back.
            assert vmm_ops.map_to_kv_tensors_with_result([offsets[0]], 0) == (True, [])
    finally:
        del allocator
        vmm_ops.shutdown_kvcached()


@pytest.mark.parametrize("page_mib", [2, 4])
@pytest.mark.parametrize("contiguous,buffers", [(False, 1), (False, 2), (True, 1), (True, 2)])
def test_prepared_map_uses_pool_page_size_and_buffer_geometry(page_mib, contiguous, buffers):
    import torch

    vmm = _compiled_vmm_ops()
    page = page_mib * 1024 * 1024
    layers = 2
    # The per-pool page can differ from the process default (#529/#540).
    vmm.init_kvcached("cuda:0", 2 * 1024 * 1024, contiguous)
    try:
        tensors = vmm.create_kv_tensors(
            8 * page, 1, "cuda:0", layers, buffers, page_size=page)
        # Non-contiguous MLA's upper half is usable, not a mirrored V region.
        offset = 4 * page if not contiguous and buffers == 1 else 0
        prepared = vmm.prepare_map_to_kv_tensors("geometry-abort", [offset], 0)
        assert prepared["success"]
        assert prepared["targets_count"] == (1 if contiguous else layers * buffers)
        assert prepared["required_bytes"] == page * layers * buffers
        assert vmm.abort_prepared_map("geometry-abort", 0)
        assert vmm.prepare_map_to_kv_tensors("geometry-commit", [offset], 0)["success"]
        assert vmm.commit_prepared_map("geometry-commit", 0)["success"]
        views = [tensor[start:start + 128] for tensor in tensors
                 for start in ([offset] if contiguous or buffers == 1 else [0, 4 * page])]
        for index, view in enumerate(views):
            view.fill_(index + 9)
        torch.cuda.synchronize()
        adopted = vmm.prepare_map_to_kv_tensors("geometry-adopt", [offset], 0)
        assert adopted["success"] and adopted["required_bytes"] == 0
        assert vmm.commit_prepared_map("geometry-adopt", 0)["success"]
        assert vmm.prepare_unmap_from_kv_tensors([offset], "geometry-release", 0)
        assert vmm.abort_unmap_from_kv_tensors("geometry-release", 0)
        for index, view in enumerate(views):
            assert torch.all(view == index + 9).item()
        torch.cuda.synchronize()
        assert vmm.unmap_from_kv_tensors([offset], 0)
    finally:
        vmm.shutdown_kvcached()


def test_native_allocator_preserves_recoverable_callback_error():
    import uuid

    vmm_ops = _compiled_vmm_ops()
    allocator = vmm_ops.PageAllocator(
        num_layers=1, mem_size_per_layer=8 * 1024 * 1024,
        page_size=2 * 1024 * 1024, world_size=1, pp_rank=0,
        async_sched=False, contiguous_layout=False, enable_page_prealloc=False,
        num_kv_buffers=2, group_id=0, ipc_name="MISS_" + uuid.uuid4().hex[:8],
    )
    allocator.set_use_worker_ipc(True)

    def fail(*args):
        raise RuntimeError("capacity_exhausted")

    allocator.set_broadcast_map_callback(fail)
    before = allocator.get_num_free_pages()
    with pytest.raises(RuntimeError, match="capacity_exhausted"):
        allocator.alloc_page()
    assert allocator.get_num_free_pages() == before
    assert allocator.get_num_inuse_pages() == 0


def test_background_unknown_outcome_reaches_manager_capacity_check():
    import os
    import subprocess
    import sys

    _compiled_vmm_ops()
    # Native reserved-page settings are captured when the extension loads.
    env = dict(os.environ, KVCACHED_MIN_RESERVED_PAGES="1", KVCACHED_MAX_RESERVED_PAGES="1")
    result = subprocess.run(
        [sys.executable, "-c", "import runpy, sys; "
         "runpy.run_path(sys.argv[1])['_check_background_unknown_outcome']()", __file__],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _batch_allocator(vmm_ops, *, contiguous_layout=False):
    import uuid

    page_size = 2 * 1024 * 1024
    return vmm_ops.PageAllocator(
        num_layers=2, mem_size_per_layer=4 * page_size,
        page_size=page_size, world_size=1, pp_rank=0,
        async_sched=False, contiguous_layout=contiguous_layout,
        enable_page_prealloc=False, num_kv_buffers=2,
        group_id=0, ipc_name="BATCH_" + uuid.uuid4().hex[:8],
    )


@pytest.mark.parametrize("contiguous_layout", [False, True])
def test_native_batch_maps_all_new_offsets_once_and_reuses_reserved(contiguous_layout):
    _run_reserved_batch_case("_check_batch_reuse", contiguous_layout)


def _run_reserved_batch_case(name, argument):
    import json
    import os
    import subprocess
    import sys

    _compiled_vmm_ops()
    env = dict(os.environ, KVCACHED_MIN_RESERVED_PAGES="0", KVCACHED_MAX_RESERVED_PAGES="1")
    result = subprocess.run(
        [sys.executable, "-c", "import json, runpy, sys; "
         "runpy.run_path(sys.argv[1])[sys.argv[2]](json.loads(sys.argv[3]))",
         __file__, name, json.dumps(argument)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_batch_reuse(contiguous_layout):
    v = _compiled_vmm_ops()
    page_size = 2 * 1024 * 1024
    v.init_kvcached("cuda:0", page_size, contiguous_layout)
    # create_kv_tensors takes the per-layer K+V size, whereas PageAllocator
    # takes the size of one buffer. Reserve four pages for each K/V buffer.
    v.create_kv_tensors(8 * page_size, 2, "cuda:0", 2, 2, 0, False)
    allocator = _batch_allocator(v, contiguous_layout=contiguous_layout)
    allocator.set_use_worker_ipc(True)
    calls = []
    stride = 2 * 1024 * 1024 * (4 if contiguous_layout else 1)

    def map_batch(world_size, offsets):
        calls.append(list(offsets))
        assert v.map_to_kv_tensors_with_result(offsets, 0) == (True, list(offsets))

    allocator.set_broadcast_map_callback(map_batch)
    allocator.set_broadcast_unmap_callback(lambda ws, offsets: v.unmap_from_kv_tensors(offsets, 0))
    try:
        pages = allocator.alloc_pages(2)
        assert [p.page_id for p in pages] == [0, 1]
        assert calls == [[0, stride]]
        allocator.free_page(0)
        assert allocator.get_num_reserved_pages() == 1
        pages = allocator.alloc_pages(3)
        assert [p.page_id for p in pages] == [0, 2, 3]
        assert calls == [[0, stride], [2 * stride, 3 * stride]]
        assert allocator.get_num_free_pages() == 0
        # Every offset is already mapped, including the reused reserved page.
        assert v.map_to_kv_tensors_with_result([i * stride for i in range(4)], 0) == (True, [])
        allocator.free_pages([0, 1, 2, 3])
        allocator.trim()
        assert allocator.get_num_free_pages() == 4
    finally:
        del allocator
        v.shutdown_kvcached()


@pytest.mark.parametrize("failure", ["recoverable", "quarantined", "unknown"])
def test_native_batch_failure_restores_only_safe_page_ids(failure):
    _run_reserved_batch_case("_check_batch_failure", failure)


def _check_batch_failure(failure):
    from kvcached.errors import MapQuarantinedError, StateConsistencyError
    from kvcached.tp_ipc_util import MapTransactionOutcomeUnknownError

    v = _compiled_vmm_ops()
    allocator = _batch_allocator(v)
    allocator.set_use_worker_ipc(True)
    calls, unmaps = [], []
    allocator.set_broadcast_map_callback(lambda ws, offsets: calls.append(list(offsets)))
    allocator.set_broadcast_unmap_callback(lambda ws, offsets: unmaps.append(list(offsets)))
    allocator.alloc_pages(2)
    allocator.free_page(0)
    error_type = {"recoverable": RuntimeError, "quarantined": MapQuarantinedError,
                  "unknown": MapTransactionOutcomeUnknownError}[failure]
    expected_type = StateConsistencyError if failure == "unknown" else error_type

    def fail(ws, offsets):
        raise error_type("injected batch failure")

    allocator.set_broadcast_map_callback(fail)
    with pytest.raises(expected_type, match="injected batch failure"):
        allocator.alloc_pages(3)
    assert allocator.get_num_reserved_pages() == 1
    assert unmaps == []
    allocator.set_broadcast_map_callback(lambda ws, offsets: calls.append(list(offsets)))
    if failure == "recoverable":
        assert allocator.get_num_free_pages() == 3
        assert allocator.get_num_inuse_pages() == 1
        assert [p.page_id for p in allocator.alloc_pages(3)] == [0, 2, 3]
        assert len(calls[-1]) == 2
    else:
        assert allocator.get_transaction_state()["quarantined_page_ids"] == [2, 3]
        if failure == "unknown":
            with pytest.raises(StateConsistencyError):
                allocator.alloc_page()
        else:
            assert allocator.get_num_free_pages() == 1
            assert allocator.alloc_page().page_id == 0
            with pytest.raises(RuntimeError, match="free pages"):
                allocator.alloc_page()


def test_native_batch_invalid_or_exhausted_request_does_not_mutate_state():
    v = _compiled_vmm_ops()
    allocator = _batch_allocator(v)
    allocator.set_use_worker_ipc(True)
    calls = []
    allocator.set_broadcast_map_callback(lambda ws, offsets: calls.append(list(offsets)))
    assert allocator.alloc_pages(0) == []
    with pytest.raises(ValueError, match="non-negative"):
        allocator.alloc_pages(-1)
    with pytest.raises(RuntimeError, match="free pages"):
        allocator.alloc_pages(5)
    assert allocator.get_num_free_pages() == 4
    assert allocator.get_num_reserved_pages() == 0
    assert calls == []
    assert [p.page_id for p in allocator.alloc_pages(4)] == list(range(4))
    assert len(calls) == 1


def _check_background_unknown_outcome():
    import threading
    import uuid

    from kvcached import tp_ipc_util as ipc
    from kvcached.kv_cache_manager import KVCacheManager
    from kvcached.lifecycle import LifecyclePhase, LifecycleState
    from kvcached.locks import NoOpLock

    vmm_ops = _compiled_vmm_ops()
    entered = threading.Event()
    allocator = vmm_ops.PageAllocator(
        num_layers=1, mem_size_per_layer=16 * 1024 * 1024,
        page_size=2 * 1024 * 1024, world_size=1, pp_rank=0,
        async_sched=False, contiguous_layout=False, enable_page_prealloc=True,
        num_kv_buffers=2, group_id=0, ipc_name="BACKGROUND_" + uuid.uuid4().hex[:8],
    )
    allocator.set_use_worker_ipc(True)

    def fail(*args):
        try:
            ipc._fail_unresolved_map_transaction("background", [0], 0, "prepare", ["lost"])
        finally:
            entered.set()

    allocator.set_broadcast_map_callback(fail)
    manager = object.__new__(KVCacheManager)
    manager._lock = NoOpLock()
    manager.page_allocator = allocator
    manager.num_avail_blocks = 0
    manager.reserved_blocks = []
    manager.in_shrink = False
    manager._lifecycle = LifecycleState("background-unknown")
    manager._lifecycle.mark_ready()
    try:
        allocator.start_prealloc_thread()
        assert entered.wait(10), "preallocator did not exercise the callback"
        with pytest.raises(ipc.MapTransactionOutcomeUnknownError, match="background"):
            manager.available_size()
        assert manager.lifecycle_phase is LifecyclePhase.FAILED
        with pytest.raises(ipc.MapTransactionOutcomeUnknownError):
            manager.wait_ready()
    finally:
        allocator.stop_prealloc_thread()


def _compiled_vmm_ops():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for VMM transaction validation")

    try:
        from kvcached import vmm_ops
    except ImportError:
        pytest.skip("kvcached.vmm_ops extension is not built")
    required_operations = (
        "init_kvcached",
        "create_kv_tensors",
        "map_to_kv_tensors_with_result",
        "prepare_map_to_kv_tensors",
        "commit_prepared_map",
        "abort_prepared_map",
        "has_prepared_map",
        "unmap_from_kv_tensors",
        "prepare_unmap_from_kv_tensors",
        "commit_unmap_from_kv_tensors",
        "abort_unmap_from_kv_tensors",
        "current_device_pci_bus_id",
        "shutdown_kvcached",
    )
    if any(not hasattr(vmm_ops, name) for name in required_operations):
        pytest.skip("a test stub replaced the compiled kvcached.vmm_ops module")
    return vmm_ops


@pytest.mark.parametrize("outcome", ["success", "capacity", "invalid-utilization"])
def test_prepared_growth_preserves_callers_cuda_device(monkeypatch, tmp_path, outcome):
    import torch

    native = _compiled_vmm_ops()
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required for cross-device admission")
    original_device = torch.cuda.current_device()
    page = 2 * 1024 * 1024
    monkeypatch.setenv("KVCACHED_PHYSICAL_GROWTH_LOCK_DIR", str(tmp_path))
    native.init_kvcached("cuda:1", page, False)
    try:
        tensors = native.create_kv_tensors(4 * page, 1, "cuda:1", 1, 1, page_size=page)
        torch.cuda.set_device(0)
        utilization = {"success": "1", "capacity": "1e-20", "invalid-utilization": "invalid"}
        monkeypatch.setenv("KVCACHED_GPU_UTILIZATION", utilization[outcome])
        if outcome == "success":
            assert native.prepare_map_to_kv_tensors("device-scope", [0], 0)["success"]
            assert torch.cuda.current_device() == 0
            assert native.commit_prepared_map("device-scope", 0)["success"]
            tensors[0][:128].fill_(19)
            torch.cuda.synchronize(1)
            assert torch.all(tensors[0][:128] == 19).item()
        else:
            result = native.prepare_map_to_kv_tensors("device-scope", [0], 0)
            assert not result["success"]
            assert result["capacity_rejections"] == (1 if outcome == "capacity" else 0)
            assert not native.has_prepared_map("device-scope", 0)
        assert torch.cuda.current_device() == 0
    finally:
        torch.cuda.set_device(1)
        native.shutdown_kvcached()
        torch.cuda.set_device(original_device)


def _create_layout(vmm_ops, contiguous_layout, unified_pool):
    page_size = 2 * 1024 * 1024
    num_layers = 2
    num_kv_buffers = 2
    vmm_ops.init_kvcached("cuda:0", page_size, contiguous_layout)
    vmm_ops.create_kv_tensors(
        8 * 1024 * 1024,
        2,
        "cuda:0",
        num_layers,
        num_kv_buffers,
        0,
        unified_pool,
    )
    stride = page_size * num_layers * num_kv_buffers if contiguous_layout else page_size
    return [0, stride], 32 * 1024 * 1024


@pytest.mark.parametrize(
    ("contiguous_layout", "unified_pool"),
    [(True, False), (False, True), (False, False)],
    ids=["contiguous", "unified", "per-layer-kv"],
)
@pytest.mark.parametrize("failure_position", [0, 1, 2])
def test_cuda_map_batch_rolls_back_at_each_operation_position(
    contiguous_layout, unified_pool, failure_position
):
    vmm_ops = _compiled_vmm_ops()
    valid_offsets, invalid_offset = _create_layout(vmm_ops, contiguous_layout, unified_pool)
    offsets = list(valid_offsets)
    offsets.insert(failure_position, invalid_offset)

    try:
        with pytest.raises(RuntimeError, match="outside the reserved virtual address"):
            vmm_ops.map_to_kv_tensors_with_result(offsets, 0)

        for offset in valid_offsets:
            assert vmm_ops.map_to_kv_tensors_with_result([offset], 0) == (
                True,
                [offset],
            )
        assert vmm_ops.unmap_from_kv_tensors(valid_offsets, 0)
    finally:
        vmm_ops.shutdown_kvcached()


@pytest.mark.parametrize(
    ("contiguous_layout", "unified_pool"),
    [(True, False), (False, True), (False, False)],
    ids=["contiguous", "unified", "per-layer-kv"],
)
def test_map_rollback_preserves_preexisting_mapping(contiguous_layout, unified_pool):
    vmm_ops = _compiled_vmm_ops()
    valid_offsets, invalid_offset = _create_layout(vmm_ops, contiguous_layout, unified_pool)
    first, second = valid_offsets

    try:
        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [first])
        with pytest.raises(RuntimeError, match="outside the reserved virtual address"):
            vmm_ops.map_to_kv_tensors_with_result([first, second, invalid_offset], 0)

        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [])
        assert vmm_ops.map_to_kv_tensors_with_result([second], 0) == (True, [second])
        assert vmm_ops.unmap_from_kv_tensors(valid_offsets, 0)
    finally:
        vmm_ops.shutdown_kvcached()


@pytest.mark.parametrize(
    ("contiguous_layout", "unified_pool"),
    [(True, False), (False, True), (False, False)],
    ids=["contiguous", "unified", "per-layer-kv"],
)
def test_prepared_unmap_can_abort_or_commit(contiguous_layout, unified_pool):
    vmm_ops = _compiled_vmm_ops()
    valid_offsets, _invalid_offset = _create_layout(vmm_ops, contiguous_layout, unified_pool)
    first = valid_offsets[0]

    try:
        with pytest.raises(ValueError, match="transaction id must not be empty"):
            vmm_ops.prepare_unmap_from_kv_tensors([first], "", 0)
        with pytest.raises(ValueError, match="transaction id must not be empty"):
            vmm_ops.commit_unmap_from_kv_tensors("", 0)
        with pytest.raises(ValueError, match="transaction id must not be empty"):
            vmm_ops.abort_unmap_from_kv_tensors("", 0)

        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [first])
        assert vmm_ops.prepare_unmap_from_kv_tensors([first], "abort-me", 0)
        with pytest.raises(RuntimeError, match="unmap transaction is pending"):
            vmm_ops.map_to_kv_tensors_with_result([valid_offsets[1]], 0)
        assert vmm_ops.abort_unmap_from_kv_tensors("abort-me", 0)
        assert vmm_ops.abort_unmap_from_kv_tensors("abort-me", 0)
        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [])

        assert vmm_ops.prepare_unmap_from_kv_tensors([first], "commit-me", 0)
        assert vmm_ops.commit_unmap_from_kv_tensors("commit-me", 0)
        assert vmm_ops.commit_unmap_from_kv_tensors("commit-me", 0)
        with pytest.raises(RuntimeError, match="cannot abort a committed"):
            vmm_ops.abort_unmap_from_kv_tensors("commit-me", 0)
        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [first])
        assert vmm_ops.unmap_from_kv_tensors([first], 0)
    finally:
        vmm_ops.shutdown_kvcached()


@pytest.mark.parametrize(
    ("contiguous_layout", "unified_pool"),
    [(True, False), (False, True), (False, False)],
    ids=["contiguous", "unified", "per-layer-kv"],
)
def test_prepared_map_can_abort_or_commit(contiguous_layout, unified_pool):
    vmm_ops = _compiled_vmm_ops()
    valid_offsets, _invalid_offset = _create_layout(
        vmm_ops, contiguous_layout, unified_pool
    )
    first = valid_offsets[0]

    try:
        prepared = vmm_ops.prepare_map_to_kv_tensors("abort-me", [first], 0)
        assert prepared["success"]
        assert vmm_ops.has_prepared_map("abort-me", 0)
        assert vmm_ops.abort_prepared_map("abort-me", 0)
        assert not vmm_ops.has_prepared_map("abort-me", 0)

        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [first])
        assert vmm_ops.unmap_from_kv_tensors([first], 0)

        prepared = vmm_ops.prepare_map_to_kv_tensors("commit-me", [first], 0)
        assert prepared["success"]
        committed = vmm_ops.commit_prepared_map("commit-me", 0)
        assert committed["success"]
        assert not vmm_ops.has_prepared_map("commit-me", 0)
        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [])
        assert vmm_ops.unmap_from_kv_tensors([first], 0)
    finally:
        vmm_ops.shutdown_kvcached()


@pytest.mark.parametrize("ordered", [False, True])
def test_unmap_can_release_disjoint_offset_during_background_prepare(ordered):
    vmm_ops = _compiled_vmm_ops()
    offsets, _ = _create_layout(vmm_ops, False, False)
    first, second = offsets
    try:
        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [first])
        assert vmm_ops.prepare_map_to_kv_tensors("background", [second], 0)["success"]
        with pytest.raises(RuntimeError, match="prepared map"):
            vmm_ops.unmap_from_kv_tensors([second], 0)
        if ordered:
            assert vmm_ops.prepare_unmap_from_kv_tensors([first], "release", 0)
            assert vmm_ops.commit_unmap_from_kv_tensors("release", 0)
        else:
            assert vmm_ops.unmap_from_kv_tensors([first], 0)
        assert vmm_ops.commit_prepared_map("background", 0)["success"]
        assert vmm_ops.map_to_kv_tensors_with_result([second], 0) == (True, [])
        assert vmm_ops.unmap_from_kv_tensors([second], 0)
    finally:
        vmm_ops.shutdown_kvcached()


def test_prepared_map_excludes_competing_map_on_same_offset():
    vmm_ops = _compiled_vmm_ops()
    offsets, _ = _create_layout(vmm_ops, False, False)
    first, second = offsets
    try:
        assert vmm_ops.prepare_map_to_kv_tensors("owner", [first], 0)["success"]
        assert vmm_ops.prepare_map_to_kv_tensors("owner", [first], 0)["success"]
        with pytest.raises(RuntimeError, match="prepared map"):
            vmm_ops.prepare_map_to_kv_tensors("competitor", [first], 0)
        with pytest.raises(RuntimeError, match="prepared map"):
            vmm_ops.map_to_kv_tensors_with_result([first], 0)
        assert not vmm_ops.has_prepared_map("competitor", 0)
        assert vmm_ops.map_to_kv_tensors_with_result([second], 0) == (True, [second])
        assert vmm_ops.commit_prepared_map("owner", 0)["success"]
        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [])
        assert vmm_ops.unmap_from_kv_tensors(offsets, 0)
    finally:
        vmm_ops.shutdown_kvcached()


def test_prepared_map_adopts_existing_mapping_without_capacity_check():
    vmm_ops = _compiled_vmm_ops()
    valid_offsets, _invalid_offset = _create_layout(vmm_ops, False, False)
    first = valid_offsets[0]

    try:
        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [first])
        prepared = vmm_ops.prepare_map_to_kv_tensors("adopt-existing", [first], 0)
        assert prepared["success"]
        assert prepared["capacity_checks"] == 0
        assert prepared["required_bytes"] == 0
        assert vmm_ops.commit_prepared_map("adopt-existing", 0)["success"]
        assert vmm_ops.unmap_from_kv_tensors([first], 0)
    finally:
        vmm_ops.shutdown_kvcached()


@pytest.mark.parametrize(
    ("contiguous_layout", "unified_pool"),
    [(True, False), (False, True), (False, False)],
)
def test_prepared_map_deduplicates_offsets_before_reserving(
    contiguous_layout, unified_pool
):
    vmm_ops = _compiled_vmm_ops()
    offsets, _ = _create_layout(vmm_ops, contiguous_layout, unified_pool)
    first, second = offsets
    try:
        prepared = vmm_ops.prepare_map_to_kv_tensors(
            "duplicate", [first, first, second, second], 0
        )
        assert prepared["success"]
        assert prepared["offsets_count"] == 2
        assert prepared["targets_count"] == (2 if contiguous_layout else 4 if unified_pool else 8)
        assert vmm_ops.prepare_map_to_kv_tensors(
            "duplicate", offsets, 0
        )["success"]
        assert vmm_ops.commit_prepared_map("duplicate", 0)["success"]
        assert vmm_ops.map_to_kv_tensors_with_result(offsets, 0) == (True, [])
        assert vmm_ops.unmap_from_kv_tensors(offsets, 0)
    finally:
        vmm_ops.shutdown_kvcached()


def test_physical_growth_rejects_non_finite_utilization(monkeypatch):
    vmm_ops = _compiled_vmm_ops()
    valid_offsets, _invalid_offset = _create_layout(vmm_ops, False, False)
    monkeypatch.setenv("KVCACHED_GPU_UTILIZATION", "nan")

    try:
        prepared = vmm_ops.prepare_map_to_kv_tensors(
            "invalid-limit", [valid_offsets[0]], 0
        )
        assert not prepared["success"]
        assert not vmm_ops.has_prepared_map("invalid-limit", 0)
    finally:
        vmm_ops.shutdown_kvcached()


def test_physical_growth_lock_rejects_symlink(monkeypatch, tmp_path):
    vmm_ops = _compiled_vmm_ops()
    valid_offsets, _invalid_offset = _create_layout(vmm_ops, False, False)
    monkeypatch.setenv("KVCACHED_PHYSICAL_GROWTH_LOCK_DIR", str(tmp_path))
    device_id = vmm_ops.current_device_pci_bus_id()
    safe_device_id = "".join(char if char.isalnum() else "_" for char in device_id)
    lock_path = tmp_path / f"kvcached-physical-growth-{safe_device_id}.lock.budget-v2"
    lock_path.symlink_to(tmp_path / "unrelated")

    try:
        prepared = vmm_ops.prepare_map_to_kv_tensors(
            "symlink-lock", [valid_offsets[0]], 0
        )
        assert not prepared["success"]
        assert not vmm_ops.has_prepared_map("symlink-lock", 0)
    finally:
        vmm_ops.shutdown_kvcached()


def test_shared_budget_covers_local_prepared_growth_and_multiple_pools(monkeypatch, tmp_path):
    import torch

    native = _compiled_vmm_ops()
    page = 2 * 1024 * 1024
    monkeypatch.setenv("KVCACHED_PHYSICAL_GROWTH_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("KVCACHED_IPC_NAME", "shared-budget-two-pools")
    native.init_kvcached("cuda:0", page, False)
    try:
        first = native.create_kv_tensors(4 * page, 1, "cuda:0", 1, 1, 0)
        second = native.create_kv_tensors(4 * page, 1, "cuda:0", 1, 1, 1000)
        assert native.set_instance_budget_limit(2 * page, 1, 0) == "applied"
        assert native.physical_budget_state(1000)["instance_limit_bytes"] == 2 * page
        assert native.map_to_kv_tensors_with_result([0, 0], 0) == (True, [0])
        assert native.physical_budget_state()["charged_bytes"] == page
        assert native.prepare_map_to_kv_tensors("second", [0], 1000)["success"]
        assert native.prepare_map_to_kv_tensors("second", [0], 1000)["success"]
        assert native.physical_budget_state()["charged_bytes"] == 2 * page
        denied = native.prepare_map_to_kv_tensors("denied", [page], 0)
        assert not denied["success"] and denied["capacity_rejections"] == 1
        with pytest.raises(RuntimeError, match="capacity_exhausted"):
            native.map_to_kv_tensors_with_result([page], 0)
        assert native.commit_prepared_map("second", 1000)["success"]
        assert native.physical_budget_state()["charged_bytes"] == 2 * page
        assert native.map_to_kv_tensors_with_result([0], 0) == (True, [])
        for value, tensors in enumerate([first, second], 10):
            tensors[0][:128].fill_(value)
        torch.cuda.synchronize()
        assert (first[0][:128] == 10).all().item()
        assert (second[0][:128] == 11).all().item()
        assert native.unmap_from_kv_tensors([0], 0)
        assert native.physical_budget_state()["charged_bytes"] == page
        assert native.prepare_map_to_kv_tensors("abort", [page], 0)["success"]
        assert native.abort_prepared_map("abort", 0)
        assert native.physical_budget_state()["charged_bytes"] == page
        assert native.unmap_from_kv_tensors([0], 1000)
        assert native.physical_budget_state()["charged_bytes"] == 0
        assert native.recalibrate_physical_budget()
    finally:
        native.shutdown_kvcached()


def test_shared_budget_calibration_observes_external_cuda_allocation(monkeypatch, tmp_path):
    import torch

    native = _compiled_vmm_ops()
    page = 2 * 1024 * 1024
    monkeypatch.setenv("KVCACHED_PHYSICAL_GROWTH_LOCK_DIR", str(tmp_path))
    native.init_kvcached("cuda:0", page, False)
    external = None
    try:
        native.create_kv_tensors(4 * page, 1, "cuda:0", 1, 1, 0)
        assert native.prepare_map_to_kv_tensors("resident", [0], 0)["success"]
        before = native.physical_budget_state()
        assert before["charged_bytes"] == page
        external = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        external.fill_(1)
        torch.cuda.synchronize()
        assert native.recalibrate_physical_budget()
        after = native.physical_budget_state()
        assert after["budget_bytes"] < before["budget_bytes"]
        assert after["charged_bytes"] == page
        del external
        external = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        assert native.recalibrate_physical_budget()
        recovered = native.physical_budget_state()
        assert recovered["charged_bytes"] == page
        assert recovered["budget_bytes"] >= after["budget_bytes"] + 32 * 1024 * 1024
        assert native.abort_prepared_map("resident", 0)
        assert native.physical_budget_state()["charged_bytes"] == 0
    finally:
        del external
        torch.cuda.empty_cache()
        native.shutdown_kvcached()


def test_shared_budget_calibration_preserves_headroom_deficit(monkeypatch, tmp_path):
    native = _compiled_vmm_ops()
    page = 2 * 1024 * 1024
    monkeypatch.setenv("KVCACHED_PHYSICAL_GROWTH_LOCK_DIR", str(tmp_path))
    native.init_kvcached("cuda:0", page, False)
    try:
        native.create_kv_tensors(4 * page, 1, "cuda:0", 1, 1, 0)
        assert native.prepare_map_to_kv_tensors("resident-deficit", [0], 0)["success"]
        # A near-total-device safety margin exceeds observed free memory even
        # after this small KV allocation is released; it must not be regranted.
        monkeypatch.setenv("KVCACHED_GPU_UTILIZATION", "1e-20")
        assert native.recalibrate_physical_budget()
        assert native.physical_budget_state()["budget_bytes"] == 0
        assert native.abort_prepared_map("resident-deficit", 0)
        denied = native.prepare_map_to_kv_tensors("retry-deficit", [0], 0)
        assert not denied["success"] and denied["capacity_rejections"] == 1
    finally:
        native.shutdown_kvcached()


def test_page_availability_observes_instance_budget(monkeypatch, tmp_path):
    import uuid

    from kvcached.vmm_ops import PageAllocator

    native = _compiled_vmm_ops()
    page = 2 * 1024 * 1024
    monkeypatch.setenv("KVCACHED_PHYSICAL_GROWTH_LOCK_DIR", str(tmp_path))
    native.init_kvcached("cuda:0", page, False)
    allocator = None
    try:
        native.create_kv_tensors(8 * page, 1, "cuda:0", 1, 1, 0)
        allocator = PageAllocator(
            num_layers=1, mem_size_per_layer=8 * page, page_size=page,
            world_size=1, pp_rank=0, async_sched=False, contiguous_layout=False,
            enable_page_prealloc=False, num_kv_buffers=1, group_id=0,
            ipc_name="kvcached-budget-admission-" + uuid.uuid4().hex,
        )
        assert native.set_instance_budget_limit(2 * page, 1) == "applied"
        assert allocator.get_avail_physical_pages() == 2
        assert native.prepare_map_to_kv_tensors("admission-resident", [0], 0)["success"]
        assert allocator.get_avail_physical_pages() == 1
        assert native.set_instance_budget_limit(0, 2) == "applied"
        assert allocator.get_avail_physical_pages() == 0
        assert native.abort_prepared_map("admission-resident", 0)
        assert allocator.get_avail_physical_pages() == 0
        assert native.set_instance_budget_limit(3 * page, 3) == "applied"
        assert allocator.get_avail_physical_pages() == 3
    finally:
        del allocator
        native.shutdown_kvcached()


@pytest.mark.parametrize("page", [2 * 1024 * 1024, 4 * 1024 * 1024])
@pytest.mark.parametrize("contiguous,buffers", [(False, 1), (False, 2), (True, 1), (True, 2)])
def test_local_growth_reserves_pool_geometry_and_preserves_zero_cost_reuse(
    monkeypatch, tmp_path, contiguous, buffers, page
):
    import uuid

    import torch

    native = _compiled_vmm_ops()
    monkeypatch.setenv("KVCACHED_PHYSICAL_GROWTH_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("KVCACHED_IPC_NAME", "geometry-" + uuid.uuid4().hex)
    native.init_kvcached("cuda:0", 2 * 1024 * 1024, contiguous)
    try:
        tensors = native.create_kv_tensors(8 * page, 1, "cuda:0", 2, buffers, page_size=page)
        allocator = native.PageAllocator(
            2, 8 * page, page, contiguous_layout=contiguous,
            enable_page_prealloc=False, num_kv_buffers=buffers,
            ipc_name="LOCAL_" + uuid.uuid4().hex[:8],
        )
        assert native.set_instance_budget_limit(0, 1) == "applied"
        before = allocator.get_num_free_pages()
        with pytest.raises(RuntimeError, match="capacity_exhausted"):
            allocator.alloc_page()
        assert allocator.get_num_free_pages() == before
        assert allocator.get_transaction_state()["state"] == "HEALTHY"

        charge = page * 2 * buffers
        assert native.set_instance_budget_limit(2 * charge, 2) == "applied"
        first = allocator.alloc_page()
        assert first.page_id == 0
        for tensor in tensors:
            tensor[:128].fill_(7)
        torch.cuda.synchronize()
        assert native.physical_budget_state()["charged_bytes"] == charge

        # A mapped logical offset costs zero even when admission cannot run.
        offset = page * 2 * buffers if contiguous else page
        assert native.map_to_kv_tensors_with_result([offset], 0) == (True, [offset])
        assert native.physical_budget_state()["charged_bytes"] == 2 * charge
        assert native.set_instance_budget_limit(0, 3) == "applied"
        second = allocator.alloc_page()
        assert second.page_id == 1
        assert native.physical_budget_state()["charged_bytes"] == 2 * charge
        for tensor in tensors:
            assert torch.all(tensor[:128] == 7).item()
        torch.cuda.synchronize()
    finally:
        native.shutdown_kvcached()


def _local_growth_worker(index, lock_dir, ready, start, release, results):
    import os

    import torch

    from kvcached import vmm_ops as native

    os.environ["KVCACHED_PHYSICAL_GROWTH_LOCK_DIR"] = lock_dir
    os.environ["KVCACHED_IPC_NAME"] = "local-race-shared-instance"
    torch.cuda.set_device(0)
    page = 64 * 1024 * 1024
    native.init_kvcached("cuda:0", page, False)
    tensors = native.create_kv_tensors(4 * page, 1, "cuda:0", 1, 1, page_size=page)
    allocator = native.PageAllocator(
        1, 4 * page, page, contiguous_layout=False, enable_page_prealloc=False,
        num_kv_buffers=1, ipc_name=f"LOCAL_RACE_{os.getpid()}",
    )
    if index == 0:
        assert native.set_instance_budget_limit(page, 1) == "applied"
    ready.put(index)
    assert start.wait(30)
    try:
        try:
            allocated = allocator.alloc_page()
        except RuntimeError as exc:
            assert "capacity_exhausted" in str(exc)
            assert allocator.get_num_free_pages() == 4
            assert allocator.get_transaction_state()["state"] == "HEALTHY"
            results.put((index, "rejected"))
        else:
            tensors[0][:128].fill_(index + 1)
            torch.cuda.synchronize()
            results.put((index, "allocated"))
            assert release.wait(30)
            assert torch.all(tensors[0][:128] == index + 1).item()
            allocator.free_page(allocated.page_id)
    finally:
        native.shutdown_kvcached()


def test_local_processes_share_one_physical_admission_budget(tmp_path):
    """Independent processes share one instance limit sufficient for one page."""
    import multiprocessing

    _compiled_vmm_ops()
    context = multiprocessing.get_context("spawn")
    ready, results = context.Queue(), context.Queue()
    start, release = context.Event(), context.Event()
    processes = [context.Process(target=_local_growth_worker, args=(
        index, str(tmp_path), ready, start, release, results,
    )) for index in range(2)]
    try:
        for process in processes:
            process.start()
        assert {ready.get(timeout=60) for _ in processes} == {0, 1}
        start.set()
        outcomes = [results.get(timeout=60) for _ in processes]
        assert sorted(outcome for _, outcome in outcomes) == ["allocated", "rejected"]
        release.set()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
    finally:
        release.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=10)
