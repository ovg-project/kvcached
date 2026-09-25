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
    try:
        allocator.start_prealloc_thread()
        assert entered.wait(10), "preallocator did not exercise the callback"
        with pytest.raises(ipc.MapTransactionOutcomeUnknownError, match="background"):
            manager.available_size()
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
    lock_path = tmp_path / f"kvcached-physical-growth-{safe_device_id}.lock"
    lock_path.symlink_to(tmp_path / "unrelated")

    try:
        prepared = vmm_ops.prepare_map_to_kv_tensors(
            "symlink-lock", [valid_offsets[0]], 0
        )
        assert not prepared["success"]
        assert not vmm_ops.has_prepared_map("symlink-lock", 0)
    finally:
        vmm_ops.shutdown_kvcached()
