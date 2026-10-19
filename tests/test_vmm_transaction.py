# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import pytest


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
