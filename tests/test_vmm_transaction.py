# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import pytest


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
        "unmap_from_kv_tensors",
        "prepare_unmap_from_kv_tensors",
        "commit_unmap_from_kv_tensors",
        "abort_unmap_from_kv_tensors",
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
